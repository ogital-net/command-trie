# command-trie

A compact radix trie for command-line tab completion and longest-prefix
matching. Designed for the build-once / query-many lifecycle of a line
editor or REPL dispatcher.

- No dependencies. `#![no_std]` (depends only on `alloc`). `Send + Sync`.
- Frozen trie lives in four contiguous heap allocations (nodes, label
  bytes, child-id lists, and a parallel one-byte-per-edge index for the
  lookup hot path) for cache-friendly traversal.
- `u16`-indexed slabs cap the trie at ~32,767 entries — plenty for any
  realistic CLI — in exchange for a tighter memory layout.
- Lookups are non-allocating; only methods that materialize keys allocate.
- Arbitrary UTF-8 keys: any `&str` is accepted; radix splits are
  char-aligned so labels are always valid UTF-8 (no normalization, no
  case folding, no grapheme awareness).

## Install

```toml
[dependencies]
command-trie = "1"
```

## Quick start

The crate is shaped around a build-once / query-many workload:

1. Construct a [`CommandTrieBuilder`], `insert` your commands.
2. Call [`CommandTrieBuilder::build`] to freeze it into a compact,
   read-only [`CommandTrie`].
3. Query the [`CommandTrie`] repeatedly for exact lookup,
   longest-prefix match against an input line, and completion enumeration.

```rust
use command_trie::CommandTrieBuilder;

let mut b = CommandTrieBuilder::new();
b.insert("commit",  "save changes");
b.insert("command", "shell command");
b.insert("config",  "settings");
let trie = b.build();

// Exact lookup.
assert_eq!(trie.get("commit"), Some(&"save changes"));

// Dispatch: split a typed line at the longest known command.
assert_eq!(
    trie.longest_prefix_match("commit -a"),
    Some(("commit", &"save changes")),
);

// Tab completion: longest unambiguous extension of a typed prefix.
assert_eq!(trie.completion_prefix("co").as_deref(),   Some("co"));
assert_eq!(trie.completion_prefix("comm").as_deref(), Some("comm"));
assert_eq!(trie.count_completions("comm"), 2);
```

## Fish-style TAB handling

`subtrie` returns a view over every entry sharing a prefix and tells the
line editor exactly what to splice into the buffer:

```rust
# use command_trie::CommandTrieBuilder;
# let mut b = CommandTrieBuilder::new();
# b.insert("commit", ()); b.insert("command", ()); b.insert("config", ());
# let trie = b.build();
let sub = trie.subtrie("comma").unwrap();
assert_eq!(sub.extension(), "nd"); // splice this on TAB
assert!(sub.is_unique());          // and stop prompting

let sub = trie.subtrie("co").unwrap();
assert_eq!(sub.extension(), "");   // already at a branch point
assert!(!sub.is_unique());         // show the menu instead
```

## UTF-8

Keys are arbitrary `&str`. The builder splits edge labels on char
boundaries, so every label is itself valid UTF-8 and iteration order is
byte-lexicographic (which equals code-point order for valid UTF-8).

The crate is deliberately byte-oriented beyond that: there is **no**
Unicode normalization (`café` and `cafe\u{0301}` are different keys),
**no** case folding, and **no** grapheme-cluster awareness. Size and
length limits — the `u16`-indexed offsets and the `~32,767` entry cap —
are measured in bytes, not chars, so multi-byte keys consume more of the
budget than ASCII ones.

The empty key `""` is legal; it associates a value with the trie root
and behaves like any other entry under iteration and longest-prefix match.

## API surface

Build phase — [`CommandTrieBuilder<T>`]:

- `insert(&str, T) -> Option<T>`
- `remove(&str) -> Option<T>` (re-merges passthroughs)
- `get` / `contains` / `len` / `is_empty` / `clear`
- `FromIterator<(K, T)>` and `Extend<(K, T)>` for `K: AsRef<str>`
- `build() -> CommandTrie<T>`

Query phase — [`CommandTrie<T>`]:

- `get` / `contains` / `len` / `is_empty`
- `longest_prefix_match(&str) -> Option<(&str, &T)>`
- `contains_prefix(&str) -> bool`
- `completion_prefix(&str) -> Option<String>` (longest unambiguous extension)
- `count_completions(&str) -> usize`
- `completions(&str) -> Vec<(String, &T)>` and
  `for_each_completion(&str, FnMut)`
- `subtrie(&str) -> Option<SubTrie<'_, T>>`
- `iter() / for_each(FnMut)` — alphabetical traversal

`SubTrie<'_, T>` adds `common_prefix`, `extension`, `is_unique`, `value`,
`unique_value`, `len`, `iter`, `for_each`.

See [`examples/tab_completion.rs`](https://github.com/ogital-net/command-trie/blob/main/examples/tab_completion.rs)
and [`examples/command_dispatch.rs`](https://github.com/ogital-net/command-trie/blob/main/examples/command_dispatch.rs).

## Performance vs `radix_trie`

Measured on a 64-entry git-style command corpus (Apple Silicon, release,
criterion 0.8). Full numbers in
[`benches/comparison.rs`](https://github.com/ogital-net/command-trie/blob/main/benches/comparison.rs).

Lookups and prefix queries are several times faster than `radix_trie`:

| operation                          | command-trie | `radix_trie` |
|------------------------------------|-------------:|-------------:|
| `get` (short hit, `"rm"`)          |     12.5 ns  |     35.9 ns  |
| `get` (long hit, `"verify-commit"`)|     18.9 ns  |     74.2 ns  |
| `get` (miss)                       |      3.6 ns  |     29.8 ns  |
| `count_completions("co")`          |     21.6 ns  |    158.4 ns  |
| `count_completions("r")`           |     30.3 ns  |    424.3 ns  |

Heap footprint (measured with `dhat`, see
[`examples/alloc_profile.rs`](https://github.com/ogital-net/command-trie/blob/main/examples/alloc_profile.rs)):

| library          | resident heap | allocations |
|------------------|--------------:|------------:|
| **command-trie** |    **2.7 KB**|       **4** |
| `radix_trie`     |     25.2 KB  |        166  |

The four allocations are the four frozen slabs; `radix_trie` does one
heap allocation per node. At ~2.9× the raw key+value payload, the frozen
trie is close to the floor for a non-compressing trie.

At scale, the `u16`-indexed slabs keep the per-entry cost low: a corpus of
~32,000 realistic command-style keys (avg ~16 bytes, `u32` values) lands at
~711 KB resident — about 1.16× the raw key+value payload — still in four
allocations.

This crate trades flexibility for these wins: the trie is immutable
after `build()` and the public API is read-mostly.
If you need a general-purpose mutable trie, prefer `radix_trie`.

## Scope

This crate is intentionally read-mostly. The frozen trie exposes no
mutation surface; the builder exposes only `insert` / `remove` / `clear`.
There is no `iter_mut` / `get_mut` / `keys` / `values`.

The frozen trie's internal offsets are `u16`, which caps the trie at
roughly **32,767 entries** (worst case: `2N + 1` nodes per `N` entries,
bounded by `u16::MAX = 65,535`). `CommandTrieBuilder::build` panics if a
larger trie is constructed. This is well above the size of any plausible
command set and buys a noticeably tighter memory layout.

## License

BSD-2-Clause. See [`LICENSE`](https://github.com/ogital-net/command-trie/blob/main/LICENSE).
