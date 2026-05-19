#![doc = include_str!("../README.md")]
#![no_std]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;

// ============================================================
// UTF-8 helpers
// ============================================================
//
// All keys enter the trie as `&str`, i.e. valid UTF-8. The builder splits
// edges only on **char boundaries** (see `lcp` and `BuilderNode::child_index`
// below), so every edge label — and therefore every concatenation of edge
// labels reachable from a frozen node — is itself valid UTF-8. The helpers
// below centralize this invariant; read paths use them in place of a checked
// `from_utf8(...).expect(...)` to skip a redundant validation pass.

/// # Safety
/// Caller must ensure `bytes` is either an edge label produced by
/// `CommandTrieBuilder::build` or a concatenation of such labels following
/// parent → child edges. Both are guaranteed valid UTF-8 by the char-aligned
/// split invariant.
#[inline]
unsafe fn utf8_str_unchecked(bytes: &[u8]) -> &str {
    debug_assert!(core::str::from_utf8(bytes).is_ok());
    // SAFETY: char-aligned splits keep every edge label valid UTF-8; the
    // concatenation of valid UTF-8 fragments is itself valid UTF-8.
    unsafe { core::str::from_utf8_unchecked(bytes) }
}

/// # Safety
/// Same contract as [`utf8_str_unchecked`].
#[inline]
unsafe fn utf8_string_unchecked(bytes: Vec<u8>) -> String {
    debug_assert!(core::str::from_utf8(&bytes).is_ok());
    // SAFETY: see [`utf8_str_unchecked`].
    unsafe { String::from_utf8_unchecked(bytes) }
}

/// Byte length of the UTF-8 character whose leading byte is `b`. For valid
/// UTF-8 leading bytes the result is 1..=4. For continuation bytes (`0b10xxxxxx`)
/// or otherwise-invalid leading bytes the result is 1, which causes lookup to
/// fall through without panicking — invalid bytes can't be in any stored key.
#[inline]
fn utf8_char_len(b: u8) -> usize {
    // ASCII or continuation byte: 1 (continuation is invalid as a leader,
    // but returning 1 makes the lookup fall through without panicking).
    if b < 0xC0 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

// ============================================================
// Builder
// ============================================================

/// Mutable trie used during the build phase.
///
/// Call [`Self::build`] to produce a frozen [`CommandTrie`] when you are
/// done inserting.
#[derive(Clone)]
pub struct CommandTrieBuilder<T> {
    root: BuilderNode<T>,
    len: usize,
}

#[derive(Clone)]
struct BuilderNode<T> {
    /// Bytes consumed by the edge from this node's parent. Empty only for the root.
    label: Box<[u8]>,
    value: Option<T>,
    /// Children, kept sorted by `label[0]`. First bytes are unique among siblings.
    children: Vec<BuilderNode<T>>,
}

impl<T> Default for CommandTrieBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CommandTrieBuilder<T> {
    /// Create a new, empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: BuilderNode {
                label: Box::from(&[][..]),
                value: None,
                children: Vec::new(),
            },
            len: 0,
        }
    }

    /// Number of entries currently in the builder.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no entries have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Remove every entry, returning the builder to its initial state.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Insert `key` with `value`. Returns the previous value, if any.
    ///
    /// Any `&str` is accepted; the radix splits on UTF-8 char boundaries so
    /// every internal edge label remains valid UTF-8.
    pub fn insert(&mut self, key: &str, value: T) -> Option<T> {
        let prev = self.root.insert(key.as_bytes(), value);
        if prev.is_none() {
            self.len += 1;
        }
        prev
    }

    /// Remove `key` and return its value, if present. Maintains the radix
    /// invariant by pruning empty branches and merging single-child
    /// passthroughs with their parent edge.
    pub fn remove(&mut self, key: &str) -> Option<T> {
        let v = self.root.remove(key.as_bytes())?;
        self.len -= 1;
        Some(v)
    }

    /// Exact lookup against the builder.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&T> {
        let mut node = &self.root;
        let mut rem = key.as_bytes();
        loop {
            if rem.is_empty() {
                return node.value.as_ref();
            }
            let child = node.find_child(rem)?;
            if !rem.starts_with(&child.label) {
                return None;
            }
            rem = &rem[child.label.len()..];
            node = child;
        }
    }

    /// Returns `true` when `key` is present in the builder.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Freeze the builder into a compact, read-only [`CommandTrie`].
    ///
    /// Performs one DFS over the builder trie and writes the result into
    /// four contiguous slabs (`nodes`, `labels`, `children`,
    /// `child_first_bytes`). After this call the frozen trie has exactly
    /// four heap allocations regardless of trie size.
    ///
    /// # Panics
    /// If the trie's total node count, total label bytes, or total edge
    /// count would exceed [`u16::MAX`] (65,535). The frozen representation
    /// stores every offset and node id as a `u16` to halve the structural
    /// slabs. The worst case for `N` inserted entries is `2N + 1` nodes,
    /// so this caps the trie at roughly **32,767 entries** — enough for
    /// any realistic command set.
    pub fn build(self) -> CommandTrie<T> {
        let len = self.len;
        let mut nodes: Vec<FrozenNode<T>> = Vec::new();
        let mut labels: Vec<u8> = Vec::new();
        let mut children: Vec<NodeId> = Vec::new();
        let mut child_first_bytes: Vec<u8> = Vec::new();
        build_visit(
            self.root,
            &mut nodes,
            &mut labels,
            &mut children,
            &mut child_first_bytes,
        );
        CommandTrie {
            nodes: nodes.into_boxed_slice(),
            labels: labels.into_boxed_slice(),
            children: children.into_boxed_slice(),
            child_first_bytes: child_first_bytes.into_boxed_slice(),
            len,
        }
    }
}

/// Pre-order DFS: visit a node, emit its `FrozenNode` slot, reserve a
/// contiguous range of child-id slots, then recurse and fill those slots.
fn build_visit<T>(
    node: BuilderNode<T>,
    nodes: &mut Vec<FrozenNode<T>>,
    labels: &mut Vec<u8>,
    children: &mut Vec<NodeId>,
    child_first_bytes: &mut Vec<u8>,
) -> NodeId {
    let id = u16_or_panic(nodes.len());
    let label_start = u16_or_panic(labels.len());
    let label_len = u16_or_panic(node.label.len());
    labels.extend_from_slice(&node.label);

    nodes.push(FrozenNode {
        label_start,
        label_len,
        children_start: 0,
        children_len: 0,
        value: node.value,
    });

    // Reserve a contiguous run of child-id slots BEFORE recursing, so all of
    // this node's children end up adjacent in `children` regardless of how
    // many descendants each child contributes.
    let n_children = u16_or_panic(node.children.len());
    let children_start = u16_or_panic(children.len());
    for _ in 0..n_children {
        children.push(0);
        child_first_bytes.push(0);
    }
    for (i, child) in node.children.into_iter().enumerate() {
        // Capture the child's first byte BEFORE recursing, since `child` is
        // about to be moved into `build_visit`. Non-root children always have
        // a non-empty label.
        let first = child.label[0];
        let slot = children_start as usize + i;
        let child_id = build_visit(child, nodes, labels, children, child_first_bytes);
        children[slot] = child_id;
        child_first_bytes[slot] = first;
    }

    let n = &mut nodes[id as usize];
    n.children_start = children_start;
    n.children_len = n_children;
    id
}

/// Convert a `usize` index/length to `u16`, panicking with a clear message
/// if the documented size cap (see [`FrozenNode`]) is exceeded. Used during
/// `build` to turn what would otherwise be silent `as u16` truncation into
/// an explicit failure.
#[inline]
fn u16_or_panic(n: usize) -> u16 {
    u16::try_from(n).expect("command-trie size exceeds u16::MAX (see FrozenNode docs)")
}

impl<T> BuilderNode<T> {
    /// Find the child whose label starts with the same UTF-8 char as `rem`.
    ///
    /// `rem` must be non-empty and start with a valid UTF-8 leading byte
    /// (always true when called with a slice of an inserted `&str` or with
    /// remaining query bytes from `Self::get`).
    fn find_child(&self, rem: &[u8]) -> Option<&BuilderNode<T>> {
        let idx = self.child_index(rem).ok()?;
        Some(&self.children[idx])
    }

    /// Binary search for the child whose label starts with the same UTF-8 char
    /// as `rem`, returning the index or where it would be inserted.
    ///
    /// Children are sorted by their first char's UTF-8 bytes; this is identical
    /// to first-byte order for valid UTF-8, so the comparison only needs to
    /// look at each child's leading-char bytes (1..=4 bytes) against the
    /// needle's leading-char bytes.
    ///
    /// ASCII fast path: when the needle's first byte is `< 0x80` it *is* the
    /// whole leading char, and an ASCII first byte is unique among siblings
    /// (char-aligned splits guarantee distinct sibling leading chars, and an
    /// ASCII byte cannot collide with any multi-byte first byte which is
    /// always `>= 0xC0`). A plain `binary_search_by_key` on the first byte
    /// suffices and matches the pre-UTF-8 cost.
    fn child_index(&self, rem: &[u8]) -> Result<usize, usize> {
        let first = rem[0];
        if first < 0x80 {
            return self.children.binary_search_by_key(&first, |c| c.label[0]);
        }
        let needle_len = utf8_char_len(first).min(rem.len());
        let needle = &rem[..needle_len];
        self.children.binary_search_by(|c| {
            let cn = utf8_char_len(c.label[0]).min(c.label.len());
            c.label[..cn].cmp(needle)
        })
    }

    fn insert(&mut self, rem: &[u8], value: T) -> Option<T> {
        if rem.is_empty() {
            return self.value.replace(value);
        }
        match self.child_index(rem) {
            Err(at) => {
                self.children.insert(
                    at,
                    BuilderNode {
                        label: Box::from(rem),
                        value: Some(value),
                        children: Vec::new(),
                    },
                );
                None
            }
            Ok(idx) => {
                let child = &mut self.children[idx];
                let common = lcp(&child.label, rem);
                if common == child.label.len() {
                    return child.insert(&rem[common..], value);
                }
                // Split this edge at `common`. `common` is char-aligned by `lcp`,
                // so both halves are valid UTF-8.
                let old_label = core::mem::replace(&mut child.label, Box::from(&rem[..common]));
                let old_value = child.value.take();
                let old_children = core::mem::take(&mut child.children);
                let existing = BuilderNode {
                    label: Box::from(&old_label[common..]),
                    value: old_value,
                    children: old_children,
                };
                if common == rem.len() {
                    child.value = Some(value);
                    child.children = vec![existing];
                } else {
                    let new_node = BuilderNode {
                        label: Box::from(&rem[common..]),
                        value: Some(value),
                        children: Vec::new(),
                    };
                    // Sort by first byte (equivalent to first-char lex order for
                    // valid UTF-8). The two leading chars differ (otherwise `lcp`
                    // would have extended `common`), so first bytes differ unless
                    // the chars share a leading UTF-8 byte — in which case the
                    // tie is broken by the next byte.
                    child.children = if existing.label[..].cmp(&new_node.label[..])
                        == core::cmp::Ordering::Less
                    {
                        vec![existing, new_node]
                    } else {
                        vec![new_node, existing]
                    };
                }
                None
            }
        }
    }

    fn remove(&mut self, rem: &[u8]) -> Option<T> {
        if rem.is_empty() {
            return self.value.take();
        }
        let idx = self.child_index(rem).ok()?;
        if !rem.starts_with(&self.children[idx].label) {
            return None;
        }
        let label_len = self.children[idx].label.len();
        let removed = self.children[idx].remove(&rem[label_len..])?;

        let child = &self.children[idx];
        if child.value.is_none() {
            if child.children.is_empty() {
                self.children.remove(idx);
            } else if child.children.len() == 1 {
                let mut removed_child = self.children.remove(idx);
                let mut grandchild = removed_child.children.pop().unwrap();
                let mut merged =
                    Vec::with_capacity(removed_child.label.len() + grandchild.label.len());
                merged.extend_from_slice(&removed_child.label);
                merged.extend_from_slice(&grandchild.label);
                grandchild.label = merged.into_boxed_slice();
                self.children.insert(idx, grandchild);
            }
        }
        Some(removed)
    }
}

/// Longest common **char-aligned** byte prefix of `a` and `b`.
///
/// Both inputs are assumed to be valid UTF-8 (they come from `&str`s or from
/// previously char-aligned edge labels). The byte-LCP is computed normally,
/// then truncated back to the nearest char boundary. This ensures that any
/// radix split using the returned length never bisects a multi-byte codepoint.
fn lcp(a: &[u8], b: &[u8]) -> usize {
    let mut i = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    // If the first differing byte (or one-past-end in either input) is a
    // UTF-8 continuation byte in `a`, back up to the codepoint boundary.
    // For valid UTF-8 the matching prefix in `b` is identical, so checking
    // either side is sufficient.
    while i > 0 && i < a.len() && (a[i] & 0xC0) == 0x80 {
        i -= 1;
    }
    i
}

impl<K: AsRef<str>, T> FromIterator<(K, T)> for CommandTrieBuilder<T> {
    /// Build a [`CommandTrieBuilder`] from an iterator of `(key, value)` pairs.
    fn from_iter<I: IntoIterator<Item = (K, T)>>(iter: I) -> Self {
        let mut t = Self::new();
        t.extend(iter);
        t
    }
}

impl<K: AsRef<str>, T> Extend<(K, T)> for CommandTrieBuilder<T> {
    /// Insert each `(key, value)` from `iter`.
    fn extend<I: IntoIterator<Item = (K, T)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k.as_ref(), v);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for CommandTrieBuilder<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandTrieBuilder")
            .field("len", &self.len)
            .field("root", &self.root)
            .finish()
    }
}

impl<T: fmt::Debug> fmt::Debug for BuilderNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuilderNode")
            // SAFETY: `label` is a char-aligned slice of an inserted `&str`,
            // so it is valid UTF-8.
            .field("label", &unsafe { utf8_str_unchecked(&self.label) })
            .field("value", &self.value)
            .field("children", &self.children)
            .finish()
    }
}

// ============================================================
// Frozen trie
// ============================================================

type NodeId = u16;
const ROOT: NodeId = 0;

/// Compact, read-only radix trie produced by [`CommandTrieBuilder::build`].
///
/// All storage lives in four boxed slices (`nodes`, `labels`, `children`,
/// `child_first_bytes`). All query methods are non-allocating in their
/// lookup paths; only methods that materialize keys (e.g. `Iter::next`)
/// allocate.
#[derive(Clone)]
pub struct CommandTrie<T> {
    /// `nodes[0]` is the root. Pre-order DFS layout from `build`.
    nodes: Box<[FrozenNode<T>]>,
    /// All edge labels concatenated; sliced by `(label_start, label_len)`.
    labels: Box<[u8]>,
    /// All child-id lists concatenated; each node's children live in a
    /// contiguous run sorted by the first byte of their label.
    children: Box<[NodeId]>,
    /// Parallel to `children`: the first byte of each child's label.
    /// Lets `find_child` binary-search a tight `&[u8]` slice instead of
    /// chasing two pointer indirections per probe (`nodes` → `labels`).
    child_first_bytes: Box<[u8]>,
    len: usize,
}

#[derive(Clone)]
struct FrozenNode<T> {
    // All four offsets/lengths are `u16`; the builder consequently rejects
    // any trie whose total node count, total label bytes, or total edge
    // count would exceed `u16::MAX` (65,535). The worst-case node count
    // for `N` inserted entries is `2N + 1`, so this caps the trie at
    // roughly **32,767 entries** — enough for any realistic command set.
    //
    // Storing the `start + len` pair (rather than just the end and
    // deriving start from the previous node) keeps `label_of` and
    // `children_of` to a single `FrozenNode` load per access, which
    // matters on the lookup hot path.
    label_start: u16,
    label_len: u16,
    children_start: u16,
    children_len: u16,
    value: Option<T>,
}

impl<T> CommandTrie<T> {
    // SAFETY NOTE FOR INTERNAL UNCHECKED ACCESSES
    // -------------------------------------------
    // The following helpers (`label_of`, `children_of`, `value_of`,
    // `find_child`) use `get_unchecked` on `nodes`, `labels`, `children`,
    // and `child_first_bytes`. The build invariants make every index they
    // compute provably in-bounds:
    //
    //   * `nodes` always contains at least the root after `build`, so
    //     `ROOT = 0` is a valid index.
    //   * Every `NodeId` stored in `children` was emitted by `build_visit`
    //     as `nodes.len()` at the moment of the matching `nodes.push`, so
    //     it is always `< nodes.len()` in the finished trie.
    //   * `(label_start, label_len)` and `(children_start, children_len)`
    //     for each node describe ranges that `build_visit` pushed into
    //     `labels` / `children` / `child_first_bytes` *before* writing the
    //     node's fields. They are always in-bounds.
    //   * `child_first_bytes.len() == children.len()` by construction.
    //
    // External index-typed APIs (e.g. `descend_to_node`'s key bytes) are
    // not affected; they still use safe slicing.

    /// Number of entries stored in the trie.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the trie holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn label_of(&self, id: NodeId) -> &[u8] {
        // SAFETY: see the SAFETY NOTE at the top of this impl.
        unsafe {
            let n = self.nodes.get_unchecked(id as usize);
            let start = n.label_start as usize;
            let end = start + n.label_len as usize;
            self.labels.get_unchecked(start..end)
        }
    }

    #[inline]
    fn children_of(&self, id: NodeId) -> &[NodeId] {
        // SAFETY: see the SAFETY NOTE at the top of this impl.
        unsafe {
            let n = self.nodes.get_unchecked(id as usize);
            let start = n.children_start as usize;
            let end = start + n.children_len as usize;
            self.children.get_unchecked(start..end)
        }
    }

    #[inline]
    fn value_of(&self, id: NodeId) -> Option<&T> {
        // SAFETY: see the SAFETY NOTE at the top of this impl.
        unsafe { self.nodes.get_unchecked(id as usize).value.as_ref() }
    }

    /// Returns the child of `parent` whose label starts with the same UTF-8
    /// char as `rem`, if any.
    ///
    /// `rem` must be non-empty and begin with a valid UTF-8 leading byte; this
    /// is always true when called with a slice of a `&str`. The fast path is
    /// pure ASCII: the leading byte is `< 0x80`, unique among siblings, and
    /// binary search on `child_first_bytes` returns the match directly with
    /// no further work — equivalent to the pre-UTF-8 implementation. For
    /// multi-byte chars two siblings can share a leading byte (e.g.
    /// `'é'`=`C3 A9` and `'ê'`=`C3 AA`), so on a hit we widen to the
    /// equal-byte run and check the full leading char against each candidate.
    #[inline]
    fn find_child(&self, parent: NodeId, rem: &[u8]) -> Option<NodeId> {
        // SAFETY: see the SAFETY NOTE at the top of this impl.
        unsafe {
            let n = self.nodes.get_unchecked(parent as usize);
            let start = n.children_start as usize;
            let end = start + n.children_len as usize;
            let first = *rem.get_unchecked(0);
            let slab = self.child_first_bytes.get_unchecked(start..end);
            // Binary-search the dense first-byte slab — typically one cache
            // line — instead of dereferencing `nodes` then `labels` per probe.
            let idx = slab.binary_search(&first).ok()?;
            // Hot path: ASCII first byte is unique among siblings (char-aligned
            // splits guarantee siblings have distinct leading chars, and an
            // ASCII byte cannot share its slot with any other char's leading
            // byte). No UTF-8 work needed.
            if first < 0x80 {
                return Some(*self.children.get_unchecked(start + idx));
            }
            // Cold path: multi-byte char. Disambiguate against the equal-byte
            // run by comparing the full leading-char bytes.
            self.find_child_multibyte(start, slab, idx, first, rem)
        }
    }

    /// Cold-path disambiguation for multi-byte leading chars. Outlined so the
    /// ASCII fast path in `find_child` stays small enough to inline cleanly.
    #[cold]
    #[inline(never)]
    fn find_child_multibyte(
        &self,
        start: usize,
        slab: &[u8],
        idx: usize,
        first: u8,
        rem: &[u8],
    ) -> Option<NodeId> {
        let clen = utf8_char_len(first);
        // `rem` is always a suffix of a valid `&str` cut at an edge-label
        // boundary, which is itself char-aligned. A leader byte at `rem[0]`
        // therefore implies `rem.len() >= clen`. We assert this in debug
        // builds; the SAFETY block below relies on it.
        debug_assert!(rem.len() >= clen);
        // SAFETY: see the SAFETY NOTE at the top of this impl; the debug
        // assertion above and the loop guards make each unchecked index
        // in-bounds.
        unsafe {
            let needle = rem.get_unchecked(..clen);
            // Walk the (usually length-1, rarely up to 4) run of siblings
            // sharing this leading byte, comparing full leading-char bytes.
            let mut lo = idx;
            while lo > 0 && *slab.get_unchecked(lo - 1) == first {
                lo -= 1;
            }
            let mut i = lo;
            while i < slab.len() && *slab.get_unchecked(i) == first {
                let child = *self.children.get_unchecked(start + i);
                let lbl = self.label_of(child);
                if lbl.len() >= clen && lbl.get_unchecked(..clen) == needle {
                    return Some(child);
                }
                i += 1;
            }
            None
        }
    }

    /// Exact lookup.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&T> {
        let mut node = ROOT;
        let mut rem = key.as_bytes();
        loop {
            if rem.is_empty() {
                return self.value_of(node);
            }
            let child = self.find_child(node, rem)?;
            let lbl = self.label_of(child);
            if !rem.starts_with(lbl) {
                return None;
            }
            rem = &rem[lbl.len()..];
            node = child;
        }
    }

    /// Returns `true` when `key` is present in the trie.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Longest stored key that is a prefix of `input`, with its value.
    ///
    /// The returned `&str` is a slice of `input`; its `len()` is the number
    /// of bytes consumed.
    #[must_use]
    pub fn longest_prefix_match<'a>(&self, input: &'a str) -> Option<(&'a str, &T)> {
        let bytes = input.as_bytes();
        let mut node = ROOT;
        let mut consumed = 0usize;
        // Track the deepest ancestor with a value as `(consumed_bytes, value_ref)`
        // so we never need to re-`unwrap` a known-Some lookup at the end.
        let mut best: Option<(usize, &T)> = None;
        loop {
            if let Some(v) = self.value_of(node) {
                best = Some((consumed, v));
            }
            let rem = &bytes[consumed..];
            if rem.is_empty() {
                break;
            }
            let Some(child) = self.find_child(node, rem) else {
                break;
            };
            let lbl = self.label_of(child);
            if !rem.starts_with(lbl) {
                break;
            }
            consumed += lbl.len();
            node = child;
        }
        best.map(|(n, v)| (&input[..n], v))
    }

    /// Returns true if any stored key starts with `prefix`.
    #[must_use]
    pub fn contains_prefix(&self, prefix: &str) -> bool {
        match self.descend_to_node(prefix.as_bytes()) {
            Some(node) => self.value_of(node).is_some() || !self.children_of(node).is_empty(),
            None => false,
        }
    }

    /// Walks down the trie consuming `prefix` and returns the node it lands
    /// on (which may be one whose edge label extends past `prefix`). Allocates
    /// nothing. Used by the hot "does anything match?" / "how many match?"
    /// paths that don't care about reconstructing the prefix string.
    fn descend_to_node(&self, mut rem: &[u8]) -> Option<NodeId> {
        let mut node = ROOT;
        while !rem.is_empty() {
            let child = self.find_child(node, rem)?;
            let lbl = self.label_of(child);
            if rem.len() >= lbl.len() {
                if !rem.starts_with(lbl) {
                    return None;
                }
                rem = &rem[lbl.len()..];
                node = child;
            } else {
                if !lbl.starts_with(rem) {
                    return None;
                }
                node = child;
                break;
            }
        }
        Some(node)
    }

    /// Like [`Self::descend_to_node`] but also returns the bytes from the
    /// trie root to the landing node — needed by [`Self::subtrie`] which
    /// exposes them as `common_prefix`.
    fn descend_to_prefix(&self, mut rem: &[u8]) -> Option<(NodeId, Vec<u8>)> {
        let mut node = ROOT;
        let mut path: Vec<u8> = Vec::with_capacity(rem.len());
        while !rem.is_empty() {
            let child = self.find_child(node, rem)?;
            let lbl = self.label_of(child);
            if rem.len() >= lbl.len() {
                if !rem.starts_with(lbl) {
                    return None;
                }
                path.extend_from_slice(lbl);
                rem = &rem[lbl.len()..];
                node = child;
            } else {
                if !lbl.starts_with(rem) {
                    return None;
                }
                path.extend_from_slice(lbl);
                node = child;
                break;
            }
        }
        Some((node, path))
    }

    /// Iterator over all `(key, value)` pairs in alphabetical order.
    ///
    /// Each item allocates a fresh `String` for the key. For hot loops
    /// prefer [`Self::for_each`], which reuses an internal buffer.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self, ROOT, Vec::new())
    }

    /// Visit every `(key, value)` pair in alphabetical order without
    /// allocating per match.
    pub fn for_each(&self, mut f: impl FnMut(&str, &T)) {
        let mut buf = Vec::new();
        for_each_descendants(self, ROOT, &mut buf, &mut f);
    }

    /// View of the entries whose key starts with `prefix`.
    ///
    /// Returns `None` if no entry has that prefix. Prefer this when you need
    /// to ask multiple questions about the same prefix.
    #[must_use]
    pub fn subtrie<'a>(&'a self, prefix: &str) -> Option<SubTrie<'a, T>> {
        let (mut node, mut path) = self.descend_to_prefix(prefix.as_bytes())?;
        if self.value_of(node).is_none() && self.children_of(node).is_empty() {
            return None;
        }
        // Extend the shared prefix through any unambiguous (single-child,
        // no-value) passthrough chain.
        loop {
            let kids = self.children_of(node);
            if self.value_of(node).is_none() && kids.len() == 1 {
                let child = kids[0];
                path.extend_from_slice(self.label_of(child));
                node = child;
            } else {
                break;
            }
        }
        Some(SubTrie {
            trie: self,
            node,
            query_len: prefix.len(),
            common_prefix: path,
        })
    }

    /// All `(key, value)` pairs whose key starts with `prefix`.
    ///
    /// Allocates a `Vec` and a `String` per match; for hot paths prefer
    /// [`Self::for_each_completion`].
    #[must_use]
    pub fn completions<'a>(&'a self, prefix: &str) -> Vec<(String, &'a T)> {
        match self.subtrie(prefix) {
            Some(sub) => sub.into_iter().collect(),
            None => Vec::new(),
        }
    }

    /// Number of entries whose key starts with `prefix`. Allocation-free.
    #[must_use]
    pub fn count_completions(&self, prefix: &str) -> usize {
        match self.descend_to_node(prefix.as_bytes()) {
            Some(node) => count_values(self, node),
            None => 0,
        }
    }

    /// Longest string `s` such that every key matching `prefix` also starts
    /// with `s`. Always `s.starts_with(prefix)`; may extend past `prefix`
    /// when only one branch is reachable. `None` if no entries match.
    ///
    /// Allocates exactly one `String` (the returned value).
    #[must_use]
    pub fn completion_prefix(&self, prefix: &str) -> Option<String> {
        let mut rem = prefix.as_bytes();
        let mut node = ROOT;
        let mut buf: Vec<u8> = Vec::with_capacity(rem.len());
        // Inline descent so we collect bytes into a single buffer that becomes
        // the returned String -- no intermediate `SubTrie` allocation.
        while !rem.is_empty() {
            let child = self.find_child(node, rem)?;
            let lbl = self.label_of(child);
            if rem.len() >= lbl.len() {
                if !rem.starts_with(lbl) {
                    return None;
                }
                buf.extend_from_slice(lbl);
                rem = &rem[lbl.len()..];
                node = child;
            } else {
                if !lbl.starts_with(rem) {
                    return None;
                }
                buf.extend_from_slice(lbl);
                node = child;
                break;
            }
        }
        if self.value_of(node).is_none() && self.children_of(node).is_empty() {
            return None;
        }
        // Extend through unambiguous passthrough chains.
        while self.value_of(node).is_none() && self.children_of(node).len() == 1 {
            let child = self.children_of(node)[0];
            buf.extend_from_slice(self.label_of(child));
            node = child;
        }
        // SAFETY: `buf` is a concatenation of char-aligned edge labels, all valid UTF-8.
        Some(unsafe { utf8_string_unchecked(buf) })
    }

    /// Visit every completion of `prefix` without allocating per match.
    pub fn for_each_completion(&self, prefix: &str, mut f: impl FnMut(&str, &T)) {
        if let Some(sub) = self.subtrie(prefix) {
            sub.for_each(&mut f);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for CommandTrie<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `child_first_bytes` is a derived lookup index over `children`;
        // intentionally omitted from Debug output.
        f.debug_struct("CommandTrie")
            .field("len", &self.len)
            .field("nodes", &self.nodes.len())
            .field("labels_bytes", &self.labels.len())
            .field("children_edges", &self.children.len())
            .finish_non_exhaustive()
    }
}

impl<T> Default for CommandTrie<T> {
    /// An empty, frozen trie (equivalent to `CommandTrieBuilder::new().build()`).
    fn default() -> Self {
        CommandTrieBuilder::new().build()
    }
}

impl<'a, T> IntoIterator for &'a CommandTrie<T> {
    type Item = (String, &'a T);
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ============================================================
// SubTrie view
// ============================================================

/// View over the subset of a [`CommandTrie`] whose keys share a common prefix.
///
/// Construct via [`CommandTrie::subtrie`]. Guaranteed non-empty.
#[derive(Clone)]
pub struct SubTrie<'a, T> {
    trie: &'a CommandTrie<T>,
    /// Deepest node such that every entry in this view lives in its subtrie.
    node: NodeId,
    /// Length in bytes of the original `prefix` argument to [`CommandTrie::subtrie`].
    /// Always `<= common_prefix.len()`. Used by [`SubTrie::extension`].
    query_len: usize,
    /// Bytes from the trie root to `node`; also the longest common prefix
    /// shared by every entry in this view.
    common_prefix: Vec<u8>,
}

impl<'a, T> SubTrie<'a, T> {
    /// Longest prefix shared by every entry in the view.
    #[must_use]
    pub fn common_prefix(&self) -> &str {
        // SAFETY: `common_prefix` is a concatenation of char-aligned edge labels.
        unsafe { utf8_str_unchecked(&self.common_prefix) }
    }

    /// Bytes between the originally queried prefix and [`Self::common_prefix`].
    ///
    /// This is exactly what a fish-style line editor should splice into the
    /// buffer on TAB: the unambiguous auto-extension implied by what the user
    /// has typed so far. Empty when the typed prefix is already at a branch
    /// point (caller should then display the menu of completions).
    #[must_use]
    pub fn extension(&self) -> &str {
        // SAFETY: `self.query_len` is the byte length of the originally queried
        // `&str`, a char boundary; `common_prefix` is valid UTF-8 in full, so
        // the suffix is also valid UTF-8.
        unsafe { utf8_str_unchecked(&self.common_prefix[self.query_len..]) }
    }

    /// `true` when exactly one entry matches — the caller can commit it and
    /// stop prompting. O(1).
    #[must_use]
    pub fn is_unique(&self) -> bool {
        self.trie.value_of(self.node).is_some() && self.trie.children_of(self.node).is_empty()
    }

    /// Value at this view's deepest node, when that node itself holds a value
    /// (i.e. [`Self::common_prefix`] is itself a stored key). Returns `None`
    /// for a pure branch-point view.
    #[must_use]
    pub fn value(&self) -> Option<&'a T> {
        self.trie.value_of(self.node)
    }

    /// The single matching value when this view contains exactly one entry,
    /// else `None`. Combines [`Self::is_unique`] and [`Self::value`]; spares
    /// the caller a follow-up `trie.get(sub.common_prefix())`.
    #[must_use]
    pub fn unique_value(&self) -> Option<&'a T> {
        if self.is_unique() {
            self.value()
        } else {
            None
        }
    }

    /// Number of entries in the view. Walks the subtrie.
    #[must_use]
    pub fn len(&self) -> usize {
        count_values(self.trie, self.node)
    }

    /// Always `false` — a `SubTrie` only exists when at least one entry matches.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Iterator over `(key, value)` pairs in this view, alphabetically.
    ///
    /// Clones `common_prefix` into the iterator's buffer. If you only walk
    /// the view once, prefer `subtrie.into_iter()`, which moves the buffer
    /// in instead.
    #[must_use]
    pub fn iter(&self) -> Iter<'a, T> {
        Iter::new(self.trie, self.node, self.common_prefix.clone())
    }

    /// Visit every entry in the view without allocating per match.
    pub fn for_each(&self, mut f: impl FnMut(&str, &T)) {
        let mut buf = self.common_prefix.clone();
        // The starting node's label is already in `common_prefix`, so emit
        // its value here and recurse into children directly.
        if let Some(v) = self.trie.value_of(self.node) {
            // SAFETY: `buf` clones `common_prefix`, all char-aligned UTF-8.
            f(unsafe { utf8_str_unchecked(&buf) }, v);
        }
        for &child in self.trie.children_of(self.node) {
            for_each_descendants(self.trie, child, &mut buf, &mut f);
        }
    }
}

impl<'a, T> IntoIterator for &SubTrie<'a, T> {
    type Item = (String, &'a T);
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for SubTrie<'a, T> {
    type Item = (String, &'a T);
    type IntoIter = Iter<'a, T>;
    /// Moves `common_prefix` into the iterator's buffer, avoiding the clone
    /// that `(&subtrie).into_iter()` would perform.
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.trie, self.node, self.common_prefix)
    }
}

impl<T> fmt::Debug for SubTrie<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubTrie")
            .field("common_prefix", &self.common_prefix())
            .field("is_unique", &self.is_unique())
            .finish()
    }
}

// ============================================================
// Iterator
// ============================================================

/// Iterator yielding `(key, value)` pairs from a [`CommandTrie`] or [`SubTrie`].
///
/// Implements [`ExactSizeIterator`]: the remaining count is computed once at
/// construction (one DFS over the starting subtrie) and decremented per item,
/// so `collect::<Vec<_>>()` reserves the exact capacity up front.
pub struct Iter<'a, T> {
    trie: &'a CommandTrie<T>,
    stack: Vec<Frame>,
    path: Vec<u8>,
    /// On the first call, emit the starting node's own value (if any).
    /// Its label is already represented in `path`, so it can't be handled by
    /// the normal `Frame::Enter` step.
    pending_root: Option<NodeId>,
    /// Exact number of `(key, value)` pairs still to be yielded.
    remaining: usize,
}

enum Frame {
    /// Push the node's label onto `path`, then emit its value (if any), then
    /// schedule its children.
    Enter(NodeId),
    /// Pop `usize` bytes from `path`. Width matches `FrozenNode::label_len`;
    /// see the note there about the `u16` size cap.
    Exit(u16),
}

impl<'a, T> Iter<'a, T> {
    fn new(trie: &'a CommandTrie<T>, root: NodeId, initial_path: Vec<u8>) -> Self {
        let mut stack = Vec::new();
        // Push children in reverse so the first child pops first → left-to-right.
        let kids = trie.children_of(root);
        for &child in kids.iter().rev() {
            stack.push(Frame::Enter(child));
        }
        let pending_root = if trie.value_of(root).is_some() {
            Some(root)
        } else {
            None
        };
        let remaining = count_values(trie, root);
        Self {
            trie,
            stack,
            path: initial_path,
            pending_root,
            remaining,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (String, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(id) = self.pending_root.take() {
            let v = self.trie.value_of(id).expect("pending_root has a value");
            self.remaining -= 1;
            // SAFETY: `path` holds the starting node's bytes, a char-aligned
            // concatenation of edge labels — valid UTF-8.
            return Some((unsafe { utf8_string_unchecked(self.path.clone()) }, v));
        }
        while let Some(frame) = self.stack.pop() {
            match frame {
                Frame::Exit(n) => {
                    let new_len = self.path.len() - n as usize;
                    self.path.truncate(new_len);
                }
                Frame::Enter(node) => {
                    let lbl = self.trie.label_of(node);
                    self.path.extend_from_slice(lbl);
                    // SAFETY of cast: `lbl` came from `FrozenNode::label_len: u16`,
                    // so its length always fits in `u16`.
                    self.stack
                        .push(Frame::Exit(u16::try_from(lbl.len()).unwrap()));
                    for &child in self.trie.children_of(node).iter().rev() {
                        self.stack.push(Frame::Enter(child));
                    }
                    if let Some(v) = self.trie.value_of(node) {
                        self.remaining -= 1;
                        // SAFETY: `path` is the concatenation of char-aligned edge labels — valid UTF-8.
                        return Some((unsafe { utf8_string_unchecked(self.path.clone()) }, v));
                    }
                }
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<T> FusedIterator for Iter<'_, T> {}

impl<T> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iter")
            .field("remaining_frames", &self.stack.len())
            .finish()
    }
}

// ============================================================
// Internal walkers
// ============================================================

fn for_each_descendants<T>(
    trie: &CommandTrie<T>,
    node: NodeId,
    buf: &mut Vec<u8>,
    f: &mut impl FnMut(&str, &T),
) {
    let prev = buf.len();
    buf.extend_from_slice(trie.label_of(node));
    if let Some(v) = trie.value_of(node) {
        // SAFETY: `buf` is the concatenation of char-aligned edge labels — valid UTF-8.
        f(unsafe { utf8_str_unchecked(buf) }, v);
    }
    for &child in trie.children_of(node) {
        for_each_descendants(trie, child, buf, f);
    }
    buf.truncate(prev);
}

fn count_values<T>(trie: &CommandTrie<T>, node: NodeId) -> usize {
    let mut n = usize::from(trie.value_of(node).is_some());
    for &child in trie.children_of(node) {
        n += count_values(trie, child);
    }
    n
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::ToString;

    /// Convenience for tests that don't care about the build step.
    fn build_from<'a, I: IntoIterator<Item = (&'a str, i32)>>(items: I) -> CommandTrie<i32> {
        let mut b = CommandTrieBuilder::new();
        for (k, v) in items {
            b.insert(k, v);
        }
        b.build()
    }

    // -------- Builder behavior --------

    #[test]
    fn builder_insert_overwrite_remove() {
        let mut b = CommandTrieBuilder::new();
        assert_eq!(b.insert("commit", 1), None);
        assert_eq!(b.insert("commit", 2), Some(1));
        assert_eq!(b.get("commit"), Some(&2));
        assert!(b.contains("commit"));
        assert_eq!(b.remove("commit"), Some(2));
        assert_eq!(b.remove("commit"), None);
        assert!(b.is_empty());
    }

    #[test]
    fn builder_remove_prunes_and_merges() {
        let mut b = CommandTrieBuilder::new();
        b.insert("command", 1);
        b.insert("commit", 2);
        b.insert("comm", 3);
        assert_eq!(b.remove("comm"), Some(3));
        assert_eq!(b.remove("commit"), Some(2));
        assert_eq!(b.get("command"), Some(&1));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn builder_from_iter_and_extend() {
        let b: CommandTrieBuilder<i32> = [("a", 1), ("ab", 2), ("abc", 3)].into_iter().collect();
        assert_eq!(b.len(), 3);
        assert_eq!(b.get("ab"), Some(&2));
    }

    #[test]
    fn builder_get_diverges_mid_edge() {
        // First byte matches a child but the label diverges: covers
        // the `!rem.starts_with(&child.label)` path in BuilderNode::get.
        let mut b = CommandTrieBuilder::new();
        b.insert("command", 1);
        assert_eq!(b.get("comx"), None);
        assert_eq!(b.get("c"), None);
    }

    // -------- Frozen trie: basic queries --------

    #[test]
    fn frozen_get() {
        let t = build_from([("commit", 1), ("command", 2)]);
        assert_eq!(t.get("commit"), Some(&1));
        assert_eq!(t.get("command"), Some(&2));
        assert_eq!(t.get("comm"), None);
        assert_eq!(t.get("commits"), None);
        assert_eq!(t.get(""), None);
        assert!(t.contains("commit"));
        assert!(!t.contains("comm"));
    }

    #[test]
    fn frozen_query_accepts_non_ascii_keys() {
        // Read methods accept `&str` and must remain safe (no panic, no UB)
        // when queried with multi-byte UTF-8 input against an ASCII-only
        // trie. The queries below simply cannot match any stored edge and
        // should return the empty answer rather than panicking.
        let t = build_from([("commit", 1), ("command", 2), ("config", 3)]);

        // Multi-byte first byte: 0xC3 ('é' starts with 0xC3 0xA9) is > 0x7F
        // and cannot index any child edge.
        assert_eq!(t.get("café"), None);
        assert!(!t.contains("café"));

        // Non-ASCII appearing after a valid ASCII prefix.
        assert_eq!(t.get("commité"), None);
        assert_eq!(t.get("comméand"), None);

        // Pure non-ASCII / emoji.
        assert_eq!(t.get("🦀"), None);
        assert_eq!(t.get("π"), None);

        // Prefix-flavored queries behave the same: no match, no panic.
        assert!(!t.contains_prefix("café"));
        assert_eq!(t.count_completions("café"), 0);
        assert!(t.completions("café").is_empty());
        assert_eq!(t.completion_prefix("café"), None);
        assert!(t.subtrie("café").is_none());
        assert_eq!(t.longest_prefix_match("café"), None);

        // Non-ASCII tail after a real command still finds the command via
        // longest-prefix match -- the ASCII prefix is consumed cleanly and
        // the multi-byte suffix simply fails to extend any edge.
        assert_eq!(t.longest_prefix_match("commit é"), Some(("commit", &1)),);
    }

    #[test]
    fn frozen_empty_trie() {
        let t = CommandTrieBuilder::<i32>::new().build();
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert_eq!(t.get(""), None);
        assert_eq!(t.get("anything"), None);
        assert!(!t.contains_prefix(""));
        assert!(t.subtrie("").is_none());
        assert_eq!(t.completion_prefix(""), None);
        assert_eq!(t.count_completions(""), 0);
        assert!(t.completions("").is_empty());
        assert_eq!(t.iter().count(), 0);
    }

    #[test]
    fn frozen_longest_prefix_match() {
        let t = build_from([("git", 1), ("git-status", 2)]);
        assert_eq!(
            t.longest_prefix_match("git-status --short"),
            Some(("git-status", &2))
        );
        assert_eq!(t.longest_prefix_match("git foo"), Some(("git", &1)));
        assert_eq!(t.longest_prefix_match("git"), Some(("git", &1)));
        assert_eq!(t.longest_prefix_match("gi"), None);
        assert_eq!(t.longest_prefix_match("zzz"), None);
        assert_eq!(t.longest_prefix_match(""), None);
        // Returned slice's len() is the consumed byte count.
        let (matched, _) = t.longest_prefix_match("git-status xyz").unwrap();
        assert_eq!(matched.len(), 10);
    }

    #[test]
    fn frozen_contains_prefix() {
        let t = build_from([("commit", 1)]);
        assert!(t.contains_prefix(""));
        assert!(t.contains_prefix("c"));
        assert!(t.contains_prefix("comm"));
        assert!(t.contains_prefix("commit"));
        assert!(!t.contains_prefix("commits"));
        assert!(!t.contains_prefix("d"));
    }

    // -------- Frozen trie: completions / subtrie --------

    #[test]
    fn frozen_completions() {
        let t = build_from([("commit", 1), ("command", 2), ("config", 3), ("clone", 4)]);
        let mut got = t.completions("comm");
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            got,
            vec![("command".to_string(), &2), ("commit".to_string(), &1)]
        );
        assert_eq!(t.completions("").len(), 4);
        assert!(t.completions("xyz").is_empty());
    }

    #[test]
    fn frozen_completions_prefix_ends_mid_edge() {
        let t = build_from([("command", 1)]);
        let got = t.completions("co");
        assert_eq!(got, vec![("command".to_string(), &1)]);
    }

    #[test]
    fn frozen_completion_prefix_extends_past_query() {
        let t = build_from([("command", 1), ("commit", 2)]);
        assert_eq!(t.completion_prefix("c").as_deref(), Some("comm"));

        let t = build_from([("command", 1)]);
        assert_eq!(t.completion_prefix("").as_deref(), Some("command"));
        assert_eq!(t.completion_prefix("c").as_deref(), Some("command"));
        assert_eq!(t.completion_prefix("commits"), None);
    }

    #[test]
    fn frozen_subtrie_views() {
        let t = build_from([("commit", 1), ("command", 2), ("config", 3)]);
        let sub = t.subtrie("comm").unwrap();
        assert_eq!(sub.common_prefix(), "comm");
        assert_eq!(sub.len(), 2);
        assert!(!sub.is_empty());

        let mut via_iter: Vec<(String, i32)> = sub.iter().map(|(k, v)| (k, *v)).collect();
        via_iter.sort();
        let mut via_for_each: Vec<(String, i32)> = Vec::new();
        sub.for_each(|k, v| via_for_each.push((k.to_string(), *v)));
        via_for_each.sort();
        assert_eq!(via_iter, via_for_each);
        assert_eq!(
            via_iter,
            vec![("command".to_string(), 2), ("commit".to_string(), 1)]
        );

        // by-value IntoIterator
        let owned: Vec<_> = sub.into_iter().collect();
        assert_eq!(owned.len(), 2);
    }

    #[test]
    fn frozen_subtrie_on_exact_leaf() {
        let t = build_from([("commit", 1), ("command", 2)]);
        let sub = t.subtrie("commit").unwrap();
        assert_eq!(sub.common_prefix(), "commit");
        assert_eq!(sub.len(), 1);
    }

    #[test]
    fn frozen_subtrie_extension_and_is_unique() {
        let t = build_from([("commit", 1), ("command", 2), ("config", 3), ("clone", 4)]);

        // Branch point: extension is empty, more than one match.
        let sub = t.subtrie("c").unwrap();
        assert_eq!(sub.common_prefix(), "c");
        assert_eq!(sub.extension(), "");
        assert!(!sub.is_unique());

        // Passthrough extends the LCP past the typed prefix.
        let sub = t.subtrie("co").unwrap();
        assert_eq!(sub.common_prefix(), "co");
        assert_eq!(sub.extension(), "");
        assert!(!sub.is_unique());

        let sub = t.subtrie("comma").unwrap();
        assert_eq!(sub.common_prefix(), "command");
        assert_eq!(sub.extension(), "nd");
        assert!(sub.is_unique());

        // Typing exactly one full command yields a unique view with no extension.
        let sub = t.subtrie("clone").unwrap();
        assert_eq!(sub.extension(), "");
        assert!(sub.is_unique());

        // A value-bearing internal node bounds the LCP at its own key.
        let t2 = build_from([("git", 1), ("github", 2)]);
        let sub = t2.subtrie("gi").unwrap();
        assert_eq!(sub.common_prefix(), "git");
        assert_eq!(sub.extension(), "t");
        assert!(!sub.is_unique()); // "git" itself plus "github"
    }

    #[test]
    fn frozen_subtrie_value_and_unique_value() {
        let t = build_from([("commit", 1), ("command", 2), ("config", 3)]);

        // Branch-point view: no value at the deepest node.
        let sub = t.subtrie("com").unwrap();
        assert_eq!(sub.value(), None);
        assert_eq!(sub.unique_value(), None);

        // Unique view: value() and unique_value() both return the entry.
        let sub = t.subtrie("commi").unwrap();
        assert_eq!(sub.common_prefix(), "commit");
        assert_eq!(sub.value(), Some(&1));
        assert_eq!(sub.unique_value(), Some(&1));

        // Value-bearing internal node: value() exposes it; unique_value() does not.
        let t2 = build_from([("git", 10), ("github", 20)]);
        let sub = t2.subtrie("gi").unwrap();
        assert_eq!(sub.common_prefix(), "git");
        assert_eq!(sub.value(), Some(&10));
        assert_eq!(sub.unique_value(), None);
    }

    #[test]
    fn frozen_iter_alphabetical() {
        let t = build_from([("commit", 1), ("command", 2), ("config", 3), ("clone", 4)]);
        let got: Vec<_> = t.iter().map(|(k, v)| (k, *v)).collect();
        assert_eq!(
            got,
            vec![
                ("clone".to_string(), 4),
                ("command".to_string(), 2),
                ("commit".to_string(), 1),
                ("config".to_string(), 3),
            ]
        );

        // for-loop sugar via IntoIterator for &CommandTrie
        let mut n = 0;
        for _ in &t {
            n += 1;
        }
        assert_eq!(n, 4);
    }

    #[test]
    fn frozen_for_each_no_alloc() {
        let t = build_from([("a", 1), ("ab", 2), ("abc", 3)]);
        let mut got: Vec<(String, i32)> = Vec::new();
        t.for_each(|k, v| got.push((k.to_string(), *v)));
        got.sort();
        assert_eq!(
            got,
            vec![
                ("a".to_string(), 1),
                ("ab".to_string(), 2),
                ("abc".to_string(), 3),
            ]
        );
    }

    #[test]
    fn frozen_count_and_for_each_completion() {
        let t = build_from([("commit", 1), ("command", 2), ("config", 3), ("clone", 4)]);
        assert_eq!(t.count_completions("c"), 4);
        assert_eq!(t.count_completions("comm"), 2);
        assert_eq!(t.count_completions("commit"), 1);
        assert_eq!(t.count_completions("z"), 0);

        let mut got: Vec<String> = Vec::new();
        t.for_each_completion("comm", |k, _| got.push(k.to_string()));
        got.sort();
        assert_eq!(got, vec!["command".to_string(), "commit".to_string()]);
    }

    // -------- Layout invariants --------

    #[test]
    fn build_packs_into_four_allocations() {
        let t = build_from([
            ("add", 0),
            ("alias", 1),
            ("branch", 2),
            ("checkout", 3),
            ("cherry-pick", 4),
            ("clean", 5),
            ("clone", 6),
            ("commit", 7),
            ("command", 8),
            ("config", 9),
        ]);
        // Sanity: round-trip all keys.
        for (i, k) in [
            "add",
            "alias",
            "branch",
            "checkout",
            "cherry-pick",
            "clean",
            "clone",
            "commit",
            "command",
            "config",
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(t.get(k), Some(&(i as i32)));
        }
        // `child_first_bytes` is parallel to `children` and reflects the
        // first byte of each child's label.
        assert_eq!(t.child_first_bytes.len(), t.children.len());
        for (i, &child) in t.children.iter().enumerate() {
            let lbl0 = t.label_of(child)[0];
            assert_eq!(t.child_first_bytes[i], lbl0);
        }
        // Children of any node form a contiguous, sorted-by-first-byte slice.
        for id in 0..t.nodes.len() as NodeId {
            let kids = t.children_of(id);
            for w in kids.windows(2) {
                let a = t.label_of(w[0])[0];
                let b = t.label_of(w[1])[0];
                assert!(a < b, "siblings not sorted at node {id}: {a} >= {b}");
            }
        }
    }

    // -------- Compile-time guarantees --------

    const _: fn() = || {
        fn assert_send_sync<X: Send + Sync>() {}
        assert_send_sync::<CommandTrieBuilder<i32>>();
        assert_send_sync::<CommandTrie<i32>>();
        assert_send_sync::<SubTrie<'static, i32>>();
        assert_send_sync::<Iter<'static, i32>>();
    };

    // -------- Defaults, Debug, misc trait impls --------

    #[test]
    fn builder_default_and_clear() {
        let mut b: CommandTrieBuilder<i32> = CommandTrieBuilder::default();
        assert!(b.is_empty());
        b.insert("a", 1);
        b.insert("ab", 2);
        assert_eq!(b.len(), 2);
        b.clear();
        assert!(b.is_empty());
        assert_eq!(b.get("a"), None);
    }

    #[test]
    fn trie_default_is_empty() {
        let t: CommandTrie<i32> = CommandTrie::default();
        assert!(t.is_empty());
        assert_eq!(t.get("anything"), None);
        assert_eq!(t.iter().count(), 0);
    }

    #[test]
    fn debug_impls_render() {
        let mut b = CommandTrieBuilder::new();
        b.insert("commit", 1);
        b.insert("command", 2);
        // BuilderNode debug runs via the builder's children field.
        let s = format!("{b:?}");
        assert!(s.contains("CommandTrieBuilder"));
        assert!(s.contains("BuilderNode"));

        let t = b.build();
        let s = format!("{t:?}");
        assert!(s.contains("CommandTrie"));
        assert!(s.contains("len"));

        let sub = t.subtrie("comm").unwrap();
        let s = format!("{sub:?}");
        assert!(s.contains("SubTrie"));
        assert!(s.contains("comm"));

        let it = t.iter();
        let s = format!("{it:?}");
        assert!(s.contains("Iter"));
    }

    #[test]
    fn subtrie_ref_into_iter() {
        let t = build_from([("commit", 1), ("command", 2)]);
        let sub = t.subtrie("comm").unwrap();
        // Exercise IntoIterator for &SubTrie (clones common_prefix).
        let from_ref: Vec<_> = (&sub).into_iter().collect();
        assert_eq!(from_ref.len(), 2);
        // sub is still usable afterwards.
        assert_eq!(sub.len(), 2);
    }

    #[test]
    fn subtrie_for_each_emits_starting_node_value() {
        // The starting node itself carries a value: covers the
        // "emit value at root of view" branch in SubTrie::for_each.
        let t = build_from([("git", 10), ("github", 20)]);
        let sub = t.subtrie("git").unwrap();
        let mut got: Vec<(String, i32)> = Vec::new();
        sub.for_each(|k, v| got.push((k.to_string(), *v)));
        got.sort();
        assert_eq!(
            got,
            vec![("git".to_string(), 10), ("github".to_string(), 20)]
        );
    }

    #[test]
    fn insert_32k_entries_no_panic() {
        // Design constraint: the frozen trie's `u16`-indexed slabs cap the
        // worst-case node count at `u16::MAX = 65,535`, which corresponds to
        // roughly `32,767` entries (worst case: `2N + 1` nodes per `N` entries).
        //
        // Build a dense corpus at that size using realistic command-name-style
        // keys (think `print -rl -- ${(k)commands}` on a busy `$PATH`): a few
        // dozen tool prefixes, a vocabulary of subcommand stems, then a
        // numeric suffix to fan out to the target count. Average key length
        // is ~16 bytes, which is closer to real CLI completion sets than
        // the original 3-byte base-62 indices.
        const PREFIXES: &[&str] = &[
            "git-",
            "cargo-",
            "docker-",
            "kubectl-",
            "npm-",
            "pip-",
            "rustup-",
            "systemctl-",
            "journalctl-",
            "ip-",
            "nmcli-",
            "brew-",
        ];
        const STEMS: &[&str] = &[
            "list", "get", "set", "show", "describe", "create", "delete", "update", "apply",
            "watch", "rollout", "exec", "logs", "status", "info", "config", "scale", "patch",
            "expose", "annotate",
        ];
        // 12 prefixes × 20 stems × 134 buckets = 32,160 unique keys.
        const BUCKETS: u32 = 134;
        const N: u32 = PREFIXES.len() as u32 * STEMS.len() as u32 * BUCKETS;
        const _: () = assert!(N >= 32_000, "test corpus must hit the documented ~32k cap");

        fn key(n: u32, buf: &mut String) {
            use core::fmt::Write;
            buf.clear();
            let p = (n as usize) % PREFIXES.len();
            let s = ((n as usize) / PREFIXES.len()) % STEMS.len();
            let bucket = (n as usize) / (PREFIXES.len() * STEMS.len());
            buf.push_str(PREFIXES[p]);
            buf.push_str(STEMS[s]);
            buf.push('-');
            write!(buf, "{bucket:03}").unwrap();
        }

        let mut b: CommandTrieBuilder<u32> = CommandTrieBuilder::new();
        let mut buf = String::new();
        for i in 0..N {
            key(i, &mut buf);
            b.insert(&buf, i);
        }
        assert_eq!(b.len(), N as usize);

        let t = b.build();
        assert_eq!(t.len(), N as usize);

        // Spot-check a handful of lookups across the range.
        for &i in &[0u32, 1, 11, 12, 239, 240, 1023, 12345, N / 2, N - 1] {
            key(i, &mut buf);
            assert_eq!(t.get(&buf), Some(&i), "lookup failed for {i}");
        }
    }

    #[test]
    fn iter_is_fused() {
        fn assert_fused<I: FusedIterator>(_: &I) {}
        let t = CommandTrieBuilder::<i32>::new().build();
        let it = t.iter();
        assert_fused(&it);
    }

    #[test]
    fn iter_is_exact_size() {
        fn assert_exact<I: ExactSizeIterator>(_: &I) {}

        let t = build_from([("commit", 1), ("command", 2), ("config", 3), ("clone", 4)]);

        let it = t.iter();
        assert_exact(&it);
        assert_eq!(it.len(), 4);
        assert_eq!(it.size_hint(), (4, Some(4)));

        // `len` decreases monotonically and reaches 0 at exhaustion.
        let mut it = t.iter();
        for expected in (0..4).rev() {
            assert!(it.next().is_some());
            assert_eq!(it.len(), expected);
            assert_eq!(it.size_hint(), (expected, Some(expected)));
        }
        assert!(it.next().is_none());
        assert_eq!(it.len(), 0);

        // SubTrie iterators report the subtrie's exact size, including a
        // value at the starting node.
        let t2 = build_from([("git", 10), ("github", 20), ("gitlab", 30)]);
        let sub = t2.subtrie("git").unwrap();
        let it = sub.iter();
        assert_eq!(it.len(), 3);
        let collected: Vec<_> = sub.into_iter().collect();
        assert_eq!(collected.len(), 3);

        // Empty trie: len() == 0 up front.
        let empty: CommandTrie<i32> = CommandTrieBuilder::new().build();
        assert_eq!(empty.iter().len(), 0);
    }

    // -------- Fuzz vs BTreeMap (rebuild after every op) --------

    #[test]
    fn fuzz_against_btreemap() {
        use alloc::collections::BTreeMap;

        let mut state: u64 = 0x_dead_beef_cafe_f00d;
        let mut rand = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };

        let keys = [
            "", "a", "ab", "abc", "abd", "abcd", "abce", "b", "ba", "bar", "baz", "c", "co", "com",
            "comm", "command", "commit", "config", "x", "xy", "xyz",
        ];
        let probe_prefixes = [
            "", "a", "ab", "abc", "abz", "b", "c", "comm", "com", "x", "z",
        ];

        let mut builder: CommandTrieBuilder<i32> = CommandTrieBuilder::new();
        let mut model: BTreeMap<String, i32> = BTreeMap::new();

        for op in 0..500 {
            let r = rand();
            let key = keys[(r as usize) % keys.len()];

            if (r >> 32) % 4 == 0 {
                let b_prev = builder.remove(key);
                let m_prev = model.remove(key);
                assert_eq!(b_prev, m_prev, "remove({key:?}) at op {op}");
            } else {
                let v = (r >> 8) as i32;
                let b_prev = builder.insert(key, v);
                let m_prev = model.insert(key.to_string(), v);
                assert_eq!(b_prev, m_prev, "insert({key:?}, {v}) at op {op}");
            }

            assert_eq!(builder.len(), model.len(), "len at op {op}");

            // Rebuild and validate the frozen trie's read API.
            let trie = builder.clone().build();
            assert_eq!(trie.len(), model.len());

            for k in &keys {
                assert_eq!(trie.get(k), model.get(*k), "get({k:?}) at op {op}");
                assert_eq!(trie.contains(k), model.contains_key(*k));
            }

            for pfx in &probe_prefixes {
                let mut from_trie: Vec<String> =
                    trie.completions(pfx).into_iter().map(|(k, _)| k).collect();
                from_trie.sort();
                let mut from_model: Vec<String> = model
                    .keys()
                    .filter(|k| k.starts_with(pfx))
                    .cloned()
                    .collect();
                from_model.sort();
                assert_eq!(from_trie, from_model, "completions({pfx:?}) at op {op}");

                assert_eq!(
                    trie.count_completions(pfx),
                    from_model.len(),
                    "count_completions({pfx:?}) at op {op}"
                );

                if let Some(cp) = trie.completion_prefix(pfx) {
                    assert!(cp.starts_with(pfx));
                    for k in &from_model {
                        assert!(k.starts_with(&cp));
                    }
                } else {
                    assert!(from_model.is_empty());
                }
            }

            let from_trie: Vec<(String, i32)> = trie.iter().map(|(k, v)| (k, *v)).collect();
            let from_model: Vec<(String, i32)> =
                model.iter().map(|(k, v)| (k.clone(), *v)).collect();
            assert_eq!(from_trie, from_model, "iter at op {op}");
        }
    }

    // ---------------------------------------------------------------------
    // UTF-8 tests
    //
    // Splits in the radix trie are char-aligned, so any concatenation of
    // edge labels is valid UTF-8. These tests cover both the easy cases
    // (codepoints whose first byte is unique among siblings) and the
    // harder case where two siblings share a leading byte (e.g. `'é'` and
    // `'ê'`, both starting with `0xC3`), which exercises the
    // equal-byte-run walk in `find_child`.
    // ---------------------------------------------------------------------

    #[test]
    fn utf8_basic_insert_get() {
        let t = build_from([
            ("café", 1),
            ("über", 2),
            ("naïve", 3),
            ("naïveté", 4),
            ("🦀", 5),
            ("π", 6),
        ]);
        assert_eq!(t.get("café"), Some(&1));
        assert_eq!(t.get("über"), Some(&2));
        assert_eq!(t.get("naïve"), Some(&3));
        assert_eq!(t.get("naïveté"), Some(&4));
        assert_eq!(t.get("🦀"), Some(&5));
        assert_eq!(t.get("π"), Some(&6));
        assert_eq!(t.get("cafe"), None);
        assert_eq!(t.get("naïv"), None);
    }

    #[test]
    fn utf8_shared_first_byte_siblings() {
        // 'é' = C3 A9, 'ê' = C3 AA, 'è' = C3 A8 — all share leading byte
        // 0xC3, so the frozen find_child must walk the equal-byte run
        // rather than trust binary_search alone.
        let t = build_from([("éa", 1), ("êb", 2), ("èc", 3), ("ad", 4)]);
        assert_eq!(t.get("éa"), Some(&1));
        assert_eq!(t.get("êb"), Some(&2));
        assert_eq!(t.get("èc"), Some(&3));
        assert_eq!(t.get("ad"), Some(&4));
        assert_eq!(t.get("éb"), None);
        assert_eq!(t.get("ê"), None);
        assert!(t.contains_prefix("é"));
        assert!(t.contains_prefix("ê"));
        assert!(!t.contains_prefix("ë"));
    }

    #[test]
    fn utf8_split_at_shared_codepoint() {
        // Both keys start with 'é' (C3 A9), then diverge at the *next*
        // char. The builder must place the split at byte 2 (after 'é'),
        // not inside it.
        let t = build_from([("éa", 1), ("éb", 2)]);
        assert_eq!(t.get("éa"), Some(&1));
        assert_eq!(t.get("éb"), Some(&2));
        let sub = t.subtrie("é").expect("prefix 'é' should exist");
        assert_eq!(sub.common_prefix(), "é");
        assert_eq!(sub.len(), 2);
    }

    #[test]
    fn utf8_sort_order_matches_btreemap() {
        use alloc::collections::BTreeMap;
        let pairs: Vec<(&str, i32)> = vec![
            ("apple", 1),
            ("café", 2),
            ("cab", 3),
            ("über", 4),
            ("ünder", 5),
            ("naïve", 6),
            ("naive", 7),
            ("🦀rust", 8),
            ("🦀", 9),
            ("π", 10),
            ("zoo", 11),
        ];
        let model: BTreeMap<&str, i32> = pairs.iter().copied().collect();
        let t = build_from(pairs.iter().copied());
        let from_trie: Vec<(String, i32)> = t.iter().map(|(k, v)| (k, *v)).collect();
        let from_model: Vec<(String, i32)> =
            model.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        assert_eq!(from_trie, from_model);
    }

    #[test]
    fn utf8_completion_prefix_extends_through_char() {
        // The only key starting with "n" is "naïveté", so completion_prefix
        // should extend all the way through the multi-byte 'ï'.
        let t = build_from([("naïveté", 1), ("zzz", 2)]);
        assert_eq!(t.completion_prefix("n").as_deref(), Some("naïveté"));
        // Querying with a prefix that ends mid-char is impossible for a
        // `&str` input, but querying with a prefix that ends *exactly* on
        // a char boundary inside the key must work:
        assert_eq!(t.completion_prefix("naï").as_deref(), Some("naïveté"));
    }

    #[test]
    fn utf8_longest_prefix_match() {
        let t = build_from([("café", 1), ("ca", 2)]);
        assert_eq!(t.longest_prefix_match("café au lait"), Some(("café", &1)));
        assert_eq!(t.longest_prefix_match("cab"), Some(("ca", &2)));
        // "caf" sits between "ca" and "café"; only "ca" is a stored key.
        assert_eq!(t.longest_prefix_match("caf"), Some(("ca", &2)));
    }

    #[test]
    fn utf8_remove_and_reinsert() {
        let mut b = CommandTrieBuilder::new();
        b.insert("café", 1);
        b.insert("naïve", 2);
        assert_eq!(b.remove("café"), Some(1));
        assert_eq!(b.get("café"), None);
        assert_eq!(b.get("naïve"), Some(&2));
        b.insert("café", 11);
        let t = b.build();
        assert_eq!(t.get("café"), Some(&11));
        assert_eq!(t.get("naïve"), Some(&2));
    }

    #[test]
    fn utf8_iter_roundtrip_emoji_heavy() {
        let keys = ["🦀", "🦀rust", "🦀🦀", "🔥", "🔥fire", "ascii"];
        let mut b = CommandTrieBuilder::new();
        for (i, k) in keys.iter().enumerate() {
            b.insert(k, i as i32);
        }
        let t = b.build();
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(t.get(k), Some(&(i as i32)), "lookup {k}");
        }
        // round-trip
        let collected: Vec<String> = t.iter().map(|(k, _)| k).collect();
        let mut expected: Vec<String> = keys.iter().map(|s| s.to_string()).collect();
        expected.sort();
        assert_eq!(collected, expected);
    }

    #[test]
    fn utf8_three_byte_char_paths() {
        // '中' = E4 B8 AD — exercises the 3-byte arm of `utf8_char_len`
        // through both the builder cold path and the frozen multi-byte
        // lookup path.
        let mut b = CommandTrieBuilder::new();
        b.insert("中", 1);
        b.insert("中a", 2);
        b.insert("中b", 3);
        b.insert("間", 4);
        let t = b.build();
        assert_eq!(t.get("中"), Some(&1));
        assert_eq!(t.get("中a"), Some(&2));
        assert_eq!(t.get("中b"), Some(&3));
        assert_eq!(t.get("間"), Some(&4));
        assert_eq!(t.get("中c"), None);
        assert_eq!(t.longest_prefix_match("中ax"), Some(("中a", &2)),);
    }

    #[test]
    fn utf8_lcp_backs_off_into_codepoint_boundary() {
        // "Xé" (58 C3 A9) and "Xè" (58 C3 A8) share an ASCII 'X' then
        // diverge inside the second byte of a 2-byte codepoint. The byte
        // LCP is 2, but the char-aligned `lcp` must back off to 1 so the
        // split point lands on a codepoint boundary.
        let mut b = CommandTrieBuilder::new();
        b.insert("Xé", 1);
        b.insert("Xè", 2);
        let t = b.build();
        assert_eq!(t.get("Xé"), Some(&1));
        assert_eq!(t.get("Xè"), Some(&2));
        // The shared edge label must be "X", not "X" + a stray leading
        // byte of the multi-byte char; iteration order confirms both
        // entries are reachable and distinct.
        let keys: Vec<String> = t.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["Xè".to_string(), "Xé".to_string()]);
    }
}
