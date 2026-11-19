/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A slot-indexed, in-memory copy of the quantized codes ("quant cache").
//!
//! Searches read one quantized code per distance comparison — thousands per query — and a bf-tree
//! lookup is a chain of dependent memory reads (inner nodes, leaf, binary search, record). This
//! cache holds every code in one flat allocation addressed by slot, so a read is a single
//! computed-address access whose misses can overlap across a hop's neighbors.
//!
//! The bf-tree quant store stays the store of record (snapshots and recovery are unchanged); the
//! cache is written after it on insert, cleared before it on delete, and rebuilt from it on load.
//!
//! # Concurrency
//!
//! Each slot has a 64-bit state word used as a per-slot sequence lock:
//!
//! * bit 0 (`WRITING`): a writer holds the slot,
//! * bit 1 (`LIVE`): the slot holds a code,
//! * bits 2.. : a sequence number incremented by every write or delete.
//!
//! Readers copy the code with relaxed atomic loads and accept it only if the state word is
//! unchanged and not `WRITING` across the copy, so a reader never returns a torn code even while
//! the slot is being rewritten (slot reuse after a delete). Writers to the same slot are serialized
//! by the `WRITING` bit. Code bytes are stored in `AtomicU64` words so that concurrent access is
//! well defined; on x86-64 relaxed loads and stores compile to plain moves.

use std::sync::atomic::{fence, AtomicU64, Ordering};

const WRITING: u64 = 1;
const LIVE: u64 = 2;
const SEQ_ONE: u64 = 4;

/// Words per 64-byte cache line.
const LINE_WORDS: usize = 8;

/// Cache lines to prefetch per slot (the same bound DiskANN's in-memory stores use).
const PREFETCH_LINES: usize = 4;

/// Outcome of a cache read, mirroring the bf-tree read results the provider distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheRead {
    /// The code was copied into the output buffer.
    Found,
    /// The slot has never held a code.
    NotFound,
    /// The slot held a code that was deleted.
    Deleted,
}

/// One 64-byte cache line of storage.
#[repr(C, align(64))]
struct Line([AtomicU64; LINE_WORDS]);

/// Slot layout: every slot starts on a cache-line boundary and holds its state word followed by
/// the code words, so one miss brings in both the state and the start of the code.
///
/// ```text
/// | state | code word 0 | code word 1 | ... | padding to a multiple of 64 bytes |
/// ```
pub(crate) struct QuantCache {
    code_bytes: usize,
    code_words: usize,
    slot_lines: usize,
    capacity: usize,
    lines: Box<[Line]>,
}

impl QuantCache {
    /// Allocate a cache for `capacity` slots of `code_bytes` bytes each. Memory is zeroed lazily
    /// by the allocator (untouched pages cost nothing until written).
    pub(crate) fn new(capacity: usize, code_bytes: usize) -> diskann::ANNResult<Self> {
        let code_words = code_bytes.div_ceil(8);
        let slot_lines = (1 + code_words).div_ceil(LINE_WORDS);
        let total_lines = capacity.checked_mul(slot_lines).ok_or_else(|| {
            diskann::ANNError::message("quant cache size overflows usize".to_string())
        })?;
        Ok(Self {
            code_bytes,
            code_words,
            slot_lines,
            capacity,
            lines: zeroed_lines(total_lines),
        })
    }

    /// Number of slots.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes of memory reserved (codes, state words and alignment padding).
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.lines.len() * std::mem::size_of::<Line>()
    }

    /// Bytes per code.
    pub(crate) fn code_bytes(&self) -> usize {
        self.code_bytes
    }

    /// The words of slot `i`: index 0 is the state, `1..=code_words` the code.
    #[inline]
    fn slot(&self, i: usize) -> &[AtomicU64] {
        let lines = &self.lines[i * self.slot_lines..(i + 1) * self.slot_lines];
        // SAFETY: `Line` is `repr(C)` around `[AtomicU64; LINE_WORDS]` with size exactly
        // `LINE_WORDS * 8` (64) bytes and no padding, so consecutive lines are a contiguous run of
        // `AtomicU64`s that lives as long as `self`.
        unsafe {
            std::slice::from_raw_parts(lines.as_ptr().cast::<AtomicU64>(), lines.len() * LINE_WORDS)
        }
    }

    /// Hint the CPU to start loading slot `i` (state and the first cache lines of the code).
    /// Does nothing if `i` is out of range.
    #[inline]
    pub(crate) fn prefetch(&self, i: usize) {
        if i < self.capacity {
            let lines = &self.lines[i * self.slot_lines..(i + 1) * self.slot_lines];
            diskann_vector::prefetch_hint_max::<PREFETCH_LINES, _>(lines);
        }
    }

    /// Copy slot `i`'s code into `out` (`out.len()` must equal `code_bytes`).
    ///
    /// # Panics
    /// If `i >= capacity()` or `out` has the wrong length.
    #[inline]
    pub(crate) fn read(&self, i: usize, out: &mut [u8]) -> CacheRead {
        assert_eq!(
            out.len(),
            self.code_bytes,
            "quant cache: wrong output length"
        );
        let slot = self.slot(i);
        let state = &slot[0];
        let words = &slot[1..=self.code_words];
        loop {
            let s1 = state.load(Ordering::Acquire);
            if s1 & WRITING != 0 {
                std::hint::spin_loop();
                continue;
            }
            if s1 & LIVE == 0 {
                return if s1 >= SEQ_ONE {
                    CacheRead::Deleted
                } else {
                    CacheRead::NotFound
                };
            }
            copy_out(words, out);
            fence(Ordering::Acquire);
            if state.load(Ordering::Relaxed) == s1 {
                return CacheRead::Found;
            }
        }
    }

    /// Whether slot `i` currently holds a code.
    #[inline]
    pub(crate) fn is_live(&self, i: usize) -> bool {
        let state = &self.slot(i)[0];
        loop {
            let s = state.load(Ordering::Acquire);
            if s & WRITING == 0 {
                return s & LIVE != 0;
            }
            std::hint::spin_loop();
        }
    }

    /// Store `code` in slot `i`, making it live.
    ///
    /// # Panics
    /// If `i >= capacity()` or `code` has the wrong length.
    pub(crate) fn write(&self, i: usize, code: &[u8]) {
        assert_eq!(
            code.len(),
            self.code_bytes,
            "quant cache: wrong code length"
        );
        let slot = self.slot(i);
        let prev = lock(&slot[0]);
        for (w, chunk) in slot[1..=self.code_words].iter().zip(code.chunks(8)) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            w.store(u64::from_ne_bytes(b), Ordering::Relaxed);
        }
        unlock(&slot[0], prev, true);
    }

    /// Mark slot `i` deleted. A later read returns [`CacheRead::Deleted`].
    pub(crate) fn delete(&self, i: usize) {
        let state = &self.slot(i)[0];
        let prev = lock(state);
        unlock(state, prev, false);
    }
}

/// Acquire a slot's writer bit; returns the state before locking.
fn lock(state: &AtomicU64) -> u64 {
    loop {
        let s = state.load(Ordering::Relaxed);
        if s & WRITING != 0 {
            std::hint::spin_loop();
            continue;
        }
        if state
            .compare_exchange_weak(s, s | WRITING, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            // Data stores after this must not become visible before the WRITING bit.
            fence(Ordering::Release);
            return s;
        }
    }
}

/// Publish a slot with a new sequence number.
fn unlock(state: &AtomicU64, prev: u64, live: bool) {
    let seq = (prev >> 2).wrapping_add(1);
    let next = (seq << 2) | if live { LIVE } else { 0 };
    state.store(next, Ordering::Release);
}

#[inline]
fn copy_out(words: &[AtomicU64], out: &mut [u8]) {
    let mut chunks = out.chunks_exact_mut(8);
    let mut it = words.iter();
    for chunk in &mut chunks {
        // `words` has exactly ceil(code_bytes / 8) entries.
        let w = it.next().map_or(0, |w| w.load(Ordering::Relaxed));
        chunk.copy_from_slice(&w.to_ne_bytes());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
        let w = it.next().map_or(0, |w| w.load(Ordering::Relaxed));
        rem.copy_from_slice(&w.to_ne_bytes()[..rem.len()]);
    }
}

fn zeroed_lines(n: usize) -> Box<[Line]> {
    // SAFETY: an all-zero bit pattern is a valid `Line` (eight `AtomicU64`s holding 0).
    unsafe { Box::<[Line]>::new_zeroed_slice(n).assume_init() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    #[test]
    fn read_write_delete_reuse() {
        for code_bytes in [1usize, 7, 8, 13, 48, 392] {
            let c = QuantCache::new(4, code_bytes).unwrap();
            let mut out = vec![0u8; code_bytes];
            assert_eq!(c.read(0, &mut out), CacheRead::NotFound);
            assert!(!c.is_live(0));

            let code: Vec<u8> = (0..code_bytes).map(|b| (b * 7 + 1) as u8).collect();
            c.write(0, &code);
            assert!(c.is_live(0));
            assert_eq!(c.read(0, &mut out), CacheRead::Found);
            assert_eq!(out, code);
            // Neighbouring slots are untouched.
            assert_eq!(c.read(1, &mut out), CacheRead::NotFound);

            c.delete(0);
            assert!(!c.is_live(0));
            assert_eq!(c.read(0, &mut out), CacheRead::Deleted);

            // Reuse the slot with a different code.
            let code2: Vec<u8> = (0..code_bytes).map(|b| (b * 3 + 5) as u8).collect();
            c.write(0, &code2);
            assert_eq!(c.read(0, &mut out), CacheRead::Found);
            assert_eq!(out, code2);

            // Deleting a never-written slot reports Deleted afterwards.
            c.delete(3);
            assert_eq!(c.read(3, &mut out), CacheRead::Deleted);
        }
    }

    #[test]
    fn reserved_bytes_and_capacity() {
        let c = QuantCache::new(10, 13).unwrap();
        assert_eq!(c.capacity(), 10);
        assert_eq!(c.code_bytes(), 13);
        // 1 state word + 2 code words, padded to one 64-byte line per slot.
        assert_eq!(c.reserved_bytes(), 10 * 64);
        // 392-byte codes: 1 + 49 words -> 7 lines (448 bytes) per slot.
        assert_eq!(QuantCache::new(3, 392).unwrap().reserved_bytes(), 3 * 448);
    }

    #[test]
    fn slots_are_cache_line_aligned_and_prefetch_is_safe() {
        let c = QuantCache::new(5, 392).unwrap();
        for i in 0..5 {
            assert_eq!(
                c.slot(i).as_ptr() as usize % 64,
                0,
                "slot {i} not 64-byte aligned"
            );
            c.prefetch(i);
        }
        // Out of range: ignored.
        c.prefetch(5);
        c.prefetch(usize::MAX);
    }

    /// Writers keep rewriting one slot with uniform codes (every byte equal) and deleting it;
    /// readers must only ever see a uniform code, never a mix of two writes.
    #[test]
    fn concurrent_rewrites_are_never_torn() {
        let code_bytes = 392;
        let c = Arc::new(QuantCache::new(2, code_bytes).unwrap());
        let stop = Arc::new(AtomicBool::new(false));

        let writer = {
            let c = c.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut b = 0u8;
                while !stop.load(Ordering::Relaxed) {
                    b = b.wrapping_add(1);
                    c.write(0, &vec![b; code_bytes]);
                    if b.is_multiple_of(5) {
                        c.delete(0);
                    }
                }
            })
        };

        let readers: Vec<_> = (0..3)
            .map(|_| {
                let c = c.clone();
                std::thread::spawn(move || {
                    let mut out = vec![0u8; code_bytes];
                    let mut found = 0usize;
                    for _ in 0..200_000 {
                        if c.read(0, &mut out) == CacheRead::Found {
                            let first = out[0];
                            assert!(out.iter().all(|&x| x == first), "torn read");
                            found += 1;
                        }
                    }
                    found
                })
            })
            .collect();

        let found: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
        stop.store(true, Ordering::Relaxed);
        writer.join().unwrap();
        assert!(found > 0, "readers never saw a live code");
    }
}
