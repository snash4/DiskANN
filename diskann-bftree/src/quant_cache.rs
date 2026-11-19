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

pub(crate) struct QuantCache {
    code_bytes: usize,
    words_per_slot: usize,
    words: Box<[AtomicU64]>,
    state: Box<[AtomicU64]>,
}

impl QuantCache {
    /// Allocate a cache for `capacity` slots of `code_bytes` bytes each. Memory is zeroed lazily
    /// by the allocator (untouched pages cost nothing until written).
    pub(crate) fn new(capacity: usize, code_bytes: usize) -> diskann::ANNResult<Self> {
        let words_per_slot = code_bytes.div_ceil(8);
        let total_words = capacity.checked_mul(words_per_slot).ok_or_else(|| {
            diskann::ANNError::message("quant cache size overflows usize".to_string())
        })?;
        Ok(Self {
            code_bytes,
            words_per_slot,
            words: zeroed_atomics(total_words),
            state: zeroed_atomics(capacity),
        })
    }

    /// Number of slots.
    pub(crate) fn capacity(&self) -> usize {
        self.state.len()
    }

    /// Bytes of memory reserved for codes and per-slot state.
    pub(crate) fn reserved_bytes(&self) -> usize {
        (self.words.len() + self.state.len()) * std::mem::size_of::<AtomicU64>()
    }

    /// Bytes per code.
    pub(crate) fn code_bytes(&self) -> usize {
        self.code_bytes
    }

    fn slot_words(&self, i: usize) -> &[AtomicU64] {
        let start = i * self.words_per_slot;
        &self.words[start..start + self.words_per_slot]
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
        let state = &self.state[i];
        let words = self.slot_words(i);
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
        loop {
            let s = self.state[i].load(Ordering::Acquire);
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
        let prev = self.lock(i);
        let words = self.slot_words(i);
        for (w, chunk) in words.iter().zip(code.chunks(8)) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            w.store(u64::from_ne_bytes(b), Ordering::Relaxed);
        }
        self.unlock(i, prev, true);
    }

    /// Mark slot `i` deleted. A later read returns [`CacheRead::Deleted`].
    pub(crate) fn delete(&self, i: usize) {
        let prev = self.lock(i);
        self.unlock(i, prev, false);
    }

    /// Acquire the slot's writer bit; returns the state before locking.
    fn lock(&self, i: usize) -> u64 {
        let state = &self.state[i];
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
                // Data stores below must not become visible before the WRITING bit.
                fence(Ordering::Release);
                return s;
            }
        }
    }

    /// Publish the slot with a new sequence number.
    fn unlock(&self, i: usize, prev: u64, live: bool) {
        let seq = (prev >> 2).wrapping_add(1);
        let next = (seq << 2) | if live { LIVE } else { 0 };
        self.state[i].store(next, Ordering::Release);
    }
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

fn zeroed_atomics(n: usize) -> Box<[AtomicU64]> {
    // SAFETY: an all-zero bit pattern is a valid `AtomicU64` (value 0).
    unsafe { Box::<[AtomicU64]>::new_zeroed_slice(n).assume_init() }
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
        // 2 words of codes + 1 state word per slot.
        assert_eq!(c.reserved_bytes(), 10 * 3 * 8);
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
