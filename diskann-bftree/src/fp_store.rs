//! Pluggable full-precision (FP) vector store.
//!
//! The FP vectors of a **quantized** index are read only at rerank (a batched point-lookup by slot)
//! and at a liveness check, and written on insert/replace/delete. That is a key-value access pattern,
//! so the FP store is made pluggable behind [`FpVectorStore`]:
//!
//! - [`BfTreeFpStore`] (default) — FP vectors in a bf-tree `VectorProvider`, exactly as before. The
//!   `fp_backend = bftree` path is byte-for-byte unchanged.
//! - `GarnetFpStore` (a later impl in this module) — FP vectors in a standalone Garnet KV store over
//!   RESP; in this mode bf-tree stores **no** FP vectors.
//!
//! Only the FP store is pluggable. The quantized codes (`quant_vectors`) and the graph
//! (`neighbor_provider`) always stay in bf-tree.
//!
//! The trait is deliberately about **bytes addressed by slot**. Index-wide metadata that the provider
//! historically read off the FP store (`dim`, `total`, `starting_points`) is exposed here too so a
//! non-bf-tree backend can answer it, but persistence (`save`/`load`) is **not** on the trait — it is
//! handled by a backend branch in `SaveWith`/`LoadWith`, because a Garnet backend has no bf-tree
//! snapshot to write (it is repopulated from the WAL).

// Until Phase 2 routes the provider's `full_vectors` access through this trait, nothing here is
// constructed, so `dead_code` fires. Intentional and temporary — removed once `set_element`/`Rerank`
// call the trait.
#![allow(dead_code)]

use diskann::utils::VectorRepr;
use diskann::ANNResult;

use crate::vectors::VectorProvider;
use crate::AccessError;

/// A store of full-precision vectors, addressed by **internal slot** (`id.as_index()`).
///
/// Implementations must be cheap to share across the concurrent read path (`Send + Sync`). The
/// element type `T` is the index's full-precision element (`f32` in the service today).
pub(crate) trait FpVectorStore<T: VectorRepr>: Send + Sync {
    // --- Data plane -------------------------------------------------------------------------------

    /// Fetch one vector by slot. `Err(AccessError::Transient)` means "no live vector here"
    /// (deleted / not yet written); callers treat that as absent, not fatal.
    fn get_vector_sync(&self, i: usize) -> Result<Vec<T>, AccessError>;

    /// Fetch one vector into a caller-provided buffer (avoids allocation on the hot read path).
    fn get_vector_into(&self, i: usize, buffer: &mut [T]) -> Result<(), AccessError>;

    /// Write (or overwrite) the vector at `i`. Used on insert and replace.
    fn set_vector_sync(&self, i: usize, v: &[T]) -> ANNResult<()>;

    /// Remove the vector at `i` (delete, or eviction of a reused slot's prior occupant).
    fn delete_vector(&self, i: usize);

    // --- Metadata (index-wide facts the FP store historically answered) ---------------------------

    /// Vector dimensionality.
    fn dim(&self) -> usize;

    /// Total capacity (max vectors + frozen start points), as the provider counts it.
    fn total(&self) -> usize;

    /// The frozen start-point ids.
    fn starting_points<I: crate::BfTreeId>(&self) -> ANNResult<Vec<I>>;
}

/// The default FP store: full-precision vectors held in a bf-tree `VectorProvider<T>`.
///
/// This is a thin forwarder to today's `VectorProvider`, so `fp_backend = bftree` is byte-identical to
/// the pre-fork behavior. It also exposes the underlying provider (via [`inner`]/[`inner_mut`]) so the
/// bf-tree-specific save/load path (`config()`, `bftree()`, `save_bftree`) can reach it — those are
/// intentionally NOT on the `FpVectorStore` trait (a Garnet backend has none of them).
pub(crate) struct BfTreeFpStore<T: VectorRepr> {
    inner: VectorProvider<T>,
}

impl<T: VectorRepr> BfTreeFpStore<T> {
    pub(crate) fn new(inner: VectorProvider<T>) -> Self {
        Self { inner }
    }

    /// Access the underlying bf-tree vector provider (for save/load and config — bftree mode only).
    pub(crate) fn inner(&self) -> &VectorProvider<T> {
        &self.inner
    }

    /// Mutable access to the underlying provider (construction/load paths).
    pub(crate) fn inner_mut(&mut self) -> &mut VectorProvider<T> {
        &mut self.inner
    }
}

impl<T: VectorRepr> FpVectorStore<T> for BfTreeFpStore<T> {
    fn get_vector_sync(&self, i: usize) -> Result<Vec<T>, AccessError> {
        self.inner.get_vector_sync(i)
    }

    fn get_vector_into(&self, i: usize, buffer: &mut [T]) -> Result<(), AccessError> {
        self.inner.get_vector_into(i, buffer)
    }

    fn set_vector_sync(&self, i: usize, v: &[T]) -> ANNResult<()> {
        self.inner.set_vector_sync(i, v)
    }

    fn delete_vector(&self, i: usize) {
        self.inner.delete_vector(i)
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn total(&self) -> usize {
        self.inner.total()
    }

    fn starting_points<I: crate::BfTreeId>(&self) -> ANNResult<Vec<I>> {
        self.inner.starting_points::<I>()
    }
}
