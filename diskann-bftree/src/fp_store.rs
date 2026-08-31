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

// ================================================================================================
// GarnetFpStore — the KV backend (skeleton)
// ================================================================================================

/// Full-precision vectors in a standalone Garnet KV store (RESP protocol).
///
/// **Skeleton (Phase 2):** the struct and trait impl are complete so `FpStore::Garnet` type-checks,
/// but the RESP client is not wired yet — reads report "vector not found" (transient), writes fail
/// with a clear error. Phase 5 ports the real client (redis crate, MGET/SET/DEL, timeout + circuit
/// breaker — already written and compiling in bftree-stream's `engine/fp/garnet.rs`) and threads the
/// endpoint configuration through `BfTreeProviderParameters`. Nothing constructs this arm until then.
pub(crate) struct GarnetFpStore<T: VectorRepr> {
    /// Vector dimensionality (metadata answered locally; Garnet stores opaque blobs).
    dim: usize,
    /// Capacity bookkeeping mirrored from construction params (Garnet has no notion of capacity).
    max_vectors: usize,
    num_start_points: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: VectorRepr> GarnetFpStore<T> {
    /// Construct the skeleton store. Phase 5 replaces this with a connecting constructor
    /// (endpoints, timeout, breaker) ported from bftree-stream.
    pub(crate) fn new(dim: usize, max_vectors: usize, num_start_points: usize) -> Self {
        Self {
            dim,
            max_vectors,
            num_start_points,
            _marker: std::marker::PhantomData,
        }
    }

    fn unwired() -> diskann::ANNError {
        diskann::ANNError::message(
            "garnet FP backend not wired yet (Phase 5): no RESP client configured".to_string(),
        )
    }
}

impl<T: VectorRepr> FpVectorStore<T> for GarnetFpStore<T> {
    fn get_vector_sync(&self, i: usize) -> Result<Vec<T>, AccessError> {
        // Transient "not found": callers on the read path treat this as an absent/deleted vector.
        Err(diskann::error::RankedError::Transient(crate::VectorUnavailable {
            id: i,
            err: crate::VectorError::NotFound,
        }))
    }

    fn get_vector_into(&self, i: usize, _buffer: &mut [T]) -> Result<(), AccessError> {
        Err(diskann::error::RankedError::Transient(crate::VectorUnavailable {
            id: i,
            err: crate::VectorError::NotFound,
        }))
    }

    fn set_vector_sync(&self, _i: usize, _v: &[T]) -> ANNResult<()> {
        Err(Self::unwired())
    }

    fn delete_vector(&self, _i: usize) {
        // Nothing to delete in the unwired skeleton.
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn total(&self) -> usize {
        self.max_vectors + self.num_start_points
    }

    fn starting_points<I: crate::BfTreeId>(&self) -> ANNResult<Vec<I>> {
        // Start points are stored alongside the vectors in the bf-tree backend; the garnet backend
        // will answer them from the same slot convention (max_vectors..total) in Phase 5.
        Ok(((self.max_vectors)..(self.max_vectors + self.num_start_points))
            .map(I::from_index)
            .collect())
    }
}

// ================================================================================================
// FpStore — the provider's FP field: an internally-dispatching enum over the two backends
// ================================================================================================

/// The provider's full-precision store: bf-tree (default) or Garnet, selected at construction.
///
/// This is a concrete enum rather than `Box<dyn FpVectorStore>` deliberately:
/// - several provider sites read **fields** of the bf-tree `VectorProvider` (`num_start_points`,
///   `max_vectors`, `num_get_calls`) which a trait object cannot expose;
/// - the (not-yet-decoupled) save/load path needs the bf-tree-only `config()`/`bftree()` accessors.
///
/// The enum forwards the data-plane + metadata methods to the active backend, and exposes the
/// bf-tree-only accessors as passthroughs that are valid in the `BfTree` arm and unreachable in the
/// `Garnet` arm — safe because until Phase 4 makes save/load backend-aware, nothing constructs
/// `Garnet` (Phase 5) and the save/load callers are gated to bftree mode.
pub(super) enum FpStore<T: VectorRepr> {
    BfTree(BfTreeFpStore<T>),
    Garnet(GarnetFpStore<T>),
}

impl<T: VectorRepr> FpStore<T> {
    // --- Data plane (dispatch to the active backend) ---------------------------------------------

    pub(super) fn get_vector_sync(&self, i: usize) -> Result<Vec<T>, AccessError> {
        match self {
            FpStore::BfTree(s) => s.get_vector_sync(i),
            FpStore::Garnet(s) => s.get_vector_sync(i),
        }
    }

    pub(super) fn get_vector_into(&self, i: usize, buffer: &mut [T]) -> Result<(), AccessError> {
        match self {
            FpStore::BfTree(s) => s.get_vector_into(i, buffer),
            FpStore::Garnet(s) => s.get_vector_into(i, buffer),
        }
    }

    pub(super) fn set_vector_sync(&self, i: usize, v: &[T]) -> ANNResult<()> {
        match self {
            FpStore::BfTree(s) => s.set_vector_sync(i, v),
            FpStore::Garnet(s) => s.set_vector_sync(i, v),
        }
    }

    pub(super) fn delete_vector(&self, i: usize) {
        match self {
            FpStore::BfTree(s) => s.delete_vector(i),
            FpStore::Garnet(s) => s.delete_vector(i),
        }
    }

    // --- Metadata (dispatch) ---------------------------------------------------------------------

    pub(super) fn dim(&self) -> usize {
        match self {
            FpStore::BfTree(s) => s.dim(),
            FpStore::Garnet(s) => s.dim(),
        }
    }

    pub(super) fn total(&self) -> usize {
        match self {
            FpStore::BfTree(s) => s.total(),
            FpStore::Garnet(s) => s.total(),
        }
    }

    pub(super) fn starting_points<I: crate::BfTreeId>(&self) -> ANNResult<Vec<I>> {
        match self {
            FpStore::BfTree(s) => s.starting_points::<I>(),
            FpStore::Garnet(s) => s.starting_points::<I>(),
        }
    }

    /// Capacity excluding start points (mirrors `VectorProvider.max_vectors`).
    pub(super) fn max_vectors(&self) -> usize {
        match self {
            FpStore::BfTree(s) => s.inner().max_vectors,
            FpStore::Garnet(s) => s.max_vectors,
        }
    }

    /// Number of frozen start points (mirrors `VectorProvider.num_start_points`).
    pub(super) fn num_start_points(&self) -> usize {
        match self {
            FpStore::BfTree(s) => s.inner().num_start_points,
            FpStore::Garnet(s) => s.num_start_points,
        }
    }

    /// Stats counter: number of `get_vector` calls (bf-tree instruments this; garnet reports 0
    /// until Phase 5 adds its own counter).
    pub(super) fn num_get_calls(&self) -> usize {
        match self {
            FpStore::BfTree(s) => s.inner().num_get_calls.get(),
            FpStore::Garnet(_) => 0,
        }
    }

    // --- bf-tree-only accessors (save/load path; valid ONLY in the BfTree arm) -------------------
    //
    // These exist so the not-yet-decoupled save/load path keeps compiling unchanged. Phase 4 makes
    // that path backend-aware, gating these calls to bftree mode; until Phase 5 nothing constructs
    // the Garnet arm, so the unreachable!() cannot fire in practice.

    /// The bf-tree config of the FP store. **bftree backend only.**
    pub(super) fn config(&self) -> &bf_tree::Config {
        match self {
            FpStore::BfTree(s) => s.inner().config(),
            FpStore::Garnet(_) => {
                unreachable!("FP save/load is bf-tree-only until Phase 4; garnet has no bf-tree config")
            }
        }
    }

    /// The underlying bf-tree of the FP store. **bftree backend only.**
    pub(super) fn bftree(&self) -> &bf_tree::BfTree {
        match self {
            FpStore::BfTree(s) => s.inner().bftree(),
            FpStore::Garnet(_) => {
                unreachable!("FP save/load is bf-tree-only until Phase 4; garnet has no bf-tree")
            }
        }
    }
}
