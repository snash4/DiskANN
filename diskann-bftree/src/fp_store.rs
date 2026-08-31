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

use diskann::error::{RankedError, TransientError};
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

    /// Fetch the vectors for `slots` in **one batched call**. `out[i]` corresponds to `slots[i]`:
    /// `Some(vec)` if present, `None` if the slot has no live vector (deleted / not yet written).
    /// `out` is cleared and resized to `slots.len()`.
    ///
    /// This is the rerank read. A KV backend answers it with a single multi-get round-trip; the
    /// bf-tree backend loops internally (identical behavior to per-slot reads). Hard errors (not
    /// per-slot absence) abort the batch.
    fn get_many(&self, slots: &[usize], out: &mut Vec<Option<Vec<T>>>) -> ANNResult<()>;

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

    fn get_many(&self, slots: &[usize], out: &mut Vec<Option<Vec<T>>>) -> ANNResult<()> {
        out.clear();
        out.reserve(slots.len());
        for &i in slots {
            match self.inner.get_vector_sync(i) {
                Ok(v) => out.push(Some(v)),
                Err(RankedError::Transient(t)) => {
                    // Deleted/missing slot: absent, not fatal — same semantics as the previous
                    // per-candidate `allow_transient("stale candidate during rerank")`.
                    t.acknowledge("stale candidate during rerank");
                    out.push(None);
                }
                Err(RankedError::Error(e)) => return Err(e),
            }
        }
        Ok(())
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
// Circuit breaker (std-only; ported from bftree-stream engine/fp/breaker.rs)
// ================================================================================================

/// Consecutive-failure circuit breaker with time-based half-open recovery.
///
/// FP rerank is on the query critical path and, in garnet mode, there is no bf-tree FP fallback —
/// so a slow/unreachable Garnet must not stall queries. Once failures pile up the breaker opens and
/// callers degrade (quantized-only rerank) until a cooldown elapses; then a probe call half-opens it.
pub(crate) struct CircuitBreaker {
    consecutive_failures: std::sync::atomic::AtomicU32,
    threshold: u32,
    cooldown: std::time::Duration,
    /// Nanos-since-epoch at which the breaker opened; 0 = closed.
    opened_at_nanos: std::sync::atomic::AtomicU64,
    epoch: std::time::Instant,
    trip_count: std::sync::atomic::AtomicU64,
}

impl CircuitBreaker {
    pub(crate) fn new(threshold: u32, cooldown: std::time::Duration) -> Self {
        Self {
            consecutive_failures: std::sync::atomic::AtomicU32::new(0),
            threshold: threshold.max(1),
            cooldown,
            opened_at_nanos: std::sync::atomic::AtomicU64::new(0),
            epoch: std::time::Instant::now(),
            trip_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn now_nanos(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    /// Whether a call should be attempted now (closed or half-open).
    pub(crate) fn allows_call(&self) -> bool {
        use std::sync::atomic::Ordering;
        let opened = self.opened_at_nanos.load(Ordering::Acquire);
        if opened == 0 {
            return true;
        }
        // Open; allow the probe once the cooldown elapsed (half-open).
        self.now_nanos().saturating_sub(opened) >= self.cooldown.as_nanos() as u64
    }

    pub(crate) fn on_success(&self) {
        use std::sync::atomic::Ordering;
        self.consecutive_failures.store(0, Ordering::Release);
        self.opened_at_nanos.store(0, Ordering::Release);
    }

    pub(crate) fn on_failure(&self) {
        use std::sync::atomic::Ordering;
        let opened = self.opened_at_nanos.load(Ordering::Acquire);
        if opened != 0 {
            // Failed while open/half-open (a probe failure): restart the cooldown.
            self.opened_at_nanos.store(self.now_nanos().max(1), Ordering::Release);
            self.trip_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;
        if failures >= self.threshold {
            let _ = self.opened_at_nanos.compare_exchange(
                0,
                self.now_nanos().max(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            self.trip_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn trip_count(&self) -> u64 {
        self.trip_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ================================================================================================
// GarnetFpStore — the KV backend (real implementation, sync RESP client)
// ================================================================================================

/// Configuration for the Garnet FP backend.
#[derive(Debug, Clone)]
pub struct GarnetConfig {
    /// RESP endpoints, e.g. `["redis://127.0.0.1:6379"]` or `["host:port"]`. First endpoint is used;
    /// the list form keeps the surface stable for later multi-endpoint support.
    pub endpoints: Vec<String>,
    /// Per-call read/write timeout in milliseconds.
    pub timeout_ms: u64,
    /// Number of pooled connections (concurrent reranks round-robin over them).
    pub pool_size: usize,
    /// Consecutive failures before the breaker opens.
    pub breaker_threshold: u32,
    /// Key prefix namespacing the FP store (default `"fp:"`).
    pub key_prefix: String,
}

/// Which backend stores the full-precision vectors — the user-facing selection carried in
/// `BfTreeProviderParameters`.
///
/// `BfTree` (default) is byte-identical to the pre-fork behavior. `Garnet` stores FP in a standalone
/// Garnet KV over RESP and bf-tree stores **no** FP vectors; only meaningful for quantized indices
/// (a full-precision index reads FP on every traversal hop, which must stay local).
#[derive(Debug, Clone, Default)]
pub enum FpBackendSelection {
    #[default]
    BfTree,
    Garnet(GarnetConfig),
}

/// Full-precision vectors in a standalone Garnet KV store, over RESP.
///
/// Uses the **synchronous** redis client deliberately: the provider's read/write paths are sync (the
/// same threads already do blocking bf-tree disk I/O), and a sync client avoids threading a tokio
/// runtime handle through `BfTreeProviderParameters`. A small round-robin pool of connections serves
/// concurrent reranks; each call is bounded by the configured read/write timeout and guarded by the
/// circuit breaker.
///
/// Wire format: key = `"{prefix}{slot}"`, value = the vector's raw `T` bytes
/// (`bytemuck` cast — `VectorRepr: VectorElement: bytemuck::Pod` guarantees castability).
pub(crate) struct GarnetFpStore<T: VectorRepr> {
    pool: Vec<std::sync::Mutex<redis::Connection>>,
    next: std::sync::atomic::AtomicUsize,
    prefix: String,
    breaker: CircuitBreaker,
    dim: usize,
    max_vectors: usize,
    num_start_points: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: VectorRepr> GarnetFpStore<T> {
    /// Connect to Garnet and build the connection pool.
    pub(crate) fn connect(
        cfg: &GarnetConfig,
        dim: usize,
        max_vectors: usize,
        num_start_points: usize,
    ) -> ANNResult<Self> {
        let endpoint = cfg.endpoints.first().ok_or_else(|| {
            diskann::ANNError::message("garnet fp backend: endpoints is empty".to_string())
        })?;
        let url = if endpoint.starts_with("redis://") || endpoint.starts_with("rediss://") {
            endpoint.clone()
        } else {
            format!("redis://{endpoint}")
        };
        let client = redis::Client::open(url.as_str()).map_err(|e| {
            diskann::ANNError::message(format!("garnet fp backend: bad endpoint {endpoint}: {e}"))
        })?;

        let timeout = std::time::Duration::from_millis(cfg.timeout_ms.max(1));
        let pool_size = cfg.pool_size.max(1);
        let mut pool = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let conn = client.get_connection().map_err(|e| {
                diskann::ANNError::message(format!(
                    "garnet fp backend: connect to {endpoint} failed: {e}"
                ))
            })?;
            conn.set_read_timeout(Some(timeout)).map_err(|e| {
                diskann::ANNError::message(format!("garnet fp backend: set_read_timeout: {e}"))
            })?;
            conn.set_write_timeout(Some(timeout)).map_err(|e| {
                diskann::ANNError::message(format!("garnet fp backend: set_write_timeout: {e}"))
            })?;
            pool.push(std::sync::Mutex::new(conn));
        }

        Ok(Self {
            pool,
            next: std::sync::atomic::AtomicUsize::new(0),
            prefix: cfg.key_prefix.clone(),
            breaker: CircuitBreaker::new(
                cfg.breaker_threshold.max(1),
                // Recovery cooldown: a small multiple of the call timeout, floored to 500ms.
                std::time::Duration::from_millis((cfg.timeout_ms.max(1) * 20).max(500)),
            ),
            dim,
            max_vectors,
            num_start_points,
            _marker: std::marker::PhantomData,
        })
    }

    fn key(&self, slot: usize) -> String {
        format!("{}{}", self.prefix, slot)
    }

    /// Run one redis operation on a pooled connection, recording the outcome in the breaker.
    fn run<R>(
        &self,
        op: impl FnOnce(&mut redis::Connection) -> redis::RedisResult<R>,
    ) -> Result<R, redis::RedisError> {
        use std::sync::atomic::Ordering;
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.pool.len();
        let mut guard = match self.pool[idx].lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        match op(&mut guard) {
            Ok(v) => {
                self.breaker.on_success();
                Ok(v)
            }
            Err(e) => {
                self.breaker.on_failure();
                Err(e)
            }
        }
    }

    /// Whether the store is currently serving (breaker closed/half-open). When `false`, the rerank
    /// degrades to quantized-only ordering instead of waiting on Garnet.
    pub(crate) fn is_available(&self) -> bool {
        self.breaker.allows_call()
    }

    /// Breaker trip count (observability).
    pub(crate) fn breaker_trips(&self) -> u64 {
        self.breaker.trip_count()
    }

    fn decode(&self, bytes: &[u8]) -> Option<Vec<T>> {
        if bytes.len() != self.dim * std::mem::size_of::<T>() {
            return None; // corrupt/short value: treat the slot as absent rather than panic
        }
        // Copy into a T-allocated buffer via its u8 view: alignment-safe regardless of the redis
        // buffer's alignment (casting T->u8 is always valid; u8->T is not).
        let mut v = vec![T::default(); self.dim];
        bytemuck::cast_slice_mut::<T, u8>(&mut v).copy_from_slice(bytes);
        Some(v)
    }
}

impl<T: VectorRepr> FpVectorStore<T> for GarnetFpStore<T> {
    fn get_vector_sync(&self, i: usize) -> Result<Vec<T>, AccessError> {
        if !self.breaker.allows_call() {
            return Err(RankedError::Transient(crate::VectorUnavailable {
                id: i,
                err: crate::VectorError::NotFound,
            }));
        }
        let key = self.key(i);
        match self.run(|conn| redis::cmd("GET").arg(&key).query::<Option<Vec<u8>>>(conn)) {
            Ok(Some(bytes)) => match self.decode(&bytes) {
                Some(v) => Ok(v),
                None => Err(RankedError::Transient(crate::VectorUnavailable {
                    id: i,
                    err: crate::VectorError::NotFound,
                })),
            },
            Ok(None) => Err(RankedError::Transient(crate::VectorUnavailable {
                id: i,
                err: crate::VectorError::NotFound,
            })),
            Err(e) => Err(RankedError::Error(diskann::ANNError::message(format!(
                "garnet fp get({i}): {e}"
            )))),
        }
    }

    fn get_vector_into(&self, i: usize, buffer: &mut [T]) -> Result<(), AccessError> {
        let v = self.get_vector_sync(i)?;
        if v.len() != buffer.len() {
            return Err(RankedError::Error(diskann::ANNError::message(format!(
                "garnet fp get_into({i}): dim mismatch {} vs {}",
                v.len(),
                buffer.len()
            ))));
        }
        buffer.copy_from_slice(&v);
        Ok(())
    }

    fn get_many(&self, slots: &[usize], out: &mut Vec<Option<Vec<T>>>) -> ANNResult<()> {
        out.clear();
        if slots.is_empty() {
            return Ok(());
        }
        if !self.breaker.allows_call() {
            // Signal degradation to the caller (Rerank passes candidates through unranked).
            return Err(diskann::ANNError::message(
                "garnet fp backend unavailable (breaker open)".to_string(),
            ));
        }
        let keys: Vec<String> = slots.iter().map(|&s| self.key(s)).collect();
        let values: Vec<Option<Vec<u8>>> = self
            .run(|conn| {
                let mut cmd = redis::cmd("MGET");
                for k in &keys {
                    cmd.arg(k);
                }
                cmd.query::<Vec<Option<Vec<u8>>>>(conn)
            })
            .map_err(|e| diskann::ANNError::message(format!("garnet fp mget: {e}")))?;
        out.reserve(slots.len());
        for val in values {
            out.push(val.and_then(|bytes| self.decode(&bytes)));
        }
        // MGET must return one entry per key; pad defensively if the server returned fewer.
        while out.len() < slots.len() {
            out.push(None);
        }
        Ok(())
    }

    fn set_vector_sync(&self, i: usize, v: &[T]) -> ANNResult<()> {
        // FP-write-precedes-visibility: a failed FP write must fail the insert loudly.
        let key = self.key(i);
        let bytes: &[u8] = bytemuck::cast_slice(v);
        self.run(|conn| {
            redis::cmd("SET")
                .arg(&key)
                .arg(bytes)
                .query::<()>(conn)
        })
        .map_err(|e| diskann::ANNError::message(format!("garnet fp set({i}): {e}")))
    }

    fn delete_vector(&self, i: usize) {
        // bf-tree's delete_vector returns (); mirror that. Failures are recorded in the breaker;
        // a missed DEL leaves an orphaned value that recovery/repopulation overwrites.
        let key = self.key(i);
        let _ = self.run(|conn| redis::cmd("DEL").arg(&key).query::<usize>(conn));
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn total(&self) -> usize {
        self.max_vectors + self.num_start_points
    }

    fn starting_points<I: crate::BfTreeId>(&self) -> ANNResult<Vec<I>> {
        // Start points live at the fixed slot convention (max_vectors..total), same as bf-tree.
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

    /// Batched fetch — the rerank read. One call per query regardless of candidate count.
    pub(super) fn get_many(&self, slots: &[usize], out: &mut Vec<Option<Vec<T>>>) -> ANNResult<()> {
        match self {
            FpStore::BfTree(s) => s.get_many(slots, out),
            FpStore::Garnet(s) => s.get_many(slots, out),
        }
    }

    /// Whether a rerank that fails against this backend may **degrade** to quantized-only ordering
    /// instead of failing the query. bf-tree errors stay hard (today's behavior); the networked
    /// garnet backend degrades (breaker-open / transient network failure must not fail searches).
    pub(super) fn degrade_on_error(&self) -> bool {
        matches!(self, FpStore::Garnet(_))
    }

    /// Whether the garnet backend is active (used to gate the not-yet-decoupled save/load path).
    pub(super) fn is_garnet(&self) -> bool {
        matches!(self, FpStore::Garnet(_))
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
