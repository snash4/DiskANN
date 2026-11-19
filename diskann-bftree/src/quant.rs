/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Bf-Tree quant vector provider.

use crate::{AccessError, VectorError, VectorUnavailable};
use bf_tree::{BfTree, Config};
use diskann::{error::IntoANNResult, utils::VectorRepr, ANNError, ANNResult};
use diskann_quantization::{
    alloc::{GlobalAllocator, Poly, ScopedAllocator},
    spherical::iface::{
        DistanceComputer, Opaque, OpaqueMut, Quantizer, QueryComputer, QueryLayout,
    },
};
use diskann_utils::lazy_format;
use diskann_vector::PreprocessedDistanceFunction;

use super::ConfigError;
use crate::quant_cache::{CacheRead, QuantCache};
use crate::{bftree_insert, TestCallCount};

pub struct QuantQueryComputer(QueryComputer<GlobalAllocator>);

impl QuantQueryComputer {
    pub(crate) fn evaluate(&self, x: &[u8]) -> ANNResult<f32> {
        match self.0.evaluate_similarity(Opaque::new(x)) {
            Ok(distance) => Ok(distance),
            Err(err) => Err(ANNError::new(err)),
        }
    }
}

pub struct QuantVectorProvider {
    quant_vector_index: BfTree,
    pub(crate) quantizer: Poly<dyn Quantizer>,
    pub(super) num_get_calls: TestCallCount,
    /// Optional slot-indexed in-memory copy of the codes (see [`crate::quant_cache`]). When
    /// present, every read is served from it; the bf-tree remains the store of record.
    cache: Option<QuantCache>,
}

impl QuantVectorProvider {
    pub fn new_with_config(quantizer: Poly<dyn Quantizer>, config: Config) -> ANNResult<Self> {
        crate::validate_record_size(
            "quant_vector_provider",
            &config,
            std::mem::size_of::<usize>(),
            quantizer.bytes(),
        )?;

        let quant_vector_index = BfTree::with_config(config, None).map_err(ConfigError)?;

        Ok(Self {
            quant_vector_index,
            quantizer,
            num_get_calls: TestCallCount::default(),
            cache: None,
        })
    }

    /// Access the BfTree config
    pub(crate) fn config(&self) -> &Config {
        self.quant_vector_index.config()
    }

    /// Access the underlying BfTree
    pub(crate) fn bftree(&self) -> &BfTree {
        &self.quant_vector_index
    }

    /// Create a new instance from an existing BfTree (for loading from snapshot)
    ///
    pub(crate) fn new_from_bftree(
        quantizer: Poly<dyn Quantizer>,
        quant_vector_index: BfTree,
    ) -> Self {
        Self {
            quant_vector_index,
            quantizer,
            num_get_calls: TestCallCount::default(),
            cache: None,
        }
    }

    /// Serve reads from a slot-indexed in-memory copy of the codes for slots `0..capacity`,
    /// filled from the bf-tree now (in parallel) and kept in sync by every later write and delete.
    ///
    /// Call once, after construction or load and before the provider is shared. Calling it again
    /// rebuilds the cache. Memory: about `capacity × code bytes` (rounded up to 8 bytes) plus 8
    /// bytes per slot.
    pub fn enable_cache(&mut self, capacity: usize) -> ANNResult<()> {
        let cache = QuantCache::new(capacity, self.quantizer.bytes())?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(capacity.max(1));
        let chunk = capacity.div_ceil(threads.max(1)).max(1);
        let this = &*self;
        let cache_ref = &cache;
        let result: ANNResult<()> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..capacity)
                .step_by(chunk)
                .map(|start| {
                    let end = (start + chunk).min(capacity);
                    scope.spawn(move || -> ANNResult<()> {
                        let mut buf = vec![0u8; cache_ref.code_bytes()];
                        for i in start..end {
                            match this.read_bftree_into(i, &mut buf) {
                                Ok(()) => cache_ref.write(i, &buf),
                                Err(AccessError::Transient(VectorUnavailable {
                                    err: VectorError::Deleted,
                                    ..
                                })) => cache_ref.delete(i),
                                Err(AccessError::Transient(_)) => {}
                                Err(AccessError::Error(e)) => return Err(e),
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in handles {
                h.join()
                    .map_err(|_| ANNError::message("quant cache fill thread panicked"))??;
            }
            Ok(())
        });
        result?;
        self.cache = Some(cache);
        Ok(())
    }

    /// Liveness of slot `i` from the cache without copying its code; `None` when the cache is
    /// disabled or `i` is outside it (callers then fall back to a full read).
    pub(crate) fn cached_liveness(&self, i: usize) -> Option<bool> {
        match &self.cache {
            Some(c) if i < c.capacity() => Some(c.is_live(i)),
            _ => None,
        }
    }

    /// Whether reads are served from the in-memory cache.
    pub fn cache_enabled(&self) -> bool {
        self.cache.is_some()
    }

    /// Bytes reserved by the in-memory cache (0 when disabled).
    pub fn cache_bytes(&self) -> usize {
        self.cache.as_ref().map_or(0, QuantCache::reserved_bytes)
    }

    fn cache_slot(&self, i: usize) -> ANNResult<Option<&QuantCache>> {
        match &self.cache {
            Some(c) if i < c.capacity() => Ok(Some(c)),
            Some(c) => {
                let capacity = c.capacity();
                Err(ANNError::message(lazy_format!(
                    move,
                    "vector id {} is outside the quant cache capacity {}",
                    i,
                    capacity,
                )))
            }
            None => Ok(None),
        }
    }

    pub(crate) fn delete_vector(&self, i: usize) {
        // Hide the code from searches first, then remove it from the store of record.
        if let Ok(Some(c)) = self.cache_slot(i) {
            c.delete(i);
        }
        let key = bytemuck::bytes_of(&i);
        self.quant_vector_index.delete(key);
    }

    /// Return the dimension of the full-precision data associated with this provider
    pub fn full_dim(&self) -> usize {
        self.quantizer.full_dim()
    }

    /// Create a query computer for the provided query vector
    pub fn query_computer<T>(&self, query: &[T]) -> ANNResult<QuantQueryComputer>
    where
        T: VectorRepr,
    {
        let query_f32 = T::as_f32(query).into_ann_result()?;
        let inner = self
            .quantizer
            .fused_query_computer(
                &query_f32,
                QueryLayout::FullPrecision,
                true,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .map_err(ANNError::new)?;
        Ok(QuantQueryComputer(inner))
    }

    /// Create a distance computer for the underlying schema
    pub fn distance_computer(&self) -> ANNResult<DistanceComputer> {
        self.quantizer
            .distance_computer(GlobalAllocator)
            .map_err(ANNError::new)
    }

    pub(crate) fn get_vector_into(&self, i: usize, buffer: &mut [u8]) -> Result<(), AccessError> {
        use thiserror::Error;

        let expected = self.quantizer.bytes();
        if buffer.len() != expected {
            #[derive(Debug, Error)]
            #[error("expected a buffer with dim {0}, instead got {1}")]
            struct WrongDim(usize, usize);

            return Err(AccessError::Error(ANNError::new(WrongDim(
                expected,
                buffer.len(),
            ))));
        }

        self.num_get_calls.increment();
        if let Some(c) = self.cache_slot(i).map_err(AccessError::Error)? {
            return match c.read(i, buffer) {
                CacheRead::Found => Ok(()),
                CacheRead::Deleted => Err(AccessError::Transient(VectorUnavailable {
                    id: i,
                    err: VectorError::Deleted,
                })),
                CacheRead::NotFound => Err(AccessError::Transient(VectorUnavailable {
                    id: i,
                    err: VectorError::NotFound,
                })),
            };
        }
        self.read_bftree_into(i, buffer)
    }

    /// Read slot `i` from the bf-tree, bypassing the cache. `buffer.len()` must equal the code size.
    pub(crate) fn read_bftree_into(&self, i: usize, buffer: &mut [u8]) -> Result<(), AccessError> {
        let expected = self.quantizer.bytes();
        match self.quant_vector_index.read(bytemuck::bytes_of(&i), buffer) {
            bf_tree::LeafReadResult::Found(read_size) => {
                if read_size as usize != expected {
                    return Err(AccessError::Error(ANNError::message(lazy_format!(
                        move,
                        "The bf-tree entry for vector id {} is marked as found but has size {} instead of the expected size {}",
                        i, read_size, expected,
                    ))));
                }
            }
            bf_tree::LeafReadResult::Deleted => {
                return Err(AccessError::Transient(VectorUnavailable {
                    id: i,
                    err: VectorError::Deleted,
                }));
            }
            bf_tree::LeafReadResult::InvalidKey => {
                return Err(AccessError::Error(ANNError::message(lazy_format!(
                    move,
                    "The bf-tree entry for vector id {} is marked as invalid",
                    i,
                ))));
            }
            bf_tree::LeafReadResult::NotFound => {
                return Err(AccessError::Transient(VectorUnavailable {
                    id: i,
                    err: VectorError::NotFound,
                }));
            }
        };

        Ok(())
    }

    /// Return the quant vector at index `i`
    #[cfg(test)]
    pub(crate) fn get_vector_sync(&self, i: usize) -> Result<Vec<u8>, AccessError> {
        let mut value = vec![0u8; self.quantizer.bytes()];
        self.get_vector_into(i, &mut value)?;
        Ok(value)
    }

    /// Compress the vector, `v`, and set the compressed quant vector with Id, `i`, to it
    ///
    /// Errors if:
    ///
    /// * `v.dim() != self.full_dim()`: The slice must have the proper length.
    /// * PQ compression encounters an error (such as the presence of `NaN`s).
    pub(crate) fn set_vector_sync<T>(&self, i: usize, v: &[T]) -> ANNResult<()>
    where
        T: Copy + VectorRepr,
    {
        let vf32: &[f32] = &T::as_f32(v).into_ann_result()?;

        if vf32.len() != self.full_dim() {
            return Err(ANNError::message(
                "Vector f32 dimension is not equal to the expected dimension.",
            ));
        }

        // Serialize the key into a byte string, &[u8]
        let key = bytemuck::bytes_of(&i);

        let dim = self.quantizer.bytes();
        let quant_vector = &mut vec![0u8; dim];
        self.quantizer
            .compress(
                vf32,
                OpaqueMut::new(quant_vector),
                ScopedAllocator::global(),
            )
            .map_err(ANNError::new)?;

        let cache = self.cache_slot(i)?;
        bftree_insert(&self.quant_vector_index, key, quant_vector)?;
        if let Some(c) = cache {
            c.write(i, quant_vector);
        }

        Ok(())
    }

    /// Set the quant vector with Id, `i`, to `v`
    ///
    /// Errors if:
    ///
    /// * `v.len() != self.pq_chunks()`: `v` must have the right length.
    #[cfg(test)]
    pub(crate) fn set_quant_vector(&self, i: usize, v: &[u8]) -> ANNResult<()> {
        if v.len() != self.quantizer.bytes() {
            return Err(ANNError::message(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }

        // Update pq vector with id = i to v
        let key = bytemuck::bytes_of(&i);

        let cache = self.cache_slot(i)?;
        bftree_insert(&self.quant_vector_index, key, v)?;
        if let Some(c) = cache {
            c.write(i, v);
        }

        Ok(())
    }
}

/// Train a spherical quantizer on simple data and return it as a `Poly<dyn Quantizer>`.
#[cfg(test)]
pub(crate) fn create_test_quantizer(dim: usize) -> Poly<dyn Quantizer> {
    use diskann_quantization::{
        algorithms::TransformKind,
        alloc::poly,
        spherical::{iface, PreScale, SphericalQuantizer, SupportedMetric},
    };
    use diskann_utils::views::Init;
    use diskann_utils::views::Matrix;
    use rand::{rngs::StdRng, SeedableRng};

    // Create training data with spread-out values.
    let nrows = 8;
    let mut counter = 0.0f32;
    let data = Matrix::new(
        Init(move || {
            counter += 0.5;
            counter
        }),
        nrows,
        dim,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let quantizer = SphericalQuantizer::train(
        data.as_view(),
        TransformKind::Null,
        SupportedMetric::SquaredL2,
        PreScale::None,
        &mut rng,
        GlobalAllocator,
    )
    .unwrap();

    let imp = iface::Impl::<1>::new(quantizer).unwrap();
    poly!(Quantizer, imp, GlobalAllocator).unwrap()
}

///////////
// Tests //
///////////
/// These unit tests target the functionality of Bf-Tree quant vector provider alone
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use diskann_quantization::spherical::iface::Opaque;
    use diskann_vector::DistanceFunction;
    use tokio::task::JoinSet;

    use super::*;

    /// Test edge cases of the Bf-Tree quant vector provider
    #[tokio::test]
    async fn common_errors() {
        let dim = 5;
        let quantizer = create_test_quantizer(dim);
        let quant_bytes = quantizer.bytes();

        let bf_tree_config = Config::default();
        let provider = QuantVectorProvider::new_with_config(quantizer, bf_tree_config).unwrap();

        // try to set an out of bounds vector
        let _ = provider.set_quant_vector(20, &[]).unwrap_err();

        // try to set an out of bounds vector via set_vector_sync
        let _ = provider.set_vector_sync::<f32>(20, &[]).unwrap_err();

        // try to set a quant vector with the wrong dimension
        let _ = provider.set_quant_vector(0, &[]).unwrap_err();

        // verify expected quant vector byte count
        assert_eq!(quant_bytes, provider.quantizer.bytes());
    }

    fn create_test_provider() -> QuantVectorProvider {
        let dim = 2;

        let quantizer = create_test_quantizer(dim);

        let bf_tree_config = Config::default();
        let provider = QuantVectorProvider::new_with_config(quantizer, bf_tree_config).unwrap();

        assert_eq!(provider.full_dim(), dim);

        // Set vectors.
        provider.set_vector_sync(0, &[-1.5, -1.5]).unwrap();
        provider.set_vector_sync(1, &[-0.5, -0.5]).unwrap();
        provider.set_vector_sync(2, &[0.5, 0.5]).unwrap();
        provider.set_vector_sync(3, &[1.5, 1.5]).unwrap();
        provider.set_vector_sync(4, &[2.5, 2.5]).unwrap();
        provider
    }

    /// Test the distance computation functions of the provider
    #[tokio::test]
    async fn test_similarity_function() {
        let provider = create_test_provider();
        let quant_bytes = provider.quantizer.bytes();

        // Verify compressed vectors are the expected size.
        for i in 0..5 {
            let v = provider.get_vector_sync(i).unwrap();
            assert_eq!(v.len(), quant_bytes);
        }

        // Error checking.
        assert!(provider.set_vector_sync(2, &[0.0]).is_err());

        // Query Computer — verify it returns finite distances.
        let c = provider.query_computer(&[-0.5f32, -0.5]).unwrap();
        let dist = c.evaluate(&provider.get_vector_sync(3).unwrap()).unwrap();
        assert!(dist.is_finite(), "query distance should be finite");

        // Distance Computer — verify distances between compressed vectors are finite
        // and that identical vectors produce zero distance.
        let d = provider.distance_computer().unwrap();
        let v0 = provider.get_vector_sync(0).unwrap();
        let v3 = provider.get_vector_sync(3).unwrap();
        let dist = d
            .evaluate_similarity(Opaque::new(&v0), Opaque::new(&v3))
            .unwrap();
        assert!(dist.is_finite(), "distance should be finite");

        // Same vector should have small self-distance (may not be exactly zero
        // due to quantization loss, especially at low bit-widths).
        let self_dist = d
            .evaluate_similarity(Opaque::new(&v0), Opaque::new(&v0))
            .unwrap();
        assert!(
            self_dist.abs() < 1.0,
            "self-distance should be small, got {}",
            self_dist
        );
    }

    /// Test the interleaved and parallel traversal of the Bf-Tree
    /// by invoking the async accessors of the quant vector provider
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let dim = 2;
        let quantizer = create_test_quantizer(dim);

        let bf_tree_config = Config::default();
        let provider =
            Arc::new(QuantVectorProvider::new_with_config(quantizer, bf_tree_config).unwrap());
        let mut set = JoinSet::new();
        for i in 0..11 {
            let vector = vec![i as f32, (i + 1) as f32];
            let provider_clone = Arc::clone(&provider);
            set.spawn(async move { provider_clone.set_vector_sync(i as usize, &vector).unwrap() });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        // Verify that each vector was stored and can be retrieved with the correct size.
        let quant_bytes = provider.quantizer.bytes();
        let mut expected_buf = vec![0u8; quant_bytes];

        for i in 0..11 {
            let stored = provider.get_vector_sync(i).unwrap();
            assert_eq!(stored.len(), quant_bytes);

            // Compress the same input again and verify we get the same output
            // (spherical compression is deterministic).
            provider
                .quantizer
                .compress(
                    &[i as f32, (i + 1) as f32],
                    OpaqueMut::new(&mut expected_buf),
                    ScopedAllocator::global(),
                )
                .unwrap();
            assert_eq!(stored, expected_buf);
        }
    }
}
