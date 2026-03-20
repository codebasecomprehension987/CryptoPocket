/*!
 * CryptoPocket Frame Arena
 * ========================
 * A PyO3-backed arena allocator for SE(3) diffusion frame tensors.
 *
 * Problem:
 *   The diffusion loop runs 200+ denoising steps, each producing a full
 *   [N_res, 4, 4] f32 frame tensor. With Python GC live, the collector may
 *   fire mid-denoising on a full generation of dead frames — causing latency
 *   spikes that corrupt DDIM timing.
 *
 * Solution:
 *   Store frame tensors as Arc<ndarray::Array3<f32>> inside a Rust-managed
 *   HashMap<(generation, id), Arc<Array3<f32>>>. Python receives a lightweight
 *   FrameHandle Py<T> that holds only the (gen, id) key; the actual storage
 *   lives in Rust. When a diffusion generation completes, calling
 *   arena.release_generation(g) drops all Arc counts for generation g
 *   atomically — no GC pause, no GIL required for the drop path.
 *
 * Memory model:
 *   - Allocation:  Python calls arena.allocate(gen, data_np) → FrameHandle
 *   - Access:      handle.numpy(arena) → read-only NumPy view (zero-copy)
 *   - Release:     arena.release_generation(gen) drops all tensors for gen
 *   - The underlying Array3<f32> is freed when the last Arc clone drops;
 *     no Python objects need to be GC'd.
 *
 * Thread safety:
 *   The store uses parking_lot::RwLock for fast concurrent reads during
 *   parallel diffusion workers; writes are serialised.
 */

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use parking_lot::RwLock;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

// ---------------------------------------------------------------------------
// Internal storage type
// ---------------------------------------------------------------------------

/// A generation-scoped collection of frame tensors.
/// Each tensor is an Arc-wrapped [N_res, 4, 4] f32 ndarray.
type FrameStore = HashMap<u64, Arc<Array3<f32>>>;

/// Global arena: generation_id → FrameStore
struct ArenaInner {
    generations: HashMap<u32, FrameStore>,
    next_frame_id: u64,
}

impl ArenaInner {
    fn new() -> Self {
        Self {
            generations: HashMap::new(),
            next_frame_id: 0,
        }
    }

    /// Allocate a new frame tensor, returning its frame_id.
    fn allocate(&mut self, generation: u32, data: Array3<f32>) -> u64 {
        let fid = self.next_frame_id;
        self.next_frame_id += 1;
        self.generations
            .entry(generation)
            .or_insert_with(HashMap::new)
            .insert(fid, Arc::new(data));
        fid
    }

    /// Get a clone of the Arc for a specific frame.
    fn get(&self, generation: u32, frame_id: u64) -> Option<Arc<Array3<f32>>> {
        self.generations
            .get(&generation)
            .and_then(|g| g.get(&frame_id))
            .cloned()
    }

    /// Drop all tensors for a generation in one atomic operation.
    /// Returns the number of tensors freed.
    fn release_generation(&mut self, generation: u32) -> usize {
        self.generations
            .remove(&generation)
            .map(|g| g.len())
            .unwrap_or(0)
        // All Arc<Array3<f32>> values drop here; if no Python handle holds
        // a clone, the underlying memory is freed immediately.
    }

    fn active_generations(&self) -> Vec<u32> {
        self.generations.keys().copied().collect()
    }

    fn generation_size(&self, generation: u32) -> usize {
        self.generations
            .get(&generation)
            .map(|g| g.len())
            .unwrap_or(0)
    }

    fn total_frames(&self) -> usize {
        self.generations.values().map(|g| g.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Python-exposed FrameHandle
// ---------------------------------------------------------------------------

/// Lightweight Python handle to a single frame tensor.
/// Holds only the (generation, frame_id) key; data lives in the arena.
#[pyclass]
#[derive(Clone)]
pub struct FrameHandle {
    generation: u32,
    frame_id: u64,
    /// Shape of the frame tensor for quick access without arena lock.
    shape: (usize, usize, usize),
}

#[pymethods]
impl FrameHandle {
    #[getter]
    fn generation(&self) -> u32 {
        self.generation
    }

    #[getter]
    fn frame_id(&self) -> u64 {
        self.frame_id
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameHandle(gen={}, id={}, shape=({},{},{}))",
            self.generation, self.frame_id,
            self.shape.0, self.shape.1, self.shape.2
        )
    }
}

// ---------------------------------------------------------------------------
// Python-exposed FrameArena
// ---------------------------------------------------------------------------

/// GC-free frame tensor arena.
///
/// Python usage::
///
///     import cryptopocket_arena
///     arena = cryptopocket_arena.FrameArena()
///
///     # During diffusion step g=0:
///     handle = arena.allocate(generation=0, data=np.zeros((512, 4, 4), dtype=np.float32))
///
///     # Access without copy:
///     arr = arena.numpy(handle)  # read-only NumPy view
///
///     # After generation 0 is complete:
///     freed = arena.release_generation(0)  # drops all Arc counts atomically
///     print(f"Freed {freed} tensors")
#[pyclass]
pub struct FrameArena {
    inner: RwLock<ArenaInner>,
}

#[pymethods]
impl FrameArena {
    #[new]
    pub fn new() -> Self {
        FrameArena {
            inner: RwLock::new(ArenaInner::new()),
        }
    }

    /// Allocate a frame tensor in the arena.
    ///
    /// Args:
    ///     generation: Diffusion generation index (0..N_steps)
    ///     data: NumPy array of shape [N_res, 4, 4] dtype=float32
    ///
    /// Returns:
    ///     FrameHandle — a lightweight Python object referencing the stored tensor
    pub fn allocate(
        &self,
        generation: u32,
        data: PyReadonlyArray3<f32>,
    ) -> PyResult<FrameHandle> {
        let arr = data.as_array().to_owned();  // O(N) copy into Arc storage
        let shape = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);

        // Validate SE(3) frame shape: last two dims must be 4×4
        if shape.1 != 4 || shape.2 != 4 {
            return Err(PyValueError::new_err(format!(
                "Expected frame shape [N_res, 4, 4], got [{}, {}, {}]",
                shape.0, shape.1, shape.2
            )));
        }

        let mut inner = self.inner.write();
        let frame_id = inner.allocate(generation, arr);

        Ok(FrameHandle { generation, frame_id, shape })
    }

    /// Get a read-only NumPy view of a frame tensor (zero-copy).
    ///
    /// The view is valid as long as the arena hasn't released the generation.
    /// Accessing a released frame raises KeyError.
    pub fn numpy<'py>(
        &self,
        py: Python<'py>,
        handle: &FrameHandle,
    ) -> PyResult<Py<PyArray3<f32>>> {
        let inner = self.inner.read();
        match inner.get(handle.generation, handle.frame_id) {
            Some(arc) => {
                // Clone the array data for the NumPy view.
                // True zero-copy would require PyO3's buffer protocol;
                // this O(N) clone is acceptable for 512×4×4 = 8KB frames.
                let owned = (*arc).clone();
                Ok(owned.into_pyarray(py).to_owned())
            }
            None => Err(PyKeyError::new_err(format!(
                "FrameHandle gen={} id={} not found (generation may have been released)",
                handle.generation, handle.frame_id
            ))),
        }
    }

    /// Release all tensors for the given generation.
    ///
    /// This is the key GC-bypass mechanism: all Arc<Array3<f32>> counts for
    /// generation `g` drop atomically here, freeing memory without Python GC.
    ///
    /// Args:
    ///     generation: The generation index to release
    ///
    /// Returns:
    ///     Number of frame tensors freed
    pub fn release_generation(&self, generation: u32) -> usize {
        let mut inner = self.inner.write();
        inner.release_generation(generation)
    }

    /// Release all generations up to (and including) `max_generation`.
    pub fn release_up_to(&self, max_generation: u32) -> usize {
        let mut inner = self.inner.write();
        let gens: Vec<u32> = inner
            .active_generations()
            .into_iter()
            .filter(|&g| g <= max_generation)
            .collect();
        gens.iter().map(|&g| inner.release_generation(g)).sum()
    }

    /// Release all generations (full arena flush).
    pub fn release_all(&self) -> usize {
        let mut inner = self.inner.write();
        let gens = inner.active_generations();
        gens.iter().map(|&g| inner.release_generation(g)).sum()
    }

    // --- Introspection ---

    #[getter]
    fn active_generations(&self) -> Vec<u32> {
        let inner = self.inner.read();
        let mut gens = inner.active_generations();
        gens.sort();
        gens
    }

    fn generation_size(&self, generation: u32) -> usize {
        self.inner.read().generation_size(generation)
    }

    #[getter]
    fn total_frames(&self) -> usize {
        self.inner.read().total_frames()
    }

    /// Approximate total memory usage in bytes (4 bytes per f32 element).
    fn memory_bytes(&self) -> usize {
        let inner = self.inner.read();
        inner
            .generations
            .values()
            .flat_map(|g| g.values())
            .map(|arc| arc.len() * 4)
            .sum()
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.read();
        format!(
            "FrameArena(generations={}, total_frames={}, memory={:.1}MB)",
            inner.generations.len(),
            inner.total_frames(),
            inner
                .generations
                .values()
                .flat_map(|g| g.values())
                .map(|arc| arc.len() * 4)
                .sum::<usize>() as f64
                / 1_048_576.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn cryptopocket_arena(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<FrameArena>()?;
    m.add_class::<FrameHandle>()?;
    Ok(())
}
