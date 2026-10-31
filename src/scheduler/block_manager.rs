//! Paged KV-cache block allocator
//!
//! Manages physical blocks of KV cache (like vLLM's BlockManager).

pub struct BlockManager {
    // TODO: free_blocks, block_size, num_gpu_blocks, etc.
}

impl BlockManager {
    pub fn new(num_gpu_blocks: usize, block_size: usize) -> Self {
        let _ = (num_gpu_blocks, block_size);
        Self {}
    }

    pub fn allocate(&mut self, _num_blocks: usize) -> Option<Vec<u32>> {
        // TODO: allocate physical block ids
        None
    }

    pub fn free(&mut self, _block_ids: &[u32]) {
        // TODO
    }
}
