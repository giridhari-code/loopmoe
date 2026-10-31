//! Paged KV-cache block allocator
//!
//! Design mirrors DeepInfer `BlockManager` / free-list with ref-counting
//! (csrc/engine/block_manager.hpp):
//! - blocks are 1-indexed (0 = sentinel)
//! - free list is a doubly-linked list
//! - allocate / pin / free with ref_count

#[derive(Debug, Clone)]
struct Block {
    ref_count: i32,
    next: u32,
    prev: u32,
}

#[derive(Debug)]
pub struct BlockManager {
    blocks: Vec<Block>,
    head: u32,
    tail: u32,
    free_size: usize,
    pub num_blocks: usize,
    pub block_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStatus {
    Ok,
    /// Not enough free blocks now, but may free later (preempt / finish).
    Later,
    /// Hard impossibility for this request size.
    Never,
}

impl BlockManager {
    /// `num_gpu_blocks` physical blocks, each holding `block_size` tokens.
    pub fn new(num_gpu_blocks: usize, block_size: usize) -> Self {
        // index 0 is unused sentinel (DeepInfer style)
        let mut blocks = Vec::with_capacity(num_gpu_blocks + 1);
        blocks.push(Block {
            ref_count: 0,
            next: 0,
            prev: 0,
        });
        for _ in 0..num_gpu_blocks {
            blocks.push(Block {
                ref_count: 0,
                next: 0,
                prev: 0,
            });
        }
        let mut bm = Self {
            blocks,
            head: 0,
            tail: 0,
            free_size: 0,
            num_blocks: num_gpu_blocks,
            block_size,
        };
        // push all physical blocks into free list
        for i in 1..=num_gpu_blocks as u32 {
            bm.push_back(i);
        }
        bm
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_size
    }

    pub fn can_allocate(&self, num_blocks: usize) -> AllocStatus {
        if num_blocks > self.num_blocks {
            AllocStatus::Never
        } else if num_blocks <= self.free_size {
            AllocStatus::Ok
        } else {
            AllocStatus::Later
        }
    }

    /// Allocate `n` free blocks. Returns None if not enough free.
    pub fn allocate(&mut self, n: usize) -> Option<Vec<u32>> {
        if n == 0 {
            return Some(Vec::new());
        }
        if n > self.free_size {
            return None;
        }
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.pop_front()?;
            self.blocks[id as usize].ref_count = 1;
            ids.push(id);
        }
        Some(ids)
    }

    /// Extra reference (prefix cache / share).
    pub fn pin(&mut self, block_id: u32) {
        if block_id == 0 || block_id as usize >= self.blocks.len() {
            return;
        }
        let b = &mut self.blocks[block_id as usize];
        if b.ref_count > 0 {
            b.ref_count += 1;
        }
    }

    pub fn free(&mut self, block_id: u32) {
        if block_id == 0 || block_id as usize >= self.blocks.len() {
            return;
        }
        let b = &mut self.blocks[block_id as usize];
        if b.ref_count <= 0 {
            return;
        }
        b.ref_count -= 1;
        if b.ref_count == 0 {
            self.push_back(block_id);
        }
    }

    pub fn free_many(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            self.free(id);
        }
    }

    /// Blocks needed to hold `num_tokens`.
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        if self.block_size == 0 {
            return 0;
        }
        num_tokens.div_ceil(self.block_size)
    }

    fn push_back(&mut self, i: u32) {
        let bi = i as usize;
        self.blocks[bi].prev = self.tail;
        self.blocks[bi].next = 0;
        if self.tail != 0 {
            let t = self.tail as usize;
            self.blocks[t].next = i;
        } else {
            self.head = i;
        }
        self.tail = i;
        self.free_size += 1;
    }

    fn pop_front(&mut self) -> Option<u32> {
        if self.head == 0 {
            return None;
        }
        let result = self.head;
        self.erase(result);
        Some(result)
    }

    fn erase(&mut self, i: u32) {
        let bi = i as usize;
        let next = self.blocks[bi].next;
        let prev = self.blocks[bi].prev;
        if next != 0 {
            self.blocks[next as usize].prev = prev;
        } else {
            self.tail = prev;
        }
        if prev != 0 {
            self.blocks[prev as usize].next = next;
        } else {
            self.head = next;
        }
        self.blocks[bi].next = 0;
        self.blocks[bi].prev = 0;
        self.free_size = self.free_size.saturating_sub(1);
    }
}
