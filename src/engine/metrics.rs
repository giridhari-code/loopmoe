//! tokens/sec, latency, queue depth

use std::time::Instant;

pub struct Metrics {
    pub start: Instant,
    pub total_tokens: u64,
    pub num_requests: u64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            total_tokens: 0,
            num_requests: 0,
        }
    }

    pub fn tokens_per_sec(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_tokens as f64 / elapsed
        } else {
            0.0
        }
    }
}
