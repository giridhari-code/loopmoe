//! MTP varlen state — DeepInfer `mtp_varlen.hpp` inspired.
//!
//! Per-sequence draft buffer + accept-rate feedback used to schedule how many
//! draft tokens to verify on the next step (variable length MTP).

use super::config::TeslaConfig;
use super::confidence::{schedule_verify_len, schedule_verify_len_smart};

/// Per-sequence MTP / Tesla runtime state (DeepInfer Sequence.mtp).
#[derive(Debug, Clone, Default)]
pub struct MtpState {
    /// Draft tokens produced this step (cleared after verify).
    pub draft_tokens: Vec<u32>,
    pub confidences: Vec<f32>,
    pub margins: Vec<f32>,
    /// Running accept-rate estimate [0, 1].
    pub accept_rate: f32,
    /// Previous step accept rates (for async-style pre-schedule).
    pub prev_accept_rate: f32,
    pub schedule_cnt: u64,
    pub process_cnt: u64,
    pub has_draft_info: bool,
    /// Total drafted / accepted across the request (stats).
    pub total_drafted: u64,
    pub total_accepted: u64,
    pub total_scheduled: u64,
}

impl MtpState {
    pub fn clear_draft(&mut self) {
        self.draft_tokens.clear();
        self.confidences.clear();
        self.margins.clear();
        self.has_draft_info = false;
    }

    pub fn set_draft(
        &mut self,
        tokens: Vec<u32>,
        confidences: Vec<f32>,
        margins: Vec<f32>,
    ) {
        self.draft_tokens = tokens;
        self.confidences = confidences;
        self.margins = margins;
        self.has_draft_info = !self.draft_tokens.is_empty();
        self.schedule_cnt = self.schedule_cnt.saturating_add(1);
    }

    /// How many draft tokens to verify this step (DeepInfer confidence schedule).
    pub fn schedule_verify_len(&self, cfg: &TeslaConfig) -> usize {
        if !cfg.enable_mtp_varlen {
            return cfg.effective_nextn().min(self.draft_tokens.len());
        }
        if self.draft_tokens.is_empty() {
            return 0;
        }

        let base = if cfg.use_margin && !self.margins.is_empty() {
            schedule_verify_len_smart(
                &self.confidences,
                Some(&self.margins),
                cfg.effective_nextn(),
                cfg.min_confidence,
                cfg.min_survival,
                cfg.min_margin,
            )
        } else {
            schedule_verify_len(
                &self.confidences,
                cfg.effective_nextn(),
                cfg.min_confidence,
                cfg.min_survival,
            )
        };

        // Accept-rate feedback: if recent acceptance is low, verify fewer tokens
        // (DeepInfer pads / trims draft using accept_rates queues).
        let rate = if self.process_cnt > 0 {
            self.accept_rate
        } else {
            1.0
        };
        let adjusted = if rate < 0.3 {
            (base as f32 * 0.5).ceil() as usize
        } else if rate < 0.6 {
            (base as f32 * 0.75).ceil() as usize
        } else {
            base
        };
        adjusted.max(1).min(self.draft_tokens.len()).min(cfg.effective_nextn())
    }

    /// Update EMA accept rate after a verify step.
    pub fn on_verify_result(&mut self, accepted: usize, scheduled: usize, cfg: &TeslaConfig) {
        self.process_cnt = self.process_cnt.saturating_add(1);
        self.total_accepted += accepted as u64;
        self.total_scheduled += scheduled as u64;
        self.total_drafted += self.draft_tokens.len() as u64;

        let step_rate = if scheduled == 0 {
            0.0
        } else {
            accepted as f32 / scheduled as f32
        };
        self.prev_accept_rate = self.accept_rate;
        let alpha = cfg.accept_rate_ema.clamp(0.01, 1.0);
        self.accept_rate = if self.process_cnt == 1 {
            step_rate
        } else {
            alpha * step_rate + (1.0 - alpha) * self.accept_rate
        };
        self.clear_draft();
    }

    pub fn overall_accept_rate(&self) -> f32 {
        if self.total_scheduled == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_scheduled as f32
        }
    }
}
