//! EOS, max_tokens, stop strings

pub struct StopCriteria {
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
}

impl StopCriteria {
    pub fn should_stop(&self, _generated: &[u32], _decoded: &str) -> bool {
        // TODO
        false
    }
}
