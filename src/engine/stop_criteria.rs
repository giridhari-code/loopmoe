//! EOS, max_tokens, stop strings

#[derive(Debug, Clone, Default)]
pub struct StopCriteria {
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
}

impl StopCriteria {
    pub fn should_stop(&self, generated: &[u32], _decoded: &str) -> bool {
        if generated.len() >= self.max_tokens {
            return true;
        }
        if let Some(&last) = generated.last() {
            if self.stop_token_ids.contains(&last) {
                return true;
            }
        }
        false
    }
}
