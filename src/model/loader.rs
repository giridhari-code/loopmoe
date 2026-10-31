//! safetensors → weights

pub struct WeightLoader;

impl WeightLoader {
    pub fn load(_path: &str) -> anyhow::Result<()> {
        // TODO: map safetensors, convert to desired dtype / layout
        Ok(())
    }
}
