#[derive(Debug)]
pub struct FeatureFlags {
    pub vr_ar_features: bool,
    pub wasm_web: bool,
    pub desktop_platforms: bool,
}

impl FeatureFlags {
    pub fn new() -> Self {
        Self {
            vr_ar_features: false,
            wasm_web: false,
            desktop_platforms: true,
        }
    }

    pub fn enable_vr_ar(&mut self) {
        self.vr_ar_features = true;
    }

    pub fn enable_wasm_web(&mut self) {
        self.wasm_web = true;
    }
}