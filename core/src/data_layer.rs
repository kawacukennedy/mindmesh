use crate::storage::OnDiskLayout;
use crate::analytics::AnalyticsWorker;

#[derive(Debug)]
pub struct DataAssetLayer {
    pub on_disk_layout: OnDiskLayout,
    pub in_memory_model: crate::storage::StorageManager,
    pub telemetry_audit: TelemetryAudit,
}

#[derive(Debug)]
pub struct TelemetryAudit {
    pub logs: Vec<String>,
    pub metrics: Vec<f64>,
}

impl DataAssetLayer {
    pub fn new() -> Self {
        Self {
            on_disk_layout: OnDiskLayout {
                manifest: crate::Manifest {
                    project_name: "MindMesh".to_string(),
                    version: "1.0.0".to_string(),
                    seed: 0,
                    neuron_count: 0,
                    mapping_presets: vec![],
                    created_at: chrono::Utc::now().to_rfc3339(),
                    parameters: serde_json::Value::Null,
                },
                clusters: vec![],
                journal: vec![],
                metadata_db: std::collections::HashMap::new(),
                thumbnails: vec![],
                exports: vec![],
                logs: vec![],
            },
            in_memory_model: crate::storage::StorageManager::new(),
            telemetry_audit: TelemetryAudit {
                logs: vec![],
                metrics: vec![],
            },
        }
    }

    pub fn audit_flow(&mut self, component: &str, action: &str) {
        self.telemetry_audit.logs.push(format!("{}: {}", component, action));
    }
}