use crate::{Network, BrainSnapshot, Manifest, SnapshotFormat};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub struct StorageManager {
    pub in_memory_model: HashMap<String, Network>,
    pub on_disk_layout: OnDiskLayout,
}

#[derive(Debug)]
pub struct OnDiskLayout {
    pub manifest: Manifest,
    pub clusters: Vec<Cluster>,
    pub journal: Vec<JournalEntry>,
    pub metadata_db: HashMap<String, String>,
    pub thumbnails: Vec<Thumbnail>,
    pub exports: Vec<Export>,
    pub logs: Vec<String>,
}

#[derive(Debug)]
pub struct Cluster {
    pub id: u64,
    pub neurons: Vec<u64>,
}

#[derive(Debug)]
pub struct JournalEntry {
    pub timestamp: u64,
    pub action: String,
    pub details: String,
}

#[derive(Debug)]
pub struct Thumbnail {
    pub id: u64,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Export {
    pub id: u64,
    pub format: SnapshotFormat,
    pub data: Vec<u8>,
}

impl StorageManager {
    pub fn new() -> Self {
        Self {
            in_memory_model: HashMap::new(),
            on_disk_layout: OnDiskLayout {
                manifest: Manifest {
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
                metadata_db: HashMap::new(),
                thumbnails: vec![],
                exports: vec![],
                logs: vec![],
            },
        }
    }

    pub fn save_snapshot(&mut self, network: &Network, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        network.save_snapshot(path, SnapshotFormat::Mm, "Auto save", false, true)?;
        self.on_disk_layout.journal.push(JournalEntry {
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            action: "save_snapshot".to_string(),
            details: path.to_string(),
        });
        Ok(())
    }

    pub fn load_snapshot(&mut self, path: &str) -> Result<Network, Box<dyn std::error::Error>> {
        let snapshot = Network::load_snapshot(path, None)?;
        self.on_disk_layout.journal.push(JournalEntry {
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            action: "load_snapshot".to_string(),
            details: path.to_string(),
        });
        Ok(snapshot.network)
    }

    pub fn load_into_memory(&mut self, key: &str, network: Network) {
        self.in_memory_model.insert(key.to_string(), network);
    }

    pub fn get_from_memory(&self, key: &str) -> Option<&Network> {
        self.in_memory_model.get(key)
    }
}