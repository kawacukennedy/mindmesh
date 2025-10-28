use crate::{Network, Analytics as CoreAnalytics};
use std::collections::HashMap;

#[derive(Debug)]
pub struct AnalyticsWorker {
    pub metrics: HashMap<String, Vec<f64>>,
    pub patterns: Vec<String>,
    pub telemetry: Vec<TelemetryEntry>,
}

#[derive(Debug)]
pub struct TelemetryEntry {
    pub timestamp: u64,
    pub component: String,
    pub metric: String,
    pub value: f64,
}

impl AnalyticsWorker {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            patterns: vec![],
            telemetry: vec![],
        }
    }

    pub fn analyze_network(&mut self, network: &Network) {
        let firing_count = network.neurons.iter().filter(|n| n.should_fire()).count() as f64;
        let activity = firing_count / network.neurons.len() as f64;
        self.metrics.entry("activity".to_string()).or_insert(vec![]).push(activity);

        let energy = network.neurons.iter().map(|n| n.metadata.energy_cost).sum::<f64>();
        self.metrics.entry("energy".to_string()).or_insert(vec![]).push(energy);

        // Detect patterns
        if activity > 0.8 {
            self.patterns.push("High Activity".to_string());
        }

        // Telemetry
        self.telemetry.push(TelemetryEntry {
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            component: "Simulator".to_string(),
            metric: "activity".to_string(),
            value: activity,
        });
    }

    pub fn get_suggestions(&self) -> Vec<String> {
        let mut suggestions = vec![];
        if let Some(activities) = self.metrics.get("activity") {
            if activities.last().unwrap_or(&0.0) < &0.1 {
                suggestions.push("Increase input stimulation".to_string());
            }
        }
        suggestions
    }
}