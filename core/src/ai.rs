use crate::Network;
use std::collections::HashMap;

#[derive(Debug)]
pub struct AIAssistant {
    pub suggestions: Vec<String>,
    pub predictions: Vec<f64>,
    pub model: HashMap<String, f64>, // Simple model
}

impl AIAssistant {
    pub fn new() -> Self {
        Self {
            suggestions: vec![],
            predictions: vec![],
            model: HashMap::new(),
        }
    }

    pub fn analyze_and_suggest(&mut self, network: &Network) {
        let activity = network.neurons.iter().filter(|n| n.should_fire()).count() as f64 / network.neurons.len() as f64;
        if activity < 0.2 {
            self.suggestions.push("Consider adding more connections or adjusting plasticity".to_string());
        } else if activity > 0.8 {
            self.suggestions.push("High activity detected; consider pruning weak connections".to_string());
        }

        // Simple prediction
        self.predictions = network.neurons.iter().map(|n| n.output * 1.1).collect();
    }

    pub fn get_explainability(&self) -> String {
        format!("Current suggestions: {:?}", self.suggestions)
    }
}