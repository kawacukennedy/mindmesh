#![allow(unused)]

pub mod plugins;
pub mod p2p;
pub mod hardware;

use serde::{Deserialize, Serialize};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Write};
use flate2::{write::GzEncoder, read::GzDecoder, Compression};
use aes_gcm::{Aes256Gcm, Key, Nonce, KeyInit, aead::Aead};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NeuronType {
    Excitatory,
    Inhibitory,
    Modulatory,
    Sensory,
    Motor,
    Interneuron,
    PatternSpecific,
    Adaptive,
    MetaNeuron,
    VirtualNeuron,
    QuantumNeuron,
    SensorInterface,
    AiAssist,
    EmergentPattern,
    Predictive,
    TemporalSpike,
    QuantumInspired,
    BciInterface,
    ArProjection,
}

impl fmt::Display for NeuronType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            NeuronType::Excitatory => "Excitatory",
            NeuronType::Inhibitory => "Inhibitory",
            NeuronType::Modulatory => "Modulatory",
            NeuronType::Sensory => "Sensory",
            NeuronType::Motor => "Motor",
            NeuronType::Interneuron => "Interneuron",
            NeuronType::PatternSpecific => "Pattern Specific",
            NeuronType::Adaptive => "Adaptive",
            NeuronType::MetaNeuron => "Meta Neuron",
            NeuronType::VirtualNeuron => "Virtual Neuron",
            NeuronType::QuantumNeuron => "Quantum Neuron",
            NeuronType::SensorInterface => "Sensor Interface",
            NeuronType::AiAssist => "AI Assist",
            NeuronType::EmergentPattern => "Emergent Pattern",
            NeuronType::Predictive => "Predictive",
            NeuronType::TemporalSpike => "Temporal Spike",
            NeuronType::QuantumInspired => "Quantum Inspired",
            NeuronType::BciInterface => "BCI Interface",
            NeuronType::ArProjection => "AR Projection",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    StochasticProbabilistic,
    TemporalSpikeBased,
    PiecewiseLinear,
    CustomUserDefined,
    AiAdaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub last_activation: f64,
    pub firing_threshold: f64,
    pub activity_score: f64,
    pub cluster_id: Option<u64>,
    pub hierarchical_level: u32,
    pub tags: Vec<String>,
    pub creation_time: u64,
    pub evolution_score: f64,
    pub energy_cost: f64,
    pub ai_relevance_score: f64,
    pub pattern_association: Vec<String>,
    pub temporal_frequency: f64,
}

impl Default for Metadata {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            last_activation: 0.0,
            firing_threshold: rng.gen_range(0.1..1.0),
            activity_score: 0.0,
            cluster_id: None,
            hierarchical_level: 0,
            tags: vec![],
            creation_time: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            evolution_score: 0.0,
            energy_cost: rng.gen_range(0.01..0.1),
            ai_relevance_score: 0.0,
            pattern_association: vec![],
            temporal_frequency: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: u64,
    pub neuron_type: NeuronType,
    pub activation_fn: ActivationFunction,
    pub metadata: Metadata,
    pub input_sum: f64,
    pub output: f64,
    pub position: (f32, f32, f32), // x, y, z for visualization
    pub velocity: (f32, f32, f32), // For physics simulation
    pub membrane_potential: f64, // For LIF and spiking models
    pub leak_rate: f64, // Tau for LIF
    pub refractory_period: f64,
    pub last_spike_time: f64,
}

impl Neuron {
    pub fn new(id: u64, neuron_type: NeuronType, activation_fn: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            id,
            neuron_type,
            activation_fn,
            metadata: Metadata::default(),
            input_sum: 0.0,
            output: 0.0,
            position: (rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)),
            velocity: (0.0, 0.0, 0.0),
            membrane_potential: 0.0,
            leak_rate: 0.02, // Default tau
            refractory_period: 2.0, // In time steps
            last_spike_time: -10.0, // Far in past
        }
    }

    pub fn activate(&mut self, time: f64) -> f64 {
        self.output = match self.neuron_type {
            NeuronType::QuantumNeuron => rand::random(),
            _ => match self.activation_fn {
                ActivationFunction::Sigmoid => 1.0 / (1.0 + (-self.input_sum).exp()),
                ActivationFunction::Tanh => self.input_sum.tanh(),
                ActivationFunction::Relu => self.input_sum.max(0.0),
                ActivationFunction::LeakyRelu => if self.input_sum > 0.0 { self.input_sum } else { 0.01 * self.input_sum },
                ActivationFunction::StochasticProbabilistic => {
                    let prob = 1.0 / (1.0 + (-self.input_sum).exp());
                    if rand::random::<f64>() < prob { 1.0 } else { 0.0 }
                },
                ActivationFunction::TemporalSpikeBased => {
                    // LIF model: Leaky Integrate-and-Fire
                    self.membrane_potential += self.input_sum - self.leak_rate * self.membrane_potential;
                    if self.membrane_potential > self.metadata.firing_threshold && (time - self.last_spike_time) > self.refractory_period {
                        self.last_spike_time = time;
                        self.membrane_potential = 0.0; // Reset
                        1.0
                    } else {
                        0.0
                    }
                },
                ActivationFunction::PiecewiseLinear => {
                    if self.input_sum < -1.0 {
                        0.0
                    } else if self.input_sum > 1.0 {
                        1.0
                    } else {
                        (self.input_sum + 1.0) / 2.0
                    }
                },
                ActivationFunction::CustomUserDefined => {
                    // Placeholder for custom function - for now, use sigmoid
                    1.0 / (1.0 + (-self.input_sum).exp())
                },
                ActivationFunction::AiAdaptive => {
                    // Adaptive based on AI relevance score
                    let adaptive_factor = self.metadata.ai_relevance_score.max(0.1);
                    (1.0 / (1.0 + (-self.input_sum * adaptive_factor).exp())).min(1.0).max(0.0)
                },
            },
        };
        self.metadata.last_activation = self.output;
        self.metadata.temporal_frequency = (self.metadata.temporal_frequency + self.output) / 2.0; // Update frequency
        self.output
    }

    pub fn should_fire(&self) -> bool {
        self.output >= self.metadata.firing_threshold
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MappingStrategy {
    AutoEmbed,
    HashSeed,
    ManualPaint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mapping {
    pub strategy: MappingStrategy,
    pub parameters: HashMap<String, f64>,
    pub manual_brush: Option<Vec<(f32, f32, f32)>>, // x, y, intensity for manual paint
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Input {
    Text(String),
    Numbers(Vec<f64>),
    AsciiArt(String),
    Bitmap(Vec<u8>),
    Vector(Vec<f64>),
    Gesture(Vec<f64>),
    Voice(Vec<f64>),
    Sensor(Vec<f64>),
    Custom(Vec<f64>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub timestamp: u64,
    pub input: Input,
    pub output: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub short_term: Vec<Interaction>,
    pub long_term: HashMap<String, f64>,
    pub decay_rate: f64,
    pub max_short_term: usize,
}

impl Memory {
    pub fn new(max_short_term: usize, decay_rate: f64) -> Self {
        Self {
            short_term: vec![],
            long_term: HashMap::new(),
            decay_rate,
            max_short_term,
        }
    }

    pub fn add_interaction(&mut self, input: Input, output: Vec<f64>) {
        let interaction = Interaction {
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            input,
            output,
        };
        self.short_term.push(interaction);
        if self.short_term.len() > self.max_short_term {
            self.short_term.remove(0);
        }
    }

    pub fn consolidate(&mut self) {
        // Simple consolidation: average outputs for similar inputs
        // Simplified
        for interaction in &self.short_term {
            if let Input::Text(ref text) = interaction.input {
                let key = format!("text:{}", text);
                let entry = self.long_term.entry(key).or_insert(0.0);
                *entry = (*entry + interaction.output.iter().sum::<f64>() / interaction.output.len() as f64) / 2.0;
            }
        }
        // Decay
        for val in self.long_term.values_mut() {
            *val *= 1.0 - self.decay_rate;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlasticityType {
    Hebbian,
    Stdp,
    Ltp,
    Ltd,
    ReinforcementAdjusted,
    DynamicRewiring,
    MetaLearningAdjusted,
    AiPredictiveRewiring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from_id: u64,
    pub to_id: u64,
    pub weight: f64,
    pub plasticity: PlasticityType,
    pub last_update: u64,
}

impl Connection {
    pub fn new(from_id: u64, to_id: u64, weight: f64, plasticity: PlasticityType) -> Self {
        Self {
            from_id,
            to_id,
            weight,
            plasticity,
            last_update: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    pub fn update_weight(&mut self, pre_activity: f64, post_activity: f64, learning_rate: f64) {
        match self.plasticity {
            PlasticityType::Hebbian => {
                self.weight += learning_rate * pre_activity * post_activity;
            }
            PlasticityType::Stdp => {
                // STDP: strengthen if pre fires before post, weaken otherwise
                let dt = (self.last_update as f64 - pre_activity as f64 * 1000.0) - (self.last_update as f64 - post_activity as f64 * 1000.0); // Simplified timing
                if dt > 0.0 && dt < 20.0 { // pre before post
                    self.weight += learning_rate * (0.01 * (20.0 - dt).exp());
                } else if dt < 0.0 && dt > -20.0 { // post before pre
                    self.weight -= learning_rate * (0.005 * (-dt).exp());
                }
            }
            PlasticityType::Ltp => {
                if post_activity > 0.0 {
                    self.weight += learning_rate * pre_activity;
                }
            }
            PlasticityType::Ltd => {
                if post_activity == 0.0 && pre_activity > 0.0 {
                    self.weight -= learning_rate * 0.01;
                }
            }
            PlasticityType::ReinforcementAdjusted => {
                // Simplified reinforcement learning
                let reward = if post_activity > 0.5 { 0.1 } else { -0.05 };
                self.weight += learning_rate * pre_activity * reward;
            }
            PlasticityType::DynamicRewiring => {
                // Dynamic rewiring based on activity correlation
                let correlation = pre_activity * post_activity;
                self.weight += learning_rate * correlation;
                if correlation < 0.1 {
                    self.weight *= 0.99; // Decay weak connections
                }
            }
            PlasticityType::MetaLearningAdjusted => {
                // Meta-learning: adjust based on global network performance
                let meta_factor = 1.0; // Would be computed from network metrics
                self.weight += learning_rate * pre_activity * post_activity * meta_factor;
            }
            PlasticityType::AiPredictiveRewiring => {
                // AI-guided: predict optimal weight change
                let predicted_change = pre_activity * post_activity * 0.1; // Simplified prediction
                self.weight += learning_rate * predicted_change;
            }
        }
        self.weight = self.weight.clamp(-1.0, 1.0);
        self.last_update = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        // Note: Journal weight updates in Network::step
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub neurons: Vec<Neuron>,
    pub connections: Vec<Connection>,
    pub next_id: u64,
    pub memory: Memory,
    pub learning_rate: f64,
    pub emergent_patterns: Vec<String>,
    pub total_energy: f64,
    pub simulation_speed: f64,
    pub mutation_rate: f64,
    pub adaptive_parameters: AdaptiveParams,
    pub analytics: Analytics,
    pub settings: SimulationSettings,
    pub time: f64,
    pub input_mappings: HashMap<String, Mapping>,
    pub journal: Vec<JournalEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParams {
    pub neuron_thresholds: f64,
    pub connection_weights: f64,
    pub cluster_connectivity: f64,
    pub meta_neuron_tuning: f64,
    pub energy_thresholds: f64,
    pub ai_guidance_weighting: f64,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            neuron_thresholds: 0.5,
            connection_weights: 0.1,
            cluster_connectivity: 0.05,
            meta_neuron_tuning: 0.01,
            energy_thresholds: 1.0,
            ai_guidance_weighting: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analytics {
    pub neuron_activity_histogram: Vec<f64>,
    pub cluster_heatmaps: Vec<Vec<f64>>,
    pub connection_distributions: Vec<f64>,
    pub rare_event_detections: Vec<String>,
    pub meta_neuron_trends: Vec<f64>,
    pub energy_tracking: Vec<f64>,
    pub temporal_correlation_analysis: Vec<f64>,
    pub predictive_anomaly_detection: Vec<f64>,
}

impl Default for Analytics {
    fn default() -> Self {
        Self {
            neuron_activity_histogram: vec![],
            cluster_heatmaps: vec![],
            connection_distributions: vec![],
            rare_event_detections: vec![],
            meta_neuron_trends: vec![],
            energy_tracking: vec![],
            temporal_correlation_analysis: vec![],
            predictive_anomaly_detection: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSettings {
    pub speed: f64,
    pub precision: f64,
    pub pause: bool,
    pub rewind: bool,
    pub fast_forward: bool,
    pub reset: bool,
    pub autosave_frequency: u64,
    pub energy_efficient_mode: bool,
    pub cluster_resolution: f64,
    pub gpu_cpu_load_balancing: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub project_name: String,
    pub version: String,
    pub seed: u64,
    pub neuron_count: usize,
    pub mapping_presets: Vec<String>,
    pub created_at: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSnapshot {
    pub network: Network,
    pub manifest: Manifest,
    pub version: String,
    pub timestamp: String,
    pub description: String,
    pub format: SnapshotFormat,
    pub encrypted: bool,
    pub compressed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotFormat {
    Mm,
    Mmsnapshot,
    CompressedHtml,
    Gif,
    Mp4,
    VrArScene,
    DeltaSnapshot,
}

impl Network {
    pub fn save_snapshot(&self, path: &str, format: SnapshotFormat, description: &str, encrypt: bool, compress: bool) -> Result<(), Box<dyn std::error::Error>> {
        let manifest = Manifest {
            project_name: "MindMesh".to_string(),
            version: "1.0.0".to_string(),
            seed: 123456789, // TODO: use actual seed
            neuron_count: self.neurons.len(),
            mapping_presets: self.input_mappings.keys().cloned().collect(),
            created_at: Utc::now().to_rfc3339(),
            parameters: serde_json::json!({
                "plasticity": "STDP",
                "pruning_policy": "activity_30d",
                "visual_preset": "artistic"
            }),
        };
        let snapshot = BrainSnapshot {
            network: self.clone(),
            manifest,
            version: "1.0.0".to_string(),
            timestamp: Utc::now().to_rfc3339(),
            description: description.to_string(),
            format,
            encrypted: encrypt,
            compressed: compress,
        };

        let data = bincode::serialize(&snapshot)?;

        let processed_data = if compress {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&data)?;
            encoder.finish()?
        } else {
            data
        };

        let final_data = if encrypt {
            let mut key_bytes = [0u8; 32];
            rand::thread_rng().fill(&mut key_bytes);
            let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
            let cipher = Aes256Gcm::new(key);
            let nonce = Nonce::from(rand::random::<[u8; 12]>());
            let ciphertext = cipher.encrypt(&nonce, processed_data.as_ref()).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))) as Box<dyn std::error::Error>)?;
            // Prepend nonce and key (in real app, key should be derived from password)
            let mut encrypted = nonce.to_vec();
            encrypted.extend_from_slice(&key_bytes);
            encrypted.extend(ciphertext);
            encrypted
        } else {
            processed_data
        };

        let mut file = File::create(path)?;
        file.write_all(&final_data)?;
        Ok(())
    }

    pub fn load_snapshot(path: &str, key: Option<&[u8]>) -> Result<BrainSnapshot, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let decrypted_data = if let Some(_key_bytes) = key {
            if buffer.len() < 12 + 32 { // nonce + key
                return Err("Invalid encrypted data".into());
            }
            let nonce = &buffer[0..12];
            let stored_key = &buffer[12..44];
            let ciphertext = &buffer[44..];
            let cipher = Aes256Gcm::new_from_slice(stored_key).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))) as Box<dyn std::error::Error>)?;
            cipher.decrypt(Nonce::from_slice(nonce), ciphertext).map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))) as Box<dyn std::error::Error>)?
        } else {
            buffer
        };

        let decompressed_data = if decrypted_data.len() > 0 && decrypted_data[0] == 0x1f && decrypted_data[1] == 0x8b {
            let mut decoder = GzDecoder::new(&decrypted_data[..]);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            decompressed
        } else {
            decrypted_data
        };

        let snapshot: BrainSnapshot = bincode::deserialize(&decompressed_data)?;
        Ok(snapshot)
    }

    pub fn compare_snapshots(&self, other: &Network) -> SnapshotComparison {
        let neuron_diff = self.neurons.len() as i64 - other.neurons.len() as i64;
        let connection_diff = self.connections.len() as i64 - other.connections.len() as i64;
        let energy_diff = self.total_energy - other.total_energy;
        let pattern_diff = self.emergent_patterns.len() as i64 - other.emergent_patterns.len() as i64;

        SnapshotComparison {
            neuron_count_diff: neuron_diff,
            connection_count_diff: connection_diff,
            energy_diff,
            emergent_patterns_diff: pattern_diff,
        }
    }

    pub fn create_delta_snapshot(&self, previous: &Network) -> DeltaSnapshot {
        let new_neurons: Vec<Neuron> = self.neurons.iter()
            .filter(|n| !previous.neurons.iter().any(|pn| pn.id == n.id))
            .cloned()
            .collect();

        let new_connections: Vec<Connection> = self.connections.iter()
            .filter(|c| !previous.connections.iter().any(|pc| pc.from_id == c.from_id && pc.to_id == c.to_id))
            .cloned()
            .collect();

        let changed_weights: Vec<(u64, u64, f64)> = self.connections.iter()
            .filter_map(|c| {
                if let Some(pc) = previous.connections.iter().find(|pc| pc.from_id == c.from_id && pc.to_id == c.to_id) {
                    if (pc.weight - c.weight).abs() > 0.001 {
                        Some((c.from_id, c.to_id, c.weight))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        DeltaSnapshot {
            timestamp: Utc::now().to_rfc3339(),
            new_neurons,
            new_connections,
            changed_weights,
            energy_change: self.total_energy - previous.total_energy,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotComparison {
    pub neuron_count_diff: i64,
    pub connection_count_diff: i64,
    pub energy_diff: f64,
    pub emergent_patterns_diff: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JournalEntry {
    AddNeuron(Neuron),
    AddConnection(Connection),
    UpdateWeight(u64, u64, f64),
    RemoveConnection(u64, u64),
    Step,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaSnapshot {
    pub timestamp: String,
    pub new_neurons: Vec<Neuron>,
    pub new_connections: Vec<Connection>,
    pub changed_weights: Vec<(u64, u64, f64)>,
    pub energy_change: f64,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            speed: 1.0,
            precision: 0.001,
            pause: false,
            rewind: false,
            fast_forward: false,
            reset: false,
            autosave_frequency: 300, // 5 minutes
            energy_efficient_mode: false,
            cluster_resolution: 1.0,
            gpu_cpu_load_balancing: 0.5,
        }
    }
}

impl Network {
    pub fn new() -> Self {
        Self {
            neurons: vec![],
            connections: vec![],
            next_id: 1,
            memory: Memory::new(1000, 0.01),
            learning_rate: 0.01,
            emergent_patterns: vec![],
            total_energy: 0.0,
            simulation_speed: 1.0,
            mutation_rate: 0.05,
            adaptive_parameters: AdaptiveParams::default(),
            analytics: Analytics::default(),
              settings: SimulationSettings::default(),
              time: 0.0,
              input_mappings: HashMap::new(),
              journal: vec![],
        }
    }

    pub fn add_neuron(&mut self, neuron_type: NeuronType, activation_fn: ActivationFunction) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let neuron = Neuron::new(id, neuron_type, activation_fn);
        self.journal.push(JournalEntry::AddNeuron(neuron.clone()));
        self.neurons.push(neuron);
        id
    }

    pub fn add_connection(&mut self, from_id: u64, to_id: u64, weight: f64, plasticity: PlasticityType) {
        let connection = Connection::new(from_id, to_id, weight, plasticity);
        self.journal.push(JournalEntry::AddConnection(connection.clone()));
        self.connections.push(connection);
    }

    pub fn step(&mut self) {
        // Calculate new inputs
        let mut new_inputs = vec![0.0; self.neurons.len()];
        for conn in &self.connections {
            let from_idx = (conn.from_id - 1) as usize;
            let to_idx = (conn.to_id - 1) as usize;
            if from_idx < self.neurons.len() && to_idx < self.neurons.len() {
                new_inputs[to_idx] += self.neurons[from_idx].output * conn.weight;
            }
        }
        // Set inputs and activate
        let mut step_energy = 0.0;
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.input_sum = new_inputs[i];
            neuron.activate(self.time);
            step_energy += neuron.metadata.energy_cost;
        }
        self.time += 1.0;

        self.total_energy += step_energy;

        // Detect emergent patterns
        let firing_count = self.neurons.iter().filter(|n| n.should_fire()).count();
        if firing_count > self.neurons.len() / 2 && firing_count > 5 {
            self.emergent_patterns.push("High Activity Pattern".to_string());
        }
        // Detect rare events
        if firing_count < 2 && self.neurons.len() > 10 {
            self.analytics.rare_event_detections.push("Low Activity Event".to_string());
        }

        // Update analytics
        self.analytics.neuron_activity_histogram.push(firing_count as f64 / self.neurons.len() as f64);
        self.analytics.energy_tracking.push(step_energy);
        if self.analytics.neuron_activity_histogram.len() > 1000 {
            self.analytics.neuron_activity_histogram.remove(0);
        }
        if self.analytics.energy_tracking.len() > 1000 {
            self.analytics.energy_tracking.remove(0);
        }

        // Self-evolution: mutate parameters occasionally
        if rand::random::<f64>() < self.mutation_rate {
            self.adaptive_parameters.neuron_thresholds += (rand::random::<f64>() - 0.5) * 0.01;
            self.adaptive_parameters.neuron_thresholds = self.adaptive_parameters.neuron_thresholds.clamp(0.1, 1.0);
        }

        // Update weights
        for conn in &mut self.connections {
            let from_idx = (conn.from_id - 1) as usize;
            let to_idx = (conn.to_id - 1) as usize;
            if from_idx < self.neurons.len() && to_idx < self.neurons.len() {
                let pre = self.neurons[from_idx].output;
                let post = self.neurons[to_idx].output;
                let old_weight = conn.weight;
                conn.update_weight(pre, post, self.learning_rate);
                if (conn.weight - old_weight).abs() > 0.001 {
                    self.journal.push(JournalEntry::UpdateWeight(conn.from_id, conn.to_id, conn.weight));
                }
            }
        }

        // Update positions with force-directed layout
        self.update_positions();

        // Memory consolidation
        self.memory.consolidate();
    }

    pub fn parallel_step(&mut self) {
        // Parallel input calculation
        let new_inputs: Vec<f64> = self.connections.par_iter().fold(
            || vec![0.0; self.neurons.len()],
            |mut acc, conn| {
                let from_idx = (conn.from_id - 1) as usize;
                let to_idx = (conn.to_id - 1) as usize;
                if from_idx < self.neurons.len() && to_idx < self.neurons.len() {
                    acc[to_idx] += self.neurons[from_idx].output * conn.weight;
                }
                acc
            },
        ).reduce(
            || vec![0.0; self.neurons.len()],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );
        // Set inputs and activate in parallel
        self.neurons.par_iter_mut().zip(new_inputs).for_each(|(neuron, input)| {
            neuron.input_sum = input;
            neuron.activate(self.time);
        });
        self.time += 1.0;
    }

    pub fn grow_connections(&mut self, growth_rate: f64) {
        let mut rng = rand::thread_rng();
        let mut new_connections = vec![];
        for from_neuron in &self.neurons {
            if rng.gen::<f64>() < growth_rate * from_neuron.metadata.activity_score {
                let to_id = self.neurons[rng.gen_range(0..self.neurons.len())].id;
                if from_neuron.id != to_id && !self.connections.iter().any(|c| c.from_id == from_neuron.id && c.to_id == to_id) {
                    let plasticity = match rng.gen_range(0..8) {
                        0 => PlasticityType::Hebbian,
                        1 => PlasticityType::Stdp,
                        2 => PlasticityType::Ltp,
                        3 => PlasticityType::Ltd,
                        4 => PlasticityType::ReinforcementAdjusted,
                        5 => PlasticityType::DynamicRewiring,
                        6 => PlasticityType::MetaLearningAdjusted,
                        _ => PlasticityType::AiPredictiveRewiring,
                    };
                    new_connections.push(Connection::new(from_neuron.id, to_id, rng.gen_range(-0.5..0.5), plasticity));
                }
            }
        }
        self.connections.extend(new_connections);
    }

    pub fn prune_connections(&mut self, threshold: f64) {
        self.connections.retain(|c| c.weight.abs() > threshold);
    }

    pub fn autonomous_synaptogenesis(&mut self) {
        // Autonomous growth of new connections
        self.grow_connections(0.01);
    }

    pub fn user_guided_growth(&mut self, from_id: u64, to_id: u64) {
        if !self.connections.iter().any(|c| c.from_id == from_id && c.to_id == to_id) {
            self.connections.push(Connection::new(from_id, to_id, 0.5, PlasticityType::Hebbian));
        }
    }

    pub fn long_range_cross_cluster(&mut self) {
        let mut rng = rand::thread_rng();
        if self.neurons.len() > 10 {
            let from_idx = rng.gen_range(0..self.neurons.len());
            let to_idx = rng.gen_range(0..self.neurons.len());
            if from_idx != to_idx {
                let from_id = self.neurons[from_idx].id;
                let to_id = self.neurons[to_idx].id;
                if !self.connections.iter().any(|c| c.from_id == from_id && c.to_id == to_id) {
                    self.connections.push(Connection::new(from_id, to_id, rng.gen_range(-0.1..0.1), PlasticityType::DynamicRewiring));
                }
            }
        }
    }

    pub fn energy_efficient_pruning(&mut self) {
        // Prune based on energy cost and activity
        self.connections.retain(|c| {
            if let Some(from_idx) = self.neurons.iter().position(|n| n.id == c.from_id) {
                let energy_factor = self.neurons[from_idx].metadata.energy_cost;
                c.weight.abs() > 0.01 * energy_factor
            } else {
                false
            }
        });
    }

    pub fn ai_guided_pruning(&mut self) {
        // AI-guided pruning based on predictive importance
        self.connections.retain(|c| {
            // Simplified: keep connections with recent updates
            let time_since_update = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() - c.last_update;
            time_since_update < 3600 || c.weight.abs() > 0.1 // Keep if recent or strong
        });
    }

    pub fn probabilistic_synaptic_reallocation(&mut self) {
        // Rewire connections probabilistically
        let mut rng = rand::thread_rng();
        for conn in &mut self.connections {
            if rng.gen::<f64>() < 0.001 { // Low probability rewiring
                let new_to_id = self.neurons[rng.gen_range(0..self.neurons.len())].id;
                if new_to_id != conn.from_id {
                    conn.to_id = new_to_id;
                    conn.weight = rng.gen_range(-0.5..0.5);
                }
            }
        }
    }

    pub fn meta_learning_rewiring(&mut self) {
        // Meta-learning based rewiring
        let avg_activity = self.neurons.iter().map(|n| n.output).sum::<f64>() / self.neurons.len() as f64;
        if avg_activity < 0.1 {
            // Increase connectivity for low activity
            self.grow_connections(0.05);
        } else if avg_activity > 0.8 {
            // Prune for high activity
            self.prune_connections(0.05);
        }
    }

    pub fn ai_predictive_rewiring(&mut self) {
        // AI-guided predictive rewiring
        for conn in &mut self.connections {
            if let Some(from_idx) = self.neurons.iter().position(|n| n.id == conn.from_id) {
                if let Some(to_idx) = self.neurons.iter().position(|n| n.id == conn.to_id) {
                    let correlation = self.neurons[from_idx].output * self.neurons[to_idx].output;
                    if correlation < 0.0 {
                        // Negative correlation: consider rewiring
                        if rand::random::<f64>() < 0.01 {
                            let new_to_id = self.neurons[rand::random::<usize>() % self.neurons.len()].id;
                            if new_to_id != conn.from_id {
                                conn.to_id = new_to_id;
                                conn.weight = rand::random::<f64>() * 0.2 - 0.1;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn adaptive_parameter_tuning(&mut self) {
        // Adapt parameters based on performance
        let firing_rate = self.neurons.iter().filter(|n| n.should_fire()).count() as f64 / self.neurons.len() as f64;
        if firing_rate < 0.05 {
            self.adaptive_parameters.neuron_thresholds *= 0.99;
        } else if firing_rate > 0.5 {
            self.adaptive_parameters.neuron_thresholds *= 1.01;
        }
        self.adaptive_parameters.neuron_thresholds = self.adaptive_parameters.neuron_thresholds.clamp(0.1, 1.0);
    }

    pub fn emergent_pattern_tracking(&mut self) {
        // Track emergent patterns
        let activity_levels: Vec<f64> = self.neurons.iter().map(|n| n.output).collect();
        let mean = activity_levels.iter().sum::<f64>() / activity_levels.len() as f64;
        let variance = activity_levels.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / activity_levels.len() as f64;

        if variance > 0.5 {
            self.emergent_patterns.push("High Variance Pattern".to_string());
        }
        if activity_levels.iter().filter(|&&x| x > 0.8).count() > activity_levels.len() / 4 {
            self.emergent_patterns.push("Synchronized Burst".to_string());
        }
    }

    pub fn autonomous_experiments(&mut self) {
        // Run autonomous experiments
        if rand::random::<f64>() < 0.001 { // Rare autonomous actions
            match rand::random::<u32>() % 4 {
                0 => self.grow_connections(0.1),
                1 => self.prune_connections(0.01),
                2 => self.probabilistic_synaptic_reallocation(),
                _ => self.adaptive_parameter_tuning(),
            }
        }
    }

    pub fn process_input(&mut self, input: Input, mapping: Option<&Mapping>, plugins: &mut [Box<dyn plugins::Plugin>]) {
        // Map input to sensory neurons
        let sensory_neurons: Vec<usize> = self.neurons.iter().enumerate()
            .filter(|(_, n)| matches!(n.neuron_type, NeuronType::Sensory))
            .map(|(i, _)| i)
            .collect();

        let data = match input {
            Input::Text(ref text) => {
                match mapping {
                    Some(Mapping { strategy: MappingStrategy::AutoEmbed, .. }) => {
                        // Simple embedding: char codes
                        text.chars().map(|c| c as u32 as f64 / 255.0).collect()
                    }
                    Some(Mapping { strategy: MappingStrategy::HashSeed, .. }) => {
                        // Hash to seed
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        text.hash(&mut hasher);
                        let hash = hasher.finish();
                        (0..10).map(|i| ((hash >> (i * 6)) & 63) as f64 / 63.0).collect()
                    }
                    Some(Mapping { strategy: MappingStrategy::ManualPaint, manual_brush: Some(brush), .. }) => {
                        // Use brush intensities
                        brush.iter().map(|&(_, _, intensity)| intensity as f64).collect()
                    }
                    _ => text.chars().map(|c| c as u32 as f64 / 255.0).collect(),
                }
            },
            Input::Numbers(ref nums) => nums.clone(),
            Input::AsciiArt(ref art) => {
                // Convert ASCII art to vector (simplified)
                art.chars().filter(|&c| c != '\n').map(|c| if c == '#' { 1.0 } else { 0.0 }).collect()
            },
            Input::Bitmap(ref bitmap) => {
                // Convert bitmap to vector
                bitmap.iter().map(|&b| b as f64 / 255.0).collect()
            },
            Input::Vector(ref vec) => vec.clone(),
            Input::Gesture(ref vec) => vec.clone(),
            Input::Voice(ref vec) => vec.clone(),
            Input::Sensor(ref vec) => vec.clone(),
            Input::Custom(ref vec) => vec.clone(),
        };

        // Distribute data to sensory neurons
        for (i, &idx) in sensory_neurons.iter().enumerate() {
            if i < data.len() {
                self.neurons[idx].input_sum += data[i];
            }
        }

        // If no sensory neurons, create some
        if sensory_neurons.is_empty() && !data.is_empty() {
            let num_to_create = data.len().min(10);
            for i in 0..num_to_create {
                let id = self.add_neuron(NeuronType::Sensory, ActivationFunction::Relu);
                if let Some(neuron) = self.neurons.last_mut() {
                    neuron.input_sum = data[i] * 0.1;
                }
            }
        }

        // Run steps based on simulation speed
        let steps = (10.0 * self.simulation_speed) as usize;
        for _ in 0..steps {
            self.step();
            self.autonomous_experiments();
            self.emergent_pattern_tracking();
        }

        // Collect output from motor neurons
        let motor_neurons: Vec<usize> = self.neurons.iter().enumerate()
            .filter(|(_, n)| matches!(n.neuron_type, NeuronType::Motor))
            .map(|(i, _)| i)
            .collect();
        let output: Vec<f64> = if motor_neurons.is_empty() {
            self.neurons.iter().map(|n| n.output).collect()
        } else {
            motor_neurons.iter().map(|&i| self.neurons[i].output).collect()
        };

        self.memory.add_interaction(input, output);

        // Self-optimization
        self.connection_weight_compression();
        self.energy_efficient_firing();
        self.pattern_prioritization();
    }

    pub fn connection_weight_compression(&mut self) {
        // Compress connection weights for efficiency
        for conn in &mut self.connections {
            if conn.weight.abs() < 0.01 {
                conn.weight *= 0.9; // Compress small weights
            }
        }
    }

    pub fn energy_efficient_firing(&mut self) {
        // Adjust thresholds for energy efficiency
        for neuron in &mut self.neurons {
            if self.total_energy > 1000.0 {
                neuron.metadata.firing_threshold *= 1.01; // Increase threshold to reduce firing
            }
        }
    }

    pub fn pattern_prioritization(&mut self) {
        // Prioritize important patterns
        let high_activity_neurons: Vec<usize> = self.neurons.iter().enumerate()
            .filter(|(_, n)| n.output > 0.7)
            .map(|(i, _)| i)
            .collect();

        for &idx in &high_activity_neurons {
            self.neurons[idx].metadata.ai_relevance_score += 0.01;
        }
    }

    pub fn update_positions(&mut self) {
        let damping = 0.9;
        let attraction_strength = 0.01;
        let repulsion_strength = 1.0;
        let max_force = 1.0;

        // Calculate forces
        let mut forces: Vec<(f32, f32, f32)> = vec![(0.0, 0.0, 0.0); self.neurons.len()];

        // Repulsion between all pairs
        for i in 0..self.neurons.len() {
            for j in (i+1)..self.neurons.len() {
                let dx = self.neurons[j].position.0 - self.neurons[i].position.0;
                let dy = self.neurons[j].position.1 - self.neurons[i].position.1;
                let dz = self.neurons[j].position.2 - self.neurons[i].position.2;
                let dist_sq = dx*dx + dy*dy + dz*dz + 1.0; // Avoid division by zero
                let dist = dist_sq.sqrt();
                let force = repulsion_strength / dist_sq;
                let fx = force * dx / dist;
                let fy = force * dy / dist;
                let fz = force * dz / dist;
                forces[i].0 -= fx.min(max_force).max(-max_force);
                forces[i].1 -= fy.min(max_force).max(-max_force);
                forces[i].2 -= fz.min(max_force).max(-max_force);
                forces[j].0 += fx.min(max_force).max(-max_force);
                forces[j].1 += fy.min(max_force).max(-max_force);
                forces[j].2 += fz.min(max_force).max(-max_force);
            }
        }

        // Attraction for connections
        for conn in &self.connections {
            if let Some(from_idx) = self.neurons.iter().position(|n| n.id == conn.from_id) {
                if let Some(to_idx) = self.neurons.iter().position(|n| n.id == conn.to_id) {
                    let dx = self.neurons[to_idx].position.0 - self.neurons[from_idx].position.0;
                    let dy = self.neurons[to_idx].position.1 - self.neurons[from_idx].position.1;
                    let dz = self.neurons[to_idx].position.2 - self.neurons[from_idx].position.2;
                    let dist = (dx*dx + dy*dy + dz*dz).sqrt() + 1.0;
                    let force = attraction_strength * conn.weight.abs() as f32 * dist;
                    let fx = force * dx / dist;
                    let fy = force * dy / dist;
                    let fz = force * dz / dist;
                    forces[from_idx].0 += fx.min(max_force).max(-max_force);
                    forces[from_idx].1 += fy.min(max_force).max(-max_force);
                    forces[from_idx].2 += fz.min(max_force).max(-max_force);
                    forces[to_idx].0 -= fx.min(max_force).max(-max_force);
                    forces[to_idx].1 -= fy.min(max_force).max(-max_force);
                    forces[to_idx].2 -= fz.min(max_force).max(-max_force);
                }
            }
        }

        // Update velocities and positions
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.velocity.0 = (neuron.velocity.0 + forces[i].0) * damping;
            neuron.velocity.1 = (neuron.velocity.1 + forces[i].1) * damping;
            neuron.velocity.2 = (neuron.velocity.2 + forces[i].2) * damping;
            neuron.position.0 += neuron.velocity.0;
            neuron.position.1 += neuron.velocity.1;
            neuron.position.2 += neuron.velocity.2;
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let network: Network = bincode::deserialize(&buffer)?;
        Ok(network)
    }

    pub fn replay_journal(&mut self, journal: &[JournalEntry]) {
        for entry in journal {
            match entry {
                JournalEntry::AddNeuron(neuron) => {
                    self.neurons.push(neuron.clone());
                    if neuron.id >= self.next_id {
                        self.next_id = neuron.id + 1;
                    }
                }
                JournalEntry::AddConnection(conn) => {
                    self.connections.push(conn.clone());
                }
                JournalEntry::UpdateWeight(from, to, weight) => {
                    if let Some(conn) = self.connections.iter_mut().find(|c| c.from_id == *from && c.to_id == *to) {
                        conn.weight = *weight;
                    }
                }
                JournalEntry::RemoveConnection(from, to) => {
                    self.connections.retain(|c| !(c.from_id == *from && c.to_id == *to));
                }
                JournalEntry::Step => {
                    self.step();
                }
            }
        }
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_creation() {
        let neuron = Neuron::new(1, NeuronType::Excitatory, ActivationFunction::Sigmoid);
        assert_eq!(neuron.id, 1);
    }

    #[test]
    fn neuron_activation() {
        let mut neuron = Neuron::new(1, NeuronType::Excitatory, ActivationFunction::Relu);
        neuron.input_sum = 1.0;
        let output = neuron.activate(0.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn network_step() {
        let mut network = Network::new();
        let id1 = network.add_neuron(NeuronType::Excitatory, ActivationFunction::Relu);
        let id2 = network.add_neuron(NeuronType::Excitatory, ActivationFunction::Relu);
        network.add_connection(id1, id2, 1.0, PlasticityType::Hebbian);
        network.neurons[0].output = 1.0; // Set initial output
        network.step();
        assert!(network.neurons[1].output > 0.0);
    }

    #[test]
    fn large_network_performance() {
        let mut network = Network::new();
        // Create 1000 neurons
        let mut ids = Vec::new();
        for i in 0..1000 {
            let id = network.add_neuron(NeuronType::Excitatory, ActivationFunction::Relu);
            ids.push(id);
        }
        // Add connections (sparse)
        for i in 0..ids.len() {
            for j in (i+1)..(i+11).min(ids.len()) {
                network.add_connection(ids[i], ids[j], 0.1, PlasticityType::Hebbian);
            }
        }
        // Time 100 steps
        let start = std::time::Instant::now();
        for _ in 0..100 {
            network.step();
        }
        let duration = start.elapsed();
        println!("100 steps with 1000 neurons took: {:?}", duration);
        // Performance note: In release build, this should be much faster
    }
}
