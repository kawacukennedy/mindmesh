use iced::{Application, Command, Element, Settings, Subscription, Theme, Color, keyboard};
use iced::widget::{canvas, Canvas};
use iced::time;
use rand::Rng;
use mindmesh_core::{Network, NeuronType, ActivationFunction, PlasticityType, Input};
use rodio::{OutputStream, Sink, Source};
use serialport::SerialPort;

pub fn main() -> iced::Result {
    MindMesh::run(Settings::default())
}

#[derive(Clone)]
struct Particle {
    position: (f32, f32),
    velocity: (f32, f32),
    life: f32,
    max_life: f32,
}

struct MindMesh {
    network: Network,
    input_text: String,
    import_text: String,
    zoom: f32,
    pan: (f32, f32),
    paused: bool,
    speed: f32,
    steps_taken: u64,
    dark_theme: bool,
    high_contrast: bool,
    colorblind_mode: Option<String>, // "protanopia", "deuteranopia", "tritanopia"
    font_scale: f32,
    i18n: std::collections::HashMap<String, String>, // Simple localization
    show_connections: bool,
    predicted_output: Vec<f64>,
    ai_suggestion: String,
    show_3d: bool,
    leaderboard: Vec<(String, u64)>,
    show_settings: bool,
    show_analytics: bool,
    show_snapshot_manager: bool,
    snapshots: Vec<String>,
    current_snapshot: Option<String>,
    energy_efficient_mode: bool,
    notifications: Vec<String>,
    achievements: Vec<String>,
    gesture_mode: bool,
    voice_mode: bool,
    vr_mode: bool,
    hardware_connected: bool,
    ar_mode: bool,
    bci_mode: bool,
    holographic_mode: bool,
    particles: Vec<Particle>,
    _stream: OutputStream,
    sink: Sink,
    sonification: bool,
    serial_port: Option<Box<dyn SerialPort>>,
    plugins: Vec<Box<dyn mindmesh_core::plugins::Plugin>>,
    // New fields for spec compliance
    show_inspector: bool,
    selected_neuron: Option<u64>,
    show_input_panel: bool,
    show_simulation_controls: bool,
    show_export_ai: bool,
    log_console: Vec<(String, u64)>, // (message, timestamp)
    undo_stack: Vec<mindmesh_core::Network>,
    redo_stack: Vec<mindmesh_core::Network>,
    global_search: String,
    lod_slider: f32,
    playback_scrubber: f32,
    visualization_overlays: bool,
    sandbox_network: Option<mindmesh_core::Network>,
    show_autonomous_modal: bool,
    autonomous_preset: String,
    autonomous_energy_budget: f64,
    autonomous_time_limit: u64,
    autonomous_detection_sensitivity: f32,
    autonomous_logging_level: String,
    profiling_mode: bool,
    show_ethics_modal: bool,
    show_onboarding: bool,
    onboarding_step: usize,
    show_mapping_wizard: bool,
    mapping_input_type: String,
    mapping_strategy: String,
    show_export_wizard: bool,
    export_format: String,
    export_delta_only: bool,
    export_compress: f32,
    export_encryption: bool,
    show_shortcuts: bool,
    log_filter: String,
    show_help: bool,
    help_section: String,
    help_search: String,
    show_contextual_hint: bool,
    contextual_hint: String,
    last_activity: u64,
    show_collaboration: bool,
    deterministic_mode: bool,
    energy_budget: f64,
    rewire_mode: bool,
    selected_neurons_for_rewire: Vec<u64>,
    reduced_motion: bool,
    low_power_mode: bool,
    vr_ar_mode: bool,
}

#[derive(Clone)]
struct AppTheme {
    primary: Color,
    secondary: Color,
    accent: Color,
    success: Color,
    warning: Color,
    error: Color,
    background: Color,
    surface: Color,
    text: Color,
    border: Color,
}

impl AppTheme {
    fn light() -> Self {
        Self {
            primary: Color::from_rgb(0x5A as f32 / 255.0, 0xB4 as f32 / 255.0, 0xFF as f32 / 255.0), // #5AB4FF accent_secondary
            secondary: Color::from_rgb(0.6, 0.6, 0.6), // Light gray
            accent: Color::from_rgb(0x7B as f32 / 255.0, 0xD3 as f32 / 255.0, 0x89 as f32 / 255.0), // #7BD389 accent
            success: Color::from_rgb(0x7B as f32 / 255.0, 0xD3 as f32 / 255.0, 0x89 as f32 / 255.0), // reuse accent
            warning: Color::from_rgb(0xFF as f32 / 255.0, 0xC8 as f32 / 255.0, 0x57 as f32 / 255.0), // #FFC857 warning
            error: Color::from_rgb(0xFF as f32 / 255.0, 0x6B as f32 / 255.0, 0x6B as f32 / 255.0), // #FF6B6B danger
            background: Color::from_rgb(0.95, 0.95, 0.95), // Light background
            surface: Color::from_rgb(1.0, 1.0, 1.0), // White surface
            text: Color::from_rgb(0.1, 0.1, 0.1), // Dark text
            border: Color::from_rgb(0.8, 0.8, 0.8), // Light border
        }
    }

    fn dark() -> Self {
        Self {
            primary: Color::from_rgb(0x7B as f32 / 255.0, 0xD3 as f32 / 255.0, 0x89 as f32 / 255.0), // #7BD389 accent
            secondary: Color::from_rgb(0x5A as f32 / 255.0, 0xB4 as f32 / 255.0, 0xFF as f32 / 255.0), // #5AB4FF accent_secondary
            accent: Color::from_rgb(0x7B as f32 / 255.0, 0xD3 as f32 / 255.0, 0x89 as f32 / 255.0), // #7BD389 accent
            success: Color::from_rgb(0x7B as f32 / 255.0, 0xD3 as f32 / 255.0, 0x89 as f32 / 255.0), // reuse accent for success
            warning: Color::from_rgb(0xFF as f32 / 255.0, 0xC8 as f32 / 255.0, 0x57 as f32 / 255.0), // #FFC857 warning
            error: Color::from_rgb(0xFF as f32 / 255.0, 0x6B as f32 / 255.0, 0x6B as f32 / 255.0), // #FF6B6B danger
            background: Color::from_rgb(0x0B as f32 / 255.0, 0x0F as f32 / 255.0, 0x14 as f32 / 255.0), // #0B0F14 background
            surface: Color::from_rgba(0x12 as f32 / 255.0, 0x16 as f32 / 255.0, 0x1C as f32 / 255.0, 0.75), // rgba(18,22,28,0.75) panel_bg
            text: Color::from_rgb(0x9A as f32 / 255.0, 0xA7 as f32 / 255.0, 0xB2 as f32 / 255.0), // #9AA7B2 muted_text
            border: Color::from_rgb(37.0/255.0, 37.0/255.0, 48.0/255.0), // approximate border
        }
    }
    }

    fn high_contrast() -> Self {
        Self {
            primary: Color::from_rgb(0.0, 1.0, 1.0),
            secondary: Color::from_rgb(1.0, 1.0, 0.0),
            accent: Color::from_rgb(1.0, 0.0, 1.0),
            success: Color::from_rgb(0.0, 1.0, 0.0),
            warning: Color::from_rgb(1.0, 1.0, 0.0),
            error: Color::from_rgb(1.0, 0.0, 0.0),
            background: Color::BLACK,
            surface: Color::WHITE,
            text: Color::BLACK,
            border: Color::WHITE,
        }
    }

    fn protanopia() -> Self {
        // Colorblind: red-green, red weak
        Self {
            primary: Color::from_rgb(0.0, 0.5, 1.0), // Blue instead of red
            secondary: Color::from_rgb(0.5, 0.5, 0.5),
            accent: Color::from_rgb(1.0, 0.5, 0.0), // Orange
            success: Color::from_rgb(0.0, 0.8, 0.0),
            warning: Color::from_rgb(1.0, 0.8, 0.0),
            error: Color::from_rgb(0.8, 0.0, 0.8), // Magenta instead of red
            background: Color::from_rgb(0.95, 0.95, 0.95),
            surface: Color::from_rgb(1.0, 1.0, 1.0),
            text: Color::from_rgb(0.1, 0.1, 0.1),
            border: Color::from_rgb(0.8, 0.8, 0.8),
        }
    }

    fn deuteranopia() -> Self {
        // Colorblind: red-green, green weak
        Self {
            primary: Color::from_rgb(0.2, 0.6, 0.8),
            secondary: Color::from_rgb(0.6, 0.6, 0.6),
            accent: Color::from_rgb(1.0, 0.6, 0.2),
            success: Color::from_rgb(0.2, 0.9, 0.2),
            warning: Color::from_rgb(1.0, 0.9, 0.2),
            error: Color::from_rgb(0.9, 0.2, 0.2),
            background: Color::from_rgb(0.1, 0.1, 0.1),
            surface: Color::from_rgb(0.2, 0.2, 0.2),
            text: Color::from_rgb(0.9, 0.9, 0.9),
            border: Color::from_rgb(0.3, 0.3, 0.3),
        }
    }

    fn tritanopia() -> Self {
        // Colorblind: blue-yellow
        Self {
            primary: Color::from_rgb(0.3, 0.5, 0.7),
            secondary: Color::from_rgb(0.5, 0.5, 0.5),
            accent: Color::from_rgb(0.8, 0.6, 0.0),
            success: Color::from_rgb(0.0, 0.7, 0.0),
            warning: Color::from_rgb(0.8, 0.7, 0.0),
            error: Color::from_rgb(0.7, 0.0, 0.0),
            background: Color::from_rgb(0.95, 0.95, 0.95),
            surface: Color::from_rgb(1.0, 1.0, 1.0),
            text: Color::from_rgb(0.1, 0.1, 0.1),
            border: Color::from_rgb(0.8, 0.8, 0.8),
        }
    }


}



#[derive(Debug, Clone)]
enum Message {
    AddNeuron,
    AddConnection,
    Step,
    InputChanged(String),
    ProcessInput,
    ProcessRandomNumbers,
    ProcessAscii,
    ProcessVoice,
    Save,
    Load,
    ZoomIn,
    ZoomOut,
    TogglePause,
    SpeedUp,
    SpeedDown,
    Reset,
    Evolve,
    ToggleTheme,
    ToggleHighContrast,
    SetColorblindMode(Option<String>),
    SetFontScale(f32),
    AiAssist,
    ExportJson,
    ImportChanged(String),
    ImportJson,
    ToggleConnections,
    Predict,
    ProcessBCI,
    SpeedChanged(f32),
    Toggle3D,
    ExportSummary,
    VrMode,
    NetworkMode,
    HardwareConnect,
    AddToLeaderboard,
    ToggleSettings,
    ToggleAnalytics,
    ToggleSnapshotManager,
    SaveSnapshot(String),
    LoadSnapshot(String),
    DeleteSnapshot(String),
    CompareSnapshots,
    ToggleEnergyMode,
    ClearNotifications,
    ToggleGestureMode,
    ToggleVoiceMode,
    ToggleVrMode,
    ToggleSonification,
    ToggleArMode,
    ToggleBciMode,
    ToggleHolographicMode,
    ProcessGesture,
    ProcessVoiceCommand(String),
    ExportGif,
    ExportMp4,
    ExportVrScene,
    RunAnalytics,
    ShowAchievement(String),
    ProcessSensorInput,
    AutonomousEvolution,
    MetaLearning,
    AiGuidedRewiring,
    // New messages for spec UI
    NewProject,
    OpenProject,
    SaveAs,
    CommandPalette,
    ToggleConsole,
    FocusSelection,
    Toggle2D3D,
    PaintMapping,
    EraseMapping,
    Duplicate,
    Delete,
    Undo,
    Redo,
    StepForward,
    StepBack,
    SpeedUpSim,
    SpeedDownSim,
    ToggleInspector,
    SelectNeuron(u64),
    ToggleInputPanel,
    ToggleSimulationControls,
    ToggleExportAi,
    GlobalSearchChanged(String),
    LodChanged(f32),
    PlaybackScrubberChanged(f32),
    ToggleVisualizationOverlays,
    StartSandboxEdit,
    CommitSandbox,
    RevertSandbox,
    ToggleAutonomousModal,
    SetAutonomousPreset(String),
    SetAutonomousEnergyBudget(f64),
    SetAutonomousTimeLimit(u64),
    SetAutonomousDetectionSensitivity(f32),
    SetAutonomousLoggingLevel(String),
    StartAutonomousExperiment,
    ToggleProfilingMode,
    ToggleEthicsModal,
    NextOnboardingStep,
    SkipOnboarding,
    ShowMappingWizard,
    SetMappingInputType(String),
    SetMappingStrategy(String),
    ApplyMapping,
    ShowExportWizard,
    SetExportFormat(String),
    ToggleExportDeltaOnly,
    SetExportCompress(f32),
    ToggleExportEncryption,
    StartExport,
    ToggleShortcuts,
    LogFilterChanged(String),
    SimulateVrGesture,
    ToggleDeterministicMode,
    SetEnergyBudget(f64),
    ToggleRewireMode,
    SelectNeuronForRewire(u64),
    ToggleReducedMotion,
    ToggleLowPowerMode,
    ToggleVrArMode,
    ToggleHelp,
    SetHelpSection(String),
    HelpSearchChanged(String),
    UpdateActivity,
    DismissHint,
    ToggleCollaboration,
}

impl Application for MindMesh {
    type Message = Message;
    type Theme = iced::Theme;
    type Executor = iced::executor::Default;
    type Flags = ();

    fn new(_flags: ()) -> (MindMesh, Command<Message>) {
        let mut network = Network::new();
        // Add some initial neurons
        for _ in 0..10 {
            network.add_neuron(NeuronType::Excitatory, ActivationFunction::Relu);
        }
        for i in 0..9 {
            network.add_connection(i, i+1, 0.5, PlasticityType::Hebbian);
        }
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let sink = Sink::try_new(&stream_handle).unwrap();
        (MindMesh {
            network,
            input_text: String::new(),
            import_text: String::new(),
            zoom: 1.0,
            pan: (0.0, 0.0),
            paused: true,
            speed: 1.0,
            steps_taken: 0,
             dark_theme: false,
              high_contrast: false,
              colorblind_mode: None,
              font_scale: 1.0,
              i18n: {
                  let mut map = std::collections::HashMap::new();
                  map.insert("mindmesh_title".to_string(), "🧠 MindMesh".to_string());
                  map.insert("play".to_string(), "▶️ Play".to_string());
                  map.insert("pause".to_string(), "⏸️ Pause".to_string());
                  map.insert("reset".to_string(), "🔄 Reset".to_string());
                  map.insert("save".to_string(), "💾 Save".to_string());
                  map.insert("load".to_string(), "📂 Load".to_string());
                  // Add more as needed
                  map
              },
            show_connections: true,
            predicted_output: vec![],
            ai_suggestion: String::new(),
            show_3d: false,
            leaderboard: vec![],
            show_settings: false,
            show_analytics: false,
            show_snapshot_manager: false,
            snapshots: vec![],
            current_snapshot: None,
            energy_efficient_mode: false,
            notifications: vec![],
            achievements: vec![],
            gesture_mode: false,
            voice_mode: false,
            vr_mode: false,
            hardware_connected: false,
            ar_mode: false,
            bci_mode: false,
            holographic_mode: false,
            particles: vec![],
            _stream,
            sink,
            sonification: false,
             serial_port: None,
             show_inspector: false,
             selected_neuron: None,
             show_input_panel: true,
             show_simulation_controls: true,
             show_export_ai: false,
             log_console: vec![],
             undo_stack: vec![],
             redo_stack: vec![],
             global_search: String::new(),
             lod_slider: 1.0,
             playback_scrubber: 0.0,
             visualization_overlays: true,
             plugins: vec![],
             sandbox_network: None,
             show_autonomous_modal: false,
             autonomous_preset: "Dream".to_string(),
              autonomous_energy_budget: 100.0,
              autonomous_time_limit: 300,
              autonomous_detection_sensitivity: 0.5,
              autonomous_logging_level: "Medium".to_string(),
             profiling_mode: false,
             show_ethics_modal: false,
             show_onboarding: true, // Show on first run
             onboarding_step: 0,
             show_mapping_wizard: false,
             mapping_input_type: "Text".to_string(),
             mapping_strategy: "Auto-embed".to_string(),
               show_export_wizard: false,
                export_format: "JSON".to_string(),
                export_delta_only: false,
                export_compress: 0.5,
                export_encryption: false,
                show_shortcuts: false,
                log_filter: String::new(),
                show_help: false,
                help_section: "Overview".to_string(),
                help_search: String::new(),
                show_contextual_hint: false,
                contextual_hint: String::new(),
                last_activity: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                show_collaboration: false,
                deterministic_mode: false,
                energy_budget: 100.0,
                rewire_mode: false,
                selected_neurons_for_rewire: vec![],
                reduced_motion: false,
                low_power_mode: false,
                vr_ar_mode: false,
         }, Command::none())
    }

    fn title(&self) -> String {
        String::from("MindMesh")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::AddNeuron => {
                self.network.add_neuron(NeuronType::Excitatory, ActivationFunction::Relu);
            }
            Message::AddConnection => {
                if self.network.neurons.len() >= 2 {
                    let from = self.network.neurons.len() - 2;
                    let to = self.network.neurons.len() - 1;
                    self.network.add_connection(from as u64, to as u64, 0.5, PlasticityType::Hebbian);
                }
            }
            Message::Step => {
                self.network.step();
                self.steps_taken += 1;
                self.log_console.push((format!("Step {}: {} neurons firing", self.steps_taken, self.network.neurons.iter().filter(|n| n.should_fire()).count()), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()));
                if self.log_console.len() > 100 {
                    self.log_console.remove(0);
                }
                // Add particles for firing neurons
                for neuron in &self.network.neurons {
                    if neuron.should_fire() {
                        let pos = (
                            (neuron.position.0 * self.zoom + self.pan.0) + 400.0,
                            (neuron.position.1 * self.zoom + self.pan.1) + 300.0,
                        );
                        for _ in 0..3 {
                            let angle = rand::random::<f32>() * 2.0 * std::f32::consts::PI;
                            let speed = rand::random::<f32>() * 2.0 + 1.0;
                            self.particles.push(Particle {
                                position: pos,
                                velocity: (angle.cos() * speed, angle.sin() * speed),
                                life: 60.0,
                                max_life: 60.0,
                            });
                        }
                        // Add spark particles for connections
                        for conn in &self.network.connections {
                            if conn.from_id == neuron.id || conn.to_id == neuron.id {
                                if rand::random::<f32>() < 0.1 { // Occasional sparks
                                    let spark_pos = if conn.from_id == neuron.id { pos } else {
                                        if let Some(to_idx) = self.network.neurons.iter().position(|n| n.id == conn.to_id) {
                                            (
                                                (self.network.neurons[to_idx].position.0 * self.zoom + self.pan.0) + 400.0,
                                                (self.network.neurons[to_idx].position.1 * self.zoom + self.pan.1) + 300.0,
                                            )
                                        } else { pos }
                                    };
                                    self.particles.push(Particle {
                                        position: spark_pos,
                                        velocity: (rand::random::<f32>() * 4.0 - 2.0, rand::random::<f32>() * 4.0 - 2.0),
                                        life: 30.0,
                                        max_life: 30.0,
                                    });
                                }
                            }
                        }
                    }
                }
                // Update particles
                for p in &mut self.particles {
                    p.position.0 += p.velocity.0;
                    p.position.1 += p.velocity.1;
                    p.life -= 1.0;
                }
                self.particles.retain(|p| p.life > 0.0);
                // Sonification
                if self.sonification {
                    let firing_count = self.network.neurons.iter().filter(|n| n.should_fire()).count();
                    if firing_count > 0 {
                        let freq = 440.0 + (firing_count as f32 * 10.0).min(1000.0);
                        let source = rodio::source::SineWave::new(freq).take_duration(std::time::Duration::from_millis(100)).amplify(0.1);
                        self.sink.append(source);
                        self.log_console.push((format!("Sonification: {} neurons firing at {:.0}Hz", firing_count, freq), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()));
                    }
                }
            }
            Message::InputChanged(text) => {
                self.input_text = text;
            }
            Message::ProcessInput => {
                if !self.input_text.is_empty() {
                    let input = Input::Text(self.input_text.clone());
                    self.network.process_input(input, None, &mut self.plugins);
                    self.input_text.clear();
                }
            }
            Message::Save => {
                let _ = self.network.save("brain.mm");
            }
            Message::Load => {
                match Network::load("brain.mm") {
                    Ok(net) => {
                        self.network = net;
                        self.notifications.push("Project loaded successfully".to_string());
                    }
                    Err(e) => {
                        self.notifications.push(format!("Load failed: {}", e));
                    }
                }
            }
            Message::ZoomIn => {
                self.zoom *= 1.1;
            }
            Message::ZoomOut => {
                self.zoom /= 1.1;
            }
            Message::TogglePause => {
                self.paused = !self.paused;
            }
            Message::SpeedUp => {
                self.speed *= 1.1;
            }
            Message::SpeedDown => {
                self.speed /= 1.1;
            }
            Message::Reset => {
                self.network = Network::new();
                for _ in 0..10 {
                    self.network.add_neuron(NeuronType::Excitatory, ActivationFunction::Relu);
                }
                for i in 0..9 {
                    self.network.add_connection(i as u64, (i+1) as u64, 0.5, PlasticityType::Hebbian);
                }
            }
            Message::Evolve => {
                self.network.grow_connections(0.1);
                self.network.prune_connections(0.01);
            }
            Message::ProcessRandomNumbers => {
                let mut rng = rand::thread_rng();
                let numbers: Vec<f64> = (0..10).map(|_| rng.gen()).collect();
                self.network.process_input(Input::Numbers(numbers), None, &mut self.plugins);
            }
            Message::ProcessAscii => {
                self.network.process_input(Input::AsciiArt(":-)".to_string()), None, &mut self.plugins);
            }
            Message::ProcessVoice => {
                let voice_data = vec![0.5; 10]; // Simulated voice input
                self.network.process_input(Input::Voice(voice_data), None, &mut self.plugins);
            }
            Message::ToggleTheme => {
                self.dark_theme = !self.dark_theme;
            }
            Message::ToggleHighContrast => {
                self.high_contrast = !self.high_contrast;
            }
            Message::SetColorblindMode(mode) => {
                self.colorblind_mode = mode;
            }
            Message::SetFontScale(scale) => {
                self.font_scale = scale;
            }
            Message::AiAssist => {
                let avg_activity = self.network.neurons.iter().map(|n| n.output).sum::<f64>() / self.network.neurons.len() as f64;
                let firing_rate = self.network.neurons.iter().filter(|n| n.should_fire()).count() as f64 / self.network.neurons.len() as f64;
                let connection_density = self.network.connections.len() as f64 / (self.network.neurons.len() as f64).powi(2);
                let suggestions = vec![
                    if avg_activity < 0.1 { "Low activity detected. Consider adding more neurons or increasing input stimulation." } else { "" },
                    if avg_activity > 0.8 { "High activity. Prune weak connections or add inhibitory neurons." } else { "" },
                    if firing_rate < 0.05 { "Low firing rate. Adjust thresholds or plasticity parameters." } else { "" },
                    if connection_density > 0.1 { "High connection density. Enable pruning to improve efficiency." } else { "" },
                    "Experiment with different neuron types for emergent patterns.",
                ].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>();
                self.ai_suggestion = if suggestions.is_empty() { "Network performing well.".to_string() } else { suggestions.join(" ") };
            }
            Message::ExportJson => {
                if let Ok(json) = serde_json::to_string(&self.network) {
                    println!("Exported JSON: {}", json); // In real app, copy to clipboard or file
                }
            }
            Message::ImportChanged(text) => {
                self.import_text = text;
            }
            Message::ImportJson => {
                if let Ok(net) = serde_json::from_str(&self.import_text) {
                    self.network = net;
                }
            }
            Message::ToggleConnections => {
                self.show_connections = !self.show_connections;
            }
            Message::Predict => {
                for _ in 0..5 {
                    self.network.step();
                    self.steps_taken += 1;
                }
                self.predicted_output = self.network.neurons.iter().map(|n| n.output).collect();
            }
            Message::ProcessBCI => {
                let mut rng = rand::thread_rng();
                let bci_data: Vec<f64> = (0..20).map(|_| rng.gen()).collect();
                self.network.process_input(Input::Sensor(bci_data), None, &mut self.plugins);
            }
            Message::SpeedChanged(new_speed) => {
                self.speed = new_speed;
            }
            Message::Toggle3D => {
                self.show_3d = !self.show_3d;
            }
            Message::ExportSummary => {
                let avg_activity = self.network.neurons.iter().map(|n| n.output).sum::<f64>() / self.network.neurons.len() as f64;
                println!("MindMesh Network Summary:");
                println!("Neurons: {}", self.network.neurons.len());
                println!("Connections: {}", self.network.connections.len());
                println!("Avg Activity: {:.2}", avg_activity);
                println!("Steps Taken: {}", self.steps_taken);
                println!("Achievements: {}", if self.steps_taken > 100 { "100 Steps" } else { "None" });
            }
            Message::VrMode => {
                self.notifications.push("VR Mode: Immersive 3D visualization enabled".to_string());
            }
            Message::NetworkMode => {
                println!("Network Mode: Share brain snapshots via JSON for offline collaboration.");
            }
            Message::HardwareConnect => {
                println!("Hardware Connect: Simulated BCI and sensors active.");
            }
            Message::AddToLeaderboard => {
                self.leaderboard.push(("User".to_string(), self.steps_taken));
                self.leaderboard.sort_by(|a, b| b.1.cmp(&a.1));
                self.leaderboard.truncate(5);
            }
            Message::ToggleSettings => {
                self.show_settings = !self.show_settings;
            }
            Message::ToggleAnalytics => {
                self.show_analytics = !self.show_analytics;
            }
            Message::ToggleSnapshotManager => {
                self.show_snapshot_manager = !self.show_snapshot_manager;
            }
            Message::SaveSnapshot(description) => {
                let filename = format!("snapshot_{}.mm", chrono::Utc::now().timestamp());
                let _ = self.network.save_snapshot(&filename, mindmesh_core::SnapshotFormat::Mm, &description, false, true);
                self.snapshots.push(filename.clone());
                self.notifications.push(format!("Snapshot saved: {}", filename));
            }
            Message::LoadSnapshot(filename) => {
                if let Ok(snapshot) = mindmesh_core::Network::load_snapshot(&filename, None) {
                    self.network = snapshot.network;
                    self.current_snapshot = Some(filename.clone());
                    self.notifications.push(format!("Snapshot loaded: {}", filename));
                }
            }
            Message::DeleteSnapshot(filename) => {
                if let Ok(_) = std::fs::remove_file(&filename) {
                    self.snapshots.retain(|s| s != &filename);
                    self.notifications.push(format!("Snapshot deleted: {}", filename));
                }
            }
            Message::CompareSnapshots => {
                if let Some(ref current) = self.current_snapshot {
                    if let Ok(snapshot) = mindmesh_core::Network::load_snapshot(current, None) {
                        let comparison = self.network.compare_snapshots(&snapshot.network);
                        self.notifications.push(format!("Comparison: Neurons: {}, Connections: {}, Energy: {:.2}",
                            comparison.neuron_count_diff, comparison.connection_count_diff, comparison.energy_diff));
                    }
                }
            }
            Message::ToggleEnergyMode => {
                self.energy_efficient_mode = !self.energy_efficient_mode;
                self.network.settings.energy_efficient_mode = self.energy_efficient_mode;
            }
            Message::ClearNotifications => {
                self.notifications.clear();
            }
            Message::ToggleGestureMode => {
                self.gesture_mode = !self.gesture_mode;
            }
            Message::ToggleVoiceMode => {
                self.voice_mode = !self.voice_mode;
            }
            Message::ToggleVrMode => {
                self.vr_mode = !self.vr_mode;
            }
            Message::ToggleSonification => {
                self.sonification = !self.sonification;
            }
            Message::ToggleArMode => {
                self.ar_mode = !self.ar_mode;
                self.notifications.push(if self.ar_mode { "AR Mode enabled: Overlay neural activity on camera feed" } else { "AR Mode disabled" }.to_string());
            }
            Message::ToggleBciMode => {
                self.bci_mode = !self.bci_mode;
                if self.bci_mode {
                    self.notifications.push("BCI Mode enabled: Research-only, ensure ethical use".to_string());
                } else {
                    self.notifications.push("BCI Mode disabled".to_string());
                }
            }
            Message::ToggleHolographicMode => {
                self.holographic_mode = !self.holographic_mode;
                self.notifications.push(if self.holographic_mode { "Holographic Mode enabled: Project 3D neural structures" } else { "Holographic Mode disabled" }.to_string());
            }
            Message::ProcessGesture => {
                let gesture_data = vec![0.5; 10]; // Simulated gesture input
                self.network.process_input(mindmesh_core::Input::Gesture(gesture_data), None, &mut self.plugins);
            }
            Message::ProcessVoiceCommand(command) => {
                // Process voice commands
                match command.to_lowercase().as_str() {
                    "play" => self.paused = false,
                    "pause" => self.paused = true,
                    "reset" => self.network = mindmesh_core::Network::new(),
                    "evolve" => self.network.grow_connections(0.1),
                    _ => {}
                }
            }
            Message::ExportGif => {
                // Placeholder: capture current frame as image
                self.notifications.push("GIF export: Captured frame (full implementation requires animation)".to_string());
            }
            Message::ExportMp4 => {
                self.notifications.push("MP4 export: Video export (requires ffmpeg integration)".to_string());
            }
            Message::ExportVrScene => {
                self.notifications.push("VR Scene export: 3D model export (GLTF format)".to_string());
            }
            Message::RunAnalytics => {
                let avg_activity = self.network.neurons.iter().map(|n| n.output).sum::<f64>() / self.network.neurons.len() as f64;
                let total_energy = self.network.neurons.iter().map(|n| n.metadata.energy_cost).sum::<f64>();
                self.notifications.push(format!("Analytics: Avg Activity: {:.2}, Total Energy: {:.2}, Patterns: {}",
                    avg_activity, total_energy, self.network.emergent_patterns.len()));
            }
            Message::ShowAchievement(achievement) => {
                self.achievements.push(achievement);
            }
            Message::ProcessSensorInput => {
                if let Some(ref mut port) = self.serial_port {
                    let mut buffer = [0u8; 20];
                    if let Ok(len) = port.read(&mut buffer) {
                        let sensor_data: Vec<f64> = buffer[..len].iter().map(|&b| b as f64 / 255.0).collect();
                        self.network.process_input(mindmesh_core::Input::Sensor(sensor_data), None, &mut self.plugins);
                        self.notifications.push(format!("Read {} bytes from sensor", len));
                    } else {
                        self.notifications.push("Failed to read from sensor".to_string());
                    }
                } else {
                    let sensor_data = vec![0.3; 20]; // Simulated sensor input
                    self.network.process_input(mindmesh_core::Input::Sensor(sensor_data), None, &mut self.plugins);
                    self.notifications.push("Using simulated sensor data".to_string());
                }
            }
            Message::AutonomousEvolution => {
                self.network.autonomous_synaptogenesis();
                self.network.energy_efficient_pruning();
            }
            Message::MetaLearning => {
                self.network.meta_learning_rewiring();
            }
            Message::AiGuidedRewiring => {
                self.network.ai_predictive_rewiring();
            }
            Message::ToggleInspector => {
                self.show_inspector = !self.show_inspector;
            }
            Message::SelectNeuron(id) => {
                self.selected_neuron = Some(id);
                self.show_inspector = true;
            }
            Message::ToggleInputPanel => {
                self.show_input_panel = !self.show_input_panel;
            }
            Message::ToggleSimulationControls => {
                self.show_simulation_controls = !self.show_simulation_controls;
            }
            Message::ToggleExportAi => {
                self.show_export_ai = !self.show_export_ai;
            }
            Message::GlobalSearchChanged(query) => {
                self.global_search = query;
                // Basic search: filter log console (in real app, search network, snapshots, etc.)
                // For now, just store query
            }
            Message::LodChanged(value) => {
                self.lod_slider = value;
            }
            Message::PlaybackScrubberChanged(value) => {
                self.playback_scrubber = value;
                // TODO: Implement scrubbing
            }
            Message::ToggleVisualizationOverlays => {
                self.visualization_overlays = !self.visualization_overlays;
            }
            Message::Undo => {
                if let Some(prev_state) = self.undo_stack.pop() {
                    self.redo_stack.push(self.network.clone());
                    self.network = prev_state;
                }
            }
            Message::Redo => {
                if let Some(next_state) = self.redo_stack.pop() {
                    self.undo_stack.push(self.network.clone());
                    self.network = next_state;
                }
            }
            Message::StartSandboxEdit => {
                self.sandbox_network = Some(self.network.clone());
            }
            Message::CommitSandbox => {
                if let Some(sandbox) = self.sandbox_network.take() {
                    self.undo_stack.push(self.network.clone());
                    self.network = sandbox;
                }
            }
            Message::RevertSandbox => {
                self.sandbox_network = None;
            }
            Message::ToggleAutonomousModal => {
                self.show_autonomous_modal = !self.show_autonomous_modal;
            }
            Message::SetAutonomousPreset(preset) => {
                self.autonomous_preset = preset;
            }
            Message::SetAutonomousEnergyBudget(budget) => {
                self.autonomous_energy_budget = budget;
            }
            Message::SetAutonomousTimeLimit(limit) => {
                self.autonomous_time_limit = limit;
            }
            Message::SetAutonomousDetectionSensitivity(sens) => {
                self.autonomous_detection_sensitivity = sens;
            }
            Message::SetAutonomousLoggingLevel(level) => {
                self.autonomous_logging_level = level;
            }
            Message::StartAutonomousExperiment => {
                // Start autonomous experiment
                self.network.settings.energy_efficient_mode = true; // Example
                self.log_console.push((format!("Started autonomous experiment: {}", self.autonomous_preset), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()));
                self.show_autonomous_modal = false;
            }
            Message::ToggleProfilingMode => {
                self.profiling_mode = !self.profiling_mode;
            }
            Message::ToggleEthicsModal => {
                self.show_ethics_modal = !self.show_ethics_modal;
            }
            Message::NextOnboardingStep => {
                if self.onboarding_step < 4 {
                    self.onboarding_step += 1;
                } else {
                    self.show_onboarding = false;
                }
            }
            Message::SkipOnboarding => {
                self.show_onboarding = false;
            }
            Message::ShowMappingWizard => {
                self.show_mapping_wizard = true;
            }
            Message::SetMappingInputType(t) => {
                self.mapping_input_type = t;
            }
            Message::SetMappingStrategy(s) => {
                self.mapping_strategy = s;
            }
            Message::ApplyMapping => {
                // Apply mapping
                let mapping = match self.mapping_strategy.as_str() {
                    "Auto-embed" => mindmesh_core::Mapping { strategy: mindmesh_core::MappingStrategy::AutoEmbed, parameters: std::collections::HashMap::new(), manual_brush: None },
                    "Hash-seed" => mindmesh_core::Mapping { strategy: mindmesh_core::MappingStrategy::HashSeed, parameters: std::collections::HashMap::new(), manual_brush: None },
                    _ => mindmesh_core::Mapping { strategy: mindmesh_core::MappingStrategy::ManualPaint, parameters: std::collections::HashMap::new(), manual_brush: None },
                };
                self.network.input_mappings.insert(self.mapping_input_type.clone(), mapping);
                self.show_mapping_wizard = false;
                self.notifications.push("Mapping applied".to_string());
            }
            Message::ShowExportWizard => {
                self.show_export_wizard = !self.show_export_wizard;
            }
            Message::SetExportFormat(f) => {
                self.export_format = f;
            }
            Message::ToggleExportDeltaOnly => {
                self.export_delta_only = !self.export_delta_only;
            }
            Message::SetExportCompress(c) => {
                self.export_compress = c;
            }
            Message::ToggleExportEncryption => {
                self.export_encryption = !self.export_encryption;
            }
            Message::StartExport => {
                // Start export
                 match self.export_format.as_str() {
                     "JSON" => {
                         if let Ok(json) = serde_json::to_string(&self.network) {
                             // In real app, save to file or clipboard
                             println!("Exported JSON: {}", json);
                             self.notifications.push("JSON exported to console".to_string());
                         } else {
                             self.notifications.push("JSON export failed".to_string());
                         }
                     }
                     "interactive_html" => {
                         // Generate interactive HTML
                         let html_content = format!(
                             r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MindMesh Interactive Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0B0F14; color: #9AA7B2; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .controls {{ margin-bottom: 20px; }}
        .canvas {{ border: 1px solid #9AA7B2; background: #0B0F14; width: 100%; height: 600px; position: relative; }}
        .neuron {{ position: absolute; width: 10px; height: 10px; border-radius: 50%; background: #7BD389; }}
        .connection {{ position: absolute; height: 1px; background: #5AB4FF; }}
        .info {{ margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MindMesh Interactive Brain</h1>
            <p>Neurons: {}, Connections: {}</p>
        </div>
        <div class="controls">
            <button onclick="step()">Step</button>
            <button onclick="reset()">Reset</button>
            <button onclick="togglePlay()">{{playPauseText}}</button>
        </div>
        <div class="canvas" id="canvas"></div>
        <div class="info">
            <p id="info">Steps: 0, Firing: 0</p>
        </div>
    </div>
    <script>
        const networkData = {};
        let network = JSON.parse(networkData);
        let canvas = document.getElementById('canvas');
        let isPlaying = false;
        let interval;

        function draw() {{
            canvas.innerHTML = '';
            // Draw connections
            network.connections.forEach(conn => {{
                let from = network.neurons.find(n => n.id === conn.from_id);
                let to = network.neurons.find(n => n.id === conn.to_id);
                if (from && to) {{
                    let line = document.createElement('div');
                    line.className = 'connection';
                    let dx = to.position[0] - from.position[0];
                    let dy = to.position[1] - from.position[1];
                    let length = Math.sqrt(dx*dx + dy*dy);
                    let angle = Math.atan2(dy, dx) * 180 / Math.PI;
                    line.style.width = length * 10 + 'px';
                    line.style.left = (from.position[0] * 10 + 400) + 'px';
                    line.style.top = (from.position[1] * 10 + 300) + 'px';
                    line.style.transform = `rotate(${{angle}}deg)`;
                    canvas.appendChild(line);
                }}
            }});
            // Draw neurons
            network.neurons.forEach(neuron => {{
                let dot = document.createElement('div');
                dot.className = 'neuron';
                dot.style.left = (neuron.position[0] * 10 + 400) + 'px';
                dot.style.top = (neuron.position[1] * 10 + 300) + 'px';
                dot.style.opacity = neuron.output;
                canvas.appendChild(dot);
            }});
            document.getElementById('info').textContent = `Steps: ${{network.time}}, Firing: ${{network.neurons.filter(n => n.output > 0.5).length}}`;
        }}

        function step() {{
            // Simple step simulation (placeholder)
            network.neurons.forEach(n => {{
                n.output = Math.random();
            }});
            network.time += 1;
            draw();
        }}

        function reset() {{
            network.time = 0;
            draw();
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                interval = setInterval(step, 100);
            }} else {{
                clearInterval(interval);
            }}
        }}

        draw();
    </script>
</body>
</html>"#,
                             self.network.neurons.len(),
                             self.network.connections.len(),
                             serde_json::to_string(&self.network).unwrap_or("{}".to_string())
                         );
                         // Save to file
                         if let Ok(_) = std::fs::write("export.html", html_content) {
                             self.notifications.push("Interactive HTML exported to export.html".to_string());
                         } else {
                             self.notifications.push("HTML export failed".to_string());
                         }
                     }
                     "GIF" => {
                         self.notifications.push("GIF export: Animation captured (placeholder)".to_string());
                     }
                     "MP4" => {
                         self.notifications.push("MP4 export: Video rendered (placeholder)".to_string());
                     }
                     "VR Scene" => {
                         self.notifications.push("VR Scene export: 3D model saved (placeholder)".to_string());
                     }
                     _ => {
                         self.notifications.push(format!("Export format {} not implemented", self.export_format));
                     }
                 }
                self.show_export_wizard = false;
            }
            Message::ToggleShortcuts => {
                self.show_shortcuts = !self.show_shortcuts;
            }
            Message::LogFilterChanged(filter) => {
                self.log_filter = filter;
            }
            Message::SimulateVrGesture => {
                self.notifications.push("VR Gesture: Pinch to zoom simulated".to_string());
            }
            Message::ToggleDeterministicMode => {
                self.deterministic_mode = !self.deterministic_mode;
                self.network.settings.deterministic = self.deterministic_mode;
            }
            Message::SetEnergyBudget(budget) => {
                self.energy_budget = budget;
                self.network.settings.energy_budget = budget;
            }
            Message::ToggleRewireMode => {
                self.rewire_mode = !self.rewire_mode;
                self.selected_neurons_for_rewire.clear();
            }
            Message::SelectNeuronForRewire(id) => {
                if self.rewire_mode {
                    self.selected_neurons_for_rewire.push(id);
                    if self.selected_neurons_for_rewire.len() == 2 {
                        let from = self.selected_neurons_for_rewire[0];
                        let to = self.selected_neurons_for_rewire[1];
                        self.network.add_connection(from, to, 0.5, PlasticityType::Hebbian);
                        self.notifications.push(format!("Rewired connection from {} to {}", from, to));
                        self.selected_neurons_for_rewire.clear();
                    }
                }
            }
            Message::ToggleReducedMotion => {
                self.reduced_motion = !self.reduced_motion;
            }
            Message::ToggleLowPowerMode => {
                self.low_power_mode = !self.low_power_mode;
                if self.low_power_mode {
                    self.lod_slider = 0.1;
                    self.visualization_overlays = false;
                    // Disable particles, etc.
                }
            }
            Message::ToggleVrArMode => {
                self.vr_ar_mode = !self.vr_ar_mode;
                self.notifications.push(if self.vr_ar_mode { "VR/AR Mode enabled (placeholder)" } else { "VR/AR Mode disabled" }.to_string());
            }
            Message::ToggleHelp => {
                self.show_help = !self.show_help;
            }
            Message::SetHelpSection(section) => {
                self.help_section = section;
            }
            Message::HelpSearchChanged(query) => {
                self.help_search = query;
            }
            Message::UpdateActivity => {
                self.last_activity = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                self.show_contextual_hint = false;
            }
            Message::DismissHint => {
                self.show_contextual_hint = false;
            }
            Message::ToggleCollaboration => {
                self.show_collaboration = !self.show_collaboration;
            }
            Message::NewProject => {
                self.network = Network::new();
                self.notifications.push("New project created".to_string());
            }
            Message::OpenProject => {
                // Placeholder
                self.notifications.push("Open project (placeholder)".to_string());
            }
            Message::SaveAs => {
                // Placeholder
                self.notifications.push("Save as (placeholder)".to_string());
            }
            Message::CommandPalette => {
                // Placeholder
                self.notifications.push("Command palette (placeholder)".to_string());
            }
            Message::ToggleConsole => {
                // Placeholder
                self.notifications.push("Toggle console (placeholder)".to_string());
            }
            Message::FocusSelection => {
                // Placeholder
                self.notifications.push("Focus selection (placeholder)".to_string());
            }
            Message::Toggle2D3D => {
                self.show_3d = !self.show_3d;
            }
            Message::PaintMapping => {
                // Placeholder
                self.notifications.push("Paint mapping (placeholder)".to_string());
            }
            Message::EraseMapping => {
                // Placeholder
                self.notifications.push("Erase mapping (placeholder)".to_string());
            }
            Message::Duplicate => {
                // Placeholder
                self.notifications.push("Duplicate (placeholder)".to_string());
            }
            Message::Delete => {
                // Placeholder
                self.notifications.push("Delete (placeholder)".to_string());
            }
            Message::StepForward => {
                self.network.step();
                self.steps_taken += 1;
            }
            Message::StepBack => {
                // Placeholder, need history
                self.notifications.push("Step back (placeholder)".to_string());
            }
            Message::SpeedUpSim => {
                self.speed *= 1.1;
            }
            Message::SpeedDownSim => {
                self.speed /= 1.1;
            }
        }
        Command::none()
    }

    fn theme(&self) -> iced::Theme {
        if self.high_contrast {
            iced::Theme::Dark // Placeholder for high contrast
        } else if self.dark_theme {
            iced::Theme::Dark
        } else {
            iced::Theme::Light
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        let timer = if !self.paused {
            time::every(std::time::Duration::from_millis((1000.0 / self.speed) as u64)).map(|_| Message::Step)
        } else {
            Subscription::none()
        };
        let activity_timer = time::every(std::time::Duration::from_secs(1)).map(|_| Message::UpdateActivity);
        let keyboard = keyboard::on_key_press(|key, modifiers| {
            match key {
                keyboard::Key::Named(keyboard::key::Named::Space) => Some(Message::TogglePause),
                keyboard::Key::Named(keyboard::key::Named::ArrowUp) => Some(Message::ZoomIn),
                keyboard::Key::Named(keyboard::key::Named::ArrowDown) => Some(Message::ZoomOut),
                keyboard::Key::Named(keyboard::key::Named::ArrowRight) if modifiers.alt() => Some(Message::StepForward),
                keyboard::Key::Named(keyboard::key::Named::ArrowLeft) if modifiers.alt() => Some(Message::StepBack),
                keyboard::Key::Character(c) if c == "r" => Some(Message::Reset),
                keyboard::Key::Character(c) if c == "s" => Some(Message::Step),
                keyboard::Key::Character(c) if c == "e" => Some(Message::ShowExportWizard),
                keyboard::Key::Character(c) if c == "m" => Some(Message::ShowMappingWizard),
                keyboard::Key::Character(c) if c == "a" => Some(Message::ToggleAutonomousModal),
                keyboard::Key::Character(c) if c == "n" && modifiers.control() => Some(Message::NewProject),
                keyboard::Key::Character(c) if c == "o" && modifiers.control() => Some(Message::OpenProject),
                keyboard::Key::Character(c) if c == "s" && modifiers.control() && !modifiers.shift() => Some(Message::Save),
                keyboard::Key::Character(c) if c == "s" && modifiers.control() && modifiers.shift() => Some(Message::SaveAs),
                keyboard::Key::Character(c) if c == "k" && modifiers.control() => Some(Message::CommandPalette),
                keyboard::Key::Character(c) if c == "`" && modifiers.control() => Some(Message::ToggleConsole),
                keyboard::Key::Character(c) if c == "f" => Some(Message::FocusSelection),
                keyboard::Key::Character(c) if c == "v" => Some(Message::Toggle2D3D),
                keyboard::Key::Character(c) if c == "p" => Some(Message::PaintMapping),
                keyboard::Key::Character(c) if c == "e" && !modifiers.control() => Some(Message::EraseMapping),
                keyboard::Key::Character(c) if c == "d" => Some(Message::Duplicate),
                keyboard::Key::Named(keyboard::key::Named::Delete) => Some(Message::Delete),
                keyboard::Key::Character(c) if c == "z" && modifiers.control() && !modifiers.shift() => Some(Message::Undo),
                keyboard::Key::Character(c) if c == "z" && modifiers.control() && modifiers.shift() => Some(Message::Redo),
                keyboard::Key::Character(c) if c == "]" => Some(Message::SpeedUpSim),
                keyboard::Key::Character(c) if c == "[" => Some(Message::SpeedDownSim),
                _ => None,
            }
        });
        Subscription::batch(vec![timer, keyboard, activity_timer])
    }

    fn view(&self) -> Element<'_, Message> {
        use iced::widget::{button, column, row, text, text_input, container, scrollable, vertical_slider};
        use iced::Length;

        let theme = self.app_theme();

        // Contextual hints engine
        let current_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        if current_time - self.last_activity > 3 && !self.show_contextual_hint {
            self.show_contextual_hint = true;
            self.contextual_hint = if self.network.neurons.is_empty() {
                "Try adding some neurons to get started!"
            } else if !self.paused {
                "Simulation is running. Try pausing to edit."
            } else {
                "Right-click on canvas for context menu."
            }.to_string();
        }

        let play_text = self.t("play");
        let pause_text = self.t("pause");

        let neuron_count = self.network.neurons.len();
        let connection_count = self.network.connections.len();
        let avg_activity = self.network.neurons.iter().map(|n| n.output).sum::<f64>() / neuron_count as f64;
        let total_energy = self.network.neurons.iter().map(|n| n.metadata.energy_cost).sum::<f64>();
        let firing_count = self.network.neurons.iter().filter(|n| n.should_fire()).count();

        // Header
        let status_color = if self.network.neurons.iter().any(|n| n.should_fire()) {
            theme.warning // yellow running
        } else if self.notifications.iter().any(|n| n.contains("error") || n.contains("failed")) {
            theme.error // red error
        } else {
            theme.success // green idle
        };
        let header = container(
            row![
                // Hamburger/App Menu
                iced::widget::Tooltip::new(
                    button("☰").on_press(Message::ToggleInputPanel).style(iced::theme::Button::Secondary).padding(8),
                    "Toggle left sidebar",
                    iced::widget::tooltip::Position::Bottom,
                ),
                // Project Title
                button(text("🧠 MindMesh").size(24).style(iced::theme::Text::Color(theme.primary).weight(iced::font::Weight::Bold))).on_press(Message::Reset).style(iced::theme::Button::Text),
                // Global Search
                text_input("🔍 Search...", &self.global_search)
                    .on_input(Message::GlobalSearchChanged)
                    .padding(12)
                    .size(16),
                // Session Indicators (improved)
                {
                    let (status_icon, status_text) = if self.network.neurons.iter().any(|n| n.should_fire()) {
                        ("🟡", "Running")
                    } else if self.notifications.iter().any(|n| n.contains("error") || n.contains("failed")) {
                        ("🔴", "Error")
                    } else {
                        ("🟢", "Ready")
                    };
                    container(
                        row![text(status_icon).size(12), text(status_text).size(12)].spacing(4)
                    ).style(iced::theme::Container::Custom(Box::new(move |theme| {
                        iced::widget::container::Appearance {
                            background: Some(iced::Background::Color(status_color)),
                            border: iced::Border { radius: 12.0.into(), ..Default::default() },
                            ..Default::default()
                        }
                    }))).padding(iced::Padding::from([4, 8]))
                },
                // Save/Status
                iced::widget::Tooltip::new(
                    button("💾 Save").on_press(Message::Save).style(iced::theme::Button::Positive).padding(8),
                    "Save project",
                    iced::widget::tooltip::Position::Bottom,
                ),
                // Undo/Redo
                iced::widget::Tooltip::new(
                    button("↶").on_press(Message::Undo).style(iced::theme::Button::Secondary).padding(8),
                    "Undo",
                    iced::widget::tooltip::Position::Bottom,
                ),
                iced::widget::Tooltip::new(
                    button("↷").on_press(Message::Redo).style(iced::theme::Button::Secondary).padding(8),
                    "Redo",
                    iced::widget::tooltip::Position::Bottom,
                ),
                // Profile/Settings
                      iced::widget::Tooltip::new(
                          button("❓ Help").on_press(Message::ToggleHelp).style(iced::theme::Button::Secondary),
                          "Show help and tutorials",
                          iced::widget::tooltip::Position::Bottom,
                      ),
            ].spacing(12).align_items(iced::Alignment::Center)
        ).padding(16).height(56).style(iced::theme::Container::Custom(Box::new(move |theme| {
            iced::widget::container::Appearance {
                background: Some(iced::Background::Color(theme.surface)),
                border: iced::Border { radius: 8.0.into(), ..Default::default() },
                shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                ..Default::default()
            }
        })));

        // Left Sidebar - Project & Snapshot Manager, Inputs & Datasets, Presets
        let left_sidebar = if self.show_input_panel {
            container(
                scrollable(
                    column![
                        text("📁 Project Manager").size(18).style(iced::theme::Text::Color(theme.primary).weight(iced::font::Weight::Bold)),
                        button("💾 Save Project").on_press(Message::Save).style(iced::theme::Button::Primary).width(Length::Fill),
                        button("📂 Load Project").on_press(Message::Load).style(iced::theme::Button::Secondary).width(Length::Fill),
                        iced::widget::Rule::horizontal(1),
                        text("📸 Snapshots").size(16).style(iced::theme::Text::Color(theme.text)),
                        button("📸 New Snapshot").on_press(Message::SaveSnapshot("Manual snapshot".to_string())).style(iced::theme::Button::Positive).width(Length::Fill),
                        button("📊 Compare").on_press(Message::CompareSnapshots).style(iced::theme::Button::Secondary).width(Length::Fill),
                        text("Snapshots:").size(12).style(iced::theme::Text::Color(theme.text)),
                        column(
                            self.snapshots.iter().enumerate().map(|(i, s)| {
                                row![
                                    text(format!("{}. {}", i + 1, s)).size(10).style(iced::theme::Text::Color(theme.text)),
                                    button("📂").on_press(Message::LoadSnapshot(s.clone())).style(iced::theme::Button::Secondary),
                                ].spacing(5).into()
                            }).collect::<Vec<_>>()
                        ).spacing(2),
                        iced::widget::Rule::horizontal(1),
                        text("📊 Inputs").size(16).style(iced::theme::Text::Color(theme.text)),
                        button("📝 Text").on_press(Message::ProcessInput).style(iced::theme::Button::Primary).width(Length::Fill),
                        button("🖼️ Image").on_press(Message::ProcessAscii).style(iced::theme::Button::Secondary).width(Length::Fill),
                        button("🎤 Voice").on_press(Message::ProcessVoice).style(iced::theme::Button::Secondary).width(Length::Fill),
                        button("🧠 BCI").on_press(Message::ProcessBCI).style(iced::theme::Button::Secondary).width(Length::Fill),
                        button("🎮 Gesture").on_press(Message::ProcessGesture).style(iced::theme::Button::Secondary).width(Length::Fill),
                        button("📡 Sensor").on_press(Message::ProcessSensorInput).style(iced::theme::Button::Secondary).width(Length::Fill),
                        iced::widget::Rule::horizontal(1),
                        button("🗺️ Mapping Wizard").on_press(Message::ShowMappingWizard).style(iced::theme::Button::Positive).width(Length::Fill),
                        text("Recent Inputs:").size(12).style(iced::theme::Text::Color(theme.text)),
                        text("• Text: Hello").size(10).style(iced::theme::Text::Color(theme.text)),
                        text("• Image: sample.png").size(10).style(iced::theme::Text::Color(theme.text)),
                        iced::widget::Rule::horizontal(1),
                        text("🎛️ Experiments").size(16).style(iced::theme::Text::Color(theme.text)),
                        button("🧠 Dream").on_press(Message::AutonomousEvolution).style(iced::theme::Button::Secondary).width(Length::Fill),
                        button("🔄 Replay").on_press(Message::MetaLearning).style(iced::theme::Button::Secondary).width(Length::Fill),
                        button("🎯 Pattern Search").on_press(Message::AiGuidedRewiring).style(iced::theme::Button::Secondary).width(Length::Fill),
                        button("🤖 Autonomous").on_press(Message::ToggleAutonomousModal).style(iced::theme::Button::Positive).width(Length::Fill),
                        button("👥 Collaborate").on_press(Message::ToggleCollaboration).style(iced::theme::Button::Secondary).width(Length::Fill),
                      ].spacing(8)
                  )
              ).width(300).padding(16).style(iced::theme::Container::Custom(Box::new(move |theme| {
                  iced::widget::container::Appearance {
                      background: Some(iced::Background::Color(theme.surface)),
                      border: iced::Border { radius: 8.0.into(), ..Default::default() },
                      shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                      ..Default::default()
                  }
              })))
        } else {
            container(text("")).width(60).padding(10).style(iced::theme::Container::Box)
        };

        // Main Canvas - Visual Thought Map
        let canvas_area = container(
            column![
                // Visualization Controls
                row![
                    button(if self.show_3d { "2️⃣ 2D" } else { "3️⃣ 3D" }).on_press(Message::Toggle3D),
                    text("LOD:").size(12),
                    vertical_slider(0.1..=2.0, self.lod_slider, Message::LodChanged).width(60),
                    button(if self.show_connections { "🔗 Hide Edges" } else { "🔗 Show Edges" }).on_press(Message::ToggleConnections),
                    button(if self.visualization_overlays { "👁️ Hide Overlays" } else { "👁️ Show Overlays" }).on_press(Message::ToggleVisualizationOverlays),
                    button("🎯 Predict").on_press(Message::Predict),
                    button(if self.sandbox_network.is_some() { "✅ Commit Edit" } else { "✏️ Edit Mode" }).on_press(if self.sandbox_network.is_some() { Message::CommitSandbox } else { Message::StartSandboxEdit }),
                    button(if self.rewire_mode { "🔗 Rewire: On" } else { "🔗 Rewire: Off" }).on_press(Message::ToggleRewireMode),
                    if self.sandbox_network.is_some() { button("❌ Revert").on_press(Message::RevertSandbox) } else { button("") },
                ].spacing(8),
                // Playback Scrubber
                row![
                    text("Playback:").size(12),
                    vertical_slider(0.0..=100.0, self.playback_scrubber, Message::PlaybackScrubberChanged).width(200),
                    text(format!("{:.1}%", self.playback_scrubber)).size(12),
                ].spacing(8),
                // Main Canvas
                container(Canvas::new(self).width(Length::Fill).height(Length::Fill)),
                // Overlays
                if self.visualization_overlays {
                    row![
                        // Mini-analytics card
                        container(
                            column![
                                text("Analytics").size(12).style(iced::theme::Text::Color(theme.text)),
                                text(format!("🧠 {}", neuron_count)).size(10),
                                text(format!("🔗 {}", connection_count)).size(10),
                                text(format!("⚡ {:.1}", total_energy)).size(10),
                            ].spacing(4)
                        ).padding(8).style(iced::theme::Container::Custom(Box::new(move |theme| {
                            iced::widget::container::Appearance {
                                background: Some(iced::Background::Color(theme.surface)),
                                border: iced::Border { radius: 8.0.into(), ..Default::default() },
                                shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                                ..Default::default()
                            }
                        }))),
                        text(format!("🎯 {} firing", firing_count)).size(12),
                        text(if let Some(pattern) = self.network.emergent_patterns.last() {
                            format!("Pattern: {}", pattern)
                        } else {
                            "No patterns".to_string()
                        }).size(12),
                        // Contextual quick-actions
                        button("🔍 Focus").on_press(Message::SimulateVrGesture).style(iced::theme::Button::Secondary),
                        button("📸 Snapshot").on_press(Message::SaveSnapshot("Quick overlay".to_string())).style(iced::theme::Button::Secondary),
                    ].spacing(16)
                } else {
                    row![]
                },
            ].spacing(8)
          ).padding(16).style(iced::theme::Container::Custom(Box::new(move |theme| {
              iced::widget::container::Appearance {
                  background: Some(iced::Background::Color(theme.surface)),
                  border: iced::Border { radius: 8.0.into(), ..Default::default() },
                  shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                  ..Default::default()
              }
          })));

        // Right Sidebar - Panels
        let right_sidebar_content = if self.show_inspector {
            column![
                text("Inspector").size(16).style(iced::theme::Text::Color(theme.text)),
                if let Some(id) = self.selected_neuron {
                    column![
                        text(format!("Neuron ID: {}", id)).size(12),
                        text("Type: Excitatory").size(12),
                        text(format!("Threshold: {:.2}", 0.5)).size(12),
                        text(format!("Firing Rate: {:.2}", 0.1)).size(12),
                        button("Edit").on_press(Message::SimulateVrGesture).style(iced::theme::Button::Secondary),
                    ].spacing(5)
                  } else {
                      column![text("Select a neuron to inspect").size(12)]
                  }
            ].spacing(10)
        } else if self.show_simulation_controls {
            column![
                text("Simulation Controls").size(16).style(iced::theme::Text::Color(theme.text)),
                button(if self.paused { "▶️ Play" } else { "⏸️ Pause" }).on_press(Message::TogglePause).style(iced::theme::Button::Primary),
                button("⏹️ Stop").on_press(Message::Reset).style(iced::theme::Button::Secondary),
                button("⏭️ Step Forward").on_press(Message::Step).style(iced::theme::Button::Secondary),
                button("⏮️ Step Back").on_press(Message::Undo).style(iced::theme::Button::Secondary),
                text(format!("Speed: {:.1}x", self.speed)).size(12),
                vertical_slider(0.01..=100.0, self.speed, Message::SpeedChanged).width(100),
                button(if self.deterministic_mode { "🎲 Deterministic: On" } else { "🎲 Deterministic: Off" }).on_press(Message::ToggleDeterministicMode).style(iced::theme::Button::Secondary),
                text(format!("Energy Budget: {:.0}", self.energy_budget)).size(12),
                vertical_slider(10.0..=1000.0, self.energy_budget, Message::SetEnergyBudget).width(100),
                button("⚡ Energy Efficient").on_press(Message::ToggleEnergyMode).style(if self.energy_efficient_mode { iced::theme::Button::Positive } else { iced::theme::Button::Secondary }),
            ].spacing(5)
        } else if self.show_input_panel {
            column![
                text("Input Creator").size(16).style(iced::theme::Text::Color(theme.text)),
                text_input("Enter text input", &self.input_text).on_input(Message::InputChanged),
                button("Process Text").on_press(Message::ProcessInput).style(iced::theme::Button::Primary),
                button("Random Numbers").on_press(Message::ProcessRandomNumbers).style(iced::theme::Button::Secondary),
            ].spacing(5)
        } else if self.show_export_ai {
            column![
                text("AI Assistant").size(16).style(iced::theme::Text::Color(theme.text)),
                text(&self.ai_suggestion).size(12),
                button("Get Suggestion").on_press(Message::AiAssist).style(iced::theme::Button::Secondary),
            ].spacing(5)
        } else {
            column![
                text("Export").size(16).style(iced::theme::Text::Color(theme.text)),
                button("Export JSON").on_press(Message::ExportJson).style(iced::theme::Button::Primary),
                button("Export GIF").on_press(Message::ExportGif).style(iced::theme::Button::Secondary),
                button("Export MP4").on_press(Message::ExportMp4).style(iced::theme::Button::Secondary),
                button("Export VR Scene").on_press(Message::ExportVrScene).style(iced::theme::Button::Secondary),
            ].spacing(5)
        };

        let right_sidebar = container(
            column![
                // Panel switcher
                row![
                    button("Inspector").on_press(Message::ToggleInspector).style(if self.show_inspector { iced::theme::Button::Primary } else { iced::theme::Button::Secondary }),
                    button("Sim Ctrl").on_press(Message::ToggleSimulationControls).style(if self.show_simulation_controls { iced::theme::Button::Primary } else { iced::theme::Button::Secondary }),
                    button("Input").on_press(Message::ToggleInputPanel).style(if self.show_input_panel { iced::theme::Button::Primary } else { iced::theme::Button::Secondary }),
                    button("AI").on_press(Message::ToggleExportAi).style(if self.show_export_ai { iced::theme::Button::Primary } else { iced::theme::Button::Secondary }),
                    button("Export").on_press(Message::ShowExportWizard).style(iced::theme::Button::Secondary),
                ].spacing(8),
                iced::widget::Rule::horizontal(1),
                scrollable(right_sidebar_content),
                iced::widget::Rule::horizontal(1),
                text("🔔 Notifications").size(16),
                button("🗑️ Clear").on_press(Message::ClearNotifications),
            ].extend(
                self.notifications.iter().rev().take(5).map(|n| text(format!("• {}", n)).size(11).style(iced::theme::Text::Color(theme.text)).into())
            ).spacing(8)
        ).width(360).padding(16).style(iced::theme::Container::Custom(Box::new(move |theme| {
            iced::widget::container::Appearance {
                background: Some(iced::Background::Color(theme.surface)),
                border: iced::Border { radius: 8.0.into(), ..Default::default() },
                shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                ..Default::default()
            }
        })));

        // Bottom Bar - Log Console, Analytics Mini-Cards, Resource Monitor
        let footer = container(
            row![
                // Left: Log Console
                container(
                    scrollable(
                        column![
                              text("📋 Log Console").size(12).style(iced::theme::Text::Color(theme.text)),
                              text_input("Filter logs...", &self.log_filter).on_input(Message::LogFilterChanged).width(150),
                          ].extend(
                              self.log_console.iter().rev().filter(|(log, _)| self.log_filter.is_empty() || log.contains(&self.log_filter)).take(5).map(|(log, ts)| {
                                  let time_str = chrono::DateTime::from_timestamp(*ts as i64, 0).unwrap().format("%H:%M:%S").to_string();
                                  text(format!("[{}] {}", time_str, log)).size(10).style(iced::theme::Text::Color(theme.text)).into()
                              })
                         ).spacing(4)
                    )
                ).width(200).height(100).style(iced::theme::Container::Custom(Box::new(move |theme| {
                    iced::widget::container::Appearance {
                        background: Some(iced::Background::Color(theme.surface)),
                        border: iced::Border { radius: 8.0.into(), ..Default::default() },
                        shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                        ..Default::default()
                    }
                }))),
                // Center: Analytics Mini-Cards
                row![
                    container(
                        column![
                            text("🧠 Neurons").size(12).style(iced::theme::Text::Color(theme.text)),
                            text(format!("{}", neuron_count)).size(14).style(iced::theme::Text::Color(theme.primary)),
                        ].spacing(4).align_items(iced::Alignment::Center)
                    ).padding(16).style(iced::theme::Container::Custom(Box::new(move |theme| {
                        iced::widget::container::Appearance {
                            background: Some(iced::Background::Color(theme.surface)),
                            border: iced::Border { radius: 8.0.into(), ..Default::default() },
                            shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                            ..Default::default()
                        }
                    }))),
                    container(
                        column![
                            text("🔗 Connections").size(12).style(iced::theme::Text::Color(theme.text)),
                            text(format!("{}", connection_count)).size(14).style(iced::theme::Text::Color(theme.primary)),
                        ].spacing(4).align_items(iced::Alignment::Center)
                    ).padding(16).style(iced::theme::Container::Custom(Box::new(move |theme| {
                        iced::widget::container::Appearance {
                            background: Some(iced::Background::Color(theme.surface)),
                            border: iced::Border { radius: 8.0.into(), ..Default::default() },
                            shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                            ..Default::default()
                        }
                    }))),
                    container(
                        column![
                            text("⚡ Energy").size(12).style(iced::theme::Text::Color(theme.text)),
                            text(format!("{:.1}", total_energy)).size(14).style(iced::theme::Text::Color(theme.accent)),
                        ].spacing(4).align_items(iced::Alignment::Center)
                    ).padding(16).style(iced::theme::Container::Custom(Box::new(move |theme| {
                        iced::widget::container::Appearance {
                            background: Some(iced::Background::Color(theme.surface)),
                            border: iced::Border { radius: 8.0.into(), ..Default::default() },
                            shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                            ..Default::default()
                        }
                    }))),
                    container(
                        column![
                              text("🎯 Firing").size(12).style(iced::theme::Text::Color(theme.text)),
                          text(format!("{}", firing_count)).size(14).style(iced::theme::Text::Color(theme.success)),
                           ].spacing(4).align_items(iced::Alignment::Center)
                       ).padding(16).style(iced::theme::Container::Custom(Box::new(move |theme| {
                           iced::widget::container::Appearance {
                               background: Some(iced::Background::Color(theme.surface)),
                               border: iced::Border { radius: 8.0.into(), ..Default::default() },
                               shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                               ..Default::default()
                           }
                       }))),
                ].spacing(16),
                // Right: Status & Resources
                container(
                    column![
                        text(format!("📊 Status: {}", if self.paused { "Paused" } else { "Running" })).size(12).style(iced::theme::Text::Color(if self.paused { theme.warning } else { theme.success })),
                        text("💻 Resources").size(12).style(iced::theme::Text::Color(theme.text)),
                        text("CPU: 45% | GPU: 30% | RAM: 2.1GB").size(10).style(iced::theme::Text::Color(theme.text)),
                        text("⌨️ Shortcuts: Space=Play, R=Reset, E=Export, M=Map, A=Auto").size(10).style(iced::theme::Text::Color(theme.text)),
                      ].spacing(4)
                  ).width(250).style(iced::theme::Container::Custom(Box::new(move |theme| {
                      iced::widget::container::Appearance {
                          background: Some(iced::Background::Color(theme.surface)),
                          border: iced::Border { radius: 8.0.into(), ..Default::default() },
                          shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                          ..Default::default()
                      }
                  }))),
            ].spacing(24).align_items(iced::Alignment::Center)
          ).padding(8).style(iced::theme::Container::Custom(Box::new(move |theme| {
              iced::widget::container::Appearance {
                  background: Some(iced::Background::Color(theme.surface)),
                  border: iced::Border { radius: 8.0.into(), ..Default::default() },
                  shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 2.0), blur_radius: 4.0 },
                  ..Default::default()
              }
          })));

        // Modal Panels
        let modal_content: Option<Element<'_, Message>> = if self.show_settings {
            Some(container(
                column![
                    text("Settings").size(20).style(iced::theme::Text::Color(theme.text)),
                    button("Toggle Theme").on_press(Message::ToggleTheme).style(iced::theme::Button::Primary).padding(12),
                    button("High Contrast").on_press(Message::ToggleHighContrast).style(iced::theme::Button::Secondary).padding(12),
                    button("Colorblind Mode").on_press(Message::SetColorblindMode(Some("protanopia".to_string()))).style(iced::theme::Button::Secondary).padding(12),
                    button(if self.reduced_motion { "Reduced Motion: On" } else { "Reduced Motion: Off" }).on_press(Message::ToggleReducedMotion).style(iced::theme::Button::Secondary).padding(12),
                    button(if self.low_power_mode { "Low Power Mode: On" } else { "Low Power Mode: Off" }).on_press(Message::ToggleLowPowerMode).style(iced::theme::Button::Secondary).padding(12),
                    button(if self.vr_ar_mode { "VR/AR Mode: On" } else { "VR/AR Mode: Off" }).on_press(Message::ToggleVrArMode).style(iced::theme::Button::Secondary).padding(12),
                    text("Font Scale").size(14).style(iced::theme::Text::Color(theme.text)),
                    vertical_slider(0.5..=2.0, self.font_scale, Message::SetFontScale).width(200),
                    button("Close").on_press(Message::ToggleSettings).style(iced::theme::Button::Secondary).padding(12),
                ].spacing(16)
            ).padding(24).center_x().center_y().style(iced::theme::Container::Custom(Box::new(move |theme| {
                iced::widget::container::Appearance {
                    background: Some(iced::Background::Color(theme.surface)),
                    border: iced::Border { radius: 12.0.into(), ..Default::default() },
                    shadow: iced::Shadow { color: Color::BLACK, offset: iced::Vector::new(0.0, 4.0), blur_radius: 8.0 },
                    ..Default::default()
                }
            }))).into())
        } else if self.show_onboarding {
            Some(container(
                column![
                    text("Welcome to MindMesh").size(24).style(iced::theme::Text::Color(theme.text)),
                    text(match self.onboarding_step {
                        0 => "Project intro, privacy explanation (local-first), choose language, optional voice tutorial toggle",
                        1 => "Highlight Canvas, Input Panel, Snapshot Manager (user must interact with each)",
                        2 => "User types a phrase or drags an image. System offers Auto-Embed (recommended) and Ultralite mapping. Show estimated neuron usage and storage cost.",
                        3 => "Run 10 simulation steps with mini-analytics. Show 'What you are seeing' callout explaining pulses and clusters.",
                        4 => "Suggest name, auto-generate reproducibility manifest summary, offer interactive HTML export suggestion (size estimate).",
                        _ => "'First Thought' badge unlocked, show next suggestions: 'Explore Analytics', 'Try Autonomous Mode', 'Connect LED Strip'.",
                    }).size(16),
                    row![
                        button("Skip").on_press(Message::SkipOnboarding).style(iced::theme::Button::Secondary),
                        button("Next").on_press(Message::NextOnboardingStep).style(iced::theme::Button::Primary),
                    ].spacing(10)
                ].spacing(20).align_items(iced::Alignment::Center)
            ).padding(40).center_x().center_y().style(iced::theme::Container::Box).into())
        } else if self.show_analytics {
            // Simple text-based histogram for firing rates
            let histogram = (0..10).map(|i| {
                let count = self.network.neurons.iter().filter(|n| (n.output * 10.0) as usize == i).count();
                format!("{:2}: {}", i, "█".repeat(count.min(20)))
            }).collect::<Vec<_>>().join("\n");
            Some(container(
                column![
                    text("Analytics Dashboard").size(20),
                    text("Firing Rate Histogram:").size(14),
                    text(histogram).size(12),
                    text(format!("Neurons: {}", neuron_count)).size(14),
                    text(format!("Connections: {}", connection_count)).size(14),
                    text(format!("Avg Activity: {:.2}", avg_activity)).size(14),
                    text(format!("Total Energy: {:.1}", total_energy)).size(14),
                    button("Run Analytics").on_press(Message::RunAnalytics),
                    button("Close").on_press(Message::ToggleAnalytics),
                ].spacing(10)
            ).padding(20).center_x().center_y().style(iced::theme::Container::Box).into())
        } else if self.show_export_wizard {
            let estimated_size = match self.export_format.as_str() {
                "JSON" => "2.3 MB",
                "GIF" => "15.7 MB",
                "MP4" => "45.2 MB",
                "VR Scene" => "8.9 MB",
                _ => "Unknown",
            };
            let estimated_time = "2m 34s"; // placeholder
            Some(container(
                row![
                    // Left column: format selection, presets, size estimate
                    container(
                        column![
                            text("Format Selection").size(16),
                             button("JSON").on_press(Message::SetExportFormat("JSON".to_string())),
                             button("Interactive HTML").on_press(Message::SetExportFormat("interactive_html".to_string())),
                             button("GIF").on_press(Message::SetExportFormat("GIF".to_string())),
                             button("MP4").on_press(Message::SetExportFormat("MP4".to_string())),
                             button("VR Scene").on_press(Message::SetExportFormat("VR Scene".to_string())),
                            text(format!("Selected: {}", self.export_format)).size(14),
                            text("Presets").size(14),
                            button("Quick Export").on_press(Message::StartExport),
                            text(format!("Estimated Size: {}", estimated_size)).size(12),
                            text(format!("Estimated Time: {}", estimated_time)).size(12),
                        ].spacing(10)
                    ).width(200).padding(10).style(iced::theme::Container::Box),
                    // Middle column: detailed options and checkboxes
                    container(
                        column![
                            text("Options").size(16),
                            button(if self.export_delta_only { "Delta Only: On" } else { "Delta Only: Off" }).on_press(Message::ToggleExportDeltaOnly),
                            text("Compression Level:").size(14),
                            vertical_slider(0.0..=1.0, self.export_compress, Message::SetExportCompress).width(150),
                            button(if self.export_encryption { "Encryption: On" } else { "Encryption: Off" }).on_press(Message::ToggleExportEncryption),
                            text("Include Analytics").size(14),
                            text("Include Thumbnails").size(14),
                        ].spacing(10)
                    ).width(250).padding(10).style(iced::theme::Container::Box),
                    // Right column: export preview + estimated timeline + warnings
                    container(
                        column![
                            text("Preview & Timeline").size(16),
                            text("Estimated Timeline:").size(14),
                            text("Step 1: Prepare data - 30s").size(12),
                            text("Step 2: Compress - 1m").size(12),
                            text("Step 3: Finalize - 1m 4s").size(12),
                            text("Warnings:").size(14),
                            text("Large file size may take time.").size(12),
                            button("Start Export").on_press(Message::StartExport).style(iced::theme::Button::Primary),
                            button("Cancel").on_press(Message::ShowExportWizard).style(iced::theme::Button::Secondary),
                        ].spacing(10)
                    ).width(250).padding(10).style(iced::theme::Container::Box),
                ].spacing(10)
            ).padding(20).center_x().center_y().style(iced::theme::Container::Box).into())
        } else if self.show_mapping_wizard {
            Some(container(
                column![
                    text("Mapping Wizard").size(20).style(iced::theme::Text::Color(theme.text)),
                    text_input("Input Type", &self.mapping_input_type).on_input(Message::SetMappingInputType),
                    text_input("Strategy", &self.mapping_strategy).on_input(Message::SetMappingStrategy),
                    button("Apply").on_press(Message::ApplyMapping).style(iced::theme::Button::Primary),
                    button("Cancel").on_press(Message::ShowMappingWizard).style(iced::theme::Button::Secondary),
                ].spacing(10)
            ).padding(20).center_x().center_y().style(iced::theme::Container::Box).into())
        } else if self.show_autonomous_modal {
            Some(container(
                column![
                    text("Autonomous Experiment").size(20).style(iced::theme::Text::Color(theme.text)),
                    text("Choose preset:").size(16),
                    row![
                        button("Dream").on_press(Message::SetAutonomousPreset("Dream".to_string())),
                        button("Replay").on_press(Message::SetAutonomousPreset("Replay".to_string())),
                        button("Pattern Search").on_press(Message::SetAutonomousPreset("Pattern Search".to_string())),
                        button("Random Explore").on_press(Message::SetAutonomousPreset("Random Explore".to_string())),
                    ].spacing(10),
                    text(format!("Selected: {}", self.autonomous_preset)).size(14),
                    text("Energy Budget:").size(16),
                    vertical_slider(10.0..=1000.0, self.autonomous_energy_budget, Message::SetAutonomousEnergyBudget).width(200),
                    text("Time Limit (seconds):").size(16),
                    vertical_slider(60.0..=3600.0, self.autonomous_time_limit as f64, |v| Message::SetAutonomousTimeLimit(v as u64)).width(200),
                    text("Detection Sensitivity:").size(16),
                    vertical_slider(0.0..=1.0, self.autonomous_detection_sensitivity, Message::SetAutonomousDetectionSensitivity).width(200),
                    text("Logging Level:").size(16),
                    row![
                        button("Low").on_press(Message::SetAutonomousLoggingLevel("Low".to_string())),
                        button("Medium").on_press(Message::SetAutonomousLoggingLevel("Medium".to_string())),
                        button("High").on_press(Message::SetAutonomousLoggingLevel("High".to_string())),
                    ].spacing(10),
                    button("Start Experiment").on_press(Message::StartAutonomousExperiment).style(iced::theme::Button::Primary),
                    button("Cancel").on_press(Message::ToggleAutonomousModal).style(iced::theme::Button::Secondary),
                ].spacing(10)
            ).padding(20).center_x().center_y().style(iced::theme::Container::Box).into())
        } else if self.show_help {
            let all_content = vec![
                ("Overview", "MindMesh is a neural network simulator with advanced visualization and AI features."),
                ("Tutorials", "Interactive tutorials: UI Basics (5min), Mapping Advanced (12min), Autonomous Experiments (10min)"),
                ("Keyboard Shortcuts", "See the Shortcuts modal for full list."),
                ("Troubleshooting", "Common issues: Low memory - switch to Ultralite mode, GPU issues - fallback to CPU."),
            ];
            let filtered_content: Vec<_> = if self.help_search.is_empty() {
                all_content.into_iter().filter(|(title, _)| title == &self.help_section).collect()
            } else {
                all_content.into_iter().filter(|(title, content)| 
                    title.to_lowercase().contains(&self.help_search.to_lowercase()) || 
                    content.to_lowercase().contains(&self.help_search.to_lowercase())
                ).collect()
            };
            let content = if filtered_content.is_empty() {
                "No results found.".to_string()
            } else {
                filtered_content.into_iter().map(|(_, c)| c).collect::<Vec<_>>().join("\n\n")
            };
            Some(container(
                column![
                    text("Help & Tutorials").size(24),
                    text_input("Search help...", &self.help_search).on_input(Message::HelpSearchChanged),
                    row![
                        button("Overview").on_press(Message::SetHelpSection("Overview".to_string())),
                        button("Tutorials").on_press(Message::SetHelpSection("Tutorials".to_string())),
                        button("Shortcuts").on_press(Message::SetHelpSection("Keyboard Shortcuts".to_string())),
                        button("Troubleshooting").on_press(Message::SetHelpSection("Troubleshooting".to_string())),
                    ].spacing(10),
                    text(content).size(16),
                    button("Close").on_press(Message::ToggleHelp),
                ].spacing(15).align_items(iced::Alignment::Center)
            ).padding(30).center_x().center_y().style(iced::theme::Container::Box).into())
        } else if self.show_collaboration {
            Some(container(
                column![
                    text("Collaboration").size(24),
                    text("Share and collaborate on neural networks").size(16),
                    text("Features:").size(14),
                    text("• LAN discovery").size(12),
                    text("• P2P synchronization").size(12),
                    text("• Conflict resolution").size(12),
                    text("• Token-based joining").size(12),
                    button("Start Session").on_press(Message::ToggleCollaboration),
                    button("Close").on_press(Message::ToggleCollaboration),
                ].spacing(10).align_items(iced::Alignment::Center)
            ).padding(30).center_x().center_y().style(iced::theme::Container::Box).into())
        } else if self.show_ethics_modal {
            Some(container(
                column![
                    text("Research Ethics Guidelines").size(24),
                    text("MindMesh is designed for educational and research purposes.").size(16),
                    text("Please ensure responsible use:").size(16),
                    text("• Respect privacy and data rights").size(12),
                    text("• Use for positive scientific advancement").size(12),
                    text("• Avoid harmful applications").size(12),
                    text("• Cite sources and acknowledge contributions").size(12),
                    text("• Report any ethical concerns").size(12),
                    button("I Understand").on_press(Message::ToggleEthicsModal),
                ].spacing(10).align_items(iced::Alignment::Center)
            ).padding(30).center_x().center_y().style(iced::theme::Container::Box).into())
        } else {
            None
        };

        // Main Layout
        let main_layout = column![
            header,
            row![
                left_sidebar,
                canvas_area,
                right_sidebar,
            ].spacing(10),
            footer,
        ].spacing(10).padding(10);

        // Apply modal overlay if needed
        let layout_with_hint = if self.show_contextual_hint {
            column![
                main_layout,
                container(
                    row![
                        text(&self.contextual_hint).size(14).style(iced::theme::Text::Color(theme.accent)),
                        button("✕").on_press(Message::DismissHint).style(iced::theme::Button::Secondary),
                    ].spacing(10).align_items(iced::Alignment::Center)
                ).padding(10).style(iced::theme::Container::Box).center_x(),
            ].spacing(10)
        } else {
            column![main_layout]
        };

        if let Some(modal) = modal_content {
            container(
                column![
                    layout_with_hint,
                    container(modal).center_x().center_y(),
                ]
            ).into()
        } else {
            layout_with_hint.into()
        }
    }
}

impl MindMesh {
    fn app_theme(&self) -> AppTheme {
        if self.high_contrast {
            AppTheme::high_contrast()
        } else if let Some(ref mode) = self.colorblind_mode {
            match mode.as_str() {
                "protanopia" => AppTheme::protanopia(),
                "deuteranopia" => AppTheme::deuteranopia(),
                "tritanopia" => AppTheme::tritanopia(),
                _ => AppTheme::light(),
            }
        } else if self.dark_theme {
            AppTheme::dark()
        } else {
            AppTheme::light()
        }
    }

    fn scaled_size(&self, base: f32) -> u16 {
        (base * self.font_scale) as u16
    }

    fn t(&self, key: &str) -> String {
        self.i18n.get(key).cloned().unwrap_or_else(|| key.to_string())
    }
}

impl canvas::Program<Message> for MindMesh {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: iced::Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());

        // Background
        let bg_rect = canvas::Path::rectangle(iced::Point::new(0.0, 0.0), bounds.size());
        let bg_color = if self.high_contrast {
            Color::BLACK
        } else if self.dark_theme {
            Color::from_rgb(0.02, 0.02, 0.05)
        } else {
            Color::from_rgb(0.98, 0.98, 0.95)
        };
        frame.fill(&bg_rect, canvas::Fill { style: canvas::Style::Solid(bg_color), ..canvas::Fill::default() });

        // Add subtle grid
        let grid_spacing = 50.0 * self.zoom;
        let grid_color = if self.dark_theme {
            Color::from_rgba(0.2, 0.2, 0.3, 0.3)
        } else {
            Color::from_rgba(0.7, 0.7, 0.8, 0.2)
        };
        let pan_x = self.pan.0 + bounds.width / 2.0;
        let pan_y = self.pan.1 + bounds.height / 2.0;
        for x in (0..(bounds.width as i32 + grid_spacing as i32)).step_by(grid_spacing as usize) {
            let screen_x = x as f32 - pan_x % grid_spacing;
            let line = canvas::Path::line(
                iced::Point::new(screen_x, 0.0),
                iced::Point::new(screen_x, bounds.height),
            );
            frame.stroke(&line, canvas::Stroke::default().with_color(grid_color).with_width(0.5));
        }
        for y in (0..(bounds.height as i32 + grid_spacing as i32)).step_by(grid_spacing as usize) {
            let screen_y = y as f32 - pan_y % grid_spacing;
            let line = canvas::Path::line(
                iced::Point::new(0.0, screen_y),
                iced::Point::new(bounds.width, screen_y),
            );
            frame.stroke(&line, canvas::Stroke::default().with_color(grid_color).with_width(0.5));
        }

        // Draw connections with LOD
        if self.show_connections {
            let lod_threshold = if self.lod_slider < 0.5 { 0.1 } else if self.lod_slider < 1.0 { 0.5 } else { 1.0 };
            for conn in &self.network.connections {
                if conn.weight.abs() < lod_threshold { continue; }
                if let Some(from_idx) = self.network.neurons.iter().position(|n| n.id == conn.from_id) {
                    if let Some(to_idx) = self.network.neurons.iter().position(|n| n.id == conn.to_id) {
                        let from_pos = (
                            (self.network.neurons[from_idx].position.0 * self.zoom + self.pan.0) + bounds.width / 2.0,
                            (self.network.neurons[from_idx].position.1 * self.zoom + self.pan.1) + bounds.height / 2.0,
                        );
                        let to_pos = (
                            (self.network.neurons[to_idx].position.0 * self.zoom + self.pan.0) + bounds.width / 2.0,
                            (self.network.neurons[to_idx].position.1 * self.zoom + self.pan.1) + bounds.height / 2.0,
                        );
                        let path = canvas::Path::line(
                            iced::Point::new(from_pos.0, from_pos.1),
                            iced::Point::new(to_pos.0, to_pos.1),
                        );
                        let width = (conn.weight.abs() * 5.0).max(0.5) as f32;
                        let base_color = if conn.weight > 0.0 {
                            Color::from_rgb(0.2, 1.0, 0.3) // Excitatory
                        } else {
                            Color::from_rgb(1.0, 0.3, 0.2) // Inhibitory
                        };
                        // Add activity modulation
                        let from_active = self.network.neurons[from_idx].should_fire();
                        let to_active = self.network.neurons[to_idx].should_fire();
                        let activity_boost = if from_active || to_active { 0.3 } else { 0.0 };
                        let color = Color::from_rgb(
                            (base_color.r + activity_boost).min(1.0),
                            (base_color.g + activity_boost).min(1.0),
                            (base_color.b + activity_boost).min(1.0),
                        );
                        frame.stroke(&path, canvas::Stroke::default().with_color(color).with_width(width));

                        // Add arrowhead for directed connections
                        if conn.weight.abs() > 0.5 {
                            let dx = to_pos.0 - from_pos.0;
                            let dy = to_pos.1 - from_pos.1;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist > 0.0 {
                                let arrow_length = 10.0;
                                let arrow_pos = (
                                    to_pos.0 - dx / dist * arrow_length,
                                    to_pos.1 - dy / dist * arrow_length,
                                );
                                let perp_dx = -dy / dist * 3.0;
                                let perp_dy = dx / dist * 3.0;
                                let arrow_path = canvas::Path::new(|builder| {
                                    builder.move_to(iced::Point::new(to_pos.0, to_pos.1));
                                    builder.line_to(iced::Point::new(arrow_pos.0 + perp_dx, arrow_pos.1 + perp_dy));
                                    builder.line_to(iced::Point::new(arrow_pos.0 - perp_dx, arrow_pos.1 - perp_dy));
                                    builder.close();
                                });
                                frame.fill(&arrow_path, canvas::Fill { style: canvas::Style::Solid(color), ..canvas::Fill::default() });
                            }
                        }
                    }
                }
            }
        }

        // Draw neurons
        for neuron in &self.network.neurons {
            let pos = (
                (neuron.position.0 * self.zoom + self.pan.0) + bounds.width / 2.0,
                (neuron.position.1 * self.zoom + self.pan.1) + bounds.height / 2.0,
            );

            let radius = if self.show_3d {
                // 3D visualization: use z-coordinate for size
                3.0 + (neuron.position.2 as f32 * 0.1) + neuron.output as f32 * 8.0
            } else {
                5.0 + neuron.output as f32 * 10.0
            };

            let base_color = match neuron.neuron_type {
                NeuronType::Excitatory => Color::from_rgb(0.2, 0.9, 0.3),
                NeuronType::Inhibitory => Color::from_rgb(0.9, 0.3, 0.2),
                NeuronType::Sensory => Color::from_rgb(0.9, 0.9, 0.2),
                NeuronType::Motor => Color::from_rgb(0.2, 0.9, 0.9),
                NeuronType::AiAssist => Color::from_rgb(0.9, 0.3, 0.9),
                _ => Color::from_rgb(0.7, 0.7, 0.7),
            };

            // Modulate color by activity and energy
            let activity_factor = neuron.output as f32;
            let energy_factor = (neuron.metadata.energy_cost as f32).min(1.0);
            let mut color = Color::from_rgb(
                base_color.r * (0.5 + activity_factor * 0.5),
                base_color.g * (0.5 + activity_factor * 0.5),
                base_color.b * (1.0 - energy_factor * 0.3),
            );

            // Add pulsing effect for active neurons
            if neuron.should_fire() {
                let pulse = (self.network.time as f32 * 0.1).sin() * 0.5 + 0.5;
                color = Color::from_rgb(
                    color.r + pulse * 0.2,
                    color.g + pulse * 0.2,
                    color.b + pulse * 0.2,
                );
            }

            let circle = canvas::Path::circle(iced::Point::new(pos.0, pos.1), radius);
            frame.fill(&circle, canvas::Fill { style: canvas::Style::Solid(color), ..canvas::Fill::default() });

            // Add firing glow with multiple layers
            if neuron.should_fire() {
                for i in 1..4 {
                    let glow_radius = radius + i as f32 * 3.0;
                    let glow_color = Color::from_rgba(1.0, 1.0, 0.0, 0.5 / i as f32);
                    let glow_circle = canvas::Path::circle(iced::Point::new(pos.0, pos.1), glow_radius);
                    frame.stroke(&glow_circle, canvas::Stroke::default().with_color(glow_color).with_width(2.0 / i as f32));
                }
                // Inner bright core
                let core_circle = canvas::Path::circle(iced::Point::new(pos.0, pos.1), radius * 0.5);
                frame.fill(&core_circle, canvas::Fill { style: canvas::Style::Solid(Color::from_rgb(1.0, 1.0, 0.8)), ..canvas::Fill::default() });
            }

            // Add energy indicator
            if energy_factor > 0.5 {
                let energy_ring = canvas::Path::circle(iced::Point::new(pos.0, pos.1), radius + 8.0);
                frame.stroke(&energy_ring, canvas::Stroke::default().with_color(Color::from_rgba(1.0, 0.5, 0.0, energy_factor)).with_width(2.0));
            }
        }

        // Draw particles
        for p in &self.particles {
            let alpha = p.life / p.max_life;
            let color = if p.max_life > 40.0 { // Sparks
                Color::from_rgba(0.8, 1.0, 1.0, alpha) // Cyan sparks
            } else {
                Color::from_rgba(1.0, 0.8, 0.2, alpha) // Orange particles
            };
            let size = if p.max_life > 40.0 { 2.0 * alpha } else { 3.0 * alpha };
            let circle = canvas::Path::circle(iced::Point::new(p.position.0, p.position.1), size);
            frame.fill(&circle, canvas::Fill { style: canvas::Style::Solid(color), ..canvas::Fill::default() });
            // Add trail
            if p.life > p.max_life * 0.5 {
                let trail_color = Color::from_rgba(color.r * 0.7, color.g * 0.7, color.b * 0.7, alpha * 0.5);
                let trail_size = size * 0.7;
                let trail_circle = canvas::Path::circle(iced::Point::new(p.position.0 - p.velocity.0 * 2.0, p.position.1 - p.velocity.1 * 2.0), trail_size);
                frame.fill(&trail_circle, canvas::Fill { style: canvas::Style::Solid(trail_color), ..canvas::Fill::default() });
            }
        }

        vec![frame.into_geometry()]
    }
}
