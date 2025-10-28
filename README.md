# MindMesh

MindMesh is the ultimate offline self-evolving digital brain simulator capable of simulating millions of neurons and billions of connections with advanced AI, VR/AR/holographic visualization, procedural generation, energy-efficient storage, analytics, gamification, hardware integration, collaboration, and full accessibility.

## Features

- **Core Simulation**: Rust-based neuron simulation with 16 neuron types, 9 activation functions, and 8 plasticity types.
- **Learning**: Advanced learning with multiple algorithms, memory systems (short-term, long-term, semantic, episodic, predictive), feedback loops, and self-optimization.
- **Visualization**: 2D/3D canvas visualization with zoom, pan, connection toggling, color-coded neurons/connections, firing indicators, emergent pattern highlighting, particle effects, and grid overlays.
- **Analytics**: Comprehensive analytics including neuron activity histograms, cluster heatmaps, connection distributions, rare event detection, energy tracking, temporal correlations, and predictive anomaly detection.
- **Interactivity**: Real-time editing, drag-and-drop connections, annotation, custom input patterns, gesture controls, voice commands, AI assistance, prediction, undo/redo with branching, and sandbox editing.
- **Self-Evolution**: Autonomous synaptogenesis, probabilistic rewiring, meta-learning, AI-guided rewiring, adaptive parameters, emergent pattern tracking, and autonomous experiments with safety caps.
- **Snapshot Management**: Versioned snapshots with AES-256-GCM encryption, compression, delta snapshots with journaling and replay, manifest support for reproducibility, comparison tools, and multiple export formats (MM, MMSnapshot, Compressed HTML, GIF, MP4, VR/AR scenes).
- **Gamification**: Achievements, leaderboards, emergent behavior badges, and meta-pattern competitions.
- **AI Integration**: Offline GPT-style assistant, meta-pattern prediction, goal suggestion, AI-guided rewiring, and adaptive learning.
- **Hardware Interface**: Simulated BCI, LED arrays, haptics, sensors, robotics, hardware integration with serial/MIDI/BLE adapters, and WASM plugins with sandboxing.
- **Collaboration**: LAN/P2P brain sharing, multi-user offline sessions, shared evolution experiments.
- **Accessibility**: Screen reader support, keyboard navigation, colorblind modes (protanopia, deuteranopia, tritanopia), dynamic font scaling (0.5x-2.0x), high contrast, subtitles, voice-guided tutorials, gesture-only mode, tooltips, and keyboard shortcuts help modal.
- **Performance**: Multi-threading, GPU/TPU acceleration, memory efficiency, energy tracking, profiling tools.
- **Security**: Sandboxed execution, local data only, encrypted snapshots, access control.
- **Settings**: Comprehensive simulation, visual, audio, notification controls, theme switching, and onboarding wizard.

## Developer Setup

### Prerequisites
- Rust 1.70+ (install from https://rustup.rs/)
- Cargo (comes with Rust)
- For hardware integration: Serial port access (may require permissions on some systems)

### Development Workflow
1. Clone the repository
2. Run `./scripts/build.sh` to build all components
3. Run `./dist/mindmesh-frontend` to start the application (requires GUI environment)
4. Make changes to code and rebuild as needed

### Building Components

#### Core Library
```bash
cd core
cargo build --release  # Optimized build
cargo test             # Run tests
cargo doc --open       # Generate documentation
```

#### Frontend Application
```bash
cd frontend
cargo build --release
```

#### Full Build
```bash
./scripts/build.sh  # Builds everything and creates dist/
# Or manually:
cd core && cargo build --release
cd ../frontend && cargo build --release
mkdir -p ../dist && cp target/release/mindmesh-frontend ../dist/
```

### Testing
- Core tests: `cd core && cargo test`
- Manual UI testing: Run the application and test workflows
- Integration testing: Test hardware connections and plugin loading

### Code Structure
- `core/src/lib.rs`: Main simulation engine
- `core/src/hardware.rs`: Hardware interface implementations
- `core/src/p2p.rs`: Peer-to-peer networking
- `core/src/plugins.rs`: WASM plugin system
- `frontend/src/main.rs`: GUI application

## API Documentation

Run `cargo doc --open` in the core directory to view detailed API documentation.

### Core API Overview
- `Network`: Main simulation structure
- `Neuron`: Individual neuron with type, activation, and state
- `Connection`: Synaptic connections with weight and plasticity
- `Simulation`: Time-step based evolution
- `HardwareInterface`: External device integration
- `PluginManager`: WASM plugin loading and execution

## Architecture

- `core/`: Rust library with neuron, connection, network, and simulation logic.
- `frontend/`: Iced-based UI for interacting with the brain.
- `modules/`: For optional modules (Haskell, Elixir, etc.).
- `scripts/`: Build scripts.

## Accessibility

- Keyboard navigation supported via Iced framework.
- High contrast themes can be added in future updates.

## UI Overview

The MindMesh interface is organized into several panels with vibrant gradients, shadows, and accessibility features:

- **Header**: Gradient-styled header with title, global search, session controls (play/pause, quick save, undo/redo, colorblind toggle, high contrast, help), and tooltips.
- **Left Sidebar (Project & Snapshot Manager)**: Save/load projects, snapshot management (save, load, delete, compare), datasets & inputs (import, recent inputs), presets & experiments (Dream, Replay, Pattern Search, Autonomous).
- **Canvas Area**: Main visualization with grid background, LOD slider, playback scrubber, visualization overlays, particle effects for firing neurons, sparks for connections, and real-time stats (neurons, connections, energy, firing count, patterns).
- **Right Sidebar (Panels)**: Inspector (neuron details), Input Panel (text/random/ASCII/voice, mapping wizard), Simulation Controls (play/pause/step/reset/speed/energy mode), Export & AI (JSON/GIF/MP4/VR, AI suggestions), notifications with timestamps.
- **Footer**: Log console with timestamps, analytics mini-cards (neurons/connections/energy/firing), status indicators (simulation state, resources: CPU/GPU/RAM, shortcuts).
- **Modals**: Settings (simulation speed, font scale, themes, accessibility), Analytics Dashboard (energy/patterns/memory stats), Autonomous Experiment (presets, energy budget, time limit), Onboarding Wizard (step-by-step guide), Input Mapping Wizard (type/strategy selection), Export Wizard (format/options), Ethics Modal (research guidelines), Keyboard Shortcuts Help.

## User Guide

### Getting Started
1. Build and run the application using `./scripts/build.sh`
2. Launch `./dist/mindmesh-frontend`
3. Follow the onboarding wizard to set up your first neural network

### Creating Your First Network
- Use the "Add Neuron" button or keyboard shortcuts to add neurons
- Connect neurons by dragging from one to another
- Set input patterns using the text input or random generators
- Press Space to start the simulation

### Advanced Features
- **Snapshots**: Save and load network states with versioning
- **Analytics**: View detailed statistics and patterns in the analytics modal
- **Hardware Integration**: Connect external devices via serial ports
- **Collaboration**: Share networks over LAN with other users
- **Plugins**: Extend functionality with WASM plugins

## Controls

- Space: Toggle play/pause
- Up/Down arrows: Zoom in/out
- R: Reset brain
- S: Single step
- E: Show export wizard
- M: Show mapping wizard
- A: Show autonomous modal
- ?: Show keyboard shortcuts (or click Help)
- Shift+Click: Multi-select neurons (future feature)

## Implemented Advanced Features

- Quantum neurons with stochastic activation
- Self-evolving networks with growth and pruning
- Multi-modal inputs and outputs (text, numbers, ASCII art, voice, sensor, gesture, BCI)
- Offline AI assistance and pattern prediction
- Theme-based accessibility with colorblind modes and font scaling
- Export formats (JSON, GIF, MP4, VR Scene)
- Snapshot encryption and delta journaling
- Autonomous experiments with presets and safety caps
- Undo/redo with branching and sandbox editing
- Live sonification with MIDI/OSC support
- P2P collaboration framework
- WASM plugins with sandboxing
- Hardware adapters (serial, MIDI, BLE)
- Onboarding wizard and ethics modal
- Enhanced notifications and log console with timestamps
- Keyboard shortcuts help modal

## Future Expansions

- Full VR/AR support with OpenXR
- Real-time 3D rendering with Vulkan/WebGL
- Chart-based analytics dashboard
- P2P networking for live collaboration
- Actual hardware BCI and sensor integration
- Online leaderboards and multiplayer modes
- Advanced AI with GPT-like text generation

## License

TBD