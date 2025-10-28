use std::io::{Read, Write};
use serialport::SerialPort;

pub trait HardwareAdapter {
    fn read_data(&mut self) -> Vec<f64>;
    fn write_data(&mut self, data: &[f64]);
}

pub struct SerialAdapter {
    port: Box<dyn SerialPort>,
}

impl SerialAdapter {
    pub fn new(port_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let port = serialport::new(port_name, 9600).open()?;
        Ok(SerialAdapter { port })
    }
}

impl HardwareAdapter for SerialAdapter {
    fn read_data(&mut self) -> Vec<f64> {
        let mut buffer = [0u8; 20];
        if let Ok(len) = self.port.read(&mut buffer) {
            buffer[..len].iter().map(|&b| b as f64 / 255.0).collect()
        } else {
            vec![]
        }
    }

    fn write_data(&mut self, data: &[f64]) {
        let bytes: Vec<u8> = data.iter().map(|&v| (v * 255.0) as u8).collect();
        let _ = self.port.write(&bytes);
    }
}

// Placeholder for MIDI, OSC, etc.
pub struct MidiAdapter;

impl MidiAdapter {
    pub fn new() -> Self {
        MidiAdapter
    }
}

impl HardwareAdapter for MidiAdapter {
    fn read_data(&mut self) -> Vec<f64> {
        // Placeholder
        vec![0.5; 10]
    }

    fn write_data(&mut self, _data: &[f64]) {
        // Send MIDI
    }
}