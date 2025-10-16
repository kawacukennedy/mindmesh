use std::net::{TcpListener, TcpStream, UdpSocket};
use std::io::{Read, Write};
use std::thread;
use serde::{Deserialize, Serialize};
use crate::Network;

#[derive(Serialize, Deserialize)]
pub enum P2PMessage {
    Snapshot(Network),
    Delta(Vec<u8>),
    Join(String), // session token
}

pub struct P2PServer {
    listener: TcpListener,
}

impl P2PServer {
    pub fn new(port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port))?;
        Ok(P2PServer { listener })
    }

    pub fn run(&self) {
        for stream in self.listener.incoming() {
            match stream {
                Ok(stream) => {
                    thread::spawn(move || {
                        handle_client(stream);
                    });
                }
                Err(e) => eprintln!("Connection failed: {}", e),
            }
        }
    }
}

fn handle_client(mut stream: TcpStream) {
    let mut buffer = [0; 1024];
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break, // Connection closed
            Ok(n) => {
                // Process message
                if let Ok(msg) = bincode::deserialize::<P2PMessage>(&buffer[..n]) {
                    match msg {
                        P2PMessage::Snapshot(net) => {
                            // Merge or something
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        }
    }
}

pub fn discover_peers() -> Vec<String> {
    // Simple UDP broadcast for discovery
    let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
    socket.set_broadcast(true).unwrap();
    socket.send_to(b"MINDSH", "255.255.255.255:9999").unwrap();
    // Listen for responses
    vec![] // Placeholder
}