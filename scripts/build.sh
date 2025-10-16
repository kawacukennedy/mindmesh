#!/bin/bash

echo "Building MindMesh..."

# Check for required tools
command -v cargo >/dev/null 2>&1 || { echo "Cargo is required but not installed. Aborting." >&2; exit 1; }

# Build core library
echo "Building core..."
cd core
cargo build --release
if [ $? -ne 0 ]; then
    echo "Core build failed!"
    exit 1
fi
cd ..

# Build frontend
echo "Building frontend..."
cd frontend
cargo build --release
if [ $? -ne 0 ]; then
    echo "Frontend build failed!"
    exit 1
fi
cd ..

# Create distribution directory
echo "Creating distribution..."
mkdir -p dist
cp frontend/target/release/mindmesh-frontend dist/
cp -r core/src dist/core_src
cp README.md dist/

echo "Build complete! Executable is in dist/mindmesh-frontend"