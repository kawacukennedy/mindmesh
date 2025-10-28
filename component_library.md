# MindMesh Component Library

## Overview
This document describes the reusable UI components for the MindMesh application, implemented in Rust using the Iced framework.

## Components

### 1. Header Bar
- **Purpose**: Global navigation and session controls
- **Elements**: Hamburger menu, title, search input, session indicators, save/undo/redo buttons, settings
- **Styling**: Dark theme, 48px height, gradient background

### 2. Sidebar Panels
- **Left Sidebar**: Project manager, inputs, presets
- **Right Sidebar**: Inspector, controls, AI assistant, export
- **Features**: Resizable, collapsible, keyboard navigation

### 3. Canvas
- **Purpose**: Main visualization area
- **Features**: Pan/zoom, 2D/3D toggle, overlays, particle effects
- **Interactions**: Mouse/touch gestures, keyboard shortcuts

### 4. Modal Dialogs
- **Types**: Settings, analytics, export wizard, onboarding
- **Animations**: Scale from center, 180ms duration
- **Accessibility**: ARIA roles, keyboard navigation

### 5. Buttons
- **Variants**: Primary, secondary, positive, destructive
- **States**: Normal, hover, pressed, disabled
- **Feedback**: Scale animation on click

### 6. Sliders
- **Types**: Vertical, horizontal
- **Styling**: Custom thumb, track colors
- **Accessibility**: Keyboard navigation, screen reader support

### 7. Tooltips
- **Delay**: 150ms
- **Positioning**: Auto-adjust based on screen bounds
- **Content**: Rich text with icons

## Theming
All components support the design tokens defined in `design_tokens.json`, including colors, typography, spacing, and animations.

## Usage
Components are implemented as Iced widgets and can be composed to build complex interfaces while maintaining consistency.