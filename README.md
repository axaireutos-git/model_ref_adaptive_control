# Stepper Motor Simulation with Adaptive Control and Neural Network Estimation

This repository contains a Python project that simulates the behavior of a stepper motor system under adaptive control. The system incorporates a neural network to estimate unknown external torque disturbances. The project explores the motor's dynamics, adaptive parameter evolution, and the contribution of neural network-based torque estimation.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [System Description](#system-description)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)

## Introduction
Stepper motors are widely used in robotics and automation due to their precision. This project simulates a stepper motor under varying conditions, including unknown torque disturbances. A neural network is trained to approximate the disturbance, and the motor's response is analyzed with and without this compensation.

## Features
- Simulation of stepper motor dynamics using differential equations.
- Adaptive control with parameter adjustment in real-time.
- Neural network training to estimate external torque disturbances.
- Visualization of motor response, adaptive parameters, and phase currents.

## System Description
- **Parameters**:
  - `J`: Moment of Inertia.
  - `Km`: Motor Torque Constant.
  - `b`: Friction Coefficient.
- **Adaptive Gains**:
  - `Γx`, `Γr`, `Γθ`: Gains for adjusting control parameters.
- **Neural Network**:
  - Three-layer model with `Tanh` activations trained on synthetic torque data.
- **External Torque**:
  - Simulates a nonlinear disturbance modeled as a function of the motor's angle.

## Dependencies
- `numpy`: For numerical computations.
- `scipy`: For solving differential equations.
- `matplotlib`: For visualizing results.
- `torch`: For neural network training.

## Usage
1. Run the script to simulate the stepper motor's behavior
   ```bash
   model_ref_adaptive_control.ipynb
   ```

2. Modify parameters, neural network architecture, or simulation settings in the script as needed.

### Key Functions
- `simulate_system`: Simulates the motor's response over time.
- `plot_response`: Visualizes the motor's rotation angle and angular velocity.
- `plot_parameters`: Displays the evolution of adaptive control parameters.
- `train_neural_network`: Trains the neural network to estimate external torque.

## Results
The project generates several outputs:
1. **Response Plots**: Compare the motor's response with and without neural network-based torque estimation.
2. **Adaptive Parameters**: Track the convergence of adaptive gains to ideal values.
3. **Phase Currents**: Show the electrical behavior of the motor.
