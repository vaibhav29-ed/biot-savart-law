# Biot–Savart Law Simulation

A 3D numerical simulation built with Python that calculates and visualizes magnetic fields around current-carrying conductors using the vector form of the Biot–Savart law. Developed by Vaibhav Singh as a physics and programming integration project.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Links and Contact](#links-and-contact)

## Features

- **Vector Field Calculation**: Numerically integrates the Biot–Savart law ($\vec{dB} = \frac{\mu_0 I}{4\pi} \frac{d\vec{l} \times \vec{r}}{r^3}$) across discretized current loop segments.
- **Interactive 3D Rendering**: Displays normalized magnetic field vectors as 3D cones using Plotly (`go.Cone`) alongside the wire geometry (`go.Scatter3d`).
- **3D Grid Evaluation**: Evaluates vector directions across a customizable 3D coordinate grid.

## Requirements

Running the simulation requires a Python environment with the following dependencies:

- `numpy`
- `plotly`

## Usage

Run the primary simulation script with Python:

```bash
python biot-savart-sim.py
```

Alternatively, execute the identical final script:

```bash
python "final code.py"
```

Upon execution, Plotly renders an interactive 3D visualization displaying the current loop and its surrounding magnetic field vectors.

## Project Structure

```
├── .ipynb_checkpoints/
│   ├── Untitled-checkpoint.ipynb
│   ├── Untitled1-checkpoint.ipynb
│   └── biot-savart-sim-checkpoint.py
├── Readme.md
├── biot-savart-sim.py
└── final code.py
```

- `biot-savart-sim.py`: Main script parameterizing a circular current loop and visualizing its 3D magnetic field using NumPy and Plotly.
- `final code.py`: Secondary script containing the complete simulation and Plotly visualization code.
- `.ipynb_checkpoints/`: Notebook checkpoints containing symbolic math and experimental loop formulations.

## Links and Contact

- **Interactive Demo**: [Google Colab Notebook](https://colab.research.google.com/drive/1MheHrIYXwd4vYv0Ky1p9eKL6q1xLb3d-?usp=sharing)
- **Author**: Vaibhav Singh
- **LinkedIn**: [vaibhav-singh-680910217](https://www.linkedin.com/in/vaibhav-singh-680910217)