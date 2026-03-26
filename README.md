# User Guide: Lunar Convection Simulation with Dedalus

## Overview
This script simulates thermal convection in the Moon's interior using the Dedalus framework. It is based on the spherical shell convection example but includes modifications for:
- Infinite Prandtl number
- Phase change at the LMO-cumulates boundary
- Core cooling
- Customizable initial temperature profiles and boundary conditions

The simulation is controlled via a `param.toml` configuration file.

---

## Requirements
- Python 3.7+
- Dedalus (see [Dedalus Roject](https://dedalus-project.readthedocs.io/en/latest/))
- MPI (for parallel execution)
- TOML parser (`pip install tomli`)

---

## Configuration File: `param.toml`
The simulation is configured via a TOML file. Below are the available parameters:

| Parameter      | Description                                                                 | Example Value         |
|----------------|-----------------------------------------------------------------------------|-----------------------|
| GAMMA          | Aspect ratio (Ri/Ro)                                                        | 0.24                  |
| RAYLEIGH       | Rayleigh number                                                             | 1e4                   |
| PHI            | Phase change number (set to `false` for no phase change)                    | 6                     |
| FACTOR         | Curvature factor for temperature profile                                    | 4.45                  |
| GRAVITY        | Gravity profile: `true` for uniform, `false` for radius-dependent           | true                  |
| DT             | Initial and maximum time step                                               | 1e-5                  |
| STOP           | Stop simulation time                                                        | 0.1                   |
| Nr             | Radial resolution                                                           | 128                   |
| Nphi           | Azimuthal resolution                                                        | 96                    |
| Ntheta         | Polar resolution                                                            | 64                    |
| SAFETY         | Safety factor for CFL number                                                | 3                     |
| NOISE          | Initial noise amplitude                                                     | 1e-3                  |
| BC             | Thermal boundary conditions: `Dirichlet` or `CoreCooling`                   | CoreCooling           |
| TempProf       | Initial temperature profile: `Conductive`, `Parabolic`, or `RefProf`        | RefProf               |
| DIR_SAVE       | Directory to save simulation data                                           | ./Datas               |

---

## Running the Simulation

### Basic Usage
To run the simulation using 4 MPI processes:
```bash
mpiexec -n 4 python3 shell_convection.py param.toml
```

### Restarting a Simulation
To restart from the latest checkpoint:
```bash
mpiexec -n 4 python3 shell_convection.py param.toml --restart
```

---

## Outputs
The script saves the following data in the specified `DIR_SAVE/name` directory:
- **Snapshots**: Temperature and flux fields at regular intervals.
- **Scalars**: Time series of averaged quantities (e.g., mean flux, temperature, velocity).
- **Checkpoints**: Simulation state for restarting.
The name of the directory depends of the value of parameters:
- `Datas/Rayleigh_1-0e+04_Phi_6-0e+00_gam_0-24_T_RefProf_BC_CoreCooling_g_r_safety_3-0_dt_1e-05_res_96-64-128` for exemple

### Output Files Structure
```
DIR_SAVE/
├── snapshots/
│   ├── snapshots_s1.h5
│   └── ...
├── scalars/
│   ├── scalars_s1.h5
│   └── ...
├── diagnostics/
│   └── ...
└── checkpoints/
    ├── checkpoints_s1.h5
    └── ...
```

---

## Key Features
- **Phase Change**: Enabled by setting `PHI` to a positive value.
- **Core Cooling**: Activated with `BC = CoreCooling`.
- **Gravity Profile**: Uniform or radius-dependent, controlled by `GRAVITY`.
- **Initial Temperature Profiles**: Choose between `Conductive`, `Parabolic`, or `RefProf`.

---

## Notes
- The script uses the SBDF1 timestepper by default.
- For high-resolution simulations, increase `Nr`, `Nphi`, and `Ntheta` accordingly.
- The CFL condition is automatically adjusted for stability, be carreful.

---

## Troubleshooting
- **Divergence**: If the simulation diverges, reduce the initial time step (`DT`) or increase the safety factor (`SAFETY`, can bi higer than 1).
- **Checkpoint Errors**: Ensure the checkpoint directory exists and contains valid `.h5` files.

---
