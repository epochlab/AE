# AE - Atomic Engine

Classical particle simulator with physically correct interactions at atomic scale.

## Features

- **118 Elements**: Full periodic table with accurate physical parameters
- **Classical Physics**: Lennard-Jones potential and Coulomb forces
- **K-Nearest Neighbors**: Efficient O(N²) GPU-parallelized interactions (k=8)
- **RK4 Integration**: Fourth-order Runge-Kutta for accuracy
- **GPU Acceleration**: Apple Silicon (MPS) and CUDA support via PyTorch
- **Real-time Rendering**: OpenGL visualization at 120 FPS target
- **Temperature Control**: Berendsen thermostat for NVT ensemble
- **Boundary Conditions**: Box boundaries with restitution coefficient

## Installation

### Create Environment

```bash
conda env create -f environment.yaml
conda activate ae
```

### Manual Installation

```bash
conda create -n ae python=3.11
conda activate ae
conda install pytorch torchvision torchaudio -c pytorch
conda install moderngl glfw pyrr pyyaml h5py pytest pytest-benchmark -c conda-forge
```

## Quick Start

```bash
python main.py
```

### Controls

- **ESC**: Exit simulation
- **SPACE**: Pause/Resume
- **1**: Color by element (CPK standard)
- **2**: Color by velocity
- **3**: Color by acceleration
- **4**: Color by mass
- **5**: Color by charge

## Configuration

Edit `config.yaml` to customize:

```yaml
simulation:
  integrator: "RK4"
  dt: 1.0e-15          # Timestep (1 femtosecond)
  device: "mps"        # mps, cuda, or cpu

physics:
  k_neighbors: 8
  temperature: 300.0   # Kelvin
  thermostat: "berendsen"

particles:
  count:
    H: 30
    He: 20
    Ne: 10

boundary:
  size: [100.0, 100.0, 100.0]  # Angstroms
  restitution: 0.95
```

## Physics

### Lennard-Jones Potential

```
V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
F(r) = 24ε/r[(2σ¹²/r¹²) - (σ⁶/r⁶)] r̂
```

### Coulomb Force

```
F = k_e × (q₁q₂/r²) r̂
k_e = 8.9875517923×10⁹ N⋅m²/C²
```

### Temperature

```
T = (2/3) × KE / (N × k_B)
KE = 0.5 × Σ(m × v²)
```

### Berendsen Thermostat

```
λ = √[1 + (dt/τ) × (T_target/T_current - 1)]
v_new = v_old × λ
```

## Units

- **Length**: Angstroms (Å)
- **Time**: Femtoseconds (fs)
- **Mass**: Atomic mass units (amu)
- **Energy**: kcal/mol (Lennard-Jones)
- **Temperature**: Kelvin (K)

## Testing

```bash
pytest tests/
pytest tests/test_physics.py -v
pytest tests/test_performance.py --benchmark-only
```

## Performance

Target performance on Apple Silicon (M1/M2/M3):

- **Force computation**: <5ms for 100 particles
- **Rendering**: 120 FPS
- **Total step time**: <10ms

## Project Structure

```
AE/
├── config.yaml           # Simulation configuration
├── elements.yaml         # Periodic table (118 elements)
├── environment.yaml      # Conda environment
├── main.py              # Entry point
├── src/
│   ├── particle.py      # Particle system dataclass
│   ├── physics.py       # Force calculations
│   ├── integrator.py    # RK4 integration
│   ├── renderer.py      # OpenGL rendering
│   ├── simulator.py     # Main simulation loop
│   └── utils.py         # Config/element loaders
└── tests/
    ├── test_imports.py
    ├── test_physics.py
    ├── test_graphics.py
    └── test_performance.py
```

## Window Title Format

```
AE | 120 FPS | RK4 | H:30 He:20 Ne:10 | F:2.1ms P:0.3ms R:8.3ms
```

- **FPS**: Frames per second
- **RK4**: Integration method
- **H:30 He:20 Ne:10**: Particle counts by element
- **F**: Force calculation time
- **P**: Physics step time
- **R**: Render time

## Element Data

All 118 elements included with:
- Atomic mass (amu)
- Van der Waals radius (Å)
- CPK standard colors (RGB)
- Lennard-Jones parameters (ε, σ)

Source: NIST, Wolfram Alpha, UFF (Universal Force Field)

## License

MIT

## References

- CPK Coloring: https://en.wikipedia.org/wiki/CPK_coloring
- UFF Parameters: Rappé et al., J. Am. Chem. Soc. 1992
- RK4 Integration: Press et al., Numerical Recipes
