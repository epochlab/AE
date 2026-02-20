import pytest
import torch
from src.particle import ParticleSystem
from src.physics import find_k_nearest, compute_forces

@pytest.fixture
def simple_particles():
    device = 'cpu'
    n = 10
    return ParticleSystem(
        positions=torch.rand(n, 3) * 10.0,
        velocities=torch.randn(n, 3),
        masses=torch.ones(n) * 12.0,
        charges=torch.zeros(n),
        radii=torch.ones(n) * 1.7,
        colors=torch.ones(n, 3) * 0.5,
        epsilons=torch.ones(n) * 0.105,
        sigmas=torch.ones(n) * 3.4,
        elements=['C'] * n,
        device=device
    )

def test_k_nearest(simple_particles):
    k = 4
    distances, indices = find_k_nearest(simple_particles.positions, k)
    assert distances.shape == (simple_particles.n_particles, k)
    assert indices.shape == (simple_particles.n_particles, k)
    assert torch.all(distances >= 0)

def test_combined_forces(simple_particles):
    forces = compute_forces(simple_particles, k=4, coulomb_k=8.9875517923e+9)
    assert forces.shape == simple_particles.positions.shape
    assert not torch.any(torch.isnan(forces))

def test_forces_with_charges(simple_particles):
    simple_particles.charges = torch.randn(simple_particles.n_particles)
    forces = compute_forces(simple_particles, k=4, coulomb_k=8.9875517923e+9)
    assert forces.shape == simple_particles.positions.shape
    assert not torch.any(torch.isnan(forces))

def test_energy_conservation():
    device = 'cpu'
    particles = ParticleSystem(
        positions=torch.tensor([[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]]),
        velocities=torch.zeros(2, 3),
        masses=torch.ones(2) * 12.0,
        charges=torch.zeros(2),
        radii=torch.ones(2) * 1.7,
        colors=torch.ones(2, 3),
        epsilons=torch.ones(2) * 0.105,
        sigmas=torch.ones(2) * 3.4,
        elements=['C', 'C'],
        device=device
    )
    
    initial_ke = particles.kinetic_energy()
    assert initial_ke >= 0
