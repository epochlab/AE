import pytest
import torch
import time
from src.utils import load_config, create_particles
from src.physics import compute_forces

@pytest.fixture
def benchmark_config():
    config = load_config()
    config['particles']['count'] = {'H': 50, 'He': 30, 'Ne': 10}
    return config

def test_force_computation_speed(benchmark, benchmark_config):
    device = 'cpu'
    particles = create_particles(benchmark_config, device)
    k = benchmark_config['physics']['k_neighbors']
    coulomb_k = benchmark_config['physics']['coulomb_constant']
    
    def compute():
        return compute_forces(particles, k, coulomb_k)
    
    result = benchmark(compute)
    assert result is not None

def test_particle_creation_speed(benchmark, benchmark_config):
    device = 'cpu'
    
    def create():
        return create_particles(benchmark_config, device)
    
    result = benchmark(create)
    assert result is not None

def test_scaling_with_particle_count():
    device = 'cpu'
    config = load_config()
    
    counts = [10, 50, 100]
    times = []
    
    for n in counts:
        config['particles']['count'] = {'H': n}
        particles = create_particles(config, device)
        
        start = time.time()
        compute_forces(particles, 8, 8.9875517923e9)
        elapsed = time.time() - start
        times.append(elapsed)
    
    assert all(t >= 0 for t in times)
    print(f"\nScaling: {list(zip(counts, times))}")

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_acceleration():
    config = load_config()
    config['particles']['count'] = {'H': 100}
    
    particles_cpu = create_particles(config, 'cpu')
    start_cpu = time.time()
    compute_forces(particles_cpu, 8, 8.9875517923e9)
    time_cpu = time.time() - start_cpu
    
    particles_mps = create_particles(config, 'mps')
    start_mps = time.time()
    compute_forces(particles_mps, 8, 8.9875517923e9)
    time_mps = time.time() - start_mps
    
    print(f"\nCPU: {time_cpu:.4f}s, MPS: {time_mps:.4f}s")
    assert time_cpu >= 0 and time_mps >= 0
