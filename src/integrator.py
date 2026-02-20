import torch
from src.physics import compute_forces

def rk4_step(particles, dt, k, coulomb_k):
    x0 = particles.positions.clone()
    v0 = particles.velocities.clone()
    m = particles.masses[:, None]
    
    f1 = compute_forces(particles, k, coulomb_k)
    k1_v = f1 / m
    k1_x = v0
    
    particles.positions = x0 + 0.5 * dt * k1_x
    particles.velocities = v0 + 0.5 * dt * k1_v
    f2 = compute_forces(particles, k, coulomb_k)
    k2_v = f2 / m
    k2_x = v0 + 0.5 * dt * k1_v
    
    particles.positions = x0 + 0.5 * dt * k2_x
    particles.velocities = v0 + 0.5 * dt * k2_v
    f3 = compute_forces(particles, k, coulomb_k)
    k3_v = f3 / m
    k3_x = v0 + 0.5 * dt * k2_v
    
    particles.positions = x0 + dt * k3_x
    particles.velocities = v0 + dt * k3_v
    f4 = compute_forces(particles, k, coulomb_k)
    k4_v = f4 / m
    k4_x = v0 + dt * k3_v
    
    particles.positions = x0 + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    particles.velocities = v0 + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    
    return particles
