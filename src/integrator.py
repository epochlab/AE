import torch
from src.physics import compute_forces

def euler_step(particles, dt, k, coulomb_k):
    f = compute_forces(particles, k, coulomb_k)
    a = f / particles.masses.unsqueeze(1)
    
    particles.velocities += a * dt
    particles.positions += particles.velocities * dt
    
    return particles

def velocity_verlet_step(particles, dt, k, coulomb_k):
    f = compute_forces(particles, k, coulomb_k)
    a = f / particles.masses.unsqueeze(1)
    
    particles.positions += particles.velocities * dt + 0.5 * a * dt * dt
    
    f_new = compute_forces(particles, k, coulomb_k)
    a_new = f_new / particles.masses.unsqueeze(1)
    
    particles.velocities += 0.5 * (a + a_new) * dt
    
    return particles

rk4_step = velocity_verlet_step
