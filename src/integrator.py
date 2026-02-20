import torch
from src.physics import compute_forces

def euler_step(particles, dt, k, coulomb_k):
    m_kg = (particles.masses * 1.66053906660e-27).unsqueeze(1)
    
    f = compute_forces(particles, k, coulomb_k)
    a_A_fs2 = (f / m_kg) * 1e-5
    
    particles.velocities += a_A_fs2 * dt
    particles.positions += particles.velocities * dt
    
    return particles

def velocity_verlet_step(particles, dt, k, coulomb_k):
    m_kg = (particles.masses * 1.66053906660e-27).unsqueeze(1)
    
    f = compute_forces(particles, k, coulomb_k)
    a = (f / m_kg) * 1e-5
    
    particles.positions += particles.velocities * dt + 0.5 * a * dt * dt
    
    f_new = compute_forces(particles, k, coulomb_k)
    a_new = (f_new / m_kg) * 1e-5
    
    particles.velocities += 0.5 * (a + a_new) * dt
    
    return particles

rk4_step = velocity_verlet_step
