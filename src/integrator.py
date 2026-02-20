import torch
from src.physics import compute_forces

def velocity_verlet_step(particles, dt, k, coulomb_k):
    """
    Velocity Verlet integration - faster than RK4 for real-time
    Only 2 force evaluations per step vs 4 for RK4
    """
    m = particles.masses[:, None]
    
    # Compute forces at current position
    f = compute_forces(particles, k, coulomb_k)
    a = f / m
    
    # Update positions
    particles.positions = particles.positions + particles.velocities * dt + 0.5 * a * dt * dt
    
    # Compute forces at new position
    f_new = compute_forces(particles, k, coulomb_k)
    a_new = f_new / m
    
    # Update velocities
    particles.velocities = particles.velocities + 0.5 * (a + a_new) * dt
    
    return particles

def euler_step(particles, dt, k, coulomb_k):
    """
    Simple Euler integration - fastest, 1 force evaluation
    Less accurate but good enough for visualization
    """
    m = particles.masses[:, None]
    
    f = compute_forces(particles, k, coulomb_k)
    a = f / m
    
    particles.velocities = particles.velocities + a * dt
    particles.positions = particles.positions + particles.velocities * dt
    
    return particles

# Alias for backward compatibility
rk4_step = velocity_verlet_step
