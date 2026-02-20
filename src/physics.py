import torch

def find_k_nearest(positions, k):
    n = len(positions)
    distances = torch.cdist(positions, positions)
    knn_distances, knn_indices = torch.topk(distances, min(k + 1, n), largest=False, dim=1)
    return knn_distances[:, 1:], knn_indices[:, 1:]

def lennard_jones_force(particles, distances, indices):
    n = particles.n_particles
    forces = torch.zeros_like(particles.positions)
    
    for i in range(n):
        neighbor_indices = indices[i]
        r = distances[i]
        
        epsilon_ij = torch.sqrt(particles.epsilons[i] * particles.epsilons[neighbor_indices])
        sigma_ij = 0.5 * (particles.sigmas[i] + particles.sigmas[neighbor_indices])
        
        r_clipped = torch.clamp(r, min=1e-2)
        sigma_r6 = (sigma_ij / r_clipped) ** 6
        sigma_r12 = sigma_r6 ** 2
        
        f_magnitude = 24.0 * epsilon_ij * (2.0 * sigma_r12 - sigma_r6) / r_clipped
        
        pos_diff = particles.positions[i] - particles.positions[neighbor_indices]
        r_vec_norm = torch.norm(pos_diff, dim=1, keepdim=True)
        r_vec_norm = torch.clamp(r_vec_norm, min=1e-2)
        direction = pos_diff / r_vec_norm
        
        force_vectors = f_magnitude[:, None] * direction
        forces[i] = torch.sum(force_vectors, dim=0)
    
    return forces

def coulomb_force(particles, distances, indices, k_e):
    n = particles.n_particles
    forces = torch.zeros_like(particles.positions)
    
    for i in range(n):
        neighbor_indices = indices[i]
        r = distances[i]
        
        q_i = particles.charges[i]
        q_j = particles.charges[neighbor_indices]
        
        r_clipped = torch.clamp(r, min=1e-2)
        f_magnitude = k_e * q_i * q_j / (r_clipped ** 2)
        
        pos_diff = particles.positions[i] - particles.positions[neighbor_indices]
        r_vec_norm = torch.norm(pos_diff, dim=1, keepdim=True)
        r_vec_norm = torch.clamp(r_vec_norm, min=1e-2)
        direction = pos_diff / r_vec_norm
        
        force_vectors = f_magnitude[:, None] * direction
        forces[i] = torch.sum(force_vectors, dim=0)
    
    return forces

def compute_forces(particles, k, coulomb_k):
    distances, indices = find_k_nearest(particles.positions, k)
    f_lj = lennard_jones_force(particles, distances, indices)
    f_coulomb = coulomb_force(particles, distances, indices, coulomb_k)
    return f_lj + f_coulomb

def apply_boundary(particles, boundary_size, restitution):
    for dim in range(3):
        lower = particles.positions[:, dim] < 0
        upper = particles.positions[:, dim] > boundary_size[dim]
        
        particles.positions[lower, dim] = 0
        particles.velocities[lower, dim] = -particles.velocities[lower, dim] * restitution
        
        particles.positions[upper, dim] = boundary_size[dim]
        particles.velocities[upper, dim] = -particles.velocities[upper, dim] * restitution
