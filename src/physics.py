import torch

def find_k_nearest(positions, k):
    n = len(positions)
    distances = torch.cdist(positions, positions)
    knn_distances, knn_indices = torch.topk(distances, min(k + 1, n), largest=False, dim=1)
    return knn_distances[:, 1:], knn_indices[:, 1:]

def compute_forces(particles, k, coulomb_k):
    n = particles.n_particles
    k = min(k, n - 1)
    
    distances, indices = find_k_nearest(particles.positions, k)
    
    pos_i = particles.positions.unsqueeze(1)
    pos_j = particles.positions[indices]
    pos_diff = pos_i - pos_j
    
    r_vec = torch.norm(pos_diff, dim=2, keepdim=True)
    r_vec = torch.clamp(r_vec, min=1e-2)
    direction = pos_diff / r_vec
    
    r = r_vec.squeeze(-1)
    
    epsilon_i = particles.epsilons.unsqueeze(1)
    epsilon_j = particles.epsilons[indices]
    epsilon_ij = torch.sqrt(epsilon_i * epsilon_j)
    
    sigma_i = particles.sigmas.unsqueeze(1)
    sigma_j = particles.sigmas[indices]
    sigma_ij = 0.5 * (sigma_i + sigma_j)
    
    sigma_r6 = (sigma_ij / r) ** 6
    sigma_r12 = sigma_r6 ** 2
    f_lj_mag = 24.0 * epsilon_ij * (2.0 * sigma_r12 - sigma_r6) / r
    
    q_i = particles.charges.unsqueeze(1)
    q_j = particles.charges[indices]
    f_coulomb_mag = coulomb_k * q_i * q_j / (r ** 2)
    
    f_total_mag = f_lj_mag + f_coulomb_mag
    
    force_vectors = f_total_mag.unsqueeze(-1) * direction
    forces = torch.sum(force_vectors, dim=1)
    
    return forces

def apply_boundary(particles, boundary_size, restitution):
    boundary_tensor = torch.tensor(boundary_size, device=particles.positions.device, dtype=particles.positions.dtype)
    
    lower = particles.positions < 0
    upper = particles.positions > boundary_tensor
    
    particles.positions[lower] = 0
    particles.velocities[lower] = -particles.velocities[lower] * restitution
    
    particles.positions[upper] = boundary_tensor.expand_as(particles.positions)[upper]
    particles.velocities[upper] = -particles.velocities[upper] * restitution
