import yaml
import torch
import numpy as np
from src.particle import ParticleSystem

_element_cache = None

def load_elements():
    global _element_cache
    if _element_cache is None:
        with open('elements.yaml', 'r') as f:
            _element_cache = yaml.safe_load(f)['elements']
    return _element_cache

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_particles(config, device):
    elements_data = load_elements()
    particle_config = config['particles']
    counts = particle_config['count']
    
    positions_list = []
    velocities_list = []
    masses_list = []
    charges_list = []
    radii_list = []
    colors_list = []
    epsilons_list = []
    sigmas_list = []
    elements_list = []
    
    boundary = config['boundary']['size']
    temp = config['physics']['temperature']
    k_b = 1.380649e-23
    
    for symbol, count in counts.items():
        element = elements_data[symbol]
        
        pos = torch.rand(count, 3) * 50.0 + 25.0
        
        mass_kg = element['mass'] * 1.66053906660e-27
        v_thermal_m_s = np.sqrt(3 * k_b * temp / mass_kg)
        v_thermal_A_fs = v_thermal_m_s * 1e-5
        vel = (torch.randn(count, 3) * v_thermal_A_fs * particle_config.get('velocity_scale', 1.0))
        
        positions_list.append(pos)
        velocities_list.append(vel)
        masses_list.extend([element['mass']] * count)
        charges_list.extend([element.get('charge', 0.0)] * count)
        radii_list.extend([element['radius']] * count)
        colors_list.extend([element['color']] * count)
        epsilons_list.extend([element['lj_epsilon']] * count)
        sigmas_list.extend([element['lj_sigma']] * count)
        elements_list.extend([symbol] * count)
    
    positions = torch.cat(positions_list).to(device)
    velocities = torch.cat(velocities_list).to(device)
    masses = torch.tensor(masses_list, dtype=torch.float32).to(device)
    charges = torch.tensor(charges_list, dtype=torch.float32).to(device)
    radii = torch.tensor(radii_list, dtype=torch.float32).to(device)
    colors = torch.tensor(colors_list, dtype=torch.float32).to(device)
    epsilons = torch.tensor(epsilons_list, dtype=torch.float32).to(device)
    sigmas = torch.tensor(sigmas_list, dtype=torch.float32).to(device)
    
    return ParticleSystem(
        positions=positions,
        velocities=velocities,
        masses=masses,
        charges=charges,
        radii=radii,
        colors=colors,
        epsilons=epsilons,
        sigmas=sigmas,
        elements=elements_list,
        device=device
    )

def get_element_counts(particles):
    counts = {}
    for elem in particles.elements:
        counts[elem] = counts.get(elem, 0) + 1
    return counts

def print_initial_system(particles, config):
    counts = get_element_counts(particles)
    print("=" * 60)
    print("AE - Atomic Engine")
    print("=" * 60)
    print(f"Device: {particles.device}")
    print(f"Integrator: {config['simulation']['integrator']}")
    print(f"Timestep: {config['simulation']['dt']:.2e} s")
    print(f"K-Neighbors: {config['physics']['k_neighbors']}")
    print(f"Temperature: {config['physics']['temperature']} K")
    print(f"Boundary: {config['boundary']['size']} Å")
    print(f"Restitution: {config['boundary']['restitution']}")
    print()
    print("Particles:")
    for elem, count in sorted(counts.items()):
        print(f"  {elem}: {count}")
    print(f"  Total: {particles.n_particles}")
    print("=" * 60)
