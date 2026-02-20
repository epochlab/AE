#!/usr/bin/env python
"""Quick test of simulator with limited steps"""

import torch
from src.utils import load_config, create_particles, print_initial_system
from src.renderer import Renderer
from src.simulator import Simulator

def main():
    config = load_config()
    config['simulation']['steps'] = 100  # Just 100 steps for quick test
    
    device = config['simulation']['device']
    if device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    particles = create_particles(config, device)
    print_initial_system(particles, config)
    
    renderer = None
    if config['renderer']['enabled']:
        renderer = Renderer(config['renderer'])
    
    simulator = Simulator(particles, config, renderer)
    simulator.run()

if __name__ == "__main__":
    main()
