import time
import torch
from src.integrator import euler_step, velocity_verlet_step
from src.physics import apply_boundary
from src.utils import get_element_counts

class Simulator:
    def __init__(self, particles, config, renderer=None):
        self.particles = particles
        self.config = config
        self.renderer = renderer
        
        self.dt = config['simulation']['dt']
        self.k = config['physics']['k_neighbors']
        self.coulomb_k = config['physics']['coulomb_constant']
        self.boundary_size = config['boundary']['size']
        self.restitution = config['boundary']['restitution']
        
        self.use_thermostat = config['physics']['thermostat'] != "none"
        self.target_temp = config['physics']['temperature']
        self.thermostat_tau = config['physics']['thermostat_tau']
        
        self.step = 0
        
        integrator_name = config['simulation']['integrator'].lower()
        self.integrate = euler_step if integrator_name == 'euler' else velocity_verlet_step
        self.integrator_name = config['simulation']['integrator']
        
        self.fps = 0.0
    
    def run(self):
        last_time = time.time()
        frame_count = 0
        
        while self.step < self.config['simulation']['steps']:
            if self.renderer and not self.renderer.running:
                break
            
            if self.renderer and self.renderer.paused:
                element_counts = get_element_counts(self.particles)
                self.renderer.render(self.particles, self.fps, element_counts, self.integrator_name)
                time.sleep(0.016)
                continue
            
            self.integrate(self.particles, self.dt, self.k, self.coulomb_k)
            
            if self.particles.device == 'mps':
                torch.mps.synchronize()
            
            apply_boundary(self.particles, self.boundary_size, self.restitution)
            
            if self.use_thermostat:
                self.particles.apply_thermostat(self.target_temp, self.thermostat_tau, self.dt)
            
            if self.renderer:
                element_counts = get_element_counts(self.particles)
                self.renderer.render(self.particles, self.fps, element_counts, self.integrator_name)
            
            self.step += 1
            frame_count += 1
            
            current_time = time.time()
            if current_time - last_time >= 0.1:
                self.fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
        
        if self.renderer:
            self.renderer.cleanup()
