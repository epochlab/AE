import time
import torch
from src.integrator import rk4_step, velocity_verlet_step, euler_step
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
        
        self.stats_interval = config['output']['stats_interval']
        self.step = 0
        
        integrator_name = config['simulation']['integrator'].lower()
        if integrator_name == 'euler':
            self.integrate = euler_step
        elif integrator_name == 'verlet':
            self.integrate = velocity_verlet_step
        else:
            self.integrate = velocity_verlet_step
        
        self.timings = {'force': 0.0, 'physics': 0.0, 'render': 0.0}
        self.fps = 0.0
    
    def run(self):
        last_time = time.time()
        frame_count = 0
        
        while self.step < self.config['simulation']['steps']:
            if self.renderer and not self.renderer.running:
                break
            
            if self.renderer and self.renderer.paused:
                element_counts = get_element_counts(self.particles)
                integrator_name = self.config['simulation']['integrator']
                self.renderer.render(self.particles, self.fps, element_counts, self.timings, integrator_name)
                time.sleep(0.016)
                continue
            
            t_phys_start = time.time()
            
            t_force_start = time.time()
            self.integrate(self.particles, self.dt, self.k, self.coulomb_k)
            if self.particles.device.type == 'mps':
                torch.mps.synchronize()
            self.timings['force'] = (time.time() - t_force_start) * 1000
            
            apply_boundary(self.particles, self.boundary_size, self.restitution)
            
            if self.use_thermostat:
                self.particles.apply_thermostat(self.target_temp, self.thermostat_tau, self.dt)
            
            self.timings['physics'] = (time.time() - t_phys_start) * 1000
            
            if self.renderer:
                t_render_start = time.time()
                element_counts = get_element_counts(self.particles)
                integrator_name = self.config['simulation']['integrator']
                self.renderer.render(self.particles, self.fps, element_counts, self.timings, integrator_name)
                self.timings['render'] = (time.time() - t_render_start) * 1000
            
            self.step += 1
            frame_count += 1
            
            current_time = time.time()
            if current_time - last_time >= 1.0:
                self.fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
            
            if self.step % self.stats_interval == 0:
                self._print_stats()
        
        if self.renderer:
            self.renderer.cleanup()
    
    def _print_stats(self):
        ke = self.particles.kinetic_energy().item()
        temp = self.particles.temperature().item()
        element_counts = get_element_counts(self.particles)
        
        elem_str = ' '.join([f"{k}:{v}" for k, v in element_counts.items()])
        print(f"Step {self.step:6d} | KE: {ke:.2e} J | T: {temp:.1f} K | {elem_str} | "
              f"F:{self.timings['force']:.1f}ms P:{self.timings['physics']:.1f}ms R:{self.timings['render']:.1f}ms | {self.fps:.0f} FPS")
