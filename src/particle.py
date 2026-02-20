import torch
from dataclasses import dataclass

@dataclass
class ParticleSystem:
    positions: torch.Tensor
    velocities: torch.Tensor
    masses: torch.Tensor
    charges: torch.Tensor
    radii: torch.Tensor
    colors: torch.Tensor
    epsilons: torch.Tensor
    sigmas: torch.Tensor
    elements: list
    device: str

    @property
    def n_particles(self):
        return len(self.positions)

    def kinetic_energy(self):
        mass_kg = self.masses * 1.66053906660e-27
        v_m_s = self.velocities * 1e5
        return 0.5 * torch.sum(mass_kg[:, None] * v_m_s ** 2)

    def temperature(self, k_b=1.380649e-23):
        ke = self.kinetic_energy()
        return (2.0 / 3.0) * ke / (self.n_particles * k_b)

    def apply_thermostat(self, target_temp, tau, dt, k_b=1.380649e-23):
        current_temp = self.temperature(k_b)
        if current_temp > 0:
            lambda_factor = torch.sqrt(1.0 + (dt / tau) * (target_temp / current_temp - 1.0))
            self.velocities *= lambda_factor
