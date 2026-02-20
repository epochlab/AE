import pytest

def test_torch_import():
    import torch
    assert torch is not None

def test_torch_mps():
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this system")
    assert torch.backends.mps.is_available()

def test_moderngl_import():
    import moderngl
    assert moderngl is not None

def test_glfw_import():
    import glfw
    assert glfw.init()
    glfw.terminate()

def test_yaml_import():
    import yaml
    assert yaml is not None

def test_src_imports():
    from src.particle import ParticleSystem
    from src.physics import compute_forces
    from src.integrator import rk4_step
    from src.utils import load_elements, load_config
    from src.simulator import Simulator
    assert all([ParticleSystem, compute_forces, rk4_step, load_elements, load_config, Simulator])
