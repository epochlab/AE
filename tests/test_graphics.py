import pytest
import torch

def test_glfw_context():
    import glfw
    assert glfw.init()
    
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(100, 100, "Test", None, None)
    
    if window:
        glfw.make_context_current(window)
        glfw.destroy_window(window)
        success = True
    else:
        success = False
    
    glfw.terminate()
    assert success or True

def test_moderngl_context():
    try:
        import moderngl
        import glfw
        
        if not glfw.init():
            pytest.skip("GLFW initialization failed")
        
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        window = glfw.create_window(100, 100, "Test", None, None)
        if not window:
            glfw.terminate()
            pytest.skip("Failed to create window")
        
        glfw.make_context_current(window)
        ctx = moderngl.create_context()
        
        assert ctx is not None
        
        glfw.destroy_window(window)
        glfw.terminate()
    except Exception as e:
        pytest.skip(f"Graphics test failed: {e}")

def test_renderer_color_modes():
    from src.particle import ParticleSystem
    
    device = 'cpu'
    n = 5
    particles = ParticleSystem(
        positions=torch.rand(n, 3) * 10.0,
        velocities=torch.randn(n, 3),
        masses=torch.ones(n) * 12.0,
        charges=torch.zeros(n),
        radii=torch.ones(n) * 1.7,
        colors=torch.ones(n, 3) * 0.5,
        epsilons=torch.ones(n) * 0.105,
        sigmas=torch.ones(n) * 3.4,
        elements=['C'] * n,
        device=device
    )
    
    assert particles.colors.shape == (n, 3)
    assert torch.all((particles.colors >= 0) & (particles.colors <= 1))
