import moderngl
import glfw
import numpy as np
from pyrr import Matrix44

class Renderer:
    def __init__(self, config):
        self.config = config
        self.window_size = tuple(config['window_size'])
        self.point_size = config['point_size']
        self.color_mode = config['color_mode']
        self.bg = config['background']
        
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        self.window = glfw.create_window(self.window_size[0], self.window_size[1], "AE", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_window_user_pointer(self.window, self)
        glfw.set_key_callback(self.window, self._key_callback)
        
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec3 in_color;
                uniform mat4 mvp;
                uniform float point_size;
                out vec3 v_color;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    gl_PointSize = point_size;
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, 1.0);
                }
            '''
        )
        
        self.vbo_pos = None
        self.vbo_col = None
        self.vao = None
        self.paused = False
        self.running = True
    
    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                self.running = False
            elif key == glfw.KEY_SPACE:
                self.paused = not self.paused
            elif key == glfw.KEY_1:
                self.color_mode = "element"
            elif key == glfw.KEY_2:
                self.color_mode = "velocity"
            elif key == glfw.KEY_3:
                self.color_mode = "acceleration"
            elif key == glfw.KEY_4:
                self.color_mode = "mass"
            elif key == glfw.KEY_5:
                self.color_mode = "charge"
    
    def _compute_colors(self, particles):
        if self.color_mode == "element":
            return particles.colors
        elif self.color_mode == "velocity":
            v_mag = np.linalg.norm(particles.velocities.cpu().numpy(), axis=1)
            v_norm = (v_mag - v_mag.min()) / (v_mag.max() - v_mag.min() + 1e-8)
            colors = np.stack([v_norm, 1 - v_norm, 0.5 * np.ones_like(v_norm)], axis=1)
            return colors
        elif self.color_mode == "mass":
            m = particles.masses.cpu().numpy()
            m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
            colors = np.stack([m_norm, 0.5 * np.ones_like(m_norm), 1 - m_norm], axis=1)
            return colors
        elif self.color_mode == "charge":
            q = particles.charges.cpu().numpy()
            colors = np.zeros((len(q), 3))
            colors[q > 0] = [1, 0, 0]
            colors[q < 0] = [0, 0, 1]
            colors[q == 0] = [0.5, 0.5, 0.5]
            return colors
        else:
            return particles.colors
    
    def render(self, particles, fps, element_counts, timings):
        if glfw.window_should_close(self.window):
            self.running = False
            return
        
        positions = particles.positions.cpu().numpy().astype('f4')
        colors = self._compute_colors(particles).astype('f4')
        
        if self.vbo_pos is None:
            self.vbo_pos = self.ctx.buffer(positions.tobytes())
            self.vbo_col = self.ctx.buffer(colors.tobytes())
            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vbo_pos, '3f', 'in_position'), (self.vbo_col, '3f', 'in_color')]
            )
        else:
            self.vbo_pos.write(positions.tobytes())
            self.vbo_col.write(colors.tobytes())
        
        cam_pos = self.config['camera']['position']
        look_at = self.config['camera']['look_at']
        projection = Matrix44.perspective_projection(45.0, self.window_size[0] / self.window_size[1], 0.1, 500.0)
        view = Matrix44.look_at(cam_pos, look_at, [0, 1, 0])
        mvp = projection * view
        
        self.prog['mvp'].write(mvp.astype('f4').tobytes())
        self.prog['point_size'].value = self.point_size
        
        self.ctx.clear(self.bg[0], self.bg[1], self.bg[2])
        self.vao.render(moderngl.POINTS)
        
        element_str = ' '.join([f"{k}:{v}" for k, v in element_counts.items()])
        timing_str = f"F:{timings['force']:.1f}ms P:{timings['physics']:.1f}ms R:{timings['render']:.1f}ms"
        title = f"AE | {fps:.0f} FPS | RK4 | {element_str} | {timing_str}"
        glfw.set_window_title(self.window, title)
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def cleanup(self):
        glfw.terminate()
