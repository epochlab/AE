import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

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
        
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)
        
        vertex_src = """
            #version 330 core
            layout (location = 0) in vec3 in_position;
            layout (location = 1) in vec3 in_color;
            uniform mat4 mvp;
            uniform float point_size;
            out vec3 v_color;
            void main() {
                gl_Position = mvp * vec4(in_position, 1.0);
                gl_PointSize = point_size;
                v_color = in_color;
            }
        """
        
        fragment_src = """
            #version 330 core
            in vec3 v_color;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color, 1.0);
            }
        """
        
        vertex_shader = compileShader(vertex_src, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_src, GL_FRAGMENT_SHADER)
        
        self.shader = glCreateProgram()
        glAttachShader(self.shader, vertex_shader)
        glAttachShader(self.shader, fragment_shader)
        glLinkProgram(self.shader)
        
        if glGetProgramiv(self.shader, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.shader))
        
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        self.vao = glGenVertexArrays(1)
        self.vbo_pos = glGenBuffers(1)
        self.vbo_col = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
        
        self.paused = False
        self.running = True
        self.mvp_loc = glGetUniformLocation(self.shader, "mvp")
        self.point_size_loc = glGetUniformLocation(self.shader, "point_size")
    
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
            return particles.colors.cpu().numpy()
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
            return particles.colors.cpu().numpy()
    
    def _create_mvp_matrix(self):
        cam_pos = np.array(self.config['camera']['position'], dtype=np.float32)
        look_at = np.array(self.config['camera']['look_at'], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        
        aspect = self.window_size[0] / self.window_size[1]
        fov = np.radians(45.0)
        near, far = 0.1, 500.0
        
        f = 1.0 / np.tan(fov / 2.0)
        projection = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        z = cam_pos - look_at
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        view = np.array([
            [x[0], x[1], x[2], -np.dot(x, cam_pos)],
            [y[0], y[1], y[2], -np.dot(y, cam_pos)],
            [z[0], z[1], z[2], -np.dot(z, cam_pos)],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        return np.dot(projection, view)
    
    def render(self, particles, fps, element_counts, timings, integrator='Euler'):
        if glfw.window_should_close(self.window):
            self.running = False
            return
        
        positions = particles.positions.cpu().numpy().astype('f4')
        colors = self._compute_colors(particles).astype('f4')
        
        glClearColor(self.bg[0], self.bg[1], self.bg[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.shader)
        
        mvp = self._create_mvp_matrix()
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, mvp.T)
        glUniform1f(self.point_size_loc, self.point_size)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, len(positions))
        glBindVertexArray(0)
        
        element_str = ' '.join([f"{k}:{v}" for k, v in element_counts.items()])
        timing_str = f"F:{timings['force']:.1f}ms P:{timings['physics']:.1f}ms R:{timings['render']:.1f}ms"
        title = f"AE | {fps:.0f} FPS | {integrator} | {element_str} | {timing_str}"
        glfw.set_window_title(self.window, title)
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def cleanup(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo_pos])
        glDeleteBuffers(1, [self.vbo_col])
        glDeleteProgram(self.shader)
        glfw.terminate()
