import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader

class Renderer:
    def __init__(self, config):
        self.config = config
        self.window_size = tuple(config['window_size'])
        self.point_size = config['point_size']
        self.color_mode = config['color_mode']
        self.bg = config['background']
        
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        self.window = glfw.create_window(self.window_size[0], self.window_size[1], "AE", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")
        
        glfw.make_context_current(self.window)
        glfw.set_window_user_pointer(self.window, self)
        glfw.set_key_callback(self.window, self._key_callback)
        
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)
        
        vs = compileShader("""
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
        """, GL_VERTEX_SHADER)
        
        fs = compileShader("""
            #version 330 core
            in vec3 v_color;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color, 1.0);
            }
        """, GL_FRAGMENT_SHADER)
        
        self.shader = glCreateProgram()
        glAttachShader(self.shader, vs)
        glAttachShader(self.shader, fs)
        glLinkProgram(self.shader)
        glDeleteShader(vs)
        glDeleteShader(fs)
        
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
            elif key == glfw.KEY_C:
                modes = ["charge", "mass", "velocity", "element"]
                idx = modes.index(self.color_mode) if self.color_mode in modes else 0
                self.color_mode = modes[(idx + 1) % len(modes)]
    
    def _compute_colors(self, particles):
        if self.color_mode == "element":
            return particles.colors.cpu().numpy()
        elif self.color_mode == "velocity":
            v = particles.velocities.cpu().numpy()
            v_mag = np.linalg.norm(v, axis=1)
            v_max = v_mag.max() + 1e-8
            v_norm = v_mag / v_max
            return np.stack([v_norm, 1 - v_norm, 0.5 * np.ones_like(v_norm)], axis=1)
        elif self.color_mode == "mass":
            m = particles.masses.cpu().numpy()
            m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
            return np.stack([m_norm, 0.5 * np.ones_like(m_norm), 1 - m_norm], axis=1)
        elif self.color_mode == "charge":
            q = particles.charges.cpu().numpy()
            colors = np.zeros((len(q), 3))
            colors[q > 0] = [1, 0, 0]
            colors[q < 0] = [0, 0, 1]
            colors[q == 0] = [0.5, 0.5, 0.5]
            return colors
        return particles.colors.cpu().numpy()
    
    def _create_mvp(self):
        cam = np.array(self.config['camera']['position'], dtype=np.float32)
        look = np.array(self.config['camera']['look_at'], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        
        aspect = self.window_size[0] / self.window_size[1]
        f = 1.0 / np.tan(np.radians(45.0) / 2.0)
        
        projection = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -1.002002, -0.2002002],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        z = cam - look
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        view = np.array([
            [x[0], x[1], x[2], -np.dot(x, cam)],
            [y[0], y[1], y[2], -np.dot(y, cam)],
            [z[0], z[1], z[2], -np.dot(z, cam)],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        return np.dot(projection, view)
    
    def render(self, particles, fps, element_counts, integrator='Euler'):
        if glfw.window_should_close(self.window):
            self.running = False
            return
        
        positions = particles.positions.cpu().numpy().astype('f4')
        colors = self._compute_colors(particles).astype('f4')
        
        glClearColor(self.bg[0], self.bg[1], self.bg[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.shader)
        
        mvp = self._create_mvp()
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, mvp.T)
        glUniform1f(self.point_size_loc, self.point_size)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, len(positions))
        glBindVertexArray(0)
        
        elem_str = ' '.join([f"{k}:{v}" for k, v in element_counts.items()])
        title = f"AE | {fps:.0f} FPS | {integrator} | {elem_str}"
        glfw.set_window_title(self.window, title)
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def cleanup(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo_pos])
        glDeleteBuffers(1, [self.vbo_col])
        glDeleteProgram(self.shader)
        glfw.terminate()
