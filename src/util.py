from loguru import logger
import sys
import OpenGL.GL.shaders as shaders
import OpenGL.GL as gl
import numpy as np
import glm
import glfw

# Logger setup
logger.remove()
logger.add(sys.stderr, level="INFO")

# Global buffer capacities
_buffer_capacities = {}

# Shader-related functions
def load_shaders(vs, fs):
    """
    Load and compile shaders from file paths.
    """
    with open(vs, 'r') as vs_file, open(fs, 'r') as fs_file:
        vertex_shader = vs_file.read()
        fragment_shader = fs_file.read()

    return compile_shaders(vertex_shader, fragment_shader)

def compile_shaders(vertex_shader, fragment_shader):
    """
    Compile vertex and fragment shaders into a program.
    """
    return shaders.compileProgram(
        shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
    )

# Attribute and buffer setup functions
def set_attributes(program, keys, values, vao=None, buffer_ids=None):
    """
    Set vertex attributes for a shader program.
    """
    gl.glUseProgram(program)
    vao = vao or gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    buffer_ids = buffer_ids or [None] * len(keys)

    for i, (key, value, b) in enumerate(zip(keys, values, buffer_ids)):
        b = b or gl.glGenBuffers(1)
        buffer_ids[i] = b
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), gl.GL_STATIC_DRAW)

        length = value.shape[-1]
        pos = gl.glGetAttribLocation(program, key)
        gl.glVertexAttribPointer(pos, length, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(pos)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    return vao, buffer_ids

def set_faces_to_vao(vao, faces: np.ndarray):
    """
    Bind face indices to a VAO.
    """
    gl.glBindVertexArray(vao)
    element_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, gl.GL_STATIC_DRAW)
    return element_buffer

# Shader storage buffer functions
def set_storage_buffer_data(program, key, value, bind_idx, buffer_id=None, prev_count=0, full_update=False):
    """
    Upload data to a shader storage buffer, with optional partial updates.
    """
    gl.glUseProgram(program)
    block_index = gl.glGetProgramResourceIndex(program, gl.GL_SHADER_STORAGE_BLOCK, key)
    gl.glShaderStorageBlockBinding(program, block_index, bind_idx)

    new_count = value.shape[0]
    stride_in_bytes = value.shape[1] * 4
    data = np.ascontiguousarray(value, dtype=np.float32).ravel()

    if new_count <= prev_count or prev_count == 0 or full_update:
        offset_bytes, update_size = 0, data.nbytes
        size_needed = update_size
    else:
        offset_bytes = prev_count * stride_in_bytes
        update_size = (new_count - prev_count) * stride_in_bytes
        size_needed = new_count * stride_in_bytes

    if buffer_id is None:
        buffer_id = _create_new_buffer(size_needed, data)
    else:
        buffer_id = _update_or_reallocate_buffer(buffer_id, size_needed, offset_bytes, update_size, data)

    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
    return buffer_id, new_count

def upload_index_array(program, key, index_array, bind_idx, buffer_id=None):
    """
    Upload an index array to a shader storage buffer.
    """
    gl.glUseProgram(program)
    block_index = gl.glGetProgramResourceIndex(program, gl.GL_SHADER_STORAGE_BLOCK, key)
    gl.glShaderStorageBlockBinding(program, block_index, bind_idx)

    data = np.ascontiguousarray(index_array, dtype=np.int32).ravel()
    size_needed = data.nbytes

    if buffer_id is None:
        buffer_id = _create_new_buffer(size_needed, data)
    else:
        buffer_id = _update_or_reallocate_buffer(buffer_id, size_needed, 0, size_needed, data)

    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
    return buffer_id

def _create_new_buffer(size_needed, data):
    """
    Create a new buffer and upload data.
    """
    buffer_id = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
    gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, size_needed, data, gl.GL_STATIC_DRAW)
    _buffer_capacities[buffer_id] = size_needed
    return buffer_id

def _update_or_reallocate_buffer(buffer_id, size_needed, offset_bytes, update_size, data):
    """
    Update or reallocate a buffer if needed.
    """
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
    capacity = _buffer_capacities.get(buffer_id, 0)

    if size_needed <= capacity:
        gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, offset_bytes, update_size, data[offset_bytes // 4:])
    else:
        new_buffer_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, new_buffer_id)
        bigger_size = size_needed * 2
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, bigger_size, None, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_COPY_READ_BUFFER, buffer_id)
        gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, new_buffer_id)
        gl.glCopyBufferSubData(gl.GL_COPY_READ_BUFFER, gl.GL_COPY_WRITE_BUFFER, 0, 0, capacity)

        gl.glBufferSubData(gl.GL_COPY_WRITE_BUFFER, offset_bytes, update_size, data[offset_bytes // 4:])
        del _buffer_capacities[buffer_id]
        _buffer_capacities[new_buffer_id] = bigger_size
        gl.glDeleteBuffers(1, [buffer_id])
        buffer_id = new_buffer_id

    return buffer_id

# Uniform setters
def set_uniform_lf(shader, content, name):
    _set_uniform(shader, name, gl.glUniform1i, content)

def set_uniform_1int(shader, content, name):
    _set_uniform(shader, name, gl.glUniform1i, content)

def set_uniform_1f(shader, content, name):
    _set_uniform(shader, name, gl.glUniform1f, content)

def set_uniform_v3(shader, contents, name):
    gl.glUseProgram(shader)
    gl.glUniform3f(gl.glGetUniformLocation(shader, name), *contents)

def set_uniform_mat4(shader, content, name):
    gl.glUseProgram(shader)
    if isinstance(content, glm.mat4):
        content = np.array(content).astype(np.float32)
    else:
        content = content.T
    gl.glUniformMatrix4fv(
        gl.glGetUniformLocation(shader, name),
        1,
        gl.GL_FALSE,
        content.astype(np.float32)
    )

def _set_uniform(shader, name, func, *args):
    """
    Helper function to set a uniform value.
    """
    gl.glUseProgram(shader)
    func(gl.glGetUniformLocation(shader, name), *args)

# Utility functions
def get_time():
    """
    Get the current time from GLFW.
    """
    return glfw.get_time()
