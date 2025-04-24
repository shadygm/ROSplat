from loguru import logger
import sys
import OpenGL.GL.shaders as shaders 
import OpenGL.GL as gl
import numpy as np
import glm
import glfw

logger.remove()
logger.add(sys.stderr, level="INFO")

_buffer_capacities = {}

def load_shaders(vs, fs):
    vertex_shader = open(vs, 'r').read()        
    fragment_shader = open(fs, 'r').read()

    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER),
    )
    return active_shader

def compile_shaders(vertex_shader, fragment_shader):
    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
    )
    return active_shader

def set_attributes(program, keys, values, vao=None, buffer_ids=None):
    gl.glUseProgram(program)
    if vao is None:
        vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    if buffer_ids is None:
        buffer_ids = [None] * len(keys)

    for i, (key, value, b) in enumerate(zip(keys, values, buffer_ids)):
        if b is None:
            b = gl.glGenBuffers(1)
            buffer_ids[i] = b
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), gl.GL_STATIC_DRAW)

        length = value.shape[-1]  # how many floats per vertex
        pos = gl.glGetAttribLocation(program, key)
        gl.glVertexAttribPointer(pos, length, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(pos)
    
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    return vao, buffer_ids

def set_faces_to_vao(vao, faces: np.ndarray):
    gl.glBindVertexArray(vao)
    element_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, gl.GL_STATIC_DRAW)
    return element_buffer

def set_storage_buffer_data(
    program,
    key,
    value: np.ndarray,
    bind_idx: int,
    buffer_id=None,
    prev_count=0,
    full_update=False
):
    """
    Upload 'value' (the *entire* flattened Gaussian data array) into a shader storage buffer.
    This function decides internally whether a "partial" or "full" update is needed.

    Parameters
    ----------
    program : int
        Compiled shader program handle.
    key : str
        The name of the SSBO in the shader.
    value : np.ndarray
        The *complete* data to upload (shape: [N, stride]).
    bind_idx : int
        The binding index for this SSBO.
    buffer_id : int, optional
        Existing buffer ID; if None, a new buffer is generated.
    prev_count : int, optional
        The previous number of items in the buffer. Defaults to 0.

    Returns
    -------
    buffer_id : int
        The updated or newly created buffer ID.
    new_count : int
        The new number of items in 'value'.
    """
    gl.glUseProgram(program)

    block_index = gl.glGetProgramResourceIndex(program, gl.GL_SHADER_STORAGE_BLOCK, key)
    gl.glShaderStorageBlockBinding(program, block_index, bind_idx)

    # number of rows in 'value' => total gaussians
    new_count = value.shape[0]
    stride_in_floats = value.shape[1]
    stride_in_bytes = stride_in_floats * 4

    # Convert data to contiguous float array
    data = np.ascontiguousarray(value, dtype=np.float32).ravel()

    # We'll figure out offset logic based on whether new_count > prev_count
    if new_count <= prev_count or prev_count == 0 or full_update:
        # FULL update from offset=0
        offset_bytes = 0
        size_needed = data.nbytes
        update_size = size_needed  # all
        logger.info("Full update: new_count <= prev_count.")
    else:
        # PARTIAL: only append the newly added gaussians
        appended_count = new_count - prev_count
        offset_bytes = prev_count * stride_in_bytes
        update_size = appended_count * stride_in_bytes
        # But the total buffer must be large enough for the new_count
        size_needed = new_count * stride_in_bytes

    if buffer_id is None:
        # Create a brand-new buffer
        buffer_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
        # Overallocate if desired, here we allocate exactly what we need
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, size_needed, None, gl.GL_STATIC_DRAW)
        _buffer_capacities[buffer_id] = size_needed

        # A brand-new buffer has no old data; write all from offset=0
        gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data)
    else:
        # Re-use or possibly re-allocate
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
        capacity = _buffer_capacities.get(buffer_id, 0)

        if size_needed <= capacity:
            # Enough capacity => either full or partial sub-update
            gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, offset_bytes, update_size, data[offset_bytes // 4:])
        else:
            # Need a bigger buffer
            new_buffer_id = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, new_buffer_id)
            
            bigger_size = size_needed * 2  # Overallocate for future
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, bigger_size, None, gl.GL_STATIC_DRAW)

            # Copy old data from old buffer to new buffer (only if old data matters)
            gl.glBindBuffer(gl.GL_COPY_READ_BUFFER, buffer_id)
            gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, new_buffer_id)
            gl.glCopyBufferSubData(gl.GL_COPY_READ_BUFFER, gl.GL_COPY_WRITE_BUFFER, 0, 0, capacity)

            # Now place the newly added portion (or do a full update if new_count < prev_count)
            gl.glBufferSubData(gl.GL_COPY_WRITE_BUFFER, offset_bytes, update_size, data[offset_bytes // 4:])

            # If new_count <= prev_count, ensure the entire buffer matches 'value'
            if new_count <= prev_count:
                gl.glBufferSubData(gl.GL_COPY_WRITE_BUFFER, 0, data.nbytes, data)

            # Cleanup the old buffer references
            del _buffer_capacities[buffer_id]
            _buffer_capacities[new_buffer_id] = bigger_size
            gl.glDeleteBuffers(1, [buffer_id])
            # force GPU to clean memory now
            gl.glFinish()
            gl.glFlush()
            gl.glMemoryBarrier(gl.GL_ALL_BARRIER_BITS)
            
            
            buffer_id = new_buffer_id
            logger.info(f"Reallocated buffer: new size = {bigger_size} bytes")

    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)

    return buffer_id, new_count

def upload_index_array(program, key, index_array, bind_idx, buffer_id=None):
    """
    Always does a full re-upload of the *entire* index_array (sorted indices).
    If more space is required, the buffer is reallocated, but old data is discarded
    (unlike the Gaussian data, which we partially preserve). This is because we
    always re-generate a new index array anyway.
    """
    gl.glUseProgram(program)
    block_index = gl.glGetProgramResourceIndex(program, gl.GL_SHADER_STORAGE_BLOCK, key)
    gl.glShaderStorageBlockBinding(program, block_index, bind_idx)

    data = np.ascontiguousarray(index_array, dtype=np.int32).ravel()
    size_needed = data.nbytes

    if buffer_id is None:
        buffer_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, size_needed, data, gl.GL_STATIC_DRAW)
        _buffer_capacities[buffer_id] = size_needed
    else:
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
        capacity = _buffer_capacities.get(buffer_id, 0)
        if size_needed <= capacity:
            # Just sub-update from offset = 0
            gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, size_needed, data)
        else:
            # Reallocate new buffer
            new_buffer_id = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, new_buffer_id)

            bigger_size = size_needed  # or size_needed*2 if you want overhead
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, bigger_size, data, gl.GL_STATIC_DRAW)

            del _buffer_capacities[buffer_id]
            _buffer_capacities[new_buffer_id] = bigger_size
            gl.glDeleteBuffers(1, [buffer_id])
            buffer_id = new_buffer_id

    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
    return buffer_id

def set_uniform_lf(shader, content, name):
    gl.glUseProgram(shader)
    gl.glUniform1i(gl.glGetUniformLocation(shader ,name), content)

def set_uniform_1int(shader, content, name):
    gl.glUseProgram(shader)
    gl.glUniform1i(gl.glGetUniformLocation(shader, name), content)

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

def set_uniform_1f(shader, content, name):
    gl.glUseProgram(shader)
    gl.glUniform1f(
        gl.glGetUniformLocation(shader, name),
        content
    )

def set_uniform_v3(shader, contents, name):
    gl.glUseProgram(shader)
    gl.glUniform3f(
        gl.glGetUniformLocation(shader, name),
        contents[0], contents[1], contents[2]
    )

def get_time():
    return glfw.get_time()
