import os.path
import sys
import platform
import OpenGL.GL as GL  # type: ignore

from imgui_bundle import imgui

# Always import glfw *after* imgui_bundle
# (since imgui_bundle will set the correct path where to look for the correct version of the glfw dynamic library)
import glfw  # type: ignore

import numpy as np

VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 fragColor;

void main() {
    gl_Position = vec4(position, 1.0);
    fragColor = color;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 fragColor;
out vec4 finalColor;

void main() {
    finalColor = vec4(fragColor, 1.0);
}
"""

def glfw_error_callback(error: int, description: str) -> None:
    sys.stderr.write(f"Glfw Error {error}: {description}\n")


def create_shader_program(vertex_src, fragment_src):
    """
    Create and compile the shader program.
    """
    # Compile vertex shader
    vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vertex_shader, vertex_src)
    GL.glCompileShader(vertex_shader)
    if not GL.glGetShaderiv(vertex_shader, GL.GL_COMPILE_STATUS):
        raise RuntimeError(GL.glGetShaderInfoLog(vertex_shader).decode())

    # Compile fragment shader
    fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fragment_shader, fragment_src)
    GL.glCompileShader(fragment_shader)
    if not GL.glGetShaderiv(fragment_shader, GL.GL_COMPILE_STATUS):
        raise RuntimeError(GL.glGetShaderInfoLog(fragment_shader).decode())

    # Link shaders into a program
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(program).decode())

    # Cleanup shaders
    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)

    return program


def create_triangle_vao():
    """
    Create a VAO for a simple triangle with position and color attributes.
    """
    
    # Setup Platform/Renderer backends
    import ctypes
    
    vertices = np.array([
        # Positions        # Colors
        -0.5, -0.5, 0.0,   1.0, 0.0, 0.0,  # Bottom-left, Red
         0.5, -0.5, 0.0,   0.0, 1.0, 0.0,  # Bottom-right, Green
         0.0,  0.5, 0.0,   0.0, 0.0, 1.0   # Top-center, Blue
    ], dtype=np.float32)

    # Generate VAO and VBO
    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)

    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

    # Setup position attribute
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)

    # Setup color attribute
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    GL.glEnableVertexAttribArray(1)

    # Unbind VAO
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    return vao


def create_framebuffer(width, height):
    # Create a frame buffer object (FBO)
    fbo =GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

    # Create a texture to render to
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    # Attach the texture to the framebuffer
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, texture, 0)

    # Create a renderbuffer object for depth and stencil attachments
    rbo = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rbo)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8, width, height)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_RENDERBUFFER, rbo)

    # Check if the framebuffer is complete
    if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Framebuffer is not complete!")

    # Unbind the framebuffer
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    return fbo, texture


def main() -> None:
    # Setup window
    glfw.set_error_callback(glfw_error_callback)
    if not glfw.init():
        sys.exit(1)

    # Decide GL+GLSL versions
    if platform.system() == "Darwin":
        glsl_version = "#version 150"
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  # // 3.2+ only
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE) 
    else:
        # GL 3.0 + GLSL 130
        glsl_version = "#version 130"
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE) # // 3.2+ only
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

    # Create window with graphics context
    window = glfw.create_window(
        1280, 720, "Dear ImGui GLFW+OpenGL3 example", None, None
    )
    if window is None:
        sys.exit(1)
        
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # // Enable vsync

    # Setup Dear ImGui context
    # IMGUI_CHECKVERSION();
    imgui.create_context()
    io = imgui.get_io()
    io.config_flags |= (
        imgui.ConfigFlags_.nav_enable_keyboard.value
    )  # Enable Keyboard Controls
    
    # io.config_flags |= imgui.ConfigFlags_.nav_enable_gamepad # Enable Gamepad Controls
    io.config_flags |= imgui.ConfigFlags_.docking_enable    # Enable docking
    io.config_flags |= imgui.ConfigFlags_.viewports_enable # Enable Multi-Viewport / Platform Windows

    # Setup Dear ImGui style
    imgui.style_colors_dark()
    # imgui.style_colors_classic()

    # When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    style = imgui.get_style()
    if io.config_flags & imgui.ConfigFlags_.viewports_enable.value:
        style.window_rounding = 0.0
        window_bg_color = style.color_(imgui.Col_.window_bg.value)
        window_bg_color.w = 1.0
        style.set_color_(imgui.Col_.window_bg.value, window_bg_color)

    # Setup Platform/Renderer backends
    import ctypes

    # You need to transfer the window address to imgui.backends.glfw_init_for_opengl
    # proceed as shown below to get it.
    window_address = ctypes.cast(window, ctypes.c_void_p).value
    assert window_address is not None
    imgui.backends.glfw_init_for_opengl(window_address, True)

    imgui.backends.opengl3_init(glsl_version)

    # Our state
    show_demo_window: bool | None = True
    show_another_window = False
    clear_color = [0.45, 0.55, 0.60, 1.00]
    f = 0.0
    counter = 0

    # Create a framebuffer object
    shader_program = create_shader_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
    vao = create_triangle_vao()
    viewport_width, viewport_height = 800, 600
    fbo, fbo_texture = create_framebuffer(viewport_width, viewport_height)
    
    # Main loop
    while not glfw.window_should_close(window):

        glfw.poll_events()

        # Start the Dear ImGui frame
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()
                 
        # Try 2: SOLUTION
        imgui.dock_space_over_viewport(dockspace_id=0, 
                                       viewport = imgui.get_main_viewport())
                 
        # 1. Show the big demo window (Most of the sample code is in imgui.ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if show_demo_window:
            _show_demo_window = imgui.show_demo_window(show_demo_window)

        # 3. Show another simple window.
        def gui_another_window() -> None:
            nonlocal show_another_window
            if show_another_window:
                imgui.begin(
                    "Another Window", show_another_window
                )  # Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
                imgui.text("Hello from another window!")
                if imgui.button("Close Me"):
                    show_another_window = False
                imgui.end()

        gui_another_window()

        # Render OpenGL content to the framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, viewport_width, viewport_height)
        GL.glClearColor(0.1, 0.2, 0.3, 1.0)  # Set a clear color
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        # render_scene()  # Render the scene
        # Render to the framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, viewport_width, viewport_height)
        GL.glClearColor(0.1, 0.2, 0.3, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(shader_program)
        GL.glBindVertexArray(vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        
        # OpenGL rendering commands go here (e.g., render a cube, scene, etc.)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # ImGui window with the OpenGL viewport
        imgui.begin("Viewport")
        imgui.text("OpenGL Viewport:")
        imgui.image(fbo_texture, (viewport_width, viewport_height))  # Display the FBO texture in ImGui
        imgui.end()

        # Rendering
        imgui.render()
        display_w, display_h = glfw.get_framebuffer_size(window)
        GL.glViewport(0, 0, display_w, display_h)
        GL.glClearColor(
            clear_color[0] * clear_color[3],
            clear_color[1] * clear_color[3],
            clear_color[2] * clear_color[3],
            clear_color[3],
        )
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

        # Update and Render additional Platform Windows
        # (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        #  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        if io.config_flags & imgui.ConfigFlags_.viewports_enable.value > 0:
            backup_current_context = glfw.get_current_context()
            imgui.update_platform_windows()
            imgui.render_platform_windows_default()
            glfw.make_context_current(backup_current_context)
        
        glfw.swap_buffers(window)

    # Cleanup
    imgui.backends.opengl3_shutdown()
    imgui.backends.glfw_shutdown()
    imgui.destroy_context()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
