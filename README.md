
<div class="container">
  <header>
    <h1>ROSplat</h1>
    <p><em>The Online ROS2-Based Gaussian Splatting-Enabled Visualizer</em></p>
    <p>
      <a href="https://www.linkedin.com/in/shady-gmira-ba678121a/" target="_blank">Shady Gmira</a>
    </p>
    <img src="https://github.com/shadygm/ROSplat/blob/main/assets/images/image.png" alt="Project Image">
    <img src="https://github.com/shadygm/ROSplat/raw/main/assets/gifs/output.gif" alt="Demo Animation">
  </header>

  <section>
    <h2>Overview</h2>
    <p>
      ROSplat is the first online ROS2-based visualizer that leverages Gaussian splatting to render complex 3D scenes. It is designed to efficiently visualize millions of gaussians by using custom ROS2 messages and GPU-accelerated sorting and rendering techniques. ROSplat also supports data loading from PLY files and integrates with ROS2 tools such as bag recording.
    </p>
  </section>

  <section>
    <h2>Features</h2>
    <ul>
      <li><strong>Real-Time Visualization:</strong> Render millions of Gaussian “splats” in real time.</li>
      <li><strong>ROS2 Integration:</strong> Built on ROS2 for online data exchange of Gaussians, Images, and IMU data.</li>
      <li><strong>Custom Gaussian Messages:</strong> Uses custom message types (<em>SingleGaussian</em> and <em>GaussianArray</em>) to encapsulate properties such as position, rotation, scale, opacity, and spherical harmonics.</li>
      <li><strong>GPU-Accelerated Sorting &amp; Rendering:</strong> Offloads sorting and rendering tasks to the GPU.</li>
    </ul>
  </section>

  <section>
    <h2>Setup</h2>
    <p>This project has been developed and tested on <strong>Ubuntu 24.04 LTS</strong> using <strong>ROS2 Jazzy</strong>. It is important to note that this project performs significantly worse without a NVIDIA graphics card.</p>
    <h3>Dependencies</h3>
    <ul>
      <li><strong>Mandatory:</strong> ROS2 (tested on ROS2 Jazzy)</li>
      <li>
        <strong>Optional (for GPU-based Sorting):</strong>
        <ul>
          <li><code>cupy</code> (or a variant matching your CUDA version)</li>
          <li><code>torch</code> (if using PyTorch for GPU sorting)</li>
        </ul>
      </li>
    </ul>
    <p>To install these dependencies, run:</p>
    <pre><code>pip install -r requirements.txt # With GPU acceleration</code></pre>
    <pre><code>pip install -r requirements-no-gpu.txt # Without GPU acceleration</code></pre>

  <h3>Docker-Based Setup</h3>
    <p>If you prefer to use Docker, the project includes a setup script under the <code>docker</code> directory.</p>
    <p>Before running docker, make sure to install the following:</p>
    <pre><code>sudo apt-get install -y nvidia-container-toolkit</code></pre>
    <p>This is needed to enable intercommunication between the docker container and your host Nvidia GPU.</p>
    <pre><code>cd docker
./run_docker.sh -h    # To display help and usage instructions
./run_docker.sh -bu   # To build the Docker image and launch the container using docker-compose</code></pre>
    <p>This method ensures a consistent environment and avoids dependency issues. It is important that the version of CUDA on the host machine matches with what the docker image is built with. If you are using anything but CUDA 12.6, make sure to change the Dockerfile to use the same as the host version.</p>
    
  </section>



  <section>
    <h2>Building the Gaussian Messages</h2>
    <p>
      ROSplat defines two custom ROS2 messages to handle Gaussian data, located in the <code>gaussian_interface/msg</code> folder.
    </p>
    <blockquote>
      <strong>Note:</strong> The Gaussian messages are based on the <a href="https://github.com/graphdeco-inria/gaussian-splatting" target="_blank">original Gaussian Splatting implementation by graphdeco-inria</a>.
    </blockquote>
    <h3>Message Definitions</h3>
    <h4>1. SingleGaussian.msg</h4>
    <pre><code>geometry_msgs/msg/Point xyz
geometry_msgs/msg/Quaternion rotation
float32 opacity
geometry_msgs/msg/Vector3 scale
float32[] spherical_harmonics</code></pre>
    <h4>2. GaussianArray.msg</h4>
    <pre><code>gaussian_interface/SingleGaussian[] gaussians</code></pre>
    <h3>Building the Messages</h3>
    <p><strong>a) Build your workspace using colcon:</strong></p>
    <pre><code>colcon build --packages-select gaussian_interface</code></pre>
    <p><strong>b) Source your workspace:</strong></p>
    <pre><code>. install/setup.bash</code></pre>
    <blockquote>
      <strong>Important:</strong> Depending on your shell, you might need to adjust these commands.
    </blockquote>
  </section>

  <section>
    <h2>Usage</h2>
    <p>Once the Gaussian messages are built, you can launch the visualizer from the project's root directory:</p>
    <pre><code>cd src
python main.py</code></pre>
  </section>

  <section>
    <h2>Contributions</h2>
    <p>Contributions and feedback are welcome!</p>
  </section>

<section>
  <h2>Acknowledgments</h2>
  <p>
    I'm glad to have worked on such a challenging topic and grateful for the invaluable advice and support I received throughout this project.
  </p>
  <p>
    Special thanks to 
    <a href="https://scholar.google.com/citations?user=14GwKcMAAAAJ&amp;hl=en" target="_blank" rel="noopener noreferrer">
      Qihao Yuan
    </a> 
    and 
    <a href="https://kailaili.github.io/" target="_blank" rel="noopener noreferrer">
      Kailai Li
    </a> 
    for their guidance and encouragement as well as the constructive feedback that helped shape this work.
  </p>
  <p>
    This project was additionally influenced by 
    <a href="https://github.com/limacv" target="_blank" rel="noopener noreferrer">
      limacv
    </a>'s implementation of the  
    <a href="https://github.com/limacv/GaussianSplattingViewer" target="_blank" rel="noopener noreferrer">
      GaussianSplattingViewer
    </a> 
    repository.
  </p>
</section>


  <section>
    <h2>Contact</h2>
    <p>For questions or further information, please email: <strong>shady.gmira[at]gmail.com</strong></p>
  </section>
</div>
</body>
</html>
