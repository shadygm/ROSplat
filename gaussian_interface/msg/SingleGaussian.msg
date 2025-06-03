# Gaussian.msg

# 3D Pose
float32[3] xyz

# Rotation as Quaternion
float32[4] rotation

# Scale along each axis
float32[3] scale

# Opacity
# Quantized as a number between 0-255 instead of 0-1
uint8 opacity

# Spherical Harmonics
float32[] spherical_harmonics