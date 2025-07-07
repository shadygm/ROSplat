from setuptools import setup, find_packages

setup(
    name="rosplat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "PyOpenGL",
        "imgui-bundle",
        "pillow",
        "loguru",
        "plyfile",
        "torch",
        "gsplat",
        "cupy-cuda11x"
    ],
    entry_points={
        "console_scripts": [
            "rosplat = rosplat.main:main",
        ],
    },
)
