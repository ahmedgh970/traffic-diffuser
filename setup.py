from setuptools import setup

requirements = [
    "torch",
    "torchvision",
    "diffusers[torch]",
    "accelerate",
    "timm",
    "Pillow",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "PyYAML",
    "einops",
    "fvcore",
    "scikit-image",
    "calflops",
]

setup(
    name="traffsim",
    version="0.0.0",
    install_requires=requirements
)
