from setuptools import setup


requirements = [        
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "tqdm",
    "scikit-learn",
    "Pillow",
    "einops",
    "PyYAML",
    "timm",
    "diffusers[torch]",
    "accelerate",
    "fvcore",
]


setup(
    name="traffsim",
    version="0.0.0",
    install_requires=requirements,
)