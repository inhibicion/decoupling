from setuptools import setup, find_packages

setup(
    name="decoupling",
    version="0.1.0",
    description="Decoupling morphological and electrophysiological variations that are depth-independent from their gradual shifts with cortical depth",
    author="Felipe Yáñez",
    author_email="felipe.yanez@mpinb.mpg.de",
    url="https://github.com/inhibicion/decoupling",
    packages=find_packages(include=["decoupling", "decoupling.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "umap-learn",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"]
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache License 2.0",
    include_package_data=True,
)
