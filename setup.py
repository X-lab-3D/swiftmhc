from glob import glob

from setuptools import find_namespace_packages, setup

setup(
    name="swiftmhc",
    description="A deep learning algorithm for predicting MHC-p structure and binding affinity at the same time",
    version="1.0.0",
    packages=find_namespace_packages(include=["swiftmhc*"]),
    scripts=glob("scripts/swiftmhc_*"),
    python_requires=">=3.11",
    install_requires=[
        "pandas>=1.5.3",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.11.0,<1.15.0",
        "h5py>=3.10.0",
        "ml-collections>=0.1.1",
        "scikit-learn>=1.4.1",
        "blosum>=2.0.3",
        "modelcif>=1.0",
        "filelock>=3.13.1",
        "biopython>=1.84",
        "openmm[cuda12]>=8.1.1",
        "dm-tree>=0.1.8",
        "position-encoding",
    ],
)
