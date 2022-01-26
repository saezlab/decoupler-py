from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="decoupler",
    version="0.0.1",
    author="Pau Badia i Mompel",
    author_email="pau.badia@uni-heidelberg.de",
    description="Ensemble of computational methods to infer biological activities from omics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saezlab/decoupler-py",
    project_urls={
        "Bug Tracker": "https://github.com/saezlab/decoupler-py/issues",
    },
    install_requires=["sklearn",
                      "tqdm",
                      "anndata",
                      "fisher",
                      "skranger",
                      "numba"
                     ],
    packages=["decoupler"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)