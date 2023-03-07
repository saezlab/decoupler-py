from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="decoupler",
    version="1.3.5",
    author="Pau Badia i Mompel",
    author_email="pau.badia@uni-heidelberg.de",
    description="Ensemble of methods to infer biological activities from omics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saezlab/decoupler-py",
    project_urls={
        "Bug Tracker": "https://github.com/saezlab/decoupler-py/issues",
    },
    install_requires=["numba",
                      "tqdm",
                      "anndata",
                      "typing_extensions"],
    packages=["decoupler"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"]
)
