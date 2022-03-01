from setuptools import setup
import os


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="decoupler",
    version=get_version("decoupler/__init__.py"),
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
                      "anndata"],
    packages=["decoupler"],
    python_requires=">=3.8,<3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"]
)
