FROM continuumio/miniconda3

COPY environment.yml .
RUN apt-get update -qq && apt-get install -y \
    build-essential 

RUN conda env create -f environment.yml
ENV PATH="/opt/conda/envs/backsub/bin:$PATH"

RUN pip install --upgrade git+https://github.com/saezlab/decoupler-py

