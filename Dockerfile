FROM quay.io/biocontainers/decoupler:1.5.0--pyhdfd78af_0
RUN pip install omnipath==1.0.7 pandas==2.0.3
RUN pip install --upgrade git+https://github.com/saezlab/decoupler-py
