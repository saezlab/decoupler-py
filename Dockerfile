# Use a standard Python image as the base
FROM python:3.12.2-slim

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Now, git is available, so we can install packages directly from Git repositories
RUN pip install omnipath==1.0.7 pandas==2.0.3
RUN pip install --upgrade git+https://github.com/saezlab/decoupler-py

# Assuming decoupler and other packages are now installed, continue with your setup...

