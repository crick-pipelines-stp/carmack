FROM python:3.10

# Install apt packages
RUN apt-get update \
    && apt-get install -y git \
    && apt-get install -y curl

# Setup user
RUN useradd -ms /bin/bash dev

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add stuff to paths
ENV PATH="/opt/conda/bin:$PATH"
ENV PATH="/home/dev/.local/bin:${PATH}"

# Install conda packages
RUN conda config --add channels defaults && \
    conda config --add channels bioconda && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict
RUN conda install --yes --name base mamba
RUN mamba install --yes --name base prettier
RUN mamba clean --all -f -y

# Final nav
USER dev
WORKDIR /home/dev
