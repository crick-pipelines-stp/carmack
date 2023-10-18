FROM  rocker/tidyverse:4.2

# Install apt packages
RUN apt-get update \
      && apt-get install -y --no-install-recommends \
      build-essential \
      libbz2-dev \
      libz-dev

## Configure default locale, see https://github.com/rocker-org/rocker/issues/19
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    && locale-gen en_US.utf8 \
    && /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

# Install apt packages for building R packages
RUN apt-get install -y --no-install-recommends \
    build-essential \
    libbz2-dev \
    libz-dev \
    && rm -rf /var/lib/apt/lists/*

# Add conda to $PATH
ENV PATH /opt/conda/bin:$PATH

# Install miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh  -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Setup R env
RUN head -n -1 /etc/rstudio/rserver.conf > /etc/rstudio/rserver.conf
RUN echo 'rsession-which-r=/opt/conda/bin/R' >> /etc/rstudio/rserver.conf

# Install base R packages
RUN mamba install -y -c conda-forge r-base=4.2.*
RUN mamba install -y -c conda-forge r-essentials=4.3
RUN mamba install -y -c conda-forge r-devtools=2.4.*
RUN mamba install -y -c conda-forge r-biocmanager=1.30.*
RUN mamba install -y -c conda-forge r-catools=1.18.*
RUN mamba install -y -c conda-forge r-doparallel=1.0.*

# Install other packages
RUN mamba install -y -c conda-forge r-seurat

# Install archr
RUN R -e 'library(BiocManager); devtools::install_github("GreenleafLab/ArchR", ref="master", repos = BiocManager::repositories())'
RUN R -e 'library(ArchR); ArchR::installExtraPackages()'
