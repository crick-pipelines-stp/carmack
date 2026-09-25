# syntax=docker/dockerfile:1

########################################################################
# base: the interpreter and the dependencies that cannot come from PyPI
########################################################################
FROM mambaorg/micromamba:2.9.0-debian13 AS base

# The compiled scientific stack is taken from conda so that editing the source never
# rebuilds it, and pigz comes from the same place: it is what keeps compression off the
# critical path of every stage that writes .gz, and without it those stages fall back to
# single-threaded gzip. Nothing here is now uninstallable by pip alone -- the dependency
# that forced a conda base, umi_tools, is gone -- so this is a build-time choice rather
# than a requirement, and the image could move to python:3.12 plus an apt pigz if the
# rebuild cost were ever worth trading away. procps is there for Nextflow, not carmack:
# the wrapper a Nextflow task runs in exits 1 before the command starts when `ps` is
# missing, because it uses it to collect task metrics.
RUN micromamba install -y -n base -c conda-forge -c bioconda \
        python=3.12 \
        pip \
        pigz \
        procps-ng \
        pysam \
        numpy \
        scipy \
        matplotlib-base \
    && micromamba clean --all --yes

# This base image puts /opt/conda/bin on PATH only via its entrypoint's activation
# step, so the image's declared PATH has no python at all. Setting it explicitly makes
# every binary resolve in later RUN layers and, more importantly, at runtime when the
# entrypoint is bypassed -- which is how Nextflow invokes a container.
ENV PATH=/opt/conda/bin:$PATH

USER root
RUN mkdir -p /opt/carmack && chown $MAMBA_USER:$MAMBA_USER /opt/carmack
USER $MAMBA_USER
WORKDIR /opt/carmack


########################################################################
# installed: the package and its dependencies, with the source still present
########################################################################
FROM base AS installed

LABEL org.opencontainers.image.title="carmack" \
      org.opencontainers.image.description="Single-cell multiomic tools" \
      org.opencontainers.image.source="https://github.com/crick-pipelines-stp/carmack"

# Supplied by the caller, and must be a PEP 440 version -- pip rejects anything else,
# so a bare commit sha will not do:
#   docker build --build-arg CARMACK_VERSION="$(git describe --tags --match 'v*' | sed 's/^v//;s/-\([0-9]*\)-g.*/.\1/')" .
# Passing it in keeps .git out of the build context. Rewriting the one line leaves the
# rest of pyproject.toml intact, which a toml load/dump round-trip does not: that
# discards every comment in the file and reflows the inline tables.
ARG CARMACK_VERSION=0.0.0

COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md LICENSE ./
COPY --chown=$MAMBA_USER:$MAMBA_USER carmack ./carmack

# pip resolves only the pure-python remainder here: conda has already satisfied
# pysam, numpy, scipy and matplotlib.
RUN sed -i "s/^version = .*/version = \"${CARMACK_VERSION}\"/" pyproject.toml \
    && pip install --no-cache-dir . \
    && rm -rf build carmack.egg-info \
    && carmack --help > /dev/null \
    && python -c "from carmack.io.gzip_file import resolve_gzip_write_command as r; assert r()[0] == 'pigz', r()"


########################################################################
# test: the suite, resolved against the [tests] extra in pyproject.toml
########################################################################
FROM installed AS test

COPY --chown=$MAMBA_USER:$MAMBA_USER conftest.py pytest.ini ./
COPY --chown=$MAMBA_USER:$MAMBA_USER tests ./tests

RUN pip install --no-cache-dir ".[tests]"

CMD ["pytest"]


########################################################################
# runtime: the pipeline itself, and the default build target
########################################################################
FROM installed AS runtime

# Dropping the source tree stops carmack/ next to WORKDIR shadowing the installed
# package for anything run from this directory, which is the failure
# docs/development.md warns about. It has to happen in a leaf stage: the test stage
# resolves ".[tests]" from this same directory, and against a stripped one setuptools
# builds an empty carmack wheel that pip then installs over the real one.
RUN rm -rf carmack \
    && carmack --help > /dev/null \
    && python -c "import carmack, pathlib; assert 'site-packages' in carmack.__file__, carmack.__file__" \
    && ps --version > /dev/null

CMD ["bash"]
