FROM python:3.10

# Install APT
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        procps

# Install pip
RUN pip install --upgrade pip

# Build app folder
RUN mkdir -p /app
WORKDIR /app
ENV PATH /app:$PATH

# Copy the app folder
COPY . /app

# Install the lib
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Build work folder
RUN mkdir /home/work
WORKDIR /home/work

# Disable autorun
CMD ["/bin/bash"]