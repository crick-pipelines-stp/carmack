FROM python:3.12
LABEL authors="chris.cheshire@crick.ac.uk"

# Update pip to latest version
RUN python -m pip install --upgrade pip

# Add thesource files to the image
COPY . /usr/src/carmack
WORKDIR /usr/src/carmack

# Update version
RUN pip install toml
RUN python update_version.py

# Install program
RUN pip install .

CMD ["bash"]
