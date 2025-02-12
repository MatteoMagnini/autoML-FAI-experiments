# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies, including SWIG, pkg-config, and HDF5 headers
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /home/dev/persistent/autoML-FAI-experiments

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the Poetry files first to leverage Docker layer caching
COPY pyproject.toml poetry.lock /home/dev/persistent/autoML-FAI-experiments/

# Install dependencies via Poetry (without creating virtual environments)
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --no-root

# Copy all the files from the project's root directory to the working directory
COPY . /home/dev/persistent/autoML-FAI-experiments/
