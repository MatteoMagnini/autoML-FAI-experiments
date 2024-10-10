# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies, including SWIG, pkg-config, and HDF5 headers
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set a temporary working directory inside the container
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the Poetry files first to leverage Docker layer caching
COPY pyproject.toml poetry.lock /app/

# Install dependencies via Poetry (without creating virtual environments)
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Copy the rest of the application code to the temporary working directory
COPY . /app/

# Make sure all YAML configuration files are in the correct directory
# and run the experiments in parallel, passing output directory as /persistent
CMD find experiments/setup -name "*.yml" | \
  xargs -n 1 -P $(nproc) -I {} poetry run python -m experiments --config {} --output /persistent
