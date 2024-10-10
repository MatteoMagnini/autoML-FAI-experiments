# Use an official Python runtime as a parent image
FROM python:3.12-slim

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

# Copy the code into the persistent directory
RUN cp -r /app/* /mnt/persistent/

# Set the persistent directory as the working directory
WORKDIR /mnt/persistent

# Make sure all YAML configuration files are in the correct directory
# and run the experiments in parallel, passing output directory as /mnt/persistent
CMD find experiments/setup -name "*.yml" | \
  xargs -n 1 -P $(nproc) -I {} poetry run python -m experiments --config {} --output /mnt/persistent
