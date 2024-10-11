#!/bin/sh

# Step 1: Check if poetry is installed
if ! command -v poetry >/dev/null 2>&1; then
    echo "Poetry is not installed. Installing Poetry..."
    # Install Poetry using the official method
    curl -sSL https://install.python-poetry.org | python3 -

    # Ensure poetry is available in the current shell session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Poetry is already installed."
fi

# Step 2: Install dependencies and create the poetry environment
echo "Setting up the poetry environment..."
poetry install

# Step 3: Execute the program by finding .yml files and running the Python module
echo "Running experiments..."

find experiments/setup -name "*.yml" | \
while read filepath; do
    filename=`echo "$filepath" | sed "s:.*/::"`
    poetry run python -m experiments "$filename"
done
