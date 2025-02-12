#!/bin/sh

echo "Running experiments..."

find experiments/setup -name "adult*.yml" | \
while read filepath; do
    filename=`echo "$filepath" | sed "s:.*/::"`
    poetry run python -m experiments "$filename"
done
