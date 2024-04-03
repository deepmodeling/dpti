#!/bin/bash

# Find all files named 'in.lammps' under directories starting with 'benchmark'
find . -type f -name 'test*.py' -path './' | while read -r file; do
    # Use sed to replace '# dump' with 'dump' in the file
    sed -i '' 's/# dump/dump/g' "$file"
done
