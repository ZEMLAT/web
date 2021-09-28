#!/bin/bash
# find . -name "*.ipynb" | xargs jupyter nbconvert {} --to script

for file in $(find . -name *.ipynb)
do
   jupyter nbconvert "$file" --to script
done
