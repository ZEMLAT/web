#!/bin/bash
# find . -name "*.ipynb" | xargs jupyter nbconvert {} --to script

for file in $(find . -name *.ipynb)
do
   jupyter nbconvert "$file" --to pdf
   pdfFile="${file/.ipynb/.pdf}"
   dest="./papers/${pdfFile:2}"
   mv -Rv "$pdfFile" "$dest"
done
