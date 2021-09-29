#!/bin/bash
# find . -name "*.ipynb" | xargs jupyter nbconvert {} --to script

for file in $(find . -name *.ipynb)
do
	pdfFile="${file/.ipynb/.pdf}"

	[ ! -f "$pdfFile" ] || continue

   jupyter nbconvert "$file" --to pdf
   
   dest="./papers/pdf-nbconv/${pdfFile:2}"
   mkdir -p `dirname "$dest"`
   mv -v "$pdfFile" "$dest"
   # cp -Rv "$pdfFile" "$dest"
   # rm "$pdfFile"
done
