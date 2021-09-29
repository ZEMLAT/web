#!/bin/bash

for file in $(find . -name "*.md")
do
   htmlFile="${file/.md/.html}"
   dest="./papers/md-nbconv/${htmlFile:2}"
   [ ! -f "$dest" ] || continue

   mkdir -p `dirname "$dest"`
   markdown "$file" > "$dest"
   # pandoc "$file" -f markdown -t html -s -o "$dest"
done
