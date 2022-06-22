#!/bin/zsh

file="./links.txt"

while read -r line; do
    firefox $line
done <$file
