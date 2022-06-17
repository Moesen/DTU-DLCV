#!/bin/bash
read -p "DTU student id (sxxxxxx): " s 
echo $s 
scp -r $s@login1.hpc.dtu.dk:/dtu/datasets1/02514/isic/ ./data/
