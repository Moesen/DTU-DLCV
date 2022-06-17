#!/bin/bash
read -p "DTU student id (sxxxxxx): " s 
echo $s 
echo $p  
echo Running `scp $s@login1.hpc.dtu.dk:/dtu/datasets1/02514/hotdog_nothotdog.zip $p`
scp -r $s@login1.hpc.dtu.dk:/dtu/datasets1/02514/isic/ ./data/

