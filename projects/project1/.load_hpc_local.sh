#!/bin/bash
read -p "DTU student id (sxxxxxx): " s 
read -p "local_folder for storing file: " p 
echo $s 
echo $p  
echo Running `scp $s@login1.hpc.dtu.dk:/dtu/datasets1/02514/hotdog_nothotdog.zip $p`
scp $s@login1.hpc.dtu.dk:/dtu/datasets1/02514/hotdog_nothotdog.zip $p

