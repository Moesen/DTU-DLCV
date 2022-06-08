#!/bin/bash
read -p "DTU student id (sxxxxxx): " s 
echo $s 
echo $p  
echo Running `scp $s@login1.hpc.dtu.dk:/dtu/datasets1/02514/data_wastedetection $p`
scp $s@login1.hpc.dtu.dk:/dtu/datasets1/02514/data_wastedetection ./data/

