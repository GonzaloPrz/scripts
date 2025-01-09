#!/bin/bash
rsync -avz -e "ssh" "CNC Audio"@neuro-srv-1.udesa.edu.ar:'wsl rsync -avz /mnt/d/CNC_Audio/gonza/results/' /Users/gp/results/