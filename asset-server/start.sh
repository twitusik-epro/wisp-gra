#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagleai-photos
cd /opt/gry/Wisp/asset-server
python3 server.py
