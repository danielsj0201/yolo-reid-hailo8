#!/bin/bash

sudo ifconfig eth0 192.168.100.148 netmask 255.255.252.0 up

# Optional: wait a few seconds for system devices to settle
sleep 5

# Step 1: Navigate to Hailo example directory
cd ~/hailo-rpi5-examples || exit 1

# Step 2: Source Hailo environment
source setup_env.sh || exit 1

# Step 3: Move into virtualenv site-packages where detection_reid_pipeline.py is
cd venv_hailo_rpi5_examples/lib/python3.11/site-packages/hailo_apps_infra || exit 1

# Step 4: Run the detection + reid pipeline
python3 detection_reid_pipeline.py -i rpi
