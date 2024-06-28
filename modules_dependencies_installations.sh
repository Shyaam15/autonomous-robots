#!/bin/bash

# Update the package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y portaudio19-dev python-all-dev

# Install pip if it's not already installed
sudo apt-get install -y python-pip

# Install the specific versions of required modules
pip install pyaudio==0.2.11
pip install SpeechRecognition==3.8.1
pip install futures==1.0.0 
pip install numpy==1.16.6
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html 
pip install opencv-python==3.2.0

echo "Installation completed successfully."
