#!/bin/bash
pip install -r requirements.txt
python train.py --net unet
python train.py --net resunet
python train.py --net r2unet
python train.py --net r2attunet
python train.py --net proposed
echo "Hello World !"
