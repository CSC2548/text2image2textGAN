#!/bin/bash
for filename in cycleresults/cycleresults260/*.jpg; do
    python3 sample.py --gen_path=./checkpoints_old/pretrained-generator-20.pkl --image="$filename" # >> cyclecaptions260.txt
    # printf ", $filename \n" >> cyclecaptions260.txt
done


