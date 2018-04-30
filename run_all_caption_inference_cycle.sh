#!/bin/bash
for filename in results/*.jpg; do
    python3 sample.py --gen_path=./checkpoints/cycle_caption_gen-40.pkl --image="$filename" >> cyclecaptions140.txt
    printf ", $filename \n" >> cyclecaptions140.txt
done


