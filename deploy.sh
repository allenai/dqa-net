#!/usr/bin/env bash
for i in `seq 0 7`; do
  tmux send -t $i "CUDA_VISIBLE_DEVICES=$i python -m main.m04 --config $i --train" ENTER
done
