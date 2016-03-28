#!/usr/bin/env bash

DATA_DIR="$HOME/data"
DQA_DATA_DIR="$DATA_DIR/dqa"
GLOVE_DATA_DIR="$DATA_DIR/glove"

# DQA data download
mkdir $DATA_DIR
mkdir $DQA_DATA_DIR
wget https://s3-us-west-2.amazonaws.com/dqa-data/shining3-1500r.zip -O $DQA_DATA_DIR/shining3-1500r.zip
unzip -q $DQA_DATA_DIR/shining3-1500r.zip -d $DQA_DATA_DIR

# Glove vector download
mkdir $GLOVE_DATA_DIR
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DATA_DIR/glove.6B.zip
unzip -q $GLOVE_DATA_DIR/glove.6B.zip -d $GLOVE_DATA_DIR


# Caffe models download
