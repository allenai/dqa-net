#!/usr/bin/env bash

DQA_DATA_DIR="$HOME/data/dqa"
mkdir $DQA_DATA_DIR
wget https://s3-us-west-2.amazonaws.com/dqa-data/shining3-1500r.zip -O $DQA_DATA_DIR/shining3-1500r.zip
unzip -q $DQA_DATA_DIR/shining3-1500r.zip -d $DQA_DATA_DIR
