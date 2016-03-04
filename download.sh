#!/usr/bin/env bash

DQA_DATA_DIR="$HOME/data/dqa"
wget https://s3-us-west-2.amazonaws.com/dqa-data/shining3-1500r.zip -O $DQA_DATA_DIR/shining3-1500r.zip
unzip $DQA_DATA_DIR/shining3-1500r.zip -d $DQA_DATA_DIR
