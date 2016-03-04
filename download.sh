#!/usr/bin/env bash

DQA_DATA_DIR="~/data/dqa"
wget https://s3-us-west-2.amazonaws.com/dqa-data/shining3-1500r.zip $DQA_DATA_DIR
unzip $DQA_DATA_DIR/shining3-1500r $DQA_DATA_DIR
