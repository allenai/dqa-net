#!/usr/bin/env bash

#PROJ_HOME="~/csehomedir/projects/dqa-net"
DATA_DIR="$HOME/data"
DQA_DATA_DIR="$DATA_DIR/dqa"

MODELS_DIR="$HOME/models"
GLOVE_DIR="$MODELS_DIR/glove"
#VGG_DIR="$MODELS_DIR/vgg"

PREPRO_DIR="data"
DQA_PREPRO_DIR="$PREPRO_DIR/s3"

# DQA data download
#if [ ! -d "$DATA_DIR" ]; then
#				echo "making dir $DATA_DIR"
#				mkdir -p "$DATA_DIR"
#fi
#if [ ! -d "$DQA_DATA_DIR" ]; then
#				echo "making dir $DQA_DATA_DIR"
#				mkdir -p "$DQA_DATA_DIR"
#fi
#wget https://s3-us-west-2.amazonaws.com/dqa-data/shining3.zip -O $DQA_DATA_DIR/shining3.zip
#unzip -q $DQA_DATA_DIR/shining3.zip -d $DQA_DATA_DIR

# Glove pre-trained vectors download
#if [ ! -d "$MODELS_DIR" ]; then
#				echo "making dir $MODELS_DIR"
#				mkdir -p "$MODELS_DIR"
#fi
#if [ ! -d "$GLOVE_DIR" ]; then
#				echo "making dir $GLOVE_DIR"
#				mkdir -p "$GLOVE_DIR"
#fi
#wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
#unzip -q $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# VGG-19 models download
#mkdir $VGG_DIR
#wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel -O $VGG_DIR/vgg-19.caffemodel
#wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt -O $VGG_DIR/vgg-19.prototxt

# folds download
if [ ! -d "$PREPRO_DIR" ]; then
				echo "making dir $PREPRO_DIR"
				mkdir -p "$PREPRO_DIR"
fi
if [ ! -d "$DQA_PREPRO_DIR" ]; then
				echo "making dir $DQA_PREPRO_DIR"
				mkdir -p "$DQA_PREPRO_DIR"
fi
wget https://s3-us-west-2.amazonaws.com/dqa-data/shining3-folds.zip -O $DQA_PREPRO_DIR/shining3-folds.zip
unzip -q $DQA_PREPRO_DIR/shining3-folds.zip -d $DQA_PREPRO_DIR
