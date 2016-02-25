#!/usr/bin/env bash

python build_vocab.py $1 $2
python prepro_questions.py $1 $2
python prepro_annos.py $1 $2