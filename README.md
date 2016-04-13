# DQA-Net
This is the original source code for DQA-Net described in ["A Diagram is Worth a Dozen Images"] (http://arxiv.org/abs/1603.07396)

## Quick Start

### 1. Install Requirements
- Python (verified on 3.5.1)
- Python packages: numpy, progressbar2, nltk, tensorflow, h5py
- Lua
- Torch
- Lua packages: cunn, cudnn, cutorch, loadcaffe, hdf5

Note that the most recent official release of tensorflow (0.7.1) probably won't be compatible with this. 
You will need to build from a recent commit (verified on [e39d8fe](https://github.com/tensorflow/tensorflow/tree/e39d8feebb9666a331345cd8d960f5ade4652bba)).
DQA-Net does not use images (VQA baseline does), so you can skip Lua/Torch if you just want to run DQA-Net. See details below.

### 2. Download Data and Models
At the root folder, run:
```bash
chmod +x download.sh
./download.sh
``` 
to download DQA data, folds, Glove vectors and VGG-19 model. 
VGG-19 model is used for images, and as mentioned above, DQA-Net does not use images, so you can comment this line out if you only run DQA-Net.

### 3. Preprocess Data
Run:
```bash
python -m prepro.p05
``` 
to preprocess data. 
You can just use default directories unless you make changes to download directories in `download.sh`.

If you wish to skip image preprocessing (in case you only run DQA-Net), Run with an additional flag:
```bash
python -m prepro.p05 --prepro_images False
```
Now you will see all preprocessed json and h5 files in `data/s3` folder inside the source code's root folder.

### 4. Train and Test
To train the default model, run:
```bash
python -m main.x05 --train
```

To test the trained model on test data, run:
```bash
python -m main.x05
```

To launch tensorboard, run:
```bash
tensorboard --logdir logs/m05/None --port 6006
```
Here, `m05` is the model name, and `None` is the default configuration. All tensorboard logs will be stored in `logs/` folder. 

To visualize the attention, run:
```bash
python -m vis.list_results 5 None train 1 --port 8000 --open True
```
Here, `5` is the model number (`m05`), `None` is the default configuration, `train` indicates data type, and `1` is the epoch from which the result will be fetches.
See `evals/m05/None` folder to see possible epochs (result saving frequency can be controlled by "save_period" flag at `main/m05.py`).
After running the script, the script hosts html server at the specified port. 
`--open True` flag opens web browser at this address.

In general, use `-h` flag for the run files to see what kind options there are.
