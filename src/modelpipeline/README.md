# Model Pipeline

Model Pipeline consists of all the tools required to process data and train the model.

## Get Dataset

FaceForensics++ C23 Dataset is acquired from [Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23).

## Preprocess the data

Steps to preprocess data:
- Decode n frames of video to image
- Process Metadata
- Resize the Image and Perform some transformations

## Models

- EfficientNet-B4: EfficientNet-B4 with binary classificaiton node at the end.
- XceptionNet: XceptionNet with the binary encoding at the end.

## How to run

### Where to put your dataset

All your data should be inside data folder at the root of the project.

Directory Structure for the dataset.

```
├── data
│   ├── FaceForensics
│   └── FaceForensics_transformed
```

```bash
# First preprocess the data
python3 preprocess.py # will take about 30 min or more depending on your hardware.
python3 train.py # directly train model after that
```
