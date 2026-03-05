# YOLO Model Training and Export Tools

This repository contains a collection of tools used for training the YOLO model and exporting it to various formats.

## Dataset

The model was trained using the [License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) from Roboflow.

You can access the dataset and download it directly from the link above. The dataset contains images and annotations for training a License Plate Recognition system.

## Prerequisites

Before running the project, make sure you have the following dependencies installed:

- [UV](https://uv.io) (for running training and inference scripts)

To install UV, you can run:

```bash
pip install uv
```

## Running the Training

Once you have downloaded the dataset and installed UV, you can start the training process by running the following command:

```bash
uv run main.py
```

This will start the training process using the configuration provided in `main.py`.

## Exporting Models

Once training is complete, the model can be exported to various formats for deployment or further analysis.

For more specific instructions on exporting models or tuning the training, refer to the [documentation](#).
