1. Preprocessing data
Place the initial collected data in the “mnist_images” folder, divided into 0-9 files, execute the Python preprocess. py command to generate training format data, and place it in the “mnist” folder (the dataset provided in this project has been preprocessed)
2. Manual partitioning of training/testing sets
Divide the files generated in the first step into sub folders such as train, test, and val, or you can write scripts to automatically partition them
3. Start training, the weight file will be automatically saved in the saved models folder "saved_models"
Python pix2pix.py
4. Predict a single image, and the results will be saved in the results folder by default
Python predict.py
﻿
