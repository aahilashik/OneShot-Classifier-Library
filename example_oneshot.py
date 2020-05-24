from  oneshot import oneShot
import numpy as np
import cv2
import os

# Initialize the Model
comparator =  oneShot()

# Location of Data and Samples
dataset_path = "Z:/Python/Attendance System/new_dataset"
samples_path = "Z:/Python/Attendance System/new_samples"

# Data Loading
X, Y = [], []
for label in os.listdir(dataset_path):
    try    : os.mkdir(os.path.join(samples_path, label))
    except : pass
    i = 0
    for image in os.listdir(os.path.join(dataset_path, label)):
        if i < 2 and len(os.listdir(os.path.join(samples_path, label))) < 2: 
            cv2.imwrite(os.path.join(samples_path, label, image),
                        cv2.resize(cv2.imread(os.path.join(dataset_path, label, image)), (160,160)))
            i+=1
        X.append(np.array(cv2.resize(cv2.imread(os.path.join(dataset_path, label, image)), (160,160))))
        Y.append(label)
X = np.array(X)
Y = np.array(Y)

# Train the Oneshot Model
comparator.trainModel(X, Y, batch_size=12, epochs=5, validation_split=0.3, forEach=4)

# Predict the label/folder_name of the Image
name = comparator.detectFace(X[100], samples_path)

# Return the confidence level of each label
confidences, names = comparator.findConfidence(comparator.facenet.predict(np.array([X[10]])), samples_path)