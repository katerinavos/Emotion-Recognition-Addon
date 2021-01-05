from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC, SVC
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(8, 2)
data = []
labels = []

project_dir = os.path.abspath(os.path.join(__file__, os.pardir))
dataset_dir = os.path.join(project_dir, 'dataset')
trainset_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

haar_classifier = cv2.CascadeClassifier(os.path.join(project_dir, 'haar/haarcascade_frontalface_alt.xml'))

l = 0
i = 0

# loop over the training images
for root, dirs, files in os.walk(trainset_dir):
    for file in files:

        image_path = os.path.join(root, file)
        class_label = os.path.basename(root)
        image_array = plt.imread(image_path)

        '''START FACE DETECTION'''
        face = haar_classifier.detectMultiScale(image_array, 1.05, 6)
        for x, y, w, h in face:
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = image_array[y:y + h, x: x + w]
            cropped_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            # print("face roi", face_roi.shape, type(face_roi))
            hist = desc.describe(cropped_face)
            labels.append(class_label)
            data.append(hist)
            i += 1
        l += 1
        '''END FACE DETECTION'''

        # hist = desc.describe(image_array)

print("Faces detected in trainset:",i, "out of", l)

# open a binary file in write mode
with open("training_data.csv", "wb") as file:
    # save array to the file
    np.save(file, data)
    # close the file

with open("training_labels.csv", "wb") as file:
    # save array to the file
    np.save(file, labels)
    # close the file