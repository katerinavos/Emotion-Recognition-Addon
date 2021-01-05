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

project_dir = os.path.abspath(os.path.join(__file__, os.pardir))
dataset_dir = os.path.join(project_dir, 'dataset')
trainset_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

haar_classifier = cv2.CascadeClassifier(os.path.join(project_dir, 'haar/haarcascade_frontalface_alt.xml'))


k = 0
j = 0

with open("training_data.csv", "rb") as file:
	# read the file to numpy array
	data = np.load(file)

with open("training_labels.csv", "rb") as file:
	# read the file to numpy array
	labels = np.load(file)

# train a Linear SVM on the data
model = SVC(C=100.0, kernel='rbf', random_state=42, gamma='scale')
model.fit(data, labels)

vlabels = []

# loop over the validation images
for root, dirs, files in os.walk(validation_dir):
	for file in files:
		if file.endswith('jpg'):
			image_path = os.path.join(root, file)
			class_label = os.path.basename(root)
			image_array = plt.imread(image_path)
			vlabels.append(os.path.basename(root))


			'''START FACE DETECTION'''
			face = haar_classifier.detectMultiScale(image_array, 1.05, 6)
			for x, y, w, h in face:
				# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
				face_roi = image_array[y:y + h, x: x + w]
				cropped_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
			# print("face roi", face_roi.shape, type(face_roi))
				hist = desc.describe(cropped_face)
				#hist = desc.describe(image_array)
				prediction = model.predict(hist.reshape(1, -1))

				if class_label == prediction[0]:
					# if k > 50 and k < 70:
					# 	print(class_label, prediction[0], k)
					j += 1
				# if k == 136 or k == 1082:
				# 	print(class_label, prediction[0])
				# 	# display the image and the prediction
				# 	cv2.putText(image_array, prediction[0], (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				# 	cv2.imshow("Image", image_array)
				# 	cv2.waitKey(0)
				k += 1
			'''END FACE DETECTION'''

print(k, j)
print(j/k*100)
