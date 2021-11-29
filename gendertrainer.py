# Import statements

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Dropout, LayerNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os


# Read fold data files - They contain the values (e.g. Gender, Age, ...)

print('Read fold data files...')
fold0 = pd.read_csv("fold_0_data.txt",sep = "\t" )
fold1 = pd.read_csv("fold_1_data.txt",sep = "\t")
fold2 = pd.read_csv("fold_2_data.txt",sep = "\t")
fold3 = pd.read_csv("fold_3_data.txt",sep = "\t")
fold4 = pd.read_csv("fold_4_data.txt",sep = "\t")

# Combine data to avoid having to deal with five different files

print('Combine the fold data files...')
total_data = pd.concat([fold0, fold1, fold2, fold3, fold4], ignore_index=True)

# Define the wanted columns and remove unnecessary data

print('Removing unnecessary data...')
imp_data = total_data[['age', 'gender', 'x', 'y', 'dx', 'dy', 'tilt_ang', 'fiducial_yaw_angle']].copy()
imp_data.info()

# Create paths to the images

print('Start image path creation process...')
img_path = []
for row in total_data.iterrows():
    path = "faces/"+row[1].user_id+"/coarse_tilt_aligned_face."+str(row[1].face_id)+"."+row[1].original_image
    img_path.append(path)
imp_data['img_path'] = img_path
imp_data.head()


# Drop unknown genders

print('Remove unknown genders...')
imp_data = imp_data.dropna()
clean_data = imp_data[imp_data.gender != 'u'].copy()
clean_data.info()

# Map gender to 0 - Female/f and 1 - Male/m
print('Map gender to 0 and 1...')
gender_to_label_map = {
    'f' : 0,
    'm' : 1
}
clean_data['gender'] = clean_data['gender'].apply(lambda g: gender_to_label_map[g])

# Prepare data for training and testing

print('Prepare images for testing and training...')
cimg_size = 32
data = []
target = []

# Use haar cascade classifier to find the faces in the pictures (=> fast)
facedata = "faces.xml"
cascade = cv2.CascadeClassifier(facedata)

for index, row in clean_data.iterrows():
	cimg = cv2.imread(os.path.abspath(row['img_path']))
	faces = cascade.detectMultiScale(cimg)
	for f in faces:
		x, y, w, h = [v for v in f]
		subface = cimg[y:y + h, x:x + w]	
		gray = cv2.cvtColor(subface,cv2.COLOR_BGR2GRAY)
		resized = cv2.resize(gray,(cimg_size,cimg_size))
		data.append(resized)
		target.append(row['gender'])
	print('Prepared image number',index)

# Resize and reshape faces for training and testing

print('Resize and reshape faces...')
data = np.array(data)/255.0
data = np.reshape(data,(data.shape[0],cimg_size,cimg_size,1))
target = np.array(target)
new_target = np_utils.to_categorical(target)

# Save data for later - Data can be used for training and testing anytime

print('Saving prepared data...')
np.save('./training/data',data)
np.save('./training/target',new_target)

# Load the previously prepared data

print('Load the already prepared data...')
data=np.load('./training/data.npy')
target=np.load('./training/target.npy')


# Define a model structure
'''
	1. Sequential -> Progress step by step through each layer
	2. Conv2D ->
	3. Conv2D ->
	4. MaxPooling2D ->
	5. Conv2D ->
	6. Conv2D ->
	7. MaxPooling2D ->
	8. Droput ->
	9. Flatten ->
	10. Dense ->
	11. Dropout ->
	12. Dense ->

'''

print('Define model structure')
noOfFilters=64
sizeOfFilter1=(3,3)
sizeOfFilter2=(3,3)
sizeOfPool=(2,2)
noOfNode=64

model=Sequential()
model.add((Conv2D(32, sizeOfFilter1, input_shape=data.shape[1:],activation='relu')))
model.add((Conv2D(32, sizeOfFilter1,activation='relu')))
model.add(MaxPooling2D(pool_size=sizeOfPool))

model.add((Conv2D(64, sizeOfFilter2,activation='relu')))
model.add((Conv2D(64, sizeOfFilter2,activation='relu')))
model.add(MaxPooling2D(pool_size=sizeOfPool))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(noOfNode, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()

# Compile model
print('Compiling model...')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Split data into training and test data

print('Split prepared data into training and test data...')
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.3)

# Train the model and save it

print('Start training process...')
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
history=model.fit(train_data,train_target,epochs=100,callbacks=[callback],validation_split=0.2)
model.save('gender_model.h5')

# Evaluate trained model with test data

test_loss, test_acc = model.evaluate(test_data, test_target, verbose=2)
print('Accuracy: ', test_acc)
print('Loss: ', test_loss)

