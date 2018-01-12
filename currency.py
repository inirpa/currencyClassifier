import os
import matplotlib.pyplot as plt 
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm


train_dir = 'H:\\python\\practice\\money\\data\\train'
test_dir = 'H:\\python\\practice\\money\\data\\test'

img_size = 50
learning_rate = 1e-3

MODEL_NAME = 'money--{}--{}.model'.format(learning_rate,'6CONV-BASIC')

def img_label(img):
	word_label = img.split('.')[0].split('_')[0]
	if word_label == 'ten':
		return [1,0]
	elif word_label == 'tw':
		return [0,1]

def create_train_data():
	training_data = []

	for img in tqdm(os.listdir(train_dir)):
		label = img_label(img)
		path = os.path.join(train_dir,img)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (img_size,img_size))
		training_data.append([np.array(img),np.array(label)])

	shuffle(training_data)
	np.save('train_data',training_data)
	return training_data

def process_test_data():
	testing_data = []

	for img in tqdm(os.listdir(test_dir)):
		img_num = img.split('.')[0]
		path = os.path.join(test_dir,img)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (img_size, img_size))
		testing_data.append([np.array(img), np.array(img_num)])

	shuffle(testing_data)
	np.save('test_data.npy', testing_data)
	return testing_data	
	

train_data = create_train_data()

#model
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


covnet = input_data(shape=[None, img_size, img_size, 1], name='input')

covnet = conv_2d(covnet, 32, 2, activation='relu')
covnet = max_pool_2d(covnet, 2)

covnet = conv_2d(covnet, 64, 2, activation='relu')
covnet = max_pool_2d(covnet, 2)

covnet = conv_2d(covnet, 32, 2, activation='relu')
covnet = max_pool_2d(covnet, 2)

covnet = conv_2d(covnet, 64, 2, activation='relu')
covnet = max_pool_2d(covnet, 2)

covnet = conv_2d(covnet, 32, 2, activation='relu')
covnet = max_pool_2d(covnet, 2)

covnet = conv_2d(covnet, 64, 2, activation='relu')
covnet = max_pool_2d(covnet, 2)

covnet = fully_connected(covnet, 1024, activation='relu')
covnet = dropout(covnet, 0.8)

covnet = fully_connected(covnet, 2, activation='softmax')
covnet = regression(
	covnet, 
	optimizer='adam', 
	learning_rate=learning_rate, 
	loss='categorical_crossentropy', 
	name='targets'
	)

model = tflearn.DNN(covnet, tensorboard_dir = 'log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([ i[0] for i in train]).reshape(-1, img_size, img_size, 1)
y = [ i[1] for i in train ]

test_x = np.array([ i[0] for i in test]).reshape(-1, img_size, img_size,1)
test_y = [ i[1] for i in test]

model.fit(
	{ 'input' : X},
	{'targets' : y}, 
	n_epoch=1,
	validation_set=(
		{'input' : test_x},
		{'targets' : test_y}),
	snapshot_step = 500,
	show_metric = True,
	run_id = MODEL_NAME 
	)
model.save(MODEL_NAME)

# #if model is already saved use that to get test_data
test_data = process_test_data()
# test_data = np.load('train_data.npy')
fig = plt.figure()
correct = []
correctCount = 0
test_number = 15

for num, data in enumerate(test_data[:test_number]):
	img_num = data[1]
	img_data = data[0]

	y = fig.add_subplot(test_number/4+1, 4, num+1)
	orig = img_data
	data = img_data.reshape(img_size, img_size, 1)


	model_out = model.predict([data])[0]

	if np.argmax(model_out) == 1:
		str_label = "20"
		if str_label == "20" and str(img_num).split('_')[0] == "tw":
			correct.append("Correct")
			correctCount += 1
		else:
			correct.append("Incorrect")
		str_label =  "20 " + str(img_num)
		
	else:
		str_label = "10"
		if str_label == "10" and str(img_num).split('_')[0] == "ten":
			correct.append("Correct")
			correctCount += 1
		else:
			correct.append("Incorrect")	
		str_label =  "10 " + str(img_num)
	
	
	y.imshow(orig)		
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)

# print(correct)
print("Correct prediction",correctCount," / ",test_number)
plt.show()	




