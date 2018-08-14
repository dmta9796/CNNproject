import sys
import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

import encoder as encode
import neuralnets as neuralnet

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


def decode():
	# get some random training images
	dataiter = iter(encode.trainloader)
	images, labels = dataiter.next()
	
	# print labels
	print('Actual: ',' '.join('%5s' % encode.classes[labels[j]] for j in range(4)))
	
	#print classifications
	net=neuralnet.netfunction()
	outputs = net(images)
	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % encode.classes[predicted[j]] for j in range(4)))
	
	# show images
	imshow(torchvision.utils.make_grid(images))

def plotfeatures(images,features):
	plt.figure(1)
	size=len(features)
	for i in range(size):
		plt.subplot(1,size,i+1)
		plt.imshow(np.transpose((features[i].data.numpy()/2+0.5)))
	plt.subplot(2,size,1)
	plt.imshow(np.rollaxis((images.data.numpy()/2+0.5),0,3))
	plt.show()

def getfeaturemaps():
	dataiter = iter(encode.testloader)
	images, labels = dataiter.next()
	net=neuralnet.netfunction()
	new_net = nn.Sequential(*list(net.features.children())[:-2])
	outputs = new_net(images)
	#print(outputs[0])
	features=outputs[3][5].data.numpy()
	#print(img.size())
	features= features/ 2 + 0.5  # convert back to image and unnormalize
	plotfeatures(images[3],outputs[3])
	#print(images[0].data.numpy())
	#plt.figure(1)
	#plt.subplot(211)
	#plt.imshow(np.rollaxis(features,0,2))
	#plt.subplot(212)
	#plt.imshow(np.rollaxis((images[3].data.numpy()/2 + 0.5),0,3))
	#plt.show()
