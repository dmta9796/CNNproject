import sys
import os
import torch
from torch.autograd import Variable
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

import encoder as encode
import neuralnets as neuralnet

from PIL import Image, ImageFilter, ImageChops

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

def getforwardconvlayer():
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

def deprocess(image):
    return image * torch.Tensor([0.229, 0.224, 0.225])  + torch.Tensor([0.485, 0.456, 0.406])

def dd_helper(image, layer, iterations, lr):        
	net=neuralnet.netfunction()
	modulelist = list(net.features.modules())
	input = Variable(encode.transform(image).unsqueeze(0), requires_grad=True)
	net.zero_grad()
	for i in range(iterations):
		#         print('Iteration: ', i)
		out = input
		for j in range(layer):
			out = modulelist[j+1](out)
			loss = out.norm()
			loss.backward()
			input.data = input.data + lr * input.grad.data
	input = input.data.squeeze()
	input.transpose_(0,1)
	input.transpose_(1,2)
	input = np.clip(deprocess(input), 0, 1)
	im = Image.fromarray(np.uint8(input*255))
	return im


def deep_dream_image(image, layer, iterations, lr, octave_scale, num_octaves):
	if num_octaves>0:
		image1 = image.filter(ImageFilter.GaussianBlur(2))
		print(image1)
		if(image1.size[0]/octave_scale < 1 or image1.size[1]/octave_scale<1):
			size = image1.size
		else:
			size = (int(image1.size[0]/octave_scale), int(image1.size[1]/octave_scale))
		
		image1 = image1.resize(size,Image.ANTIALIAS)
		image1 = deep_dream_image(image1, layer, iterations, lr, octave_scale, num_octaves-1)
		size = (image.size[0], image.size[1])
		image1 = image1.resize(size,Image.ANTIALIAS)
		image = ImageChops.blend(image, image1, 0.6)
		#     print("-------------- Recursive level: ", num_octaves, '--------------')
	img_result = dd_helper(image, layer, iterations, lr)
	img_result = img_result.resize(image.size)
	plt.imshow(img_result)
	plt.show()
	return img_result


