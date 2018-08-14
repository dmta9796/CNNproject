import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.features=nn.Sequential(
			nn.Conv2d(3, 6, 5),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(6, 16, 5),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2)
		)
#take conv layers
#try looking at inverse of conv matrix
#inceptionnet 
#style transfer
#
		self.classifier = nn.Sequential(
			nn.Linear(16 * 5 * 5, 120),
			nn.ReLU(inplace=True),
			nn.Linear(120, 84),
			nn.ReLU(inplace=True),
			nn.Linear(84, 10),
			nn.Softmax(1)
		)
	
	def forward(self, x):
		x=self.features(x)
		x=x.view(-1,16 * 5 * 5)
		x=self.classifier(x)
		return x
      
net=Net()

#class NetFeatureExtractor(nn.Module):
	#def __init__(self):
		#super(NetFeatureExtractor,self).__init__()
		#self.features = nn.Sequential(*list(net.features.children())[:-3]
		
	#def forward(self, x):
		#x=self.features(x)
		#return x
		
#featurenet=NetFeatureExtractor()

def netfunction():
	return net
def criterionfunction():
	return nn.CrossEntropyLoss()
def optimizerfunction():
	return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
