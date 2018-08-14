import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

import encoder as encode
import neuralnets as neuralnet
import decoder as decode
import training as train
import testing as test

def fileexists(afile):
	if(os.path.exists(afile)):
		return True
	else:
		return False

def ismodel(afile):
	print(afile)
	if(fileexists(afile)):
		afilelist=afile.split('.')
		if(afilelist[1]=="pth"):
			return True
		else:
			return False
	return False



if __name__ == "__main__":
	#print("type in a command:")
	while(True):
		cmd=input('type in a command:')
		cmdlist=cmd.split()
	
		if(cmdlist[0]=="train"):
			device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
			print(device)
			model=train.traindata(device)
		
		elif(cmdlist[0]=="test"):
			test.accuracy(model)
			#test.accuracybyclass(model)
			#decode.decode()
			decode.getfeaturemaps()
		
		elif(cmdlist[0]=="save"):
			torch.save(model.state_dict(), cmdlist[1])
		
		elif(cmdlist[0]=="load"):
			if(ismodel(cmdlist[1])):
				model= neuralnet.netfunction()
				model.load_state_dict(torch.load(cmdlist[1]))
				model.eval()
			
		elif(cmdlist[0]=="classify"):
			print("classify")
		
		elif(cmdlist[0]=="quit"):
			print('end program')
			exit()
		else:
			print("invalid command")

