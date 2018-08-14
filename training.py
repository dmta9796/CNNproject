import encoder as encode
import neuralnets as nn
def traindata(device):
	net=nn.netfunction()
	criterion=nn.criterionfunction()
	optimizer=nn.optimizerfunction()
	trainloader=encode.trainloader
	for epoch in range(3):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data
			#inputs, labels = inputs.to(device), labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net.forward(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 200 == 199:    # print every 200 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 200))
				running_loss = 0.0
				
	print('Finished Training')
	return net
