#import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Loader:
	def __init__(self,batch_size=100):
		self.batch_size = batch_size
		#pass
	# load MNIST dataset
	def load_mnist(self,split):
	  train = True if split == 'train' else False
	  dataset = datasets.MNIST("./data", train=split, download=True, transform=transforms.ToTensor())
	  return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

	def load_train_test(self):
		train_loader = self.load_mnist('train')
		test_loader = self.load_mnist('test')
		"""
		print("\n-------------- Train set Features shape: --------------\n\t", next(iter(train_loader))[0].shape) # shape expected as: [batch_size, channel, height, width]
		print("\n-------------- Labels shape:  --------------\n\t", next(iter(train_loader))[1].shape)
		print("\n-------------- Labels example: --------------\n\t", next(iter(train_loader))[1])
		"""
		return train_loader, test_loader
		
