#import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Loader:
  """
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  #
  #						Class for loading MNIST datasets
  #
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  """
  def __init__(self,batch_size=100):
    self.batch_size = batch_size

  def load_mnist(self,split):
    train = True if split == 'train' else False
    dataset = datasets.MNIST("./data", train=split, download=True, transform=transforms.ToTensor())
    return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

  def load_train_test(self):
    train_loader = self.load_mnist('train')
    test_loader = self.load_mnist('test')
    return train_loader, test_loader
		
