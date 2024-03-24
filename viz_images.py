import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np
import model


def viz_im(model, p, device):
   """
      Creates and returns 2 lists of images from a testloader: one containing real images, the other containing model predictions.

      Parameters
      ----------
      model : nn.Module
         The trained model.   
      p : int
         Number of residual blocks.
      device : string
         The device to use.
      
      Returns
      -------
      list1 : tensor
         List with the true values of each pixel for 4 random testloader images.
      list2 : tensor
         List with predictions for each pixel in 4 random testloader images.
   """
   global trainloader, testloader, trainset, testset, mean, std, dataset

   list1 = []
   list2 = []
   rand = torch.randint(0, 625, (4,))

   for i in rand: 
      image, _ = testset[i]

      #Real image
      image_true = np.ravel(image.numpy())
      image_true = image_true.reshape(28, 28)

      list1.append(image_true)

      #Prediction
      y = model(image.to(device), p)
      y = torch.reshape(y, (256, 784)) #reshape to get a 1d vector, but it still has the 256 channels

      y_pred = np.zeros(784) #our future image

      for i in range(784):
         probs = y[:, i]
         y_pred[i] = torch.multinomial(probs, 1)
   
      image_hat = y_pred.reshape(28,28)
      list2.append(image_hat)

   return(list1, list2)



def display_images_in_line(image_list, title):
   """
      Displays images from a list.

      Parameters
      ----------
      image_list : list
         A list of images to view.
      title : string
         The title of the image to view.
   """
   fig = plt.figure(figsize=(len(image_list)*2, 2))
   for i, image_tensor in enumerate(image_list):
      plt.subplot(1, len(image_list), i+1)
      plt.imshow(image_tensor.squeeze(), cmap = 'gray')
      plt.title(title)
      plt.axis('off')
   plt.show()