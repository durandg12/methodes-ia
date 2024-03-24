import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np
import NN_code

def viz_im(model, nb, testloader, file = None, architecture, input_channels, output_channels,  h, residual_block, end_activation, p, optimizer, criterion, device, lr, trainloader, n_epochs = 10):
   """
      Fonction pour visualiser le résultat des prédictions sur le jeu de données test.

      Parameters
      ----------
      model : nn.Module
         The trained model.
      nb : int
         The index of the batch_size on which we want to test our model.
      testloader : torch.utils.data.DataLoader
         The test Dataloader.
      architecture : function
         The architecture of the neural network to be trained
      in_channels : int
         Number of input channels, 1 (black & white) or 3 (RGB).
      out_channels : int
         The number of classes in our multinomial distribution (256), the value a pixel can take.
      h : int
         Number of channels in the layers.
      residual_block : function
         The residual block to be used in our architecture, which defines the model.
      end_activation : torch.nn
         The activation function to be used at the end of our architecture.
      p : int
         Number of residual blocks.
         5 if PixelCNN
         6 if PixelRNN
      optimizer : torch.optim.optimizer.Optimizer
         The optimizer to use
      criterion : nn.modules._Loss
         The loss function to minimize
      device : string
         The device to use.
      lr : float
         The learning rate
      trainloader : torch.utils.data.DataLoader
         The train DataLoader
      n_epochs : int, default = 10
         Number of epochs to do

      

      Returns
      -------
   """
   
   if file != None:
       #Create an instance of the class
      model = Architecture_Pixel(in_channels, out_channels, h, ResidualBlock_CNN, nn.Softmax(dim=0)).to(device = device)
      model.load_state_dict(torch.load('file')) #importation des poids
       #on our files : h = 5, p = 3, lr = 0.09, optimizer = Adam, epoch = 10/15
   else :
       model = train_loop(architecture, input_channels, output_channels,  h, residual_block, end_activation, p, optimizer, criterion, device, lr, trainloader, n_epochs = 10)


   # Génération de l'image
   batch_image, _ = testloader[nb] #on récupère une image du test set
   batch_image_true = np.ravel(batch_image.numpy())

# Afficher l'image du test set
image_true = image_true.reshape(28, 28)
plt.imshow(image_true, cmap = 'gray')
plt.title('image du jeu de données')
plt.show()




y = model2(image.to(device),p) # on passe l'image en input de notre modèle

#y = torch.round((y.to(device) * STD_MNIST[0] + MEAN_MNIST[0]) * 255)
print(y.size()) #(256, 28, 28)

y = torch.reshape(y, (256, 784)) # on reshape pour avoir un vecteur 1d représentant l'image, mais ila toujours les 256 channels
y_pred = np.zeros(784) # numpy 1d qui sera notre soummission avec les bonnes valeurs de pixel
y2=y
y=y[:, :].cpu().detach().numpy()#on transforme en numpy

#Pour chaque pixel on ne garde que l'intensité qui a une proba maximal parmi les 256 probas produites par la logsoftmax

for i in range(784) :
  y_pred[i] = np.argmax(y[:,i])

np.set_printoptions(linewidth = 145, precision = 2)

image_hat = y_pred.reshape(28,28)
plt.imshow(image_hat, cmap = 'gray')
plt.title('image prédite avec argmax')
plt.show()



# Sampling

sample = np.zeros(784)

for i in range(784):
        probs = y2[:,i]


        indices = torch.multinomial(probs, 1)

        sample[ i] = indices




#On affiche l'image
image_hat = sample.reshape(28,28)
plt.imshow(image_hat, cmap = 'gray',vmin=0,vmax=255)
plt.title('image prédite avec multinomiale')
plt.show()