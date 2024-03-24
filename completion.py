import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import random
import torch
import numpy as np


def updateimage(sample):
    """
        Image update function.

        Parameters
        ----------
        sample : tensor
            Truncated image
    """
    plt.imshow(sample.squeeze().cpu().numpy(), cmap = 'gray')
    plt.axis('off')
    plt.savefig('temp.png')
    plt.title('image prédite avec multinomiale')
    plt.close()

    with open('temp.png', 'rb') as f:
        img_widget.value = f.read()


def completion(testset, pixel, model, device, p):
    """
        Function for the image completion vizualization.

        Parameters
        ----------
        testset : torchvision.datasets
            Test dataset
        pixel : int
            The pixel from which to hide the image.
        model : nn.module
            The trained model.
        device : string
            The device to use.
        p : int
            Number of residual blocks.
    """
    x = random.randint(0, 9999) #retrieve a test set image
    image,  = testset[x] 

    image_trunc = image
    image_trunc[:,pixel:,:] = 0 #hidden part of the image

    #Display
    image_trunc_plot = image_trunc.reshape(28, 28)
    plt.imshow(image_trunc_plot , cmap = 'gray')
    plt.title('Image du jeu de données')
    plt.show()


    img_widget = widgets.Image(format='png') #Image widget creation

    display(img_widget)

    with torch.no_grad():
        for i in np.arange(pixel,28):
            for j in range(28):
                y = model(image_trunc.to(device),p)
                probs = y[ :, i, j]  #Select probabilities for the current pixel

                pixel_value = torch.multinomial(probs, 1)  #Sample a pixel value from the probabilities

                image_trunc[ :, i, j] = (pixel_value/255 - MEAN_MNIST[0])/STD_MNIST[0] 

                update_image(image_trunc) #Update image