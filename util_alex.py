import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np



def load_show_mnist( ):
    """
    Charge le jeu de données MNIST.
    normalise les données

    Returns:
    trainloader: DataLoader pour les données d'entraînement.
    testloader: DataLoader pour les données de test.
    trainset: Dataset d'entraînement.
    testset: Dataset de test.
    """
    MEAN_MNIST = (0.1307,)
    STD_MNIST = (0.3081,)

    # Pour transformer nos donnees en Tensor et les Normaliser
    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_MNIST, STD_MNIST)])

    #On divise nos donnees en Train et Test puis on les transforme en Dataloader
    batch_size = 16 #Taille qu'ils utilisent dans l'article (5.2)

    trainset = torchvision.datasets.MNIST(root='./data', train=True, # On telecharge nos donnees Train
                                            download=True, transform=transform_mnist)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, # On telecharge nos donnees Test
                                        download=True, transform=transform_mnist)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    st.write('Affichage des images :')
    #affiche les trois premières images
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i in range(3):
        axes[i].imshow(torch.permute(trainset[i][0], (1, 2, 0)), cmap='gray' )
        
    st.pyplot(fig)

    return(trainloader,testloader,trainset,testset)


def load_show_cifar10():
    """
    Charge le jeu de données CIFAR-10.
    normalise les données
    Affiche les trois premières images

    Returns:
    trainloader: DataLoader pour les données d'entraînement.
    testloader: DataLoader pour les données de test.
    trainset: Dataset d'entraînement.
    testset: Dataset de test.
    """

    # Pour transformer nos donnees en Tensor et les Normaliser
    transform_CIFAR10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])

    #On divise nos donnees en Train et Test puis on les transforme en Dataloader
    batch_size2 = 16

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, # On telecharge nos donnees Train
                                            download=True, transform=transform_CIFAR10 )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size2,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, # On telecharge nos donnees Test
                                        download=True, transform=transform_CIFAR10 )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size2,
                                            shuffle=False, num_workers=2)
    
    st.write('Affichage des images :')
    # Les valeurs des pixels sont normalisées, nous devons les dénormaliser pour obtenir les valeurs originales (0-255)
    denorm = transforms.Normalize(mean = [-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616], std = [1/0.2470, 1/0.2435, 1/0.2616])
    #affiche les trois premières images
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i in range(3):
        axes[i].imshow(torch.permute(denorm(trainset[i][0]), (1, 2, 0)), cmap='gray' )
        
    st.pyplot(fig)
    return(trainloader,testloader,trainset,testset)





