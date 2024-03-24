import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from train import train_loop
from model import Architecture_Pixel,ResidualBlock_CNN


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

    return(trainloader,testloader,trainset,testset,MEAN_MNIST,STD_MNIST)


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
        axes[i].imshow(torch.permute(denorm(trainset[i][0]), (1, 2, 0)))
        
    st.pyplot(fig)
    return(trainloader,testloader,trainset,testset,[-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616],[1/0.2470, 1/0.2435, 1/0.2616])


def afficher_page_accueil():
    
    
    st.title("Bienvenue dans notre projet de modèle génératif d'image")
    st.write(' Nous explorerons le modèle Pixel CNN sur les jeux de données MNIST et CIFAR-10.')

    # Ajoutez d'autres éléments de la page d'accueil, tels que des images explicatives, etc.
    # ...

def afficher_choix_jeux_de_donnees():
    global trainloader, testloader, trainset, testset, mean, std,dataset

    st.title('Choix du jeu de données et affichage d\'images')

    # Sélection du jeu de données
    dataset = st.radio('Choisissez un jeu de données :', ('MNIST', 'CIFAR-10'))

    if dataset == 'MNIST':
        st.write('Chargement du jeu de données MNIST...')
        trainloader,testloader,trainset,testset,mean,std = load_show_mnist()

    elif dataset == 'CIFAR-10':
        st.write('Chargement du jeu de données CIFAR-10...')
        trainloader,testloader,trainset,testset,mean,std = load_show_cifar10()
    
def afficher_page_modele():
    global trainloader, testloader, trainset, testset, mean, std,dataset
    
    st.title('Pixel CNN')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam
    if dataset == 'MNIST' : 
        in_channels = 1 
        out_channels = 256
    else : 
        in_channels = 3 
        out_channels = 3*256
    # Afficher les valeurs des variables globales
    st.write(f'Vous avez choisi le jeu de données {dataset}')
    st.write(' Voulez vous utiliser notre mmodèles déjà entrainer ou voulez vous entrainer le votre')


    # Options pour l'utilisateur
    option = st.radio("Choisissez une option :", ( "Utiliser un modèle pré-entraîné","Entraîner un nouveau modèle"))
    if option == "Entraîner un nouveau modèle" :

        # Interface pour définir les paramètres d'entraînement
        epochs = st.slider("Nombre d'époques :", min_value=1, max_value=100, value=10, step=1)
        h = st.slider("Nombre de neurones :", min_value=1, max_value=128, value=5, step=1)
        lr = st.slider("Taux d'apprentissage :", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        p = st.slider("Nombre de blocs résiduels : ", min_value=1, max_value=8, value=2, step=1)
                 
        use_cuda = st.checkbox("Utiliser CUDA si disponible pour l'entraînement des modèles")

        if use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            st.write("Entraînement des modèles sur CUDA.")
        else:
            device = torch.device("cpu")
            st.write(" Entraînement des modèles sur CPU.")
        go =st.checkbox("Commencer l'entraînement")
        if go:
            start_time = time.time()
            model = train_loop(Architecture_Pixel, in_channels, out_channels, h, ResidualBlock_CNN, nn.LogSoftmax(), p, optimizer, criterion, device, lr, trainloader, epochs,mean,std )
            print(f"Training with {device} lasts: {np.round((time.time()-start_time)/60,2)} minutes\n")
    else :
        st.write(f'v')