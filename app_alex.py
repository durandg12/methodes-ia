from util_alex import  load_show_mnist, load_show_cifar10
import streamlit as st
import numpy as np

def main() :

    st.title('Choix du jeu de données et affichage d\'images')

    # Sélection du jeu de données
    dataset = st.radio('Choisissez un jeu de données :', ('MNIST', 'CIFAR-10'))

    if dataset == 'MNIST':
        st.write('Chargement du jeu de données MNIST...')
        trainloader,testloader,trainset,testset = load_show_mnist()

    elif dataset == 'CIFAR-10':
        st.write('Chargement du jeu de données CIFAR-10...')
        trainloader,testloader,trainset,testset= load_show_cifar10()
        


    # Bouton pour quitter l'application
    if st.button('Quitter'):
        st.stop()

if __name__ == "__main__":
    main()