from util_alex import  load_show_mnist, load_show_cifar10, afficher_page_accueil, afficher_choix_jeux_de_donnees, afficher_page_modele
import streamlit as st
import numpy as np



def main() :
    # Initialisation des variables globales
    trainloader = None
    testloader = None
    trainset = None
    testset = None
    mean = None
    std = None
    dataset = None
    st.sidebar.title('Navigation')
    # Ajoutez les différentes sections de votre application à la barre latérale
    section = st.sidebar.radio('Sections', ('Accueil',  'Modèle'))

    if section == 'Accueil':
        afficher_page_accueil()
        afficher_choix_jeux_de_donnees()

    elif section == 'Modèle':
        afficher_page_modele()

    

if __name__ == "__main__":
    main()

        