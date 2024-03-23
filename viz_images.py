
def viz_im(model, nb, in_channels, out_channels, h):
     """
     Fonction pour visualiser le résultat des prédictions sur le jeu de données test.

     Parameters
     ----------
     model : nn.Module
        The trained model.
     nb : int
        L'indice de l'image sur laquelle on souhaite tester notre modèle.

     Returns
     -------
     """
#Création du modèle
model2 = Architecture_Pixel(in_channels, out_channels, h, ResidualBlock_CNN, nn.Softmax(dim=0)).to(device = device) #création d'une instance de la classe
model2.load_state_dict(torch.load('PixelCNN_MNIST.pth')) #importation des poids
model2.eval()

# Génération de l'image
image, _ = testset[nb] #on récupère une image du test set
image_true = np.ravel(image.numpy())

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


#Si h petit et nombre d'epoque petit, on n'a seulement des pixels noirs (0) et blancs (253) (h = 10, epoch = 10)
# Sampling

sample = np.zeros(784)

for i in range(784):
        probs = y2[:,i]


        indices = torch.multinomial(probs, 1)
        #counts = torch.bincount(indices)

        # Trouver l'indice qui correspond à la classe la plus fréquente
        #sample[ i] = torch.argmax(counts)
        sample[ i] = indices




#On affiche l'image
image_hat = sample.reshape(28,28)
plt.imshow(image_hat, cmap = 'gray',vmin=0,vmax=255)
plt.title('image prédite avec multinomiale')
plt.show()