import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO

DATA_FLAG = "pneumoniamnist"
SIZE = 224
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
info = INFO[DATA_FLAG] #Para acceder a los metadatos de MedMNIST, obteniendo clase correspondiente
DataClass = getattr(medmnist, info["python_class"])


transform = transforms.Compose([
    transforms.ToTensor() #Define transformaciones necesarias, convirtiendo imagen en tensor de (Canales (1), Alto, Ancho). Además, escala valores de [0, 255] a [0.0, 1.0].
])
train_dataset = DataClass(split="train", transform=transform, download=True, size = SIZE) #Instancia conjunto datos entrenamiento, indicando versión a cargar, 224px
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) #Envuelve dataset en DataLoader, para poder iterar e batchs de tamaño ajustable y barajar las muestras.


images, labels = next(iter(train_loader)) #extrae primer lote de imágenes y labels
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i in range(6):
    img = images[i].squeeze().cpu().numpy() #squeeze para reducir dimensión a 2D, convierte de tensor a matriz graficable
    axes[i].imshow(img, cmap="gray") #renderiza imagen
    axes[i].set_title(f"Class: {labels[i].item()}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()