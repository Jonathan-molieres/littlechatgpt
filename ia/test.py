import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Télécharger le modèle ResNet-18 pré-entraîné
model = models.resnet18(pretrained=True)

# Charger une image de test (par exemple, une image d'un chat)
image_path = "chemin/vers/votre/image.jpg"
image = Image.open(image_path)

# Prétraitement de l'image pour la compatibilité avec ResNet-18
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(
    0
)  # Ajouter une dimension supplémentaire pour le lot

# Mettre le modèle en mode évaluation (pas de mise à jour des gradients)
model.eval()

# Passer l'image par le modèle pour obtenir les prédictions
with torch.no_grad():
    output = model(input_batch)

# Charger les étiquettes pour ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
import requests

LABELS = requests.get(LABELS_URL).json()

# Obtenir l'indice de classe prédit et afficher la prédiction
_, predicted_idx = torch.max(output, 1)
predicted_label = LABELS[predicted_idx.item()]
print("Prédiction : ", predicted_label)
