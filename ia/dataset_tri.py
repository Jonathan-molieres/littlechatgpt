import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Charger le dataset OpenWebText
dataset = load_dataset("tiny_shakespeare")

# Obtenir le texte à partir du dataset
text_data = dataset["train"]["text"]

# Fractionner le texte en phrases (vous pouvez utiliser une autre approche de tokenization si nécessaire)
sentences = "\n".join(text_data).split(".")

# Créer un dictionnaire de correspondance mot -> ID
vocab = {word: i for i, word in enumerate(set(" ".join(sentences).split()))}


# Fonction pour convertir une phrase en une séquence d'IDs de mots
def sentence_to_ids(sentence):
    return [vocab[word] for word in sentence.split()]


# Convertir les phrases en séquences d'IDs de mots
sequences = [sentence_to_ids(sentence) for sentence in sentences]

# Fractionner les séquences en séquences d'entrée (X) et de sortie (Y)
X = [sequence[:-1] for sequence in sequences]
Y = [sequence[1:] for sequence in sequences]

# Fractionner les données en ensembles d'entraînement et de validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.1, random_state=42
)


# Créer un DataLoader pour charger les données
class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.Y[index])


train_dataset = TextDataset(X_train, Y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


X_train_sorted = sorted(X_train, key=len, reverse=True)
Y_train_sorted = sorted(Y_train, key=len, reverse=True)

X_train_padded = pad_sequence(
    [torch.tensor(x) for x in X_train_sorted], batch_first=True
)
Y_train_padded = pad_sequence(
    [torch.tensor(y) for y in Y_train_sorted], batch_first=True
)

train_dataset = TextDataset(X_train_padded, Y_train_padded)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Continuer avec le reste du code (définition du modèle, entraînement, etc.)


# Définir le modèle LSTM simple
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.linear(output)


# Instancier le modèle LSTM
vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 256
model = SimpleLSTM(vocab_size, embedding_dim, hidden_dim)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Évaluation du modèle sur l'ensemble de validation
model.eval()
val_loss = 0

with torch.no_grad():
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        val_loss += loss.item()

avg_val_loss = val_loss / len(train_dataloader)
print(f"Validation Loss: {avg_val_loss:.4f}")
