import torch
import torch.nn as nn


# Définition de votre modèle
class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.linear(output)


save_path = "./storage/test.pth"

model = SimpleLSTM()

# Charger le dictionnaire d'état
state_dict = torch.load(save_path)

# Charger le dictionnaire d'état dans le modèle
model.load_state_dict(state_dict)

model.eval()


example_input = torch.tensor([[1, 2, 3, 4, 5]])
with torch.no_grad():
    predicted_output = model(example_input)


print(predicted_output)
