import torch
import torch.nn as nn
import torch.optim as optim
from dataset import TextDatasetManager

dataset_manager = TextDatasetManager(
    dataset_name="tiny_shakespeare", batch_size=32
)
train_dataloader = dataset_manager.train_dataloader
vocab = dataset_manager.vocab


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


vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 256
print(f"Vocab size: {vocab_size}")
model = SimpleLSTM(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("./storage/test.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10

# Specify the path to save the model
save_path = "./storage/test.pth"

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

# Save the model after completing all epochs
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")
