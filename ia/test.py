import torch
import torch.nn as nn


# Define your model with the same architecture as when it was saved
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


# Define the same model parameters as when it was trained
vocab_size = 21910  # Replace with your actual vocabulary size
embedding_dim = 128  # Replace with your actual embedding dimension
hidden_dim = 256  # Replace with your actual hidden dimension

# Create an instance of your model with the correct architecture
model = SimpleLSTM(vocab_size, embedding_dim, hidden_dim)

# Load the saved state dictionary into the model
save_path = "./storage/test.pth"
state_dict = torch.load(save_path)
model.load_state_dict(state_dict)

model.eval()
# Define your actual vocabulary and reverse vocabulary mappings
vocab = {
    "Once": 1,
    "upon": 2,
    "a": 3,
    "time": 4,
    # Add more words and their corresponding indices
}

reverse_vocab = {index: word for word, index in vocab.items()}

# Define an example input text
example_text = "Once upon a time"

# Preprocess the input text to convert it into token indices
# Use your actual vocabulary for tokenization
input_indices = [
    vocab.get(word, 0) for word in example_text.split()
]  # Use 0 for missing tokens

# Print the input indices for debugging
print("Input Indices:", input_indices)

# Convert the input indices into a tensor
example_input = torch.tensor([input_indices])

# Make predictions
with torch.no_grad():
    predicted_output = model(example_input)

# You can decode the predicted output using the vocabulary
decoded_output = [
    reverse_vocab.get(
        index.item(), "<UNKNOWN>"
    )  # Use "<UNKNOWN>" for missing tokens
    for index in predicted_output[0, :, :].argmax(dim=1)
]

print("Input Text:", example_text)
print("Predicted Output:", " ".join(decoded_output))
