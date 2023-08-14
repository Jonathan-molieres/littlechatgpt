import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index].clone().detach(), self.Y[index].clone().detach()


class TextDatasetManager:
    def __init__(
        self, dataset_name, batch_size, max_seq_length=None, other_name=None
    ):
        self.dataset = load_dataset(dataset_name, other_name)
        self.text_data = self.dataset["train"]["text"]
        self.sentences = "\n".join(self.text_data).split(".")
        self.vocab = {
            word: i
            for i, word in enumerate(set(" ".join(self.sentences).split()))
        }

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        sequences = [
            self.sentence_to_ids(sentence) for sentence in self.sentences
        ]
        X = [sequence[:-1] for sequence in sequences]
        Y = [sequence[1:] for sequence in sequences]
        X_train, _, Y_train, _ = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )

        self.X_train_sorted = sorted(X_train, key=len, reverse=True)
        self.Y_train_sorted = sorted(Y_train, key=len, reverse=True)

        self.X_train_padded = pad_sequence(
            [torch.tensor(x) for x in self.X_train_sorted], batch_first=True
        )
        self.Y_train_padded = pad_sequence(
            [torch.tensor(y) for y in self.Y_train_sorted], batch_first=True
        )

        self.train_dataset = TextDataset(
            self.X_train_padded, self.Y_train_padded
        )
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def sentence_to_ids(self, sentence):
        return [self.vocab[word] for word in sentence.split()]
