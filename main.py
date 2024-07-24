import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader

# Definir tokenizer
tokenizer = get_tokenizer("basic_english")

# Definir función para construir el vocabulario
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Descargar y preparar el dataset
train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Definir la transformación del texto
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

# Crear una función de procesamiento de lotes (batch)
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

# Crear DataLoader
train_iter = AG_NEWS(split='train')
dataloader = DataLoader(to_map_style_dataset(train_iter), batch_size=8, shuffle=False, collate_fn=collate_batch)

# Definir el modelo
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Parámetros del modelo
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
embed_dim = 64
model = TextClassificationModel(vocab_size, embed_dim, num_class)

# Entrenamiento del modelo
def train(dataloader):
    model.train()
    for label, text, offsets in dataloader:
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Entrenamiento del modelo por 5 épocas
for epoch in range(5):
    train(dataloader)

print("Modelo entrenado exitosamente")
