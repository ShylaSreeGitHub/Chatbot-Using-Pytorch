import json
import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from feed_forward_neural_network import NeuralNetwork
from preprocesing_data_nltk_utils import tokenize, stem, bag_of_words

# To Open the dataset in READ mode
with open(r'research_dataset.json','r') as f:
    intents = json.load(f)

#print(intents)


all_words = [] # tokenized words from patterns
tags = [] # differents tags
xy = [] # holds patterns and tags

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for inquiry in intent['inquiry']:
        w = tokenize(inquiry)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ','] # Ignore the Stop words
#print("tokenized words:", all_words)

all_words = [stem(w) for w in all_words if w not in ignore_words]
#print("stem words:",all_words)

all_words = sorted(set(all_words))


tags = sorted(set(tags))


#tags = sorted(set(tags))

#print(tags)

# bag of words
X_train = []  # we put all the bag of words in the array
y_train = []  # number for each tag
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)  # 0,1,2,3 for tags
    y_train.append(label)  # calculate crossentropyloss

X_train = np.array(X_train)
y_train = np.array(y_train)
#print(X_train[0])


# instantiated hyperparameters 
learning_rate = 0.001
num_epochs = 1000 # Number of training cycles
batch_size = 8
hidden_layer_size = 8
output_size = len(tags)
input_layer_size = len(X_train[0])


class ChatDataset(Dataset):
    def __init__(self):
        #print("***************888")
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        #print("*******", self.n_samples)
        return self.n_samples


#print(input_layer_size, len(all_words))
#print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # To check if we have the GPU support
model = NeuralNetwork(input_layer_size, hidden_layer_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs): # number of loops = number of training cycles
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss Value: {loss.item():.4f}')

# Save the Model's key data in dictionary data strucuture 
data = {
    "all_words": all_words, # All the words collected
    "input_layer_size": input_layer_size,
    "hidden_layer_size": hidden_layer_size,
    "output_size": output_size,
    "model_state": model.state_dict(),
    "tags": tags # All the words collected
}

# Define the File Name
FILE = "data_model.pth" # ".pth" is extension to PyTorch
torch.save(data, FILE) #save() to save the file

# This will searialize and save it to pickled file.

print(f'Completed the Training Step. File is saved to {FILE}') # Printing in f String

# Run the Training again