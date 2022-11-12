# -*- coding: utf-8 -*-
"""project 622.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16gBiuCUN-LtC-bUFqU4slTttRMCzayXE
"""

# !pip install transformers --quiet
# !pip install pyvacy --quiet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

# Import the transformers library for the retrieving the BERT embeddings.
import transformers
from transformers import BertModel, BertTokenizer


# Import pyvacy for privacy preserving optimizers.
from pyvacy import optim as private_optim, analysis

# Import scikit-learn packages.
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.utils import gen_batches


# Import scientific computing python packages.
import pandas as pd
import numpy as np      
import matplotlib.pyplot as plt

# Additional packages.
from tqdm import tqdm
import csv
from typing import List
from torchinfo import summary
import mlflow


# Custom functions.
from functions import get_bert_embeds_from_tokens, bert_tokenize, load_table_data, load_semeval_data, tf_tokenizer, tf_bert_tokenize
from custom_dataset import TableDataset


# Using gpu if available.
device = "cuda:0" if torch.cuda.is_available() else "cpu"


"""## Model definition and training

### Implement the model, make BERT part of the model.
"""

class erin_model(nn.Module):
    def __init__(self, in_size=768, hidden_size: int = 1, num_relations: int = 29, sequence_length:int = 50):
        super(erin_model,self).__init__()
        
        # Just add one LSTM unit as the model followed by a fully connected layer and then a softmax.
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50*hidden_size, num_relations)

    def forward(self, x):
        # First get the bert embeddings.
        # Then do the forward pass.

        x, (h_n, c_n) = self.lstm(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.softmax(x, 1)
        return output


sequence_max_length = 50
# Define BertTokenizer and BertModel
bert_tokenizer = BertTokenizer.from_pretrained('/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/vocab.txt', model_max_length=sequence_max_length, padding_side='right', local_files_only=True)#, config='/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/tokenizer_config.json')
bert_model = BertModel.from_pretrained('/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/', local_files_only=True)#, config='/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/config.json')
bert_model = bert_model.to(device)


"""
Load data based on semeval or table data.
"""
data_name = 'table'  # Change this based on what data you wanna load.
input_file_name = None
output_file_path = None
if data_name == 'table':
    # Load erin's data.
    sentences, y, label_mappings = load_table_data()  # By default it loads the smaller version of the dataset.
else:
    # Load semeval data:
    sentences, y, label_mappings = load_semeval_data(input_file_name, output_file_path)


# Define model parameters.
seeds = [0]   # Change the actual seed value here.
batch_size = 16
epochs = 10
optimizer_name = "Adam"
learning_rate = 0.001

load_epochs = epochs - 5

model_load_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model_checkpoints/tabular_data/sgd/epoch_{load_epochs}_{optimizer_name}_{learning_rate}_tf_encode.pt"
model_save_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model_checkpoints/tabular_data/sgd/epoch_{epochs}_{optimizer_name}_{learning_rate}_tf_encode.pt"
# Define the model and the required optimizer and loss function.
model = erin_model(sequence_length=sequence_max_length) # Using default model dimensions.
model.to(device)  # Make sure you have this before loading an existing model.
if optimizer_name == 'RMSProp':
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if (epochs - 5) > 0:
    print("Loading an existing model from checkpoint.", flush=True)
    checkpoint = torch.load(model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()  # This is important
    

print("Model summary: ", summary(model, input_size=(batch_size, sequence_max_length, 768)))
# optimizer = optim.DPSGD(params=model.parameters(), **training_parameters)  # Define training parameters.

tf_bert_tokenizer = tf_tokenizer()


# epsilon = analysis.epsilon(**training_parameters)
criterion = nn.CrossEntropyLoss()



# run the model for each seed.
for seed in seeds:
    print("Seed: ", seed)
    # First get the split of the training and test set.
    X_train, X_test, y_train_classes, y_test_classes  = train_test_split(sentences, y, random_state=seed, test_size=0.2)
    X_train_subset, X_val_subset, y_train_subset, y_val_subset = train_test_split(X_train, y_train_classes, random_state=seed, test_size=0.50)  # Getting a 50% split on the train set from the previous line is equal to a 40% train-val split on the whole dataset. This line is not necessary.
    
    # Specifying this to make sure that we are using the whole dataset.
    X_train_subset = X_train
    y_train_subset = y_train_classes
    
    print("Encoding training data.")
    all_train_tokens = []
    for batch in tqdm(range(0, len(X_train_subset), batch_size)):
        sentence_batch = X_train_subset[batch:batch+batch_size]
        
        # Tokenize the data.
        # train_tokens = bert_tokenize(sentence_batch, bert_tokenizer)
        train_tokens = tf_bert_tokenize(sentence_batch, tf_bert_tokenizer, max_len=sequence_max_length)
        # print("Train tokens: ", train_tokens)
        # print("Type train tokens: ", type(train_tokens))
        all_train_tokens.extend(train_tokens)
        # Get bert embeddings for the data.
    print("Training data encoding complete.", flush=True)
    
    print("Starting model training.", flush=True)
    epoch_losses = []
    for epoch in range(epochs - 5, epochs):
        print("Epoch: ", epoch)
        running_loss = 0.0
        for batch in range(0, len(all_train_tokens), batch_size):
            batch_train_tokens = all_train_tokens[batch:batch+batch_size]
            batch_y_train_classes = y_train_subset[batch:batch+batch_size]
            
            last_hidden_states_train = get_bert_embeds_from_tokens(bert_model, batch_train_tokens)  # bert_model is already on the gpu.
            
            inputs_tensor = torch.Tensor(last_hidden_states_train)
            batch_labels_tensor = torch.Tensor(batch_y_train_classes)
            
            # Put the batched data on the gpu.
            inputs_tensor = inputs_tensor.to(device)
            batch_labels_tensor = batch_labels_tensor.type(torch.LongTensor)
            batch_labels_tensor = batch_labels_tensor.to(device)
            
            optimizer.zero_grad()

            # # Forward pass.
            outputs = model(inputs_tensor)            
            
            # Calculate loss.
            loss = criterion(outputs, batch_labels_tensor)
            
            # Calculate gradients.
            loss.backward()

            # Update weights.
            optimizer.step()

            # Calculate loss for debugging.
            training_loss = loss.item()
            running_loss += training_loss
            # if i % 1000 == 999:
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(inputs) :.3f}')
            if batch % 100 == 0:
                print(f"Batch loss at batch {batch}: ", training_loss / batch_size, flush=True)
        epoch_loss = running_loss / len(all_train_tokens)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} loss : ", epoch_loss, flush=True)
    print("All epoch losses: ", epoch_losses)
    print("Finished model training.")

    ### Save model to disk.
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, model_save_path)
    

    """Run predictions on the trained model."""
    
    print("Encoding test data.")
    all_test_tokens = []
    for batch in tqdm(range(0, len(X_test), batch_size)):
        sentence_batch = X_test[batch:batch+batch_size]
        
        # Tokenize the data.
        # test_tokens = bert_tokenize(sentence_batch, bert_tokenizer)
        test_tokens = tf_bert_tokenize(sentence_batch, tf_bert_tokenizer, max_len=sequence_max_length)
        all_test_tokens.extend(test_tokens)
        # Get bert embeddings for the data.
    print("Test data encoding complete.")
    
    print("Testing on test data.")
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        for batch in range(0, len(all_test_tokens), batch_size):
            batch_test_tokens = all_test_tokens[batch:batch+batch_size]
            batch_y_test_classes = y_test_classes[batch:batch+batch_size]
            
            last_hidden_states_test = get_bert_embeds_from_tokens(bert_model, batch_test_tokens)  # bert_model is already on the gpu.
            
            inputs_tensor_test = torch.Tensor(last_hidden_states_test)
            batch_labels_tensor_test = torch.Tensor(batch_y_test_classes)
            
            # Put the batched data on the gpu.
            inputs_tensor_test = inputs_tensor_test.to(device)
            batch_labels_tensor_test = batch_labels_tensor_test.type(torch.LongTensor)
            batch_labels_tensor_test = batch_labels_tensor_test.to(device)
            
            test_outputs = model(inputs_tensor_test)
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(test_outputs.data, 1)
            total += batch_labels_tensor_test.size(0)
            correct += (predicted == batch_labels_tensor_test).sum().item()
        print(f'Test accuracy for seed {seed}: {100 * correct // total} %')