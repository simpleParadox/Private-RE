from typing import List

import matplotlib.pyplot as plt
import numpy as np
# Import scientific computing python packages.
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Import the transformers library for the retrieving the BERT embeddings.
import transformers
from mlflow import log_artifacts, log_metric, log_param
# Import opacus to for privacy.
from opacus import PrivacyEngine
from opacus.layers import dp_rnn
# Import scikit-learn packages.
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import gen_batches, shuffle
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchinfo import summary
from torchvision import datasets
from torchvision.transforms import ToTensor
# Additional packages.
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from custom_dataset import TableDataset, SemevalDataset
# Custom functions.
from semeval_funcs import (bert_tokenize, get_bert_embeds_from_tokens,
                       load_semeval_data, load_table_data, reformat,
                       tf_bert_tokenize, tf_tokenizer)
import sys                      

# Using gpu if available.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
seed_arg = int(sys.argv[2])
print("Seed arg: ", seed_arg)


"""## Model definition and training

### Implement the model, make BERT part of the model.
"""

class erin_model(nn.Module):
    def __init__(self, in_size=768, hidden_size: int = 1, num_relations: int = 19, sequence_length:int = 50, private=False):
        super(erin_model,self).__init__()
        
        # Just add one LSTM unit as the model followed by a fully connected layer and then a softmax.
        if private:
            self.lstm = dp_rnn.DPLSTM(input_size=in_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(sequence_length*hidden_size, num_relations)
        print("Private or non-private....: ", private)

    def forward(self, x):
        # First get the bert embeddings.
        # Then do the forward pass.
        x, (h_n, c_n) = self.lstm(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.softmax(x, 1)
        return output


sequence_max_length = 50

# --- Subject & object markup ---
SUB_START_CHAR = "<e1>"
SUB_END_CHAR = "</e1>"
OBJ_START_CHAR = "<e2>"
OBJ_END_CHAR = "</e2>"

added_special_token = [SUB_START_CHAR, SUB_END_CHAR, OBJ_START_CHAR, OBJ_END_CHAR]

# Define BertTokenizer and BertModel
bert_tokenizer = BertTokenizer.from_pretrained(
    '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/vocab.txt', model_max_length=sequence_max_length, padding_side='right', local_files_only=True,
    additional_special_tokens = added_special_token
    )
bert_model = BertModel.from_pretrained('/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/', local_files_only=True)#, config='/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model/bert/config.json')
bert_model.resize_token_embeddings(len(bert_tokenizer))
bert_model = bert_model.to(device)


"""
Load data based on semeval or table data.
"""
data_name = 'semeval'  # Change this based on what data you wanna load.

# define data path first

train_directory_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/semeval/train.txt'
test_directory_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/semeval/test.txt'

save_traincsvfile_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/semeval/train.tsv'
save_testcsvfile_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/data/semeval/test.tsv'

# input_file_name = None
# output_file_path = None
if data_name == 'table':
    # Load erin's data.
    sentences, y, label_mappings = load_table_data()  # By default it loads the smaller version of the dataset.
else:
    # Load semeval data:
    X_train, y_train_classes, label_mappings = load_semeval_data(train_directory_path, save_traincsvfile_path)
    X_test, y_test_classes, l_map = load_semeval_data(test_directory_path, save_testcsvfile_path)


# Define model parameters.
seeds = [0]   # Change the actual seed value here.
batch_size = 10
epochs = 100
optimizer_name = "Adam" # DP-SGD, DP-Adam, Adam, SGD, RMSProp
learning_rate = 0.001
load_epochs = epochs - 100
make_private = False
EPSILON = 4
DELTA = (1/8000)
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.5

print("Is private? ", make_private)

if make_private:
    model_load_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model_checkpoints/semeval/dpsgd/epoch_{load_epochs}_{optimizer_name}_{learning_rate}_private_seed_{seeds[0]}.pt"
    model_save_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model_checkpoints/semeval/dpsgd/epoch_{epochs}_{optimizer_name}_{learning_rate}_private_seed_{seeds[0]}.pt"
else:
    model_load_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model_checkpoints/semeval/sgd/epoch_{load_epochs}_{optimizer_name}_{learning_rate}_seed_{seeds[0]}.pt"
    model_save_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dp_re/model_checkpoints/semeval/sgd/epoch_{epochs}_{optimizer_name}_{learning_rate}_seed_{seeds[0]}.pt"

# Define the model and the required optimizer and loss function.
model = erin_model(sequence_length=sequence_max_length, private=make_private) # Using default model dimensions.
model.to(device)  # Make sure you have this before loading an existing model.
if optimizer_name == 'RMSProp':
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Log hyper-parameters using MLFlow here.



if load_epochs > 0 and make_private==False:
    print("Loading an existing model from checkpoint.", flush=True)
    checkpoint = torch.load(model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()  # This is important
    

print("Model summary: ", summary(model, input_size=(batch_size, sequence_max_length, 768)))
# optimizer = optim.DPSGD(params=model.parameters(), **training_parameters)  # Define training parameters.

# tf_bert_tokenizer = tf_tokenizer()
# epsilon = analysis.epsilon(**training_parameters)
criterion = nn.CrossEntropyLoss()

# run the model for each seed.
for seed in seeds:
    log_param("Seed", seed)
    log_param("Start epoch", load_epochs)
    log_param("End epoch", epochs)
    log_param("Optimizer", optimizer_name)
    log_param("Learning rate", learning_rate)
    log_param("Private", make_private)
    log_param("Sequence length", sequence_max_length)

    print("Seed: ", seed)
    # First get the split of the training and test set.
    if data_name == 'table':
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
        train_tokens = bert_tokenize(sentence_batch, bert_tokenizer)
        # train_tokens = tf_bert_tokenize(sentence_batch, tf_bert_tokenizer, max_len=sequence_max_length)
        # print("Train tokens: ", train_tokens)
        # print("Type train tokens: ", type(train_tokens))
        all_train_tokens.extend(train_tokens)
        # Get bert embeddings for the data.
    print("Training data encoding complete.", flush=True)
    
    print("Creating custom dataset", flush=True)
    train_dataset = SemevalDataset(all_train_tokens, y_train_subset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    if make_private:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
            poisson_sampling=False
        )
        if load_epochs > 0:
            print(f"Load model from {model_load_path}")
            checkpoint = torch.load(model_load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.train()  # This is important

        log_param("Noise multiplier", NOISE_MULTIPLIER)
        log_param("Max Grad Norm", MAX_GRAD_NORM)
    
    print("Starting model training.", flush=True)
    epoch_losses = []
    for epoch in range(load_epochs, epochs):
        print("Epoch: ", epoch)
        running_loss = 0.0
        for batch_index, data in enumerate(train_dataloader):
            inputs, batch_y_train_classes = data
            # print("Inputs from private dataloader: ", inputs)
            inputs_size = inputs['input_ids'].size(0)
            # print("Inputs batch size", inputs_size)
            inputs = reformat(inputs, inputs_size)  # Reformat data for the custom dataset.
            last_hidden_states_train = get_bert_embeds_from_tokens(bert_model, inputs)

            inputs_tensor = torch.Tensor(last_hidden_states_train)
            batch_labels_tensor = torch.Tensor(batch_y_train_classes)
            
            # Put the batched data on the gpu.
            inputs_tensor = inputs_tensor.to(device)
            batch_labels_tensor = batch_labels_tensor.type(torch.LongTensor)
            batch_labels_tensor = batch_labels_tensor.to(device)
            
            optimizer.zero_grad()

            # # Forward pass.
            outputs = model(inputs_tensor)            
            # print("outputs size: ", outputs.size())
            
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
            if batch_index % 100 == 0:
                batch_loss = training_loss / inputs_size
                print(f"Batch loss at batch {batch_index}: ", training_loss / inputs_size, flush=True)
                # log_metric("Batch losses", batch_loss)
        epoch_loss = running_loss / len(all_train_tokens)
        epoch_losses.append(epoch_loss)
        log_metric("Epoch loss", epoch_loss)
        if make_private:
            # Log the epsilon value.
            log_metric("Epsilon budget per epoch", privacy_engine.get_epsilon(DELTA))
        print(f"Epoch {epoch + 1} loss : ", epoch_loss, flush=True)
    print("All epoch losses: ", epoch_losses)
    print("Finished model training.")
    
    
    


    # ### Save model to disk.
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, model_save_path)
    

    # """Run predictions on the trained model."""
    
    print("Encoding test data.")
    all_test_tokens = []
    for batch in tqdm(range(0, len(X_test), batch_size)):
        sentence_batch = X_test[batch:batch+batch_size]
        
        # Tokenize the data.
        test_tokens = bert_tokenize(sentence_batch, bert_tokenizer)
        #test_tokens = tf_bert_tokenize(sentence_batch, tf_bert_tokenizer, max_len=sequence_max_length)
        all_test_tokens.extend(test_tokens)
        # Get bert embeddings for the data.
    print("Test data encoding complete.")

    test_dataset = TableDataset(all_test_tokens, y_test_classes)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Testing on test data.")
    all_predictions = []
    all_test_labels = []
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        for batch_index in enumerate(test_dataloader):
            test_inputs, batch_y_test_classes = data
            test_inputs_size = test_inputs['input_ids'].size(0)
            test_inputs = reformat(test_inputs, test_inputs_size)  # Reformat data for the custom dataset.
            last_hidden_states_test = get_bert_embeds_from_tokens(bert_model, test_inputs)
            
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
        
            all_predictions.extend(predicted.cpu().int().numpy())
            all_test_labels.extend(batch_labels_tensor_test.cpu().int().numpy())
        print("All predictions: ", all_predictions)
        print("All test labels: ", all_test_labels)
        # Calculate test accuracy and F1 here.
        f1 = f1_score(all_predictions, all_test_labels, average='macro')
        test_accuracy = 100 * correct / total
        print(f'Test accuracy for seed {seed}: {100 * correct / total} %')
        print(f"Test f1 is: {f1}")

    log_metric("Test Accuracy", test_accuracy)
    log_metric("Test F1", f1)