import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


import sys
sys.path.append("/Users/simpleparadox/PycharmProjects/Private-RE/")

from functions import tf_tokenizer, tf_bert_tokenize, get_bert_embeds_from_tokens


class erin_model(nn.Module):
    def __init__(self, in_size=768, hidden_size: int = 1, num_relations: int = 29, sequence_length: int = 50,
                 private=False):
        super(erin_model, self).__init__()

        # Just add one LSTM unit as the model followed by a fully connected layer and then a softmax.
        # if private:
        #     self.lstm = dp_rnn.DPLSTM(input_size=in_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # else:
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(sequence_length * hidden_size, num_relations)

    def forward(self, x):
        # First get the bert embeddings.
        # Then do the forward pass.
        x, (h_n, c_n) = self.lstm(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.softmax(x, 1)
        return output

@st.cache
def load_model_and_tokenizer(private=False):
    # Load model
    if private:
        # Load private model
        pass
    else:
        # Load non-private model
        model = erin_model(sequence_length=50, private=private) # Using default model dimensions.
        model.to("cpu")
        print("model params")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        checkpoint = torch.load("/Users/simpleparadox/PycharmProjects/Private-RE/model_checkpoints/tabular_data/sgd/epoch_5_Adam_0.001_seed_2.pt", map_location=torch.device('cpu'))

        print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    tokenizer = tf_tokenizer()
    return model, tokenizer

def load_labels(private=False):
    # Load labels
    if private:
        # Load private labels
        pass
    else:
        # Load non-private labels
        labels = pd.read_csv("/Users/simpleparadox/PycharmProjects/Private-RE/data/tabular_data/labels.csv")



def predict(model, sentence, labels, tokenizer):
    test_tokens = tf_bert_tokenize(sentence, tokenizer, max_len=50)
    last_hidden_states_test = get_bert_embeds_from_tokens(model, test_tokens)
    inputs_tensor_test = torch.Tensor(last_hidden_states_test)
    test_outputs = model(inputs_tensor_test)
    _, predicted = torch.max(test_outputs.data, 1)
    # return labels.iloc[predicted.item(), 1]
    return predicted.item()



def main():

    st.title("Non-Private Model")
    sentence = st.text_input(label="Enter text on which RE model will be executed", placeholder="Enter text here")
    model, tokenizer = load_model_and_tokenizer()
    non_private_predict_value = st.button("Predict")
    if non_private_predict_value:
        st.write(predict(model, [sentence], [], tokenizer))


main()