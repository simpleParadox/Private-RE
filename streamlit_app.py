import streamlit as st
import pandas as pd
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from opacus.layers import dp_rnn
from opacus.privacy_engine import PrivacyEngine
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset


# import sys
# sys.path.append("/Users/simpleparadox/PycharmProjects/Private-RE/")

from functions_st import tf_tokenizer, tf_bert_tokenize, get_bert_embeds_from_tokens, get_label_probs, get_reduced_label_mappings


class erin_model(nn.Module):
    def __init__(self, in_size=768, hidden_size: int = 1, num_relations: int = 29, sequence_length: int = 50,
                 private=False):
        super(erin_model, self).__init__()

        # Just add one LSTM unit as the model followed by a fully connected layer and then a softmax.
        if private:
            self.lstm = dp_rnn.DPLSTM(input_size=in_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        else:
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

@st.cache(ttl=15*3600)
def load_model(private=False, epsilon_value='0.5'):
    # Load model
    if private:
        # Load private model
        model = erin_model(sequence_length=50, private=True)  # Using default model dimensions.
        model.to('cpu')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_dataset = TensorDataset(torch.Tensor(np.random.random_sample((4000, 768))), torch.Tensor(np.random.random_sample((4000, 1))))
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_dataloader_poisson = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_epsilon=float(epsilon_value),
            target_delta=1/6590,
            epochs=5,
            max_grad_norm=1.0
        )
        seed_epsilon_mapping_best_model = {
            '0.5': '3',
            '1.0': '4',
            '5.0': '2',
            '10.0': '2',
            '20.0': '3',
            '30.0': '4',
            '40.0': '4'
        }
        model_seed = seed_epsilon_mapping_best_model[epsilon_value]
        checkpoint = torch.load(f"model_checkpoints/tabular_data/dpsgd/epoch_5_Adam_0.001_private_seed_{model_seed}_epsilon_{int(float(epsilon_value)*100.0)}.pt", map_location=torch.device('cpu'))
        # print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        # Load non-private model
        model = erin_model(sequence_length=50, private=private) # Using default model dimensions.
        model.to("cpu")
        checkpoint = torch.load("model_checkpoints/tabular_data/sgd/epoch_5_Adam_0.001_seed_2.pt", map_location=torch.device('cpu'))
        # print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    return model


@st.cache(max_entries=1)
def load_tokenizer():
    tokenizer = tf_tokenizer()
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, bert_model


def load_labels(data='tabular_data'):
    # Load labels
    if data == 'tabular_data':
        pass
    else:
        pass
        # Load non-private labels



def predict(model, sentence, tokenizer, bert_model):
    test_tokens = tf_bert_tokenize(sentence, tokenizer, max_len=50)
    last_hidden_states_test = get_bert_embeds_from_tokens(bert_model, test_tokens)
    inputs_tensor_test = torch.Tensor(last_hidden_states_test)
    test_outputs = model(inputs_tensor_test)
    _, predicted = torch.max(test_outputs.data, 1)
    probs = test_outputs.data.numpy()
    # print("Probs shape: ", probs.shape)
    # return labels.iloc[predicted.item(), 1]
    return predicted.item(), scipy.special.softmax(probs[0].tolist())



def main(model_type_selection, tokenizer, bert_model):


    # st.write(sentence)
    if model_type_selection == 'Non-Private':
        st.header("Non-Private Model")
        sentence = st.text_input(label="Enter text on which RE model will be executed", placeholder="Enter text here")
        predict_button_value = st.button("Predict")
        model = load_model(False)
        if predict_button_value:
            predicted_class, all_probs = predict(model, [sentence], tokenizer, bert_model)
            fig = get_label_probs(probs=all_probs, data='Table')
            label_mappings = get_reduced_label_mappings()
            st.write("Predicted class: ", label_mappings[predicted_class])
            st.pyplot(fig, clear_figure=True)
    else:
        epsilon_value = st.select_slider(label='Select Epsilon', options=['0.5', '1.0', '5.0', '10.0', '20.0', '30.0', '40.0'])
        st.header("Private Model")
        model = load_model(True, epsilon_value=epsilon_value)
        sentence = st.text_input(label="Enter text on which RE model will be executed", placeholder="Enter text here")
        predict_button_value = st.button("Predict")
        if predict_button_value:
            predicted_class, all_probs = predict(model, [sentence], tokenizer, bert_model)
            fig = get_label_probs(probs=all_probs, data='Table')
            label_mappings = get_reduced_label_mappings()
            st.write("Predicted class: ", label_mappings[predicted_class])
            st.pyplot(fig, clear_figure=True)



tokenizer, bert_model = load_tokenizer()
st.title("Tabular Dataset Relation Extraction")
st.subheader('Select model type')
model_type_selection = st.radio(label='Select model', options=['Non-Private', 'Private'])
main(model_type_selection, tokenizer, bert_model)

