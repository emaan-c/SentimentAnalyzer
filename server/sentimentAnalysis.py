import torch
import torch.nn as nn
from transformers import BertTokenizerFast, AutoModel


class BERT_architecture(nn.Module):

    def __init__(self, bert):
        super(BERT_architecture, self).__init__()
        self.bert = bert

        self.dropout = nn.Dropout(0.2) # Dropout layer to prevent overfitting
        self.relu = nn.ReLU() # Relu activation function
        self.fc1 = nn.Linear(768, 512) # Dense layer 1, where each input node is connected to an output node (768 input nodes and 512 output nodes), also nn.Linear applies transformation to the incoming data
        self.fc2 = nn.Linear(512, 2) # Dense layer 2
        self.softmax = nn.LogSoftmax(dim = 1) # Softmax activation function

    # Define the forward pass 
    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask = mask, return_dict = False) # Feed the input to the BERT model to obtain the hidden states
        x = self.fc1(cls_hs) # apply the first dense layer
        x = self.relu(x) # apply the relu activation function
        x = self.dropout(x) # apply the dropout layer
        x = self.fc2(x) # apply the second dense layer which will serve as the output layer
        x = self.softmax(x) # apply the softmax function to obtain the probabilities of the classes

        return x

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') # Load the tokenizer

bert = AutoModel.from_pretrained('bert-base-uncased') # load the pre-trained BERT model
model = BERT_architecture(bert) # load the pre-trained BERT model

model.load_state_dict(torch.load('bestWeights.pt')) # Load the model
model.eval() # Set the model to evaluation model

def preprocess(sentence):
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=17,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['attention_mask']

def predict(sentence):
    input_ids, attention_mask = preprocess(sentence)
    with torch.no_grad(): # Disable gradient calculation
        outputs = model(input_ids, attention_mask)
    logits = outputs.exp()
    prediction = torch.argmax(logits, dim = 1).item()
    return prediction

def interpret_prediction(prediction):
    return 'Positive' if prediction == 1 else 'Negative'

def sentiment_analysis(sentence):
    predictSentence = predict(sentence)
    return interpret_prediction(predictSentence)

# Test the function

# while True:
#     sentence = input("Enter a sentence (or type 'exit' to quit): ")
#     if sentence.lower() == 'exit':
#         break
#     print(f"Sentiment: {sentiment_analysis(sentence)}")