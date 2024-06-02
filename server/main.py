import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast, AdamW
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

df = pd.read_csv('sentiment_train.csv') # Load the dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split the dataset into three different datasets (training, validation, and testing)

firstSplit = train_test_split(df['sentence'], df['label'], random_state= 2021, test_size = 0.3, stratify = df['label']) # Use sklearn to split the dataset
train_text, temp_text, train_labels, temp_labels = firstSplit

secondSplit = train_test_split(temp_text, temp_labels, random_state= 2021, test_size = 0.5, stratify = temp_labels) # Use sklearn to split the dataset
val_text, test_text, val_labels, test_labels = secondSplit


bert = AutoModel.from_pretrained('bert-base-uncased') # load the pre-trained BERT model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') # Load the tokenizer

pad_length = 17 # average number of words in a sequence

tokensTrain = tokenizer.batch_encode_plus(train_text.tolist(), max_length = pad_length, pad_to_max_length = True, truncation = True) # Tokenize the training dataset
tokensVal = tokenizer.batch_encode_plus(val_text.tolist(), max_length = pad_length, pad_to_max_length = True, truncation = True) # Tokenize the validation dataset
tokensTest = tokenizer.batch_encode_plus(test_text.tolist(), max_length = pad_length, pad_to_max_length = True, truncation = True) # Tokenize the testing dataset

# Convert the training dataset to a tensor
train_seq = torch.tensor(tokensTrain['input_ids'])
train_mask = torch.tensor(tokensTrain['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# Convert the validation dataset to a tensor
val_seq = torch.tensor(tokensVal['input_ids'])
val_mask = torch.tensor(tokensVal['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# Convert the testing dataset to a tensor
test_seq = torch.tensor(tokensTest['input_ids'])
test_mask = torch.tensor(tokensTest['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

batch_size = 64 # Define the batch size
dataTrain = TensorDataset(train_seq, train_mask, train_y) # wrap the training tensors into a dataset
samplerTrain = RandomSampler(dataTrain) # Randomly sample the training dataset during training
dataLoaderTrain = DataLoader(dataTrain, sampler = samplerTrain, batch_size = batch_size) # Load the training dataset into a dataloader (a generator that helps to load the data in batches)
dataVal = TensorDataset(val_seq, val_mask, val_y) # wrap the validation tensors into a dataset
samplerVal = SequentialSampler(dataVal) # Sequentially sample the validation dataset during training
dataLoaderVal = DataLoader(dataVal, sampler = samplerVal, batch_size = batch_size) # Load the validation dataset into a dataloader (a generator that helps to load the data in batches)

# Define the model

for param in bert.parameters(): # Freeze the BERT model
    param.requires_grad = False 


# Define the model architecture to add layers on top of the BERT model

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
    

model = BERT_architecture(bert) # Pass the BERT model to our defined architecture
model = model.to(device) 
optimizer = AdamW(model.parameters(), lr = 1e-5) # learning rate

# Compute the class weights
class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(train_labels), y = train_labels)

weights = torch.tensor(class_weights, dtype = torch.float) # Convert the class weights to a tensor
weights = weights.to(device) # Push the class wieghts to the GPU
cross_entropy = nn.NLLLoss(weight = weights) # define the loss function, and add weights to handle the imbalance in the dataset
epochs = 10 # Number of training epochs

# Train the model/fine tune the BERT model
def train():
    model.train()
    totalLoss, totalAccuracy = 0, 0 # Initialize the loss and accuracy
    totalPreds = [] # Initialize the predictions (this is where the model's predictions will be stored)

    for step, batch in enumerate(dataLoaderTrain): # Iterate through the training dataset in batches

        # Print the progress update every 50 batches
        if step % 50 == 0 and not step == 0:
            print("batch: " + str(step) + " of  " + str(len(dataLoaderTrain)))

        store = []

        for r in batch: # Push the batch to the gpu
            store.append(r.to(device))

        sent_id, mask, labels = store # Unpack the batch

        model.zero_grad() # Clear the gradients that were calculated

        preds = model(sent_id, mask) # Get model predictions for the current batch
        loss = cross_entropy(preds, labels) # Compute the loss between the actual and predicted values
        totalLoss = totalLoss + loss.item() # Add on to the total loss
        
        loss.backward() # Backward pass to calculate the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip the gradients to prevent exploding gradients
        optimizer.step() # Update parameters

        preds = preds.detach().cpu().numpy() # model predictions are stored on GPU. So, push it to CPU
        totalPreds.append(preds) # Store the model predictions

    avgLoss = totalLoss / len(dataLoaderTrain) # Compute the training loss of the epoch
    totalPreds = np.concatenate(totalPreds, axis = 0) # Predictions are i the form of {no. of batches, size of batch, no. of classes}

    return avgLoss, totalPreds

def evaluate():
    print("\nEvaluating...")

    model.eval() # deactivate dropout layers

    totalLoss, totalAccuracy = 0, 0
    totalPreds = [] # Empty list to store the model's predictions

    for step, batch in enumerate(dataLoaderVal): # Iterate over batches

        if step % 50 == 0 and not step == 0: # Print progress every 50 batches
            print("batch: " + step + " of  " + len(dataLoaderVal)) # Print the progress

        store = [] 

        for t in batch: # Push the batch to the GPU`
            store.append(t.to(device))

        sent_id, mask, labels = store
        
        with torch.no_grad(): # Deactivate autograd

            preds = model(sent_id, mask) # Model predictions
            loss = cross_entropy(preds, labels) # Compute the validation loss between actual and predicted values
            totalLoss = totalLoss + loss.item() # Add on to the total loss
            preds = preds.detach().cpu().numpy() # Push the model predictions to the CPU
            totalPreds.append(preds)

    avgLoss = totalLoss / len(dataLoaderVal) # Compute the validation loss of the epoch
    totalPreds = np.concatenate(totalPreds, axis = 0) # Predictions are in the form of {no. of batches, size of batch, no. of classes}

    return avgLoss, totalPreds

best_valid_loss = float('inf') # Initialize the best validation loss
train_losses, valid_losses = [], [] # Initialize the training and validation losses of each epoch

for epoch in range(epochs): # Iterate over the epochs
    print('\n Epoch ' + str(epoch + 1) + ' / ' + str(epochs)) # Print the current epoch
    
    train_loss, _ = train() # Train the model
    valid_loss, _ = evaluate() # Evaluate the model

    if valid_loss < best_valid_loss: # Save the best model
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'bestWeights.pt') # Save the best model weights

    train_losses.append(train_loss) # Store the training loss of the epoch
    valid_losses.append(valid_loss) # Store the validation loss of the epoch

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

path = 'bestWeights.pt' # Load the best model weights
model.load_state_dict(torch.load(path)) # Load the best model weights

# Test the data

with torch.no_grad():
    temp1 = test_seq.to(device)
    temp2 =  test_mask.to(device)
    preds = model(temp1, temp2)
    preds = preds.detach().cpu().numpy()

pred = np.argmax(preds, axis = 1) # Get the class with the highest probability
print(classification_report(test_y, pred)) # Print the classification report