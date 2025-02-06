import os
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import json
# import re,nltk,json
### ML Librarires--------------------
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import average_precision_score,roc_auc_score, roc_curve, precision_recall_curve
###-------------------------------------------
np.random.seed(42)
# import string, spacy,unicodedata, random
import warnings
warnings.filterwarnings('ignore')

####-----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
from transformers import AutoModel,AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel
from transformers import AutoTokenizer, BertModel


# Set the device to GPU if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


set_seed(42)


## Path Organization
current_dir = os.getcwd()

# Get the root directory by navigating up the directory tree
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  

# print(root_dir)

dataset_dir = os.path.join(root_dir,'Dataset')
# print(dataset_dir)

image_dir = os.path.join(dataset_dir,'images')
# print(image_dir)

if not os.path.exists(os.path.join(root_dir,'models')):
        os.makedirs(os.path.join(root_dir,'models'))

model_dir = os.path.join(root_dir,'models')



# dataset Fetching
train = pd.read_excel(os.path.join(dataset_dir,"train_bvqa.xlsx"))
valid = pd.read_excel(os.path.join(dataset_dir,"valid_bvqa.xlsx"))
test = pd.read_excel(os.path.join(dataset_dir,"test_bvqa.xlsx"))

# print("Training Samples:",len(train))
# print("Validation Samples:",len(valid))
# print("Testing Samples:",len(test))
# print("Total Samples: ",len(train) + len(valid) + len(test))




# Initialize BERT and ViT
tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
bert_model = BertModel.from_pretrained("sagorsarker/bangla-bert-base")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


## DataLoader
class BVQADataset(Dataset):
    def __init__(self, dataframe, processor, tokenizer, data_dir, max_seq_length):
        self.data = dataframe
        self.max_seq_length = max_seq_length
        self.data_dir = data_dir
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.loc[idx, 'filename'])
        image = Image.open(img_name)
        caption = self.data.loc[idx, 'questions']
        label = int(self.data.loc[idx, 'enc_answers'])


        # Tokenize the caption using BERT tokenizer
        q_inputs = self.tokenizer(caption, return_tensors='pt',
                                padding='max_length', truncation=True, max_length=self.max_seq_length)

        im_inputs = self.processor(image, return_tensors="pt")

        return {
            'image': im_inputs['pixel_values'].squeeze(),
            'input_ids': q_inputs['input_ids'].squeeze(),
            'attention_mask': q_inputs['attention_mask'].squeeze(),
            'label': label
        }

# Create data loaders
print("---------------------------")
print("Creating Data Loaders...")
# train dataloader
train_dataset = BVQADataset(dataframe = train, processor = image_processor, 
                            tokenizer = tokenizer, data_dir = image_dir,
                              max_seq_length=15)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# validation dataloader
val_dataset = BVQADataset(dataframe = valid, processor = image_processor, 
                          tokenizer = tokenizer,data_dir = image_dir,
                            max_seq_length=15)
val_loader = DataLoader(val_dataset, batch_size=16,shuffle=False)

# test dataloader
test_dataset = BVQADataset(dataframe = test, processor = image_processor, 
                           tokenizer = tokenizer,data_dir = image_dir,
                            max_seq_length=15)
test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)

print("---------------------------")

# Model Defination

# Define the MultiheadAttention class
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, query, key, value, mask=None):
        output, attn_weights = self.attention(query, key, value, attn_mask=mask)
        return output, attn_weights

# Define the Gated Fusion
class GatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFusion, self).__init__()
        # Linear layer to transform the concatenated vector
        self.gate = nn.Linear(feature_dim * 3, feature_dim * 3)
        # Sigmoid activation to create gates
        self.sigmoid = nn.Sigmoid()

    def forward(self, vec1, vec2, vec3):
        # Concatenate the vectors along the feature dimension
        combined = torch.cat([vec1, vec2, vec3], dim=-1)  # [batch_size, feature_dim * 3]
        # Apply the linear transformation and sigmoid activation
        gate = self.sigmoid(self.gate(combined))  # [batch_size, feature_dim * 3]
        # Element-wise multiplication to create the gated vector
        gated_vector = gate * combined
        # Split and sum the gated vector to maintain the original dimensionality
        feature_dim = vec1.size(-1)
        combined_vector = (gated_vector[:, :feature_dim] +
                           gated_vector[:, feature_dim:2*feature_dim] +
                           gated_vector[:, 2*feature_dim:])
        return combined_vector

    # Define the model in PyTorch
class MCRAN(nn.Module):
    def __init__(self, num_classes,n_layers,num_heads):
        super(MCRAN, self).__init__()

        # Visual feature extractor (Vision Transformer)
        self.vit = vit_model

        # Textual feature extractor (BERT)
        self.bert = bert_model

        # Multihead attention
        self.attention = MultiheadAttention(d_model=768, nhead=num_heads)

        # Transformer block
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=n_layers)
        self.fusion = GatedFusion(feature_dim=768)

        # Fully connected layer for classification
        self.fc = nn.Linear(768, num_classes)


    def forward(self, image_input, input_ids, attention_mask):


        # Extract visual features using Vision Transformer
        image_features = self.vit(image_input).last_hidden_state  #[batch_size, 197, 768]
        # image_features = image_features.unsqueeze(1)
        # Apply average pooling to reduce the sequence length to 15
        image_features_new = F.adaptive_avg_pool1d(image_features.permute(0, 2, 1), 15).permute(0, 2, 1)
        # print(image_features.shape)


        # Extract BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_outputs.last_hidden_state
        # print(bert_output.shape)

        # Apply multihead attention between visual_features and BERT embeddings
        # Assuming that visual_features and bert_output have shape (seq_length, batch_size, feature_size)
        attention_output1,_ = self.attention(
            query=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            key=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            value=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            mask=None  # You can add a mask if needed
        )

        # # # Swap back the dimensions to (batch_size, seq_length, feature_size)
        attention_output1 = attention_output1.permute(1, 0, 2)
        attention_output1 =  attention_output1.mean(1)
        # #print("attention1:", attention_output1.shape)

        # Assuming that visual_features and bert_output have shape (seq_length, batch_size, feature_size)
        attention_output2,_ = self.attention(
            query=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            key=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            value=image_features_new.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            mask=None  # You can add a mask if needed
        )


        # # Swap back the dimensions to (batch_size, seq_length, feature_size)
        attention_output2 = attention_output2.permute(1, 0, 2)
        attention_output2 =  attention_output2.mean(1)
        #print('attention2:',attention_output2.shape)

        # Concatenate the context vector, visual features, BERT embeddings, and attention output
        fusion_input = torch.cat([image_features, bert_output], dim=1)
        #print(fusion_input.shape)

        # Transpose fusion_input to shape [seq_len, batch_size, 768] for MultiheadAttention
        fusion_input = fusion_input.permute(1, 0, 2)  # [seq_len, batch_size, 768]

        # Apply multihead attention
        attn_output, attn_weights = self.attention(fusion_input, fusion_input, fusion_input)
        #print("Attention output shape:", attn_output.shape)
        #print("Attention weights shape:", attn_weights.shape)

        # Apply transformer encoder to the concatenated features
        transformer_output = self.transformer_encoder(attn_output)
        #print("Transformer output shape:", transformer_output.shape)

        # Pool over the sequence dimension and pass through fully connected layer
        pooled_output = transformer_output.mean(dim=0)  # [batch_size, 768]

        # Apply gated fusion
        gated_output = self.fusion(pooled_output,attention_output1, attention_output2)
        # print("Gated output shape:", gated_output.shape)
        output = self.fc(gated_output )
        # print(output.shape)

        return output, attn_weights

    def get_attention_map(self, image_input, input_ids, attention_mask):
        # Forward pass to get the attention weights
        _, attn_weights = self.forward(image_input, input_ids, attention_mask)
        return attn_weights


def main(args):

    # Create an instance of the model
    num_classes = 140  # Number of output classes
    n_layers = args.nlayer  # Number of transformer encoder layers
    num_heads = args.heads  # Number of attention heads for multihead attention

    model = MCRAN(num_classes, n_layers, num_heads)
    model = model.to(device)
    
    print("Training Parameters")
    print("-----------------------")
    print("Number of Transformer Layers: ",n_layers)
    print("Number of Attention Heads: ",num_heads)
    print("Learning Rate: ",args.learning_rate)
    print("Number of Epochs: ",args.epochs)

    ## Model Traininig 

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    # Define learning rate scheduler
    num_epochs = args.epochs
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Define a function to calculate accuracy
    def calculate_accuracy(predictions, targets):
        # For multi-class classification, you can use torch.argmax to get the predicted class
        predictions = torch.argmax(predictions, dim=1)
        correct = (predictions == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy


    # Training loop
    patience = 5
    early_stopping_counter = 0
    best_val_accuracy = 0.0
    print("--------------------------")
    print("MCRAN Training Start..")
    print("--------------------------")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0

        # Wrap the train_loader with tqdm for the progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
            for batch in t:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs,_ = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_accuracy += calculate_accuracy(outputs, labels).item()

                # Update the tqdm progress bar
                t.set_postfix(loss=total_loss / (t.n + 1), acc=total_accuracy / (t.n + 1))

        # Calculate training accuracy and loss
        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)

        # Validation loop
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation", unit="batch"):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs,_ = model(images, input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds)

        # Calculate validation accuracy and loss
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy * 100:.2f}%, Val Acc: {val_accuracy * 100:.2f}%")

        # Early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0  # Reset the counter if validation improves

            torch.save(model.state_dict(), os.path.join(model_dir, args.model_name))
            print("Model Saved.")
        else:
            early_stopping_counter += 1  # Increment the counter if validation does not improve
            print(f"No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{patience}")

        # Stop training if early stopping criteria is met
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        lr_scheduler.step()  # Update learning rate

    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")


    # Testing 
    print("--------------------------")
    print("Start Evaluating..")
    print("--------------------------")
    print("Model Loading")
    # Load the saved model
    model.load_state_dict(torch.load(os.path.join(model_dir,args.model_name)))
    model.eval()

    test_labels = []
    test_preds = []

    # Use tqdm for the test data
    with torch.no_grad(), tqdm(test_loader, desc="Testing", unit="batch") as t:
        for batch in t:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)

            outputs,_ = model(images, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds)

    # print Classification Report

    ## Mapping the labels
    ans = test['answers'].tolist()
    label = test['enc_answers'].tolist()
    a = zip(ans,label)
    # Remove duplicates by converting to a set, then back to a list
    unique_data = list(set(a))
    # Sort the unique data based on the second item in each tuple
    sorted_unique_data = sorted(unique_data, key=lambda x: x[1])

    labels = list()
    for i in sorted_unique_data:
        labels.append(i[0])

    def label_map(row):
        return labels[row]

    test['H2'] = test_preds
    test['H2'] = test['H2'].apply(label_map)

    yes_no = test[test['category']=='yes/no']
    counting = test[test['category']=='counting']
    other = test[test['category']=='other']

    # Category wise Performance
    print('Evaluation Done.')
    print("--------------------------------")
    print("Accuracies: ")
    print("Yes/No: ",round(accuracy_score(yes_no['answers'], yes_no['H2']),4)*100)
    print("Counting: ",round(accuracy_score(counting['answers'], counting['H2']),4)*100)
    print("Other: ",round(accuracy_score(other['answers'], other['H2']),4)*100)
    print("Overall: ",round(accuracy_score(test_labels,test_preds),4)*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCRAN')
    
    parser.add_argument('--nlayer',dest="nlayer", type=int, default = 1,
                        help='Number of Transformer Layers - default 1')
    parser.add_argument('--heads',dest="heads", type=int, default = 8,
                        help='Number of Heads - default 8')
    parser.add_argument('--learning_rate',dest="learning_rate", type=float, 
                        default = 1e-5, help='Learning rate - default 1e-5')
    parser.add_argument('--epochs',dest="epochs", type=int, default = 10,
                        help='Epochs - default 10')
    parser.add_argument('--model_name',dest="model_name", type=str, default = 'mcran.pth',
                        help='Saved Model Name')
                     
    args = parser.parse_args()
    main(args)