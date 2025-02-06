import os
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import argparse
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



## Path Organization
root_dir= os.getcwd()

while not root_dir.endswith("Bengali-VQA"):
    root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

# print(root_dir)

dataset_dir = os.path.join(root_dir,'Dataset')
# print(dataset_dir)

image_dir = os.path.join(dataset_dir,'images')
# print(image_dir)

if not os.path.exists(os.path.join(root_dir,'models')):
        os.makedirs(os.path.join(root_dir,'models'))

model_dir = os.path.join(root_dir,'models')


def main(args):


    # dataset Fetching
    train = pd.read_excel(os.path.join(dataset_dir,"train_bvqa.xlsx"))
    valid = pd.read_excel(os.path.join(dataset_dir,"valid_bvqa.xlsx"))
    test = pd.read_excel(os.path.join(dataset_dir,"test_bvqa.xlsx"))

    # print("Training Samples:",len(train))
    # print("Validation Samples:",len(valid))
    # print("Testing Samples:",len(test))
    # print("Total Samples: ",len(train) + len(valid) + len(test))



    # Initialize BERT and WideResNet
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-multilingual-cased")
    bert_model = BertModel.from_pretrained("distilbert/distilbert-base-multilingual-cased")


    ## DataLoader
    class BVQADataset(Dataset):
        def __init__(self, dataframe, tokenizer, data_dir, max_seq_length):
            self.data = dataframe
            self.max_seq_length = max_seq_length
            self.data_dir = data_dir
            self.tokenizer = tokenizer
            # Define image preprocessing steps
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resizing for ResNet input
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, 'filename'])
            image = Image.open(img_name).convert('RGB')  # Ensure the image is in RGB format
            image = self.image_transform(image)
            caption = self.data.loc[idx, 'questions']
            label = int(self.data.loc[idx, 'enc_answers'])


            # Tokenize the caption using BERT tokenizer
            q_inputs = self.tokenizer(caption, return_tensors='pt',
                                    padding='max_length', truncation=True, max_length=self.max_seq_length)

            return {
                'image': image,
                'input_ids': q_inputs['input_ids'].squeeze(),
                'attention_mask': q_inputs['attention_mask'].squeeze(),
                'label': label
            }

    # Create data loaders

    # train dataloader
    train_dataset = BVQADataset(dataframe = train,  tokenizer = tokenizer, data_dir = image_dir,
                                max_seq_length=15)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # validation dataloader
    val_dataset = BVQADataset(dataframe = valid, 
                            tokenizer = tokenizer,data_dir = image_dir,
                                max_seq_length=15)
    val_loader = DataLoader(val_dataset, batch_size=32,shuffle=False)

    # test dataloader
    test_dataset = BVQADataset(dataframe = test,
                            tokenizer = tokenizer,data_dir = image_dir,
                                max_seq_length=15)
    test_loader = DataLoader(test_dataset, batch_size=32,shuffle=False)



    class MultiHeadAttention(nn.Module):
        def __init__(self, input_dim, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model
            self.head_dim = d_model // num_heads
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

            self.query_proj = nn.Linear(input_dim, d_model)
            self.key_proj = nn.Linear(input_dim, d_model)
            self.value_proj = nn.Linear(input_dim, d_model)
            self.fc_out = nn.Linear(d_model, d_model)

        def forward(self, x):
            batch_size, seq_length, channels, height, width = x.shape
            spatial_size = height * width
            x = x.view(batch_size, seq_length, channels, spatial_size).permute(0, 1, 3, 2)
            x = x.reshape(batch_size, seq_length * spatial_size, channels)  # [B, S, C]

            # Linear projections
            Q = self.query_proj(x)  # [B, S, d_model]
            K = self.key_proj(x)
            V = self.value_proj(x)

            # Split into heads
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, head_dim]
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
            attention = torch.softmax(scores, dim=-1)  # [B, H, S, S]
            context = torch.matmul(attention, V)  # [B, H, S, head_dim]

            # Concatenate heads
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [B, S, d_model]

            # Final linear layer
            output = self.fc_out(context)  # [B, S, d_model]
            return output

    # Define the model in PyTorch
    class MedVQA(nn.Module):
        def __init__(self, num_classes):
            super(MedVQA, self).__init__()

            # Visual feature extractor (Vision Transformer)
            self.resnet = models.wide_resnet50_2(pretrained=True)
            # Extract everything up to the last convolutional block
            self.image_layer = nn.Sequential(*list(self.resnet.children())[:-2])

            # Textual feature extractor (BERT)
            self.bert = bert_model

            # Multimodal Multi-Head Attention Layer
            self.mha = MultiHeadAttention(input_dim=2048 + 768, d_model=512, num_heads=8)

            # # Fully connected layers (updated for binary classification)
            self.fc = nn.Sequential(
                nn.Linear(512, 32),  # Input size includes visual features, BERT embeddings, and attention output
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, num_classes),  # Updated for binary classification
            )


        def forward(self, image_input, input_ids, attention_mask):

            # Extract visual features using CLIP
            image_features = self.image_layer(image_input) 
            # print(image_features.shape)


            # Extract BERT embeddings
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_hidden_states = bert_outputs.last_hidden_state
            # print(bert_hidden_states.shape)

            # Ensure alignment of dimensions for concatenation
            batch_size, seq_length, hidden_size = bert_hidden_states.size()  # [batch_size, 15, 768]
            _, channels, height, width = image_features.size()  # [batch_size, 2048, 7, 7]

            # Expand BERT embeddings to match spatial dimensions (7x7)
            bert_hidden_states_expanded = bert_hidden_states.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, seq_length, hidden_size, 1, 1]
            bert_hidden_states_expanded = bert_hidden_states_expanded.expand(-1, -1, -1, height, width)  # Shape: [batch_size, seq_length, hidden_size, 7, 7]

            # Expand image features to match sequence length (15)
            image_features_expanded = image_features.unsqueeze(1).expand(-1, seq_length, -1, -1, -1)  # Shape: [batch_size, seq_length, 2048, 7, 7]
            # print(image_features_expanded.shape)
            # Concatenate along the channel dimension
            fusion_features = torch.cat([image_features_expanded, bert_hidden_states_expanded], dim=2)  # Shape: [batch_size, seq_length, 2048+768, 7, 7]
            # print(fusion_features.shape)

            # Apply multimodal multi-head attention
            attention_output = self.mha(fusion_features)  # Shape: [batch_size, S, d_model], where S = L * H * W
            # print(attention_output.shape)
            # Aggregate across the sequence length
            aggregated_output = attention_output.mean(dim=1)  # Shape: [batch_size, d_model]
            # print(aggregated_output.shape)

            # fusion_input = torch.cat([image_features,bert_output ], dim=1)
            # # print(fusion_input.shape)

            output = self.fc(aggregated_output)  # pass into classification layer
            # print( output.shape)
            return output

    # Create an instance of the model
    num_classes = 140  # Number of output classes
    model = MedVQA(num_classes)
    model = model.to(device)    



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
                outputs = model(images, input_ids, attention_mask)
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

                outputs = model(images, input_ids, attention_mask)
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

            torch.save(model.state_dict(), os.path.join(model_dir, 'med_vqa.pth'))
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
    print("Training is Done")


    print("Model Loading for Evaluation.")
    # Load the saved model
    model.load_state_dict(torch.load(os.path.join(model_dir,'med_vqa.pth')))
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

            outputs = model(images, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds)




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
    parser = argparse.ArgumentParser(description='Hausa-VQA')

    parser.add_argument('--epochs',dest="epochs", type=int, default = 10,
                        help='Epochs - default 10')

    parser.add_argument('--learning_rate',dest="learning_rate", type=float, default = 1e-5,
                        help='Learning rate - default 1e-5')
                     
    args = parser.parse_args()
    main(args)