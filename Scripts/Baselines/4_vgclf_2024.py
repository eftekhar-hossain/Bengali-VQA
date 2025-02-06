import os
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import argparse
from collections import Counter
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
from torchvision.ops import MultiScaleRoIAlign,nms
import torch.nn.utils.rnn as rnn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
from transformers import AutoModel,AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel
from transformers import AutoTokenizer, BertModel
import clip

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

    # Initialize BERT and ViT
    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    bert_model = BertModel.from_pretrained("sagorsarker/bangla-bert-base")
    

    ## DataLoader
    class BVQA(Dataset):
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

    # # Data preprocessing and augmentation
    # data_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

        # train dataloader
    train_dataset = BVQA(dataframe = train, processor = image_processor, 
                         tokenizer = tokenizer, data_dir = image_dir,
                                max_seq_length=15)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # validation dataloader
    val_dataset = BVQA(dataframe = valid, processor = image_processor, 
                       tokenizer = tokenizer,data_dir = image_dir,
                                max_seq_length=15)
    val_loader = DataLoader(val_dataset, batch_size=32,shuffle=False)

    # test dataloader
    test_dataset = BVQA(dataframe = test, processor = image_processor, 
                        tokenizer = tokenizer,data_dir = image_dir, max_seq_length=15)
    
    test_loader = DataLoader(test_dataset, batch_size=32,shuffle=False)

    
        # Define the MultiheadAttention class
    class MultiheadAttention(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.1):
            super(MultiheadAttention, self).__init__()
            self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        def forward(self, query, key, value, mask=None):
            output, _ = self.attention(query, key, value, attn_mask=mask)
            return output



    # Define the model in PyTorch
    class MAF(nn.Module):
        def __init__(self, num_classes, num_heads):
            super(MAF, self).__init__()

            # Visual feature extractor (Vision Transformer)
            self.vit = vit_model

            # Textual feature extractor (BERT)
            self.bert = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")

            # Multihead attention
            self.attention = MultiheadAttention(d_model=768, nhead=num_heads)

            # Fully connected layers (updated for binary classification)
            self.fc = nn.Sequential(
                nn.Linear(768+768, 128),  # Input size includes visual features, BERT embeddings, and attention output
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )

        def forward(self, image_input, input_ids, attention_mask):

          
            # Extract visual features using Vision Transformer
            image_features = self.vit(image_input).last_hidden_state  #[batch_size, 197, 768]
            # image_features = image_features.unsqueeze(1)
            # Apply average pooling to reduce the sequence length to 15
            image_features = F.adaptive_avg_pool1d(image_features.permute(0, 2, 1), 15).permute(0, 2, 1)
            # Apply average pooling to reduce the sequence length to 15
            # print(image_features.shape)

            # Apply Attention on Image
            image_attention = self.attention(
                query=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                key=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                value=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                mask=None  # You can add a mask if needed
            )

            # Swap back the dimensions to (batch_size, seq_length, feature_size)
            image_attention_output = image_attention.permute(1, 0, 2)
            # print("Image Attention:",image_attention_output.shape)


            # Extract BERT embeddings
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_output = bert_outputs.last_hidden_state
            # print(bert_output.shape)


            # Apply Attention on Text
            text_attention = self.attention(
                query=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                key=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                value=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                mask=None  # You can add a mask if needed
            )

            # Swap back the dimensions to (batch_size, seq_length, feature_size)
            text_attention_output = text_attention.permute(1, 0, 2)
            # print("Text Attention:",text_attention_output.shape)


            # # Apply multihead attention between visual_features and BERT embeddings
            # # Assuming that visual_features and bert_output have shape (seq_length, batch_size, feature_size)
            attention_output = self.attention(
                query=image_attention_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                key=text_attention_output .permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                value=text_attention_output .permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                mask=None  # You can add a mask if needed
            )

            # Swap back the dimensions to (batch_size, seq_length, feature_size)
            attention_output = attention_output.permute(1, 0, 2)
            # print(attention_output.shape)

            # Concatenate the context vector, visual features, BERT embeddings, and attention output
            fusion_input = torch.cat([attention_output,  text_attention_output], dim=2)
            # print(fusion_input.shape)

            output = self.fc(fusion_input.mean(1))  # Pool over the sequence dimension
            # print(output.shape)
            return output
        
    # Create an instance of the model
    num_classes = 140  # Number of output classes
    num_heads = 4  # Number of attention heads for multihead attention
    model = MAF(num_classes, num_heads)
    model = model.to(device)    


        # Define a function to calculate accuracy
    def calculate_accuracy(predictions, targets):
        # For multi-class classification, you can use torch.argmax to get the predicted class
        predictions = torch.argmax(predictions, dim=1)
        correct = (predictions == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy



    # num_epochs = 1
    print("Learning Rate:",args.learning_rate)
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate , weight_decay = 0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Define learning rate scheduler
    num_epochs = args.epochs
    num_training_steps = num_epochs * len(train_loader)


    # Training loop
    patience = 5
    early_stopping_counter = 0
    best_val_accuracy = 0.0

    print("--------------------------------")
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

            torch.save(model.state_dict(), os.path.join(model_dir, 'vg_calf.pth'))
            print("Model Saved.")
        else:
            early_stopping_counter += 1  # Increment the counter if validation does not improve
            print(f"No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{patience}")

        # Stop training if early stopping criteria is met
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        # lr_scheduler.step()  # Update learning rate

    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
    print("Training is Done")
    print("--------------------------------")



    # Load the saved model
    # model = MultimodalModel(bert_model, resnet_model).to(device)
    print("Model is Loading..")
    model.load_state_dict(torch.load(os.path.join(model_dir, 'vg_calf.pth')))
    model.eval()
    print("Loaded.")

    test_labels = []
    test_preds = []

    print("--------------------------------")
    print("Start Evaluating..")
    # testing
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

    parser.add_argument('--learning_rate',dest="learning_rate", type=float, default = 1e-4,
                        help='Learning rate - default 1e-5')
                     
    args = parser.parse_args()
    main(args)
