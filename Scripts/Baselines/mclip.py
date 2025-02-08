# imports
import torch
import numpy as np
import os
import random
import pandas as pd
import clip
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from multilingual_clip import pt_multilingual_clip
from transformers import AutoModel, AutoTokenizer, AdamW
import sys
import argparse
# import string, spacy,unicodedata, random
import warnings
warnings.filterwarnings('ignore')

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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(device)

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


    clip_imodel, preprocess = clip.load("ViT-B/32", device=device)
    tokenizer = AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
    clip_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')  
    # clip_tmodel= 'M-CLIP/XLM-Roberta-Large-Vit-L-14'

    # texts = train_data['Captions'][0]
    # image = train_data['image_name'][0]
    # image = Image.open(os.path.join(image_dir, image))
    # image = preprocess(image).unsqueeze(0).to(device)
    # with torch.no_grad():
    #     image_features = clip_imodel.encode_image(image)

    # print("Image features shape:", image_features.shape) 

    # embeddings = clip_text .forward(texts, tokenizer)
    # print("Embeddings shape: ",embeddings.shape)

    ## DataLoader
    class BVQADataset(Dataset):
        def __init__(self, dataframe, data_dir, transform = None):
            self.data = dataframe
            self.data_dir = data_dir
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, 'filename'])
            image = Image.open(img_name)
            question = self.data.loc[idx, 'questions']
            label = int(self.data.loc[idx, 'enc_answers'])

            if self.transform:
                 image = self.transform(image.convert("RGB"))

            return {
                'image': image,
                'text': question,
                'label': label
                }

    # Create data loaders

    # train dataloader
    train_dataset = BVQADataset(dataframe = train, data_dir = image_dir, transform = preprocess)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # validation dataloader
    val_dataset = BVQADataset(dataframe = valid,data_dir = image_dir, transform = preprocess)
    val_loader = DataLoader(val_dataset, batch_size=32,shuffle=False)

    # test dataloader
    test_dataset = BVQADataset(dataframe = test, data_dir = image_dir, transform = preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32,shuffle=False)



    # for batch in train_loader:
    #     image = batch['image']
    #     text = batch['text']
    #     label = batch['label']

    #     print(image.shape)
    #     #print(text.shape)
    #     print(label.shape)

    #     break

    # Freeze the parameters of the CLIP model
    for param in clip_imodel.parameters():
        param.requires_grad = False  


    class CLIPClassifier(nn.Module):
        def __init__(self, device='cpu') -> None:
            super(CLIPClassifier, self).__init__()
            self.device = device
            
            self.clip_image= clip_imodel # Changed JIT to True for just inference
            # output of clip is 512

            # cat image and text for 1024
            self.fc = nn.Sequential(
                nn.Linear(1280, 512),  # Input size includes visual features, BERT embeddings, and attention output
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 140),  # Updated for multiclass classification
            )
            
        def forward(self, image, text):
            image_features = self.clip_image.encode_image(image).float()
            # print(image_features.shape)
            text_features = text
            # print(text_features.shape)
            features = torch.cat((image_features, text_features), dim=1)
            # print(features.shape)

            x = self.fc(features)
            # print(x.shape)

            return x
        
    model = CLIPClassifier(device=device)
    model  = model.to(device)    


    # for batch in val_loader:
    #     image = batch['image'].to(device)
    #     text = batch['text']
    #     text_embed = clip_text.forward(text, tokenizer).to(device)
    #     # label = batch['label'].to(device)
    #     with torch.no_grad():
    #         features = model(image,text_embed)
    #     break    

        

    # Define a function to calculate accuracy
    def calculate_accuracy(predictions, targets):
        # For multi-class classification, you can use torch.argmax to get the predicted class
        predictions = torch.argmax(predictions, dim=1)
        correct = (predictions == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy



    # Create an instance of the model
    learning_rate = args.lr_rate
    num_epochs = 500
    momentum = 0.9

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # AdamW gives nan value
    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # Training loop
    best_val_accuracy = 0.0
    patience = 5
    early_stopping_counter = 0

    print(f"Start Training CLIP on BVQA")
    print("--------------------------------")

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        total_accuracy = 0

        # Wrap the train_loader with tqdm for the progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
            for batch in t:
                images = batch['image'].to(device)
                #print(images.shape)
                texts= batch['text']
                text_embed = clip_text.forward(texts, tokenizer).to(device)
                # print(text_embed.shape)
                labels = batch['label'].to(device)
                # print("Actual Labels: ",labels)

                optimizer.zero_grad()
                outputs = model(images, text_embed)
                # print("Model Output:",outputs)
                loss = criterion(outputs, labels)
                # print("Loss: ",loss)
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
                #print(images.shape)
                texts= batch['text']
                text_embed = clip_text.forward(texts,tokenizer).to(device)
                # print(text_embed.shape)
                labels = batch['label'].to(device)


                outputs = model(images, text_embed)
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

            torch.save(model.state_dict(), os.path.join(model_dir, f'mclip_bvqa_{args.lr_rate}.pth'))
            print("Model Saved.")
        else:
            early_stopping_counter += 1  # Increment the counter if validation does not improve
            print(f"No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{patience}")
    
        # Stop training if early stopping criteria is met
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
    print("--------------------------------")
    

    print("Model is Loading..")
    model.load_state_dict(torch.load(os.path.join(model_dir, f'mclip_bvqa_{args.lr_rate}.pth')))
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
            texts= batch['text']
            text_embed = clip_text.forward(texts,tokenizer).to(device)
            labels = batch['label'].to(device)

            outputs = model(images, text_embed)
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
    parser = argparse.ArgumentParser(description='Multilingual Clip')

    parser.add_argument('--lrate',dest="lr_rate", type=float, default = 1e-4,
                        help='Learning rate - default 5e-5')
                     
    args = parser.parse_args()
    main(args)