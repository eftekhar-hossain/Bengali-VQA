# Open Ended Bengali Visual Question Answering

This repository is the official implementation of paper [BVQA: Connecting Language and Vision Through Multimodal Attention for Open-Ended Question Answering]()

## Pipeline of BVQA
<p align="center">
    <a><img src="Img/FIGURE 1.png" width="100%"></a> <br>
</p>
<p>
<em>Abstract view of the proposed <u>BVQA data generation method</u>. The left block shows the context, such as the caption used to prompt <strong>LLM</strong>, and the right block shows the responses (i.e., QA pairs) generated by LLM. Note that the visual image is not used to prompt LLM; it is only shown here as a reference.</em></p>

---

## Multimodal Cross-Attention Network for Bengali-VQA
<p align="center">
    <a><img src="Img/FIGURE 5.png" width="100%"></a> <br>
</p>
<p>
<em>Overall framework of our proposed Bengali VQA model, <u><strong>MCRAN</strong></u>. The model takes image and question as input and generates embeddings with their corresponding encoder. The top block in the middle part produces two cross-modal attentive representations: <strong>ICAR</strong> (Image Weighted Cross-modal Representation) and <strong>TCAR</strong> (Text Weighted Cross-modal Representation). In contrast, the bottom block creates a token-level multimodal attentive representation (<strong>MMAR</strong>). Finally, our method fuses these three attentive knowledge vectors through a gating mechanism to obtain richer multimodal features.</em></p>


## Instructions
- To reproduce the results, you need to  install `Python=3.10.x`. All the models are implemented using `Pytorch=2.4.0`. The `MCRAN` require GPU with 16GB RAM. 

- If you use any IDE, first clone (`git clone <url>`) the repository. Then, create a virtual environment and activate it.
    ```
    conda create -n Bengali-VQA Python=3.10.12
    conda activate Bengali-VQA
    ```
- Install all the dependencies.<br>
    ```
    pip install -r requirements.txt
    ```

### Dataset
The dataset can be downloaded from this link: [BVQA-Dataset](). The folder contains all the images and excel files for the train, validation, and test set. The excel file has the following columns:

- `filename`: image names
- `questions`: question for the image
- `answers` : answer for the corresponding question
- `enc_answers`: encoded version of answers
- `category`: category of the questions (i.e., **yes/no**, **counting**, and **other** )

You can also run the following command in the `conda-terminal`, to download the dataset. 
```
bash download_dataset.sh
```
Ensure you follow the given folder organization.

### Folder Organization

Folders need to organized as follows in `Bengali-VQA` directory.

```
├── Dataset
|   ├── Images
    |  ├── .jpg
    |  └── .png
|   └── .xlsx files
├── Scripts
   ├── Ablation  # Floder
   ├── Baselines # Folder
   └── mcran.py
└── requirements.txt           
```

### Traning and Evaluation of MCRAN

To run **MCRAN** on `BVQA` dataset run the following command. If you are not in the `Scripts` folder.

```
cd Scripts

python mcran.py \
  --nlayer 2 \
  --heads 6 \
  --learning_rate 1e-5 \
  --epochs 15 \
```

**Arguments**

- `--nlayer`: Specifies the number of transformer layers to use (`default: 1`).
- `--heads`: Sets the number of attention heads (`default: 8`).
- `--epochs`: Specifies the number of training epochs. (`default: 10`)
- `--learning_rate`: Sets the learning rate (`default: 1e-5`).
- `--model_name`: Specifies the saved model name (`default: mcran.pth`).

After evaluating, it will provide the accuracy in each question categories: **Yes/No**, **Number**, **Other**, and **All**.


### Ablation

`Ablation` folder contains the following scripts:

- `layer_head_ablation.py` ablation of number of transformer layers and attention heads.
- `img_encoder_ablation.py` ablation of different image encoders in the **MCRAN**. Tested with three encoders `ResNet`, `ConvNext`, `EfficientNet`.
- `txt_encoder_ablation.py` ablation of text encoder in the **MCRAN**. Only tested with `Multilingual-DistillBERT` model.

### Baselines

`Baselines` folder contains the following scripts:

- `initial_baselines.ipynb` contains the impelementation of baselines like **Vanilla VQA**, **MFB**, **MFH**, and **TDA**. All of them were implemented using TensorFlow framework.
- `hauvqa.py` implementation of the VQA model developed for Hausa language. [[Paper'23]](https://aclanthology.org/2023.findings-acl.646/)
- `medvqa.py` implemenation of an attention based model developed for medical VQA. [[Paper'23]](https://link.springer.com/article/10.1007/s11042-023-17162-3) 
- `vgclf.py` another attention based model developed for medical VQA. [[Paper'25]](https://www.sciencedirect.com/science/article/pii/S0925231224015017?casa_token=hpCJTR59XeMAAAAA:DiqnzktdaEWSMqI48dPdjL20yjPMZ8jxFyoNfpu83dZDikxzCBi0bPZ18sRuOiCu3_Rp-oKQTw) 
- `mclip.py` implemenation of a fine-tuned multilingual CLIP model. [[Paper'23]](https://aclanthology.org/2023.acl-long.728/)


### Case Study
<p align="center">
    <a><img src="Img/FIGURE 7.png" width="100%"></a> <br>
</p>
<p>
<em>Illustration of some case study from the
test set of BVQA where the proposed <strong>MCRAN</strong> performs well. First, we present the actual answer to a question followed by predicted answers of the state-of-the-art methods and our proposed method. The <span style = "color:red"><strong>red cross</strong></span> mark denotes a <u>false prediction</u>, while the <span style = "color:green"><strong>green tick </strong></span>mark is the <u>correct prediction</u>.</em></p>

# Citation