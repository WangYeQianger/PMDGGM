# PMDGGM

## Welcome to PMDGGM
**PMDGGM (Prediction of miRNA-Disease Associations Based on Hybrid Gated GNN and Multi-Data Integration)** is a model designed for predicting miRNA-disease associations based on a hybrid gated Graph Neural Network (GNN) and multi-data integration. The model combines various types of similarity matrices and feature data to efficiently predict potential associations between miRNAs and diseases through a powerful GNN architecture.

The flow chart of PMDGGM is as follows:

![示例图片](https://gitee.com/Muchenxixi/my-mark-down_-tuchuang/raw/master/img/202411062350934.png)

## Directory Structure

```markdown
├── Datasets
│   ├── features
│   │   └── features.csv           # Features of miRNA and diseases
│   ├── samples					  
│   │   ├── samples.csv            # Sample edges used in the model
│   │   └── labels.csv             # labels for edges
│   ├── similarly
│   │   ├── D_gene.csv             # Disease gene similarity
│   │   ├── D_lnc.csv              # Disease lncRNA similarity
│   │   ├── D_mesh.csv             # Disease semantic similarity
│   │   ├── M_gene.csv             # miRNA gene similarity
│   │   ├── M_lnc.csv              # miRNA lncRNA similarity
│   ├── MS.csv                     # Integrated similarity for miRNA
│   ├── DS.csv                     # Integrated similarity for diseases
│   └── M-D.csv                    # Association relationship between miRNA and diseases
├── Result                         # Experiment logs
│   ├── xxxx_xxx                   # Log results of running at a certain time
│   │   ├── plot
│   │   │   ├── imgs               # Images
│   │   │   └── csv                # CSV files
│   │   ├── pred                   # Prediction results for each fold
│   └── └── parameters.txt         # Parameters and results
├── model.py                       # Model code
├── utils.py                       # Utility code
└── train.py                       # Training code

MS and DS are used for negative sample selection with RWR.
```
## Installation and Requirements

PMDGGM has been tested in a Python 3.9 environment. It is recommended to use the same library versions as specified. Using a conda virtual environment is also recommended to avoid affecting other parts of the system. Please follow the steps below.

Key libraries and versions:

```markdown
├── torch              1.13.1
├── torch-geometric    2.3.1
├── torch-scatter      2.1.1+pt113cu116
├── torch-sparse       0.6.15+pt113cu116
├── matplotlib         3.5.3
└── scikit-learn       1.0.2        
```

### Step 1: Download Code and Data

Use the following command to download this project or download the zip file from the "Code" section at the top right:

```bash
git init 
git clone https://github.com/WangYeQianger/PMDGGM.git
```

### Step 2: Run the Model

Run the main script in the virtual environment:

```bash
python train.py
```

All results of the operation will be saved in the Result directory.

## Citation 

If you use our tool and code, please cite our article and mark the project to show your support，thank you!

Citation format: Yeqiang Wang，Sharen Yun，Yuchen Zhang，Xiujuan Lei，Prediction of miRNA-Disease Associations Based on Hybrid Gated GNN and Multi-Data Integration，International Conference on Bioinformatics and Biomedicine (BIBM)，Lisbon, Portugal, December 3-6, 2024.

Link: [click here](https://your_article_link)