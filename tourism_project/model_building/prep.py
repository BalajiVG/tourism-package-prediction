# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Initialize the Hugging Face API with the HF_TOKEN environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load the dataset directly from the Hugging Face dataset space
DATASET_PATH = "hf://datasets/BalajiVG/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Original shape: {df.shape}")

# ---- Data Cleaning ----
# 1. Drop unnecessary columns:
#    - 'Unnamed: 0' is a leftover index column with no information
#    - 'CustomerID' is a unique identifier and has no predictive power
df = df.drop(columns=['Unnamed: 0', 'CustomerID'])

# 2. Standardise inconsistent category values
#    - 'Fe Male' is a typo of 'Female' in the Gender column
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

#    - 'Unmarried' and 'Single' represent the same state — consolidate them
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

print(f"Shape after cleaning: {df.shape}")

# ---- Train/Test Split ----
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Stratified split to preserve the imbalanced target distribution
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Save locally ----
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train and test CSVs saved locally.")

# ---- Upload back to the Hugging Face dataset space ----
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="<HF_USERNAME>/tourism-package-prediction",
        repo_type="dataset",
    )
print("All split files uploaded to the Hugging Face dataset space.")
