# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from huggingface_hub import hf_hub_download

# Define constants for the dataset and output paths
repo_id = "ink85/tourism-package-prediction"
filename = "tourism.csv"

# Download the file locally (this will be from Hugging Face if already uploaded)
local_file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

df = pd.read_csv(local_file_path)
print("Dataset loaded successfully from local path.")
print("First 5 rows of the dataset:\n", df.head().to_string())   # ✅ Use print to actually output in .py script


df_null_summary = pd.concat([df.isnull().sum(), df.isnull().sum() * 100 /df.isnull().count()], axis = 1)
df_null_summary.columns = ['Null Record Count', 'Percentage of Null Records']
