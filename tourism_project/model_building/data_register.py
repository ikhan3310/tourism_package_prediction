from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
from huggingface_hub import login
from google.colab import userdata
import os

# -----------------------------
# Config
# -----------------------------
repo_id = "ink85/tourism-package-prediction"
repo_type = "dataset"

# -----------------------------
# Initialize API client
# Token is automatically picked from:
# - HF_TOKEN environment variable
# - or huggingface-cli login
# -----------------------------

# Log in to Hugging Face Hub (this will use the HF_TOKEN environment variable)
login(token=os.getenv("HF_TOKEN"))

# Initialize API client
api = HfApi()
# -----------------------------
# Check if dataset repo exists
# -----------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
