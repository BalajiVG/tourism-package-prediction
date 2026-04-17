from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",  # the local folder containing deployment files
    repo_id="BalajiVG/tourism-package-prediction",  # the target Space
    repo_type="space",                          # dataset, model, or space
    path_in_repo="",                            # upload directly to the root of the Space
)
print("Deployment files pushed to the Hugging Face Space successfully.")
