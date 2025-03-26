from kaggle.api.kaggle_api_extended import KaggleApi
import os

dataset_path = "E:/ModelLab/dataset"
os.makedirs(dataset_path, exist_ok=True)

api = KaggleApi()
api.authenticate()
api.dataset_download_files("drgfreeman/rockpaperscissors", path=dataset_path, unzip=True)

print("Dataset tải về thành công:", dataset_path)