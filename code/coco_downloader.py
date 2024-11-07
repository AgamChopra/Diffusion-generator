import os
import requests
import time
from zipfile import ZipFile

# URLs for downloading COCO data
urls = [
    'http://images.cocodataset.org/zips/train2017.zip',
    'http://images.cocodataset.org/zips/val2017.zip',
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
]

# Directory to save the files
save_dir = '/home/agam/Documents/git_projects/pytorch_datasets/coco/'

os.makedirs(save_dir, exist_ok=True)


def download_file(url, save_path):
    start_time = time.time()
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024  # 1 KB

    with open(save_path, 'wb') as file:
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                downloaded_size += len(chunk)
                elapsed_time = time.time() - start_time
                speed = downloaded_size / (1024 * elapsed_time)  # KB/s
                estimated_total_time = total_size / (speed * 1024)  # seconds

                # Print progress information
                print(f"\rDownloaded {downloaded_size / (1024 * 1024):.2f} MB "
                      f"of {total_size / (1024 * 1024):.2f} MB | "
                      f"Speed: {speed:.2f} KB/s | "
                      f"Elapsed: {elapsed_time:.2f}s | "
                      f"Estimated Total Time: {estimated_total_time:.2f}s", end="")

    print("\nDownload completed.")


for url in urls:
    file_name = os.path.join(save_dir, url.split('/')[-1])
    print(f"Starting download: {file_name}...")
    download_file(url, file_name)

    # Extract the downloaded files
    with ZipFile(file_name, 'r') as zip_ref:
        print(f"Extracting {file_name}...")
        zip_ref.extractall(save_dir)
        print(f"Extraction completed: {file_name}")
