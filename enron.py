import os
import tarfile
import urllib.request

# URL and file paths
url = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
file_name = "enron_mail_20150507.tar.gz"
extract_folder = "enron_mail"

# Check if the dataset is already downloaded
if not os.path.exists(file_name) and not os.path.exists(extract_folder):
    print("Dataset not found. Downloading...")

    # Download the file
    urllib.request.urlretrieve(url, file_name)
    print(f"Downloaded: {file_name}")

    # Extract the tar.gz file
    print("Extracting files...")
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=extract_folder)
    print(f"Extracted to folder: {extract_folder}")

    # Exit after download and extraction
    print("Download and extraction complete. Exiting script.")
    exit(0)
else:
    print("Dataset already exists. No action needed.")
