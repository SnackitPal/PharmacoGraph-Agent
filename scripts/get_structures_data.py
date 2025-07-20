# scripts/get_structures_data.py
import requests
from pathlib import Path

# Define the output directory and path
DATA_DIR = Path('../data')
OUTPUT_PATH = DATA_DIR / 'drugcentral_structures.tsv'

# URL for the structures file from the trusted Himmelstein repository
url = "https://raw.githubusercontent.com/dhimmel/SIDER4/master/data/drug-structures.tsv"

print(f"Downloading structures file from {url}...")
response = requests.get(url)
response.raise_for_status() # This will raise an error if the download fails

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(response.text)
    
print(f"✓ Success! File saved to {OUTPUT_PATH}")