# scripts/get_sider_data.py

import pandas as pd
import requests
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# URLs for SIDER data from Daniel Himmelstein's repository
urls = {
    "sider_side_effects.tsv": "https://raw.githubusercontent.com/dhimmel/SIDER4/master/data/side-effects.tsv",
    "sider_side_effect_terms.tsv": "https://raw.githubusercontent.com/dhimmel/SIDER4/master/data/side-effect-terms.tsv",
    "sider_indications.tsv": "https://raw.githubusercontent.com/dhimmel/SIDER4/master/data/indications.tsv"
}

# Download files
for filename, url in urls.items():
    print(f"Downloading {filename}...")
    response = requests.get(url)
    response.raise_for_status()  # This will raise an error if the download fails
    
    filepath = data_dir / filename
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f"✓ Saved to {filepath}")

print("\n--- Download Complete ---")

# Load and examine the main data file to confirm it works
print("\nLoading main dataset for verification...")
side_effects = pd.read_csv(data_dir / "sider_side_effects.tsv", sep='\t')
print(f"\nSide effects dataset shape: {side_effects.shape}")
print(f"Columns: {list(side_effects.columns)}")
print("\nFirst 5 rows:")
print(side_effects.head())