# FILE: download_tdc_data.py

from tdc.single_pred import Tox  # SIDER lives under the 'Tox' loader

def main():
    print("Downloading SIDER side-effect data...")
    data = Tox(name='SIDER')         # select the SIDER dataset by name
    df = data.get_data()             # returns a pandas.DataFrame
    print("Downloaded {} records.".format(len(df)))
    # (optional) inspect the first few rows
    print(df.head())
    print("\n--- DOWNLOAD COMPLETE ---")
    print("IMPORTANT: Find the raw data file (sider.csv.gz) in a folder like C:/Users/Sheetal/.tdc/SIDER/")

if __name__ == '__main__':
    main()