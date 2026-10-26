import time
import os
import glob
import sys
import pandas as pd

# Input files: pass the .csv names as arguments, otherwise convert every .csv in data/.
files = sys.argv[1:] or sorted(
    os.path.basename(path) for path in glob.glob(os.path.join(os.getcwd(), "data", "*.csv"))
)

def convert_to_parquet(file_list):
    parquet_paths = []

    for fname in file_list:
        FolderName = os.path.join(os.getcwd(), "data")
        os.makedirs(FolderName, exist_ok=True)

        csv_path = os.path.join(FolderName, fname)
        out_path = os.path.join(FolderName, fname.replace(".csv", ".parquet"))

        start = time.perf_counter()
        df = pd.read_csv(csv_path)
        end = time.perf_counter()

        print(f"Loaded {fname} in {end - start:.2f} seconds.")
        print(f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB in-memory before parquet")

        df.to_parquet(out_path, index=False)  # engine="pyarrow" se vuoi
        parquet_size = os.path.getsize(out_path) / (1024**2)

        print(f"{parquet_size:.2f} MB on disk after parquet: {os.path.basename(out_path)}")
        print(f"Columns: {list(df.columns)}\n")

        parquet_paths.append(out_path)

    return parquet_paths

df_parquet_list = []
df_parquet_list += convert_to_parquet(files)