import os
from tqdm import tqdm
gen_paths = os.listdir("results/synteg/")
for i,d in tqdm(enumerate(gen_paths),total=len(gen_paths),desc="Task:"):
    os.system(f"python synteg2csv.py -L {os.path.join('results/synteg/',d)} -S results/synteg/")
    os.system(f"mv results/synteg/syntegDataset.csv.gz results/synteg/syntegDataset{i}.csv.gz")
    assert os.path.exists(f"results/synteg/syntegDataset{i}.csv.gz"), "Gen File not exist"