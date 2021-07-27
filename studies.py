from tqdm import tqdm
from os import system

# List of studies to perform
num_trials = 30
k_sup = [5, 10, 20, 35, 50, 75, 100]
k_que = [2*k for k in k_sup]

# Iterate and perform studies
for ks, kq in tqdm(zip(k_sup, k_que), total=len(k_sup), desc="Performing studies"):
    system(f"python fitting.py --num_trials {num_trials} --k_sup {ks} --k_que {kq} --log 1")