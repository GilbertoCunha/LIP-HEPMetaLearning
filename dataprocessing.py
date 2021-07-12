from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import torch

class MetaHEPDataset(Dataset):
    """
    A Dataset class for the processed .h5 files.
    Each entry of the dataset contains a tuple of (features, weight, label).
    """
    def __init__(self, signal_path, bkg_path, meta_split, same_sample=False):
        # NOTE: BKG HAS NO SAMPLE MATCHING FCNC
        
        # Load and store the data
        signal_df = pd.read_hdf(signal_path)
        bkg_df = pd.read_hdf(bkg_path)
        if same_sample:
            bkg_df = bkg_df[bkg_df["gen_sample"] == signal_df["gen_sample"][0]]
        df = pd.concat([signal_df, bkg_df], ignore_index=True)
        mask = df["gen_meta_split"] == meta_split
        self.df = df[mask].reset_index(drop=True)
        
        # Store weights
        self.weights = self.df["gen_xsec"] / self.df.shape[0]
        
        # Store labels
        self.df["gen_label"] = self.df["gen_label"].replace({"bkg": 0.0, "signal": 1.0})
        self.labels = self.df["gen_label"]
        
        # Drop gen columns of dataframe
        drop_cols = [col for col in self.df if "gen" in col]
        self.df = self.df.drop(columns=drop_cols)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        features = torch.tensor(self.df.loc[idx, :].values)
        weight = torch.tensor(self.weights.loc[idx])
        label = torch.tensor(self.labels.loc[idx])
        return features, weight, label


def generate_tasks(signal_files, bkg_file, same_sample, sup_shots, query_shots):
    """
    A function that generates a group of tasks (number of tasks equals number of signal files)
    The return of the function is a dictionary of tasks:
        - Keys are the task's filename
        - Values are a dictionary:
            - Keys are "sup" and "que", relative to the support and query data
            - Values are PyTorch dataloaders
    """
    # Create task dict
    tasks = {}
    
    # Create the different tasks in the dictionary
    for file in tqdm(signal_files, total=len(signal_files), desc="Populating tasks"):
        # Get filename of the corresponding file
        filename = file.split("/")[-1].split(".")[0]

        # Create support and query DataLoaders for the signal file
        supset = MetaHEPDataset(file, bkg_file, "sup", same_sample=same_sample)
        queryset = MetaHEPDataset(file, bkg_file, "query", same_sample=same_sample)
        suploader = DataLoader(supset, batch_size=sup_shots, shuffle=True)
        queryloader = DataLoader(queryset, batch_size=query_shots, shuffle=True)

        # Add the dataloaders to the dictionary
        tasks[filename] = {}
        tasks[filename]["sup"] = suploader
        tasks[filename]["query"] = queryloader
        
    return tasks