from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import torch

class MetaHEPDataset(Dataset):
    """
    A Dataset class for the processed .h5 files.
    Each entry of the dataset contains a tuple of (features, weight, label).
    """
    def __init__(self, signal_path, bkg_path, meta_split):
        # NOTE: BKG HAS NO SAMPLE MATCHING FCNC
        
        # Load and store the data
        signal = signal_path.split("/")[-1].split(".")[0]
        signal_df = pd.read_hdf(signal_path)
        bkg_df = pd.read_hdf(bkg_path)
        bkg_df = bkg_df[bkg_df["gen_task"] == signal]
        df = pd.concat([signal_df, bkg_df], ignore_index=True)
        mask = df["gen_meta_split"] == meta_split
        self.df = df[mask].reset_index(drop=True)
        
        # Store weights
        self.df["weights"] = self.df["gen_xsec"] / self.df.shape[0]
        self.weights = self.df["weights"]

        # Store labels
        self.df["gen_label"] = self.df["gen_label"].replace({"bkg": 0.0, "signal": 1.0})
        self.labels = self.df["gen_label"]
        
        # Calculate class weights
        bkg_wsum = self.df[self.df["gen_label"] == 0]["weights"].sum()
        signal_wsum = self.df[self.df["gen_label"] == 1]["weights"].sum()
        self.class_weights = torch.tensor([1, bkg_wsum / signal_wsum]).float()

        # Drop gen columns of dataframe
        drop_cols = [col for col in self.df if "gen" in col] + ["level_0", "index", "weights"]
        self.df = self.df.drop(columns=drop_cols)
        
    def __len__(self):
        return self.df.shape[0]

    def get_class_weights(self):
        return self.class_weights
    
    def __getitem__(self, idx):
        features = torch.tensor(self.df.loc[idx, :].values).float()
        weight = torch.tensor(self.weights.loc[idx]).float()
        label = torch.tensor(self.labels.loc[idx]).float()
        return features, weight, label

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def generate_tasks(signal_files, bkg_file, sup_shots, que_shots):
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
    for file in tqdm(signal_files, total=len(signal_files), desc="Populating tasks", leave=False):
        # Get filename of the corresponding file
        filename = file.split("/")[-1].split(".")[0]

        # Create support and query DataLoaders for the signal file
        sup_set = MetaHEPDataset(file, bkg_file, "sup")
        que_set = MetaHEPDataset(file, bkg_file, "query")
        sup_loader = DataLoader(sup_set, batch_size=sup_shots, shuffle=True)
        que_loader = DataLoader(que_set, batch_size=que_shots, shuffle=True)

        # Add the dataloaders to the dictionary
        tasks[filename] = {"sup": {}, "que": {}}
        tasks[filename]["sup"]["data"] = iter(cycle(sup_loader))
        tasks[filename]["sup"]["weights"] = sup_set.get_class_weights()
        tasks[filename]["que"]["data"] = iter(cycle(que_loader))
        tasks[filename]["que"]["weights"] = que_set.get_class_weights()
        
    return tasks
