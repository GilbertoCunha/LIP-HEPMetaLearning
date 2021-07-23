from dataprocessing import generate_tasks
from meta import Meta
from tqdm import tqdm
import optuna as opt
import numpy as np
import scipy.stats
import argparse
import wandb
import torch
import os

def get_layer(in_features, out_features):
    return [
        ('bn', [in_features]),
        ('linear', [out_features, in_features]),
        ('leakyrelu', [1e-2, False])
    ]

def get_model(hidden_layers):
    # Initialize model
    config = []
    in_features = 69
    
    # Populate model according to hidden layers
    for out_features in hidden_layers:
        config += get_layer(in_features, out_features)
        in_features = out_features
        
    # Add final layer to the model
    config += [
        ('bn', [in_features]),
        ('linear', [1, in_features]),
        ('sigmoid', [])
    ]
    return config

def evaluate(model, tasks, eval_steps, desc="Eval Test"):
    accs, losses, rocs = [], [], []

    eval_bar = tqdm(tasks, desc=desc, total=len(tasks), leave=False)
    for task in eval_bar:
        # Create task bar and metric placeholders
        steps_bar = tqdm(range(eval_steps), desc=f"Evaluating task {task}", total=eval_steps, leave=False)
        task_accs, task_losses, task_rocs = [], [], []
        
        # Finetune the model and get loss and accuracy
        for _ in steps_bar:
            loss, acc, roc = model.finetunning(tasks[task])
            task_accs.append(acc)
            task_losses.append(loss)
            task_rocs.append(roc)
            
        # Calculate average loss and accuracies
        loss = sum(task_losses) / len(task_losses)
        acc = sum(task_accs) / len(task_accs)
        roc = sum(task_rocs) / len(task_rocs)
        accs.append(acc)
        losses.append(loss)
        rocs.append(roc)

    # Calculate average metrics across tasks
    acc = np.array(accs).astype(np.float32).mean(axis=0)
    loss = np.array(losses).astype(np.float32).mean(axis=0)
    roc = np.array(rocs).astype(np.float32).mean(axis=0)

    return acc, loss, roc

def fit(model, train_tasks, val_tasks, args):
    """
    Fits the input model to the training tasks and evaluates it in the validation tasks.
    This fit is done using Few-Shot Meta-learning using the Meta-SGD algorithm. 

    Args:
        model: The model to train
        train_tasks (dict): A dictionary of tasks to train on
        val_tasks (dict): A dictionary of tasks to validate on
        args (dict): A dictionary of arguments that dictates the training process

    Returns:
        float: The best validation loss achieved during the training process
    """

    # Start the training
    metrics = {}
    best_val_loss = float("inf")
    patience = args.patience
    epoch_bar = tqdm(range(args.epochs), desc=f"Training {model.name}", total=args.epochs, leave=False)
    for epoch in epoch_bar:
        # Create steps bar
        steps_bar = tqdm(range(args.epoch_steps), desc=f"Epoch {epoch}", total=args.epoch_steps, leave=False)
        
        # Perform training for each task
        tr_accs, tr_losses, tr_rocs = [], [], []
        for _ in steps_bar: 
            tr_loss, tr_acc, tr_roc = model(train_tasks)
            tr_accs.append(tr_acc)
            tr_losses.append(tr_loss)
            tr_rocs.append(tr_roc)
            
        # Get training mean metrics
        tr_acc = np.array(tr_accs).astype(np.float32).mean(axis=0)
        tr_loss = np.array(tr_losses).astype(np.float32).mean(axis=0)
        tr_roc = np.array(tr_rocs).astype(np.float32).mean(axis=0)

        # Get validation metrics
        val_acc, val_loss, val_roc = evaluate(model, val_tasks, args.val_steps, "Eval Val")

        # Update best metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = args.patience
        else:
            patience -= 1
            
        # Update Task tqdm bar
        metrics['tr_acc'] = tr_acc
        metrics['val_acc'] = val_acc
        metrics['tr_roc'] = tr_roc
        metrics['val_roc'] = val_roc
        metrics['tr_loss'] = tr_loss
        metrics['val_loss'] = val_loss
        metrics['patience'] = patience
        if args.log: wandb.log(metrics)

        # Update tqdm 
        epoch_bar.set_postfix(metrics)

        if patience == 0: break
        
    return best_val_loss

def objective(trial, train_signals, val_signals, bkg_file, args):
    # Manually seed torch and numpy for reproducible results
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Defining trial parameters
    sup_shots = trial.suggest_int("k_sup", 100, 100)
    que_shots = trial.suggest_int("k_que", 200, 200)
    num_layers = trial.suggest_int("num_hidden_layers", 1, 4)
    hidden_layers = []
    for i in range(num_layers):
        num_features = trial.suggest_int(f"num_features_layer_{i}", 20, 150)
        hidden_layers.append(num_features)

    # Generate tasks from the signal
    args.k_sup, args.k_que = sup_shots, que_shots
    train_tasks = generate_tasks(train_signals, bkg_file, sup_shots, que_shots)
    val_tasks = generate_tasks(val_signals, bkg_file, sup_shots, que_shots)
    
    # Define model name
    name = f"K{sup_shots}Q{que_shots}-HL{num_layers}-"
    for i in range(num_layers):
        features = hidden_layers[i]
        name += f"F{features}"

    # Choose PyTorch device and create the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Meta(name, args, get_model(hidden_layers), device).to(device)
    
    # Setup Weights and Biases logger, config hyperparams and watch model
    if args.log:
        wandb.init(name=name, project="Meta-HEP", config=args)
        wandb.watch(model)

    # Fit the model and return best loss
    return fit(model, train_tasks, val_tasks, args)

if __name__ == "__main__":
    # Define parser parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, help="number of configurations to try before stopping optuna search", default=30)
    parser.add_argument("--epochs", type=int, help="maximum number of epochs", default=1000)
    parser.add_argument("--epoch_steps", type=int, help="number of steps per epoch", default=200)
    parser.add_argument("--val_steps", type=int, help="number of steps per validation", default=100)
    parser.add_argument("--patience", type=int, help="number of steps the model has to improve before stopping", default=10)
    parser.add_argument("--meta_lr", type=float, help="exterior starting learning rate", default=1e-3)
    parser.add_argument("--inner_lr", type=float, help="interior starting learning rate", default=1e-2)
    parser.add_argument("--lr_type", type=str, help="type of interior learning rate: \"scalar\", \"vector\" or \"matrix\"", default="vector")
    parser.add_argument("--seed", type=int, help="seed for reproducible results", default=42)
    parser.add_argument("--log", type=int, help="flag for enabling or disabling wandb logging", default=1)
    args = parser.parse_args()
    
    # Datapath and background file for data-files
    datapath = "processed-data/"
    bkg_file = datapath + "bkg.h5"

    # Signal files for each task split
    train_signals = ["hg3000_hq1000", "hg3000_hq1400", "wohg_hq1200"]
    val_signals = ["hg3000_hq1200", "wohg_hq1000"]
    test_signals = ["wohg_hq1400", "fcnc"]

    # Add datapath and extention to files for each split
    train_signals = [datapath + p + ".h5" for p in train_signals]
    val_signals = [datapath + p + ".h5" for p in val_signals]
    test_signals = [datapath + p + ".h5" for p in test_signals]

    # Make weights and biases silent
    if args.log: os.environ["WANDB_SILENT"] = "true" 
    
    # Define and deploy optuna study
    study_name = "Fixed K-support and K-query optimization"
    study = opt.create_study(study_name=study_name, storage='sqlite:///meta-model.db', load_if_exists=True, direction="minimize")
    optimize = lambda trial: objective(trial, train_signals, val_signals, bkg_file, args)
    study.optimize(optimize, n_trials=args.num_trials)
