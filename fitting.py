from dataprocessing import generate_tasks
from meta import Meta
from tqdm import tqdm
import optuna as opt
import numpy as np
import argparse
import wandb
import torch
import os

def get_layer(in_features, out_features, dropout):
    """ Returns a linear layer of the model with 
    batch normalization and activation included.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        dropout (float): Dropout value for the layer

    Returns:
        Configuration of the layer
    """
    return [
        ('bn', [in_features]),
        ('linear', [out_features, in_features]),
        ('dropout', [dropout]),
        ('leakyrelu', [1e-2, False])
    ]

def get_model(hidden_layers, dropout):
    """ Generates the configuration of a model.

    Args:
        hidden_layers ([int]): Number of features per hidden layer
        dropout (float): Dropout value for the model

    Returns:
        Configuration to build the model
    """
    
    # Initialize model
    config = []
    in_features = 69

    # Populate model according to hidden layers
    for out_features in hidden_layers:
        config += get_layer(in_features, out_features, dropout)
        in_features = out_features

    # Add final layer to the model
    config += [
        ('bn', [in_features]),
        ('linear', [1, in_features]),
        ('sigmoid', [])
    ]
    return config

def get_model_name(k_sup, k_que, dropout, hidden_layers):
    """ Computes model name from main parameters

    Args:
        k_sup (int): Number of support samples per batch
        k_que (int): Number of query samples per batch
        dropout (float): Dropout value for the model
        hidden_layers ([int]): Number of features per hidden layer

    Returns:
        str: The name of the model
    """
    num_layers = len(hidden_layers)
    name = f"K{k_sup}Q{k_que}-HL{num_layers}-D{dropout:.2f}"
    for i in range(num_layers):
        features = hidden_layers[i]
        name += f"F{features}"
    return name


def evaluate(model, tasks, val_samples, desc="Eval Test"):
    """ Evaluates a meta-model on a set of tasks.

    Args:
        model (PyTorch model): The PyTorch model to evaluate on.
        tasks (dict): The dictionary of tasks to use.
        val_samples (int): Number of samples to use to perform evaluation per task.
        desc (str, optional): Description for tqdm bar. Defaults to "Eval Test".

    Returns:
        (float, float, float): accuracy, loss and roc of the model
    """
    accs, losses, rocs = [], [], []

    eval_bar = tqdm(tasks, desc=desc, total=len(tasks), leave=False)
    for task in eval_bar:
        # Create task bar and metric placeholders
        val_steps = val_samples // model.k_sup
        steps_bar = tqdm(range(val_steps), desc=f"Evaluating task {task}", total=val_steps, leave=False)
        task_accs, task_losses, task_rocs = [], [], []

        # Finetune the model and get loss and accuracy
        for _ in steps_bar:
            loss, acc, roc = model.evaluate(tasks[task])
            task_accs.append(acc)
            task_losses.append(loss)
            task_rocs.append(roc)

        # Calculate average loss and accuracies
        loss = sum(task_losses) / len(task_losses)
        acc = sum(task_accs) / len(task_accs)
        roc = sum(task_rocs) / len(task_rocs)
        
        # Append avg metrics to lists
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
    global current_best_model

    # Start the training
    metrics = {}
    best_val_roc = 0
    best_val_loss = float("inf")
    patience = args.patience
    epoch_bar = tqdm(range(args.epochs), desc=f"Training {model.name}", total=args.epochs, leave=False)
    for epoch in epoch_bar:
        # Create steps bar
        steps = args.epoch_samples // args.k_sup
        steps_bar = tqdm(range(steps), desc=f"Epoch {epoch}", total=steps, leave=False)

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
        val_acc, val_loss, val_roc = evaluate(model, val_tasks, args.val_samples, "Eval Val")

        # Update best metrics
        if val_loss < best_val_loss:
            # Update patience 
            patience = args.patience
            
            # Save best model weights
            if args.save_models:
                model.save_params("models/" + model.name + ".pt")
        else:
            patience -= 1
        if val_roc > best_val_roc:
            best_val_roc = roc
            current_best_model = model

        # Update Task tqdm bar
        metrics['tr_acc'] = tr_acc
        metrics['val_acc'] = val_acc
        metrics['tr_roc'] = tr_roc
        metrics['val_roc'] = val_roc
        metrics['tr_loss'] = tr_loss
        metrics['val_loss'] = val_loss
        metrics['patience'] = patience
        if args.log:
            wandb.log(metrics)

        # Update tqdm
        epoch_bar.set_postfix(metrics)

        if patience == 0:
            break

    return best_val_roc


def objective(trial, train_tasks, val_tasks, args):
    """
    Optuna objective function to optimize for best validation loss.
    """
    global best_model, current_best_model
    
    # Manually seed torch and numpy for reproducible results
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Defining trial parameters
    num_layers = trial.suggest_int("num_hidden_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.05, 0.2)
    hidden_layers = []
    for i in range(num_layers):
        num_features = trial.suggest_int(f"num_features_layer_{i}", 20, 150)
        hidden_layers.append(num_features)
    
    # Define model parameters
    name = get_model_name(args.k_sup, args.k_que, dropout, hidden_layers)
    config = get_model(hidden_layers, dropout)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create PyTorch model
    model = Meta(name, config, args.k_sup, args.k_que, device, args.meta_lr, args.lr_type, args.inner_lr).to(device)

    # Setup Weights and Biases logger, config hyperparams and watch model
    if args.log:
        wandb.init(name=name, project="Meta-HEP", config=args, reinit=True)
        wandb.watch(model)

    # Fit the model and return best loss
    roc = fit(model, train_tasks, val_tasks, args)
    
    # Change best model if loss is better
    if roc < best_model["roc"]:
        best_model["roc"] = roc
        best_model["model"] = current_best_model
    
    return loss


if __name__ == "__main__":
    # Define parser parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, help="number of configurations to try before stopping optuna search", default=50)
    parser.add_argument("--k_sup", type=int, help="number of data samples per support batch", default=100)
    parser.add_argument("--k_que", type=int, help="number of data samples per query batch", default=200)
    parser.add_argument("--epochs", type=int, help="maximum number of epochs", default=1000)
    parser.add_argument("--epoch_samples", type=int, help="number of training samples per epoch", default=1000)
    parser.add_argument("--val_samples", type=int, help="number of samples per validation", default=2500)
    parser.add_argument("--patience", type=int, help="number of steps the model has to improve before stopping", default=8)
    parser.add_argument("--meta_lr", type=float, help="exterior starting learning rate", default=1e-3)
    parser.add_argument("--inner_lr", type=float, help="interior starting learning rate", default=1e-2)
    parser.add_argument("--lr_type", type=str, help="type of interior learning rate: \"scalar\", \"vector\" or \"matrix\"", default="vector")
    parser.add_argument("--seed", type=int, help="seed for reproducible results", default=42)
    parser.add_argument("--log", type=int, help="flag for enabling or disabling wandb logging", default=1)
    parser.add_argument("--save_models", type=int, help="flag for saving best models", default=0)
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
    
    # Generate tasks from the signal
    train_tasks = generate_tasks(train_signals, bkg_file, args.k_sup, args.k_que)
    val_tasks = generate_tasks(val_signals, bkg_file, args.k_sup, args.k_que)

    # Make weights and biases silent
    if args.log: os.environ["WANDB_SILENT"] = "true"
    
    # Variable to hold best model
    best_model = {"roc": 0, "model": None}
    current_best_model = None

    # Define and perform optuna study
    study_name = f"K{args.k_sup}Q{args.k_que} optimization"
    study = opt.create_study(study_name=study_name, storage='sqlite:///meta-model.db', load_if_exists=True, direction="minimize")
    optimize = lambda trial: objective(trial, train_tasks, val_tasks, args)
    study.optimize(optimize, n_trials=args.num_trials)

    # Save model with the best trial
    filename = "models/" + best_model["model"].name + ".pt"
    best_model["model"].save(filename)
