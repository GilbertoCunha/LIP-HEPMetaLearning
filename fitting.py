from tqdm.notebook import tqdm
import numpy as np
import scipy.stats
import wandb


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def evaluate(model, tasks, eval_steps, desc="Eval Test"):
    accs, losses = [], []

    eval_bar = tqdm(tasks, desc=desc, total=len(tasks), leave=False)
    for task in eval_bar:
        # Create task bar and metric placeholders
        steps_bar = tqdm(range(eval_steps), desc=f"Evaluating task {task}", total=eval_steps, leave=False)
        task_accs, task_losses = [], []
        
        # Finetune the model and get loss and accuracy
        for _ in steps_bar:
            loss, acc = model.finetunning(tasks[task])
            task_accs.append(acc)
            task_losses.append(loss)
            
        # Calculate average loss and accuracies
        loss = sum(task_losses) / len(task_losses)
        acc = sum(task_accs) / len(task_accs)
        accs.append(acc)
        losses.append(loss)

    # Calculate average metrics across tasks
    acc = np.array(accs).astype(np.float32).mean(axis=0)
    loss = np.array(losses).astype(np.float32).mean(axis=0)

    return acc, loss

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
    best_val_loss = float("inf")
    patience = args["patience"]
    epoch_bar = tqdm(range(args["epochs"]), desc="Training", total=args["epochs"])
    for epoch in epoch_bar:
        # Create steps bar
        steps_bar = tqdm(range(args["epoch_steps"]), desc=f"Epoch {epoch}", total=args["epoch_steps"], leave=False)
        
        # Perform training for each task
        tr_accs, tr_losses = [], []
        for _ in steps_bar: 
            tr_loss, tr_acc = model(train_tasks)
            tr_accs.append(tr_acc)
            tr_losses.append(tr_loss)
            
        # Get training mean metrics
        tr_acc = np.array(tr_accs).astype(np.float32).mean(axis=0)
        tr_loss = np.array(tr_losses).astype(np.float32).mean(axis=0)

        # Get validation metrics
        val_acc, val_loss = evaluate(model, val_tasks, args["eval_steps"], "Eval Val")

        # Update Task tqdm bar
        metrics = {
            'tr acc': tr_acc,
            'val_acc': val_acc,
        }
        metrics['tr_loss'] = tr_loss
        metrics['val_loss'] = val_loss
        metrics['patience'] = patience
        wandb.log(metrics)

        # Update best metrics
        if val_loss < best_val_loss:
            best_val_loss = val_acc
            patience = args["patience"]
        else:
            patience -= 1

        # Update tqdm 
        epoch_bar.set_postfix(metrics)

        if patience == 0: break
        
    return best_val_loss
