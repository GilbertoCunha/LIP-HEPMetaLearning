from torch.utils.data import DataLoader
from dataprocessing import generate_tasks
from meta import Meta
from tqdm import tqdm
import numpy as np
import scipy.stats
import argparse
import torch
import wandb


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def evaluate(model, tasks, desc="Eval Test"):
    all_accs, losses = [], []

    eval_bar = tqdm(tasks, desc=desc, total=len(tasks), leave=False)
    for task in eval_bar:
        # Finetune the model and get loss and accuracy
        loss, accs = model.finetunning(tasks[task])
        all_accs.append(accs)
        losses.append(loss)

    accs = list(map(lambda a: a[-1], all_accs))
    acc = np.array(accs).mean(axis=0).astype(np.float16)
    loss = np.array(losses).mean(axis=0).astype(np.float16)

    return acc, loss


def defineModel(args):
    # Add Model definition
    config = [
        ('bn', [69]),
        ('linear', [100, 69]),
        ('leakyrelu', [1e-2, False]),
        ('bn', [100]),
        ('linear', [120, 100]),
        ('leakyrelu', [1e-2, False]),
        ('bn', [120]),
        ('linear', [80, 120]),
        ('leakyrelu', [1e-2, False]),
        ('bn', [80]),
        ('linear', [50, 80]),
        ('leakyrelu', [1e-2, False]),
        ('bn', [50]),
        ('linear', [20, 50]),
        ('leakyrelu', [1e-2, False]),
        ('bn', [20]),
        ('linear', [1, 20]),
        ('sigmoid', [])
    ]
    return config


def main():
    # Manually seed torch and numpy for reproducible results
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Open csv file to write for metric logging
    try:
        f = open("results.csv", "w")
    except FileNotFoundError:
        f = open("results.csv", "x")
    f.write("Steps,tr_loss,tr_acc,val_loss,val_acc,te_loss,te_acc\n")

    # Choose PyTorch device and create the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Meta(args, defineModel(args), device).to(device)

    # Setup Weights and Biases logger, config hyperparams and watch model
    wandb.init(project="Meta-HEP")
    name = f"K{args.k_sup}Q{args.k_que}"
    wandb.run.name = name
    wandb.config.update(args)
    wandb.watch(model)
    print(f"RUN NAME: {name}")

    # Print additional information on the model
    if args.verbose:
        tmp = filter(lambda x: x.requires_grad, model.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print(args)
        print(model)
        print('Total trainable tensors:', num)

    # Create datasets
    datapath = "processed-data/"
    bkg_file = datapath + "bkg.h5"

    # Add datapath and extention to files for each split
    train_signals = [datapath + p + ".h5" for p in args.train_signals]
    val_signals = [datapath + p + ".h5" for p in args.val_signals]
    #test_signals = [datapath + p + ".h5" for p in args.test_signals]

    # Generate tasks
    train_tasks = generate_tasks(train_signals, bkg_file, args.k_sup, args.k_que)
    val_tasks = generate_tasks(val_signals, bkg_file, args.k_sup, args.k_que)
    #test_tasks = generate_tasks(test_signals, bkg_file, args.k_sup, args.k_que)

    # Start the training
    print("\nMeta-Training:")
    best_val_acc = 0
    early_stop = args.early_stop
    epoch_bar = tqdm(range(args.epochs), desc="Training", total=len(range(args.epochs)))
    
    for epoch in epoch_bar:
        # Create steps bar
        steps_bar = tqdm(range(args.epoch_steps), desc=f"Epoch {epoch}", total=args.epoch_steps, leave=False)
        for step in steps_bar:
            total_steps = args.epoch_steps * epoch + step + 1

            # Perform training for each task
            model(train_tasks)                             

        # Get evaluation metrics
        tr_acc, tr_loss = evaluate(model, train_tasks, "Eval Train")
        val_acc, val_loss = evaluate(model, val_tasks, "Eval Val")

        # Update Task tqdm bar
        metrics = {
            'tr acc': tr_acc,
            'val_acc': val_acc,
        }
        metrics['tr_loss'] = tr_loss
        metrics['val_loss'] = val_loss
        metrics['prune'] = early_stop
        wandb.log(metrics)
        f.write(f"{total_steps},{tr_loss},{tr_acc},{val_loss},{val_acc}\n")

        # Update best metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop = args.early_stop
        else:
            early_stop -= 1

        # Update tqdm 
        epoch_bar.set_postfix(metrics)

        if early_stop == 0: break

    f.close()


if __name__ == '__main__':
    # Argparse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, help='number of epochs', default=1000)
    argparser.add_argument('--epoch_steps', type=int, help='number of steps per epoch', default=1000)
    argparser.add_argument('--train_signals', nargs="+", type=str, help='signal files to be used in training', default=["hg3000_hq1000", "hg3000_hq1400", "wohg_hq1200"])
    argparser.add_argument('--val_signals', nargs="+", type=str, help='signal files to be used in validation', default=["hg3000_hq1200", "wohg_hq1000"])
    argparser.add_argument('--test_signals', nargs="+", type=str, help='signal files to be used in testing', default=["wohg_hq1400", "fcnc"])
    argparser.add_argument('--k_sup', type=int, help='k shot for support set', default=100)
    argparser.add_argument('--k_que', type=int, help='k shot for que set', default=15)
    argparser.add_argument('--lr_type', type=str, help='scalar, vector or matrix (for learning rate)', default="vector")
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=1)
    argparser.add_argument('--eval_steps', type=int, help='number of batches to iterate in test mode', default=200)
    argparser.add_argument('--save_summary_steps', type=int, help='frequence to log model evaluation metrics', default=250)
    argparser.add_argument('--early_stop', type=int, help='stop the training after this number of evaluations without accuracy increase', default=12)
    argparser.add_argument('--verbose', type=int, help='print additional information', default=0)
    argparser.add_argument('--seed', type=int, help='seed for reproducible results', default=42)

    args = argparser.parse_args()

    main()
