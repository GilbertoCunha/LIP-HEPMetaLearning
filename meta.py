from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from learner import Learner
from copy import deepcopy
from torch import optim
from torch import nn
import numpy as np
import torch


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def get_class_weights(weights, labels):
    weights = [weights[int(y.item())] for y in labels]
    weights = torch.tensor(weights).float()
    return weights


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, name, config, k_sup, k_que, device, meta_lr=1e-3, lr_type="vector", inner_lr=1e-2):
        """

        :param args:
        """
        super(Meta, self).__init__()
        self.device = device
        self.name = name
        self.meta_lr = meta_lr
        self.k_sup = k_sup
        self.k_que = k_que

        self.net = Learner(config)

        # Create learnable per parameter learning rate
        self.type = lr_type
        if self.type == "vector":
            self.update_lr = nn.ParameterList()
            for p in self.net.parameters():
                p_lr = inner_lr * torch.ones_like(p)
                self.update_lr.append(nn.Parameter(p_lr))
            params = list(self.net.parameters()) + list(self.update_lr)
        elif self.type == "scalar":
            self.update_lr = nn.Parameter(torch.tensor(inner_lr))
            params = list(self.net.parameters())
            params += [self.update_lr]

        # Define outer optimizer (also optimize lr)
        self.meta_optim = optim.Adam(params, lr=self.meta_lr)

    @staticmethod
    def clip_grad_by_norm_(grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def get_fast_weights(self, grad):
        if self.type == "vector":
            fast_weights = list(map(
                lambda p: p[1] - p[2] * p[0], zip(grad, self.net.parameters(), self.update_lr)))
        elif self.type == "scalar":
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        return fast_weights

    def forward(self, tasks):

        # Get torch device
        device = self.device

        # List of losses and correct guesses for each fast weight update step
        losses_q = [0, 0]
        corrects = [0, 0]
        roc = 0

        # Iterate tasks
        for task in tasks:
            # Get support and query class Weights
            sup_cweights = tasks[task]["sup"]["weights"]
            que_cweights = tasks[task]["que"]["weights"]

            # Get the support and query data for this task and normalize weights
            x_sup, w_sup, y_sup = next(tasks[task]["sup"]["data"])
            x_que, w_que, y_que = next(tasks[task]["que"]["data"])
            x_sup, w_sup, y_sup = x_sup.to(device), w_sup.to(device), y_sup.to(device)
            x_que, w_que, y_que = x_que.to(device), w_que.to(device), y_que.to(device)
            w_sup = w_sup / w_sup.sum() * w_sup.shape[0]
            w_que = w_que / w_que.sum() * w_que.shape[0]

            # 1. run the i-th task and compute loss for k=0
            y_pred = self.net(x_sup, vars=None, bn_training=True)
            weights = get_class_weights(sup_cweights, y_sup).to(device)
            loss = F.binary_cross_entropy(y_pred, y_sup, reduction="none")
            loss = (loss * w_sup * weights).mean()

            # Get query loss and accuracy before fast weights
            with torch.no_grad():
                # Loss
                y_pred = self.net(x_que, self.net.parameters(), bn_training=True)
                weights = get_class_weights(que_cweights, y_que).to(device)
                loss_q = F.binary_cross_entropy(y_pred, y_que, reduction="none")
                loss_q = (loss_q * w_que * weights).mean()
                losses_q[0] += loss_q

                # Accuracy
                pred_q = torch.round(y_pred)
                correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
                corrects[0] = corrects[0] + correct

            # Get fast weights with inner optimizer
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = self.get_fast_weights(grad)

            # Predict with fast weights and get query loss
            y_pred = self.net(x_que, fast_weights, bn_training=True)
            weights = get_class_weights(que_cweights, y_que).to(device)
            loss_q = F.binary_cross_entropy(y_pred, y_que, reduction="none")
            loss_q = (loss_q * w_que * weights).mean()
            losses_q[1] += loss_q

            # Get query accuracy
            with torch.no_grad():
                pred_q = torch.round(y_pred)
                correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
                corrects[1] = corrects[1] + correct
                try:
                    roc += roc_auc_score(to_numpy(y_que), to_numpy(pred_q), sample_weight=to_numpy(w_que))
                except:
                    pass

        # Get the mean of the losses across tasks
        loss_q = losses_q[-1] / len(tasks)
        roc = roc / len(tasks)

        # Optimize model parameters according to query loss
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        # Get accuracy
        k_que = x_que.shape[0]
        accs = np.array(corrects) / (k_que * len(tasks))

        return loss_q.item(), accs[-1], roc

    def evaluate(self, task):
        # Get torch device and initialize accuracy placeholder
        device = self.device
        corrects = [0, 0]
        roc = 0

        # Get class weights for support and query data
        sup_cweights = task["sup"]["weights"]
        que_cweights = task["que"]["weights"]

        # Get the support and query data for this task and normalize weights
        x_sup, w_sup, y_sup = next(task["sup"]["data"])
        x_que, w_que, y_que = next(task["que"]["data"])
        x_sup, w_sup, y_sup = x_sup.to(device), w_sup.to(device), y_sup.to(device)
        x_que, w_que, y_que = x_que.to(device), w_que.to(device), y_que.to(device)
        w_sup = w_sup / w_sup.sum() * w_sup.shape[0]
        w_que = w_que / w_que.sum() * w_que.shape[0]

        # In order to not ruin the state of running_mean/variance and bn_weight/bias
        # We finetune on a copied model instead of the model itself
        net = deepcopy(self.net)

        # Get support loss for the task
        y_hat = net(x_sup)
        weights = get_class_weights(sup_cweights, y_sup).to(device)
        loss = F.binary_cross_entropy(y_hat, y_sup, reduction="none")
        loss = (loss * w_sup * weights).mean(dim=-1)

        # Loss and accuracy before first update
        with torch.no_grad():
            y_hat_q = net(x_que, net.parameters(), bn_training=True)
            weights = get_class_weights(que_cweights, y_que).to(device)
            loss_q = F.binary_cross_entropy(y_hat_q, y_que, reduction="none")
            loss_q = (loss_q * w_que * weights).mean(dim=-1)

            pred_q = torch.round(y_hat_q)
            correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
            corrects[0] = corrects[0] + correct

        # Inner optimizer to get fast weights
        grad = torch.autograd.grad(
            loss, net.parameters(), create_graph=True, retain_graph=True)
        fast_weights = self.get_fast_weights(grad)

        # Calculate query loss on fast weights
        y_hat_q = net(x_que, fast_weights, bn_training=True)
        weights = get_class_weights(que_cweights, y_que).to(device)
        loss_q = F.binary_cross_entropy(y_hat_q, y_que, reduction="none")
        loss_q = (loss_q * w_que * weights).mean(dim=-1)

        # Calculate query accuracy on fast weights
        with torch.no_grad():
            pred_q = torch.round(y_hat_q)
            correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
            corrects[1] = corrects[1] + correct
            try:
                roc += roc_auc_score(to_numpy(y_que), to_numpy(pred_q), sample_weight=to_numpy(w_que))
            except:
                pass

        del net

        # Calculate accuracies
        query_size = x_que.shape[0]
        accs = np.array(corrects) / query_size

        return loss_q.item(), accs[-1], roc
    
    def predict(self, x_sup, w_sup, y_sup, x_que, class_weights):
        # Get torch device
        device = self.device
        
        # Send tensors to device
        x_sup, w_sup, y_sup = x_sup.to(device), w_sup.to(device), y_sup.to(device)
        x_que, class_weights = x_que.to(device), class_weights.to(device)

        # Get loss
        y_pred = self.net(x_sup, vars=None, bn_training=False)
        loss = F.binary_cross_entropy(y_pred, y_sup, reduction="none")
        loss = (loss * w_sup * class_weights).mean()

        # Get fast weights with inner optimizer
        grad = torch.autograd.grad(loss, self.net.parameters())
        fast_weights = self.get_fast_weights(grad)

        # Predict with fast weights and get query loss
        y_pred = self.net(x_que, fast_weights, bn_training=False)

        return y_pred.detach().cpu()

    def save(self, file):
        params = {
            "vars": self.net.vars,
            "vars_bn": self.net.vars_bn,
            "inner_lr": self.update_lr
        }
        torch.save(params, file)

    def load(self, file):
        params = torch.load(file, map_location=self.device)
        self.net.vars = params["vars"]
        self.net.vars_bn = params["vars_bn"]
        self.update_lr = params["inner_lr"]


def main():
    pass


if __name__ == '__main__':
    main()
