from torch.nn import functional as F
from learner import Learner
from copy import deepcopy
from torch import optim
from torch import nn
import numpy as np
import torch


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, device):
        """

        :param args:
        """
        super(Meta, self).__init__()
        self.device = device
        self.meta_lr = args.meta_lr
        self.k_sup = args.k_sup
        self.k_que = args.k_que
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config)

        # Create learnable per parameter learning rate
        self.type = args.lr_type
        if self.type == "vector":
            self.update_lr = nn.ParameterList()
            for p in self.net.parameters():
                p_lr = args.update_lr * torch.ones_like(p)
                self.update_lr.append(nn.Parameter(p_lr))
            params = list(self.net.parameters()) + list(self.update_lr)
        elif self.type == "scalar":
            self.update_lr = nn.Parameter(torch.tensor(args.update_lr))
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
            fast_weights = list(map(lambda p: p[1] - p[2] * p[0], zip(grad, self.net.parameters(), self.update_lr)))
        elif self.type == "scalar":
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            
        return fast_weights

    def forward(self, tasks, save=[]):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        # Get torch device
        device = self.device

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

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
            w_sup = w_sup / w_sup.sum()
            w_que = w_que / w_que.sum()
            save.append(w_que)
            
            # 1. run the i-th task and compute loss for k=0
            y_pred = self.net(x_sup, vars=None, bn_training=True)
            loss = F.binary_cross_entropy(y_pred, y_sup, reduction="none", weight=sup_cweights)
            loss = (loss * w_sup).mean()
            
            # Loss and accuracy before first update
            with torch.no_grad():
                # Get Loss
                y_pred = self.net(x_que, self.net.parameters(), bn_training=True)
                loss_q = F.binary_cross_entropy(y_pred, y_que, reduction="none", weight=que_cweights)
                loss_q = (loss_q * w_que).mean()
                losses_q[0] += loss_q

                # Get accuracy
                pred_q = torch.round(y_pred)
                correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = self.get_fast_weights(grad)
            
            # Predict with fast weights
            y_pred = self.net(x_que, fast_weights, bn_training=True)
            loss_q = F.binary_cross_entropy(y_pred, y_que, reduction="none", weight=que_cweights)
            loss_q = (loss_q * w_que).mean()
            losses_q[1] += loss_q
            
            # Get accuracy
            with torch.no_grad():
                pred_q = torch.round(y_pred)
                correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # Get the support and query data for this task and normalize weights
                x_sup, w_sup, y_sup = next(tasks[task]["sup"]["data"])
                x_que, w_que, y_que = next(tasks[task]["que"]["data"])
                x_sup, w_sup, y_sup = x_sup.to(device), w_sup.to(device), y_sup.to(device)
                x_que, w_que, y_que = x_que.to(device), w_que.to(device), y_que.to(device)
                w_sup = w_sup / w_sup.sum()
                w_que = w_que / w_que.sum()
                
                # 1. run the i-th task and compute loss for k=1~K-1
                y_hat = self.net(x_sup, fast_weights, bn_training=True)
                loss = F.binary_cross_entropy(y_hat, y_sup, reduction="none", weight=sup_cweights)
                loss = (loss * w_que).mean()
                
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True, retain_graph=True)
                fast_weights = self.get_fast_weights(grad)

                y_hat = self.net(x_que, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.binary_cross_entropy(y_hat, y_que, reduction="none", weight=que_cweights)
                loss_q = (loss_q * w_que).mean()
                losses_q[k + 1] += loss_q

                # Get accuracy
                with torch.no_grad():
                    pred_q = torch.round(y_hat)
                    correct = (torch.eq(pred_q, y_que) * w_que).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / len(tasks)

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        # Get accuracy
        k_que = x_que.shape[0]
        accs = np.array(corrects) / (k_que * len(tasks))
        
        for i in range(1, len(save)):
            assert not torch.equal(save[0], save[i])

        return loss_q.item(), accs

    def finetunning(self, task):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        # Get torch device
        device = self.device
        
        # Get class weights for support and query data
        sup_cweights = task["sup"]["weights"]
        que_cweights = task["que"]["weights"]
        
        # Get the support and query data for this task and normalize weights
        x_sup, w_sup, y_sup = next(task["sup"]["data"])
        x_que, w_que, y_que = next(task["que"]["data"])
        x_sup, w_sup, y_sup = x_sup.to(device), w_sup.to(device), y_sup.to(device)
        x_que, w_que, y_que = x_que.to(device), w_que.to(device), y_que.to(device)
        w_sup = w_sup / w_sup.sum()
        w_que = w_que / w_que.sum()

        quesz = x_que.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        y_hat = net(x_sup)
        loss = F.binary_cross_entropy(y_hat, y_sup, reduction="none", weight=sup_cweights)
        loss = (loss * w_sup).mean(dim=-1)

        # Loss and accuracy before first update
        with torch.no_grad():
            y_hat_q = net(x_que, net.parameters(), bn_training=True)
            loss_q = F.binary_cross_entropy(y_hat_q, y_que, reduction="none", weight=que_cweights)
            loss_q = (loss_q * w_que).mean(dim=-1)
            
            pred_q = torch.round(y_hat_q)
            correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
            corrects[0] = corrects[0] + correct

        # Loss and accuracy after the first update
        if self.update_step_test > 0: # APPLY META
            grad = torch.autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True)
            fast_weights = self.get_fast_weights(grad)

            y_hat_q = net(x_que, fast_weights, bn_training=True)
            loss_q = F.binary_cross_entropy(y_hat_q, y_que, reduction="none", weight=que_cweights)
            loss_q = (loss_q * w_que).mean(dim=-1)
            
            with torch.no_grad():
                pred_q = torch.round(y_hat_q)
                correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step_test):
                # Get the support and query data for this task and normalize weights
                x_sup, w_sup, y_sup = next(task["sup"]["data"])
                x_que, w_que, y_que = next(task["que"]["data"])
                x_sup, w_sup, y_sup = x_sup.to(device), w_sup.to(device), y_sup.to(device)
                x_que, w_que, y_que = x_que.to(device), w_que.to(device), y_que.to(device)
                w_sup = w_sup / w_sup.sum()
                w_que = w_que / w_que.sum()
                
                # 1. run the i-th task and compute loss for k=1~K-1
                y_hat = net(x_sup, fast_weights, bn_training=True)
                loss = F.binary_cross_entropy(y_hat, y_sup, reduction="none", weight=sup_cweights)
                loss = (loss * w_sup).mean(dim=-1)
                
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = self.get_fast_weights(grad)

                # Get finetuned loss
                y_hat_q = net(x_que, fast_weights, bn_training=True)
                loss_q = F.binary_cross_entropy(y_hat_q, y_que, reduction="none", weight=que_cweights)
                loss_q = (loss_q * w_que).mean(dim=-1)

                # Get finetuned accuracy
                with torch.no_grad():
                    pred_q = torch.round(y_hat_q)
                    correct = (torch.eq(pred_q, y_que) * w_que).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / quesz

        return loss_q.item(), accs


def main():
    pass

if __name__ == '__main__':
    main()
