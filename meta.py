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
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config, args.imgc, args.imgsz)

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

    def forward(self, tasks):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        # Iterate tasks
        for i, task in tasks:
            # Get the support and query data for this task
            x_sup, w_sup, y_sup = next(iter(tasks[task]["sup"]))
            x_que, w_que, y_que = next(iter(tasks[task]["query"]))
            
            # Normalize the weights
            w_sup = w_sup / w_sup.sum()
            w_que = w_que / w_que.sum()
            
            # 1. run the i-th task and compute loss for k=0
            y_pred = self.net(x_sup, vars=None, bn_training=True)
            loss = F.binary_cross_entropy(y_pred, y_sup, reduce="none")
            loss = (loss * w_sup).mean()
            
            # Loss and accuracy before first update
            with torch.no_grad():
                # Get Loss
                y_pred = self.net(x_que, self.net.parameters(), bn_training=True)
                loss_q = F.binary_cross_entropy(y_pred, y_que, reduce="none")
                loss_q = (loss_q * w_que).mean()
                losses_q[0] += loss_q

                # Get accuracy
                pred_q = y_pred.argmax(dim=1)
                correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = self.get_fast_weights(grad)
            
            # Predict with fast weights
            y_pred = self.net(x_que, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(y_pred, y_que)
            loss_q = (loss_q * w_que).mean()
            losses_q[1] += loss_q
            
            # Get accuracy
            with torch.no_grad():
                pred_q = y_pred.argmax(dim=1)
                correct = (torch.eq(pred_q, y_que) * w_que).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                y_hat = self.net(x_sup, fast_weights, bn_training=True)
                loss = F.binary_cross_entropy(y_hat, y_sup)
                loss = (loss * w_que).mean()
                
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True, retain_graph=True)
                fast_weights = self.get_fast_weights(grad)

                y_hat = self.net(x_que, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.binary_cross_entropy(y_hat, y_que)
                loss_q = (loss_q * w_que).mean()
                losses_q[k + 1] += loss_q

                # Get accuracy
                with torch.no_grad():
                    pred_q = y_hat.argmax(dim=1)
                    correct = (torch.eq(pred_q, y_que) * w_que).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return loss_q.item(), accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            loss_q = F.cross_entropy(logits_q, y_qry)
            if self.update_step_test == 0:
                loss_q = (loss_q * querysz + loss * x_spt.size(0)) / (querysz + x_spt.size(0))
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        if self.update_step_test > 0: # APPLY META
            grad = torch.autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True)
            fast_weights = self.get_fast_weights(grad)

            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            # [setsz]
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                # scalar
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(x_spt, fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = self.get_fast_weights(grad)

                logits_q = net(x_qry, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz

        return loss_q.item(), accs


def main():
    pass

if __name__ == '__main__':
    main()
