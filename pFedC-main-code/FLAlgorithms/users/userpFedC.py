import torch
import numpy as np
import utils.losses
import copy
import logging
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from FLAlgorithms.users.userbase import User
from utils.model_utils import get_dataset_name, get_parameters
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers

class UserpFedC(User):
    def __init__(self,  args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

        self.personalized_model = copy.deepcopy(model)
        # self.local_params = copy.deepcopy(list(self.model.parameters()))
        # self.personalized_params = copy.deepcopy(list(self.model.parameters()))

    def get_label_mask(self):
        self.labels_index = np.unique(self.dataset['y'])
    # def update_label_counts(self, labels, counts):
    #     for label, count in zip(labels, counts):
    #         self.label_counts[int(label)] += count
    #
    # def clean_up_counts(self):
    #     del self.label_counts
    #     self.label_counts = {int(label): 1 for label in range(self.unique_labels)}

    def set_parameters(self, model, beta=1):
        for new_param, old_param in zip(get_parameters(model), get_parameters(self.model)):
            for i in range(len(old_param)):
                old_param[i] = new_param[i].data.clone()

        # for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
        #     if beta == 1:
        #         old_param.data = new_param.data.clone()
        #         local_param.data = new_param.data.clone()
        #     else:
        #         old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()
        #         local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()


    # need rewrite
    def test(self):
        # all_tasks = 10  # only for MNIST

        tot_loss = {}
        tot_loss['all'] = 0.0
        tot_test_acc = {}

        # for t in range(self.num_classes):
        for m in self.model:
            self.model[m].eval()

        for t in range(self.num_classes):
            tot_loss[t] = 0
            tot_test_acc[t] = 0

        for x, y in self.testloaderfull:
            rep = self.model['rep'](x)
            for t in range(self.num_classes):
                out_t = self.model[t](rep)
                loss_t = self.Bloss(out_t, y[:, :, t].to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(torch.float32))
                tot_loss['all'] += loss_t
                tot_loss[t] += loss_t
                t1 = out_t.reshape(-1).cpu().detach().numpy().round()
                t2 = y[:, :, t].reshape(-1).cpu().detach().numpy()
                tot_test_acc[t] += (t1 == t2).mean()

        return tot_test_acc, tot_loss, y.shape[0]

    def test_personalized_model(self):
        # test_acc = 0
        # loss = 0
        tot_loss = {}
        tot_loss['all'] = 0.0
        tot_test_acc = {}

        for m in self.model:
            self.model[m].eval()

        for t in range(self.num_classes):  # number of tasks
            tot_loss[t] = 0
            tot_test_acc[t] = 0

        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            # rep, _ = self.model['rep'](x, None)
            rep = self.model['rep'](x)
            for t in range(self.num_classes):
                # out_t, _ = self.model[t](rep, None)
                out_t = self.model[t](rep)
                loss_t = self.Bloss(out_t, y[:, :, t].to(
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(torch.float32))
                tot_loss['all'] += loss_t
                tot_loss[t] += loss_t
                tot_test_acc[t] += (out_t.reshape(-1).cpu().detach().numpy().round() == y[:, :, t].reshape(-1).cpu().detach().numpy()).mean()

        self.update_parameters(self.local_model)
        return tot_test_acc, y.shape[0], tot_loss


    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        # writer = SummaryWriter(log_dir='../../runs/exp_id_2021')
        # self.clean_up_counts()
        all_tasks = 10
        # for t in range(all_tasks):
        for m in self.model:
            self.model[m].train()

        n_iter = 0
        # loss_init = {}
        for epoch in range(1, self.local_epochs + 1):
            for x, y in self.trainloader:
                n_iter += 1

                # mask = None
                # masks = {}
                loss_data = {}
                grads = {}
                scale = {}

                # Compute gradients of each loss function wrt z
                # Gradient descent on task-specific parameters
                for t in range(all_tasks):
                    self.optimizer.zero_grad()
                    # out_t, masks[t] = self.model[t](rep, None)  # rep.detach() 保证rep不被freed
                    rep = self.model['rep'](x)
                    out_t = self.model[t](rep)  # rep.detach() 保证rep不被freed
                    loss_t = self.Bloss(out_t, y[:, :, t].to(
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(torch.float32))
                    loss_data[t] = loss_t.data.item()
                    loss_t.backward()  # retain_graph参数为True去保留中间参数从而两个loss的backward()不会相互影响。
                    # self.optimizer.step()  # self.plot_Celeb)
                    grads[t] = []
                    for param in self.model['rep'].parameters():
                        if param.grad is not None:
                            grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

                for i, t in enumerate(range(self.num_classes)):
                #    scale[t] = float(sol[i])
                    scale[t] = float(0.1)

                # Gradient descent on shared parameters
                # loss = 0
                self.optimizer.zero_grad()
                # rep, _ = self.model['rep'](x, mask)
                rep = self.model['rep'](x)
                for i, t in enumerate(range(all_tasks)):
                    # out_t, masks[t] = self.model[t](rep, masks[t])
                    out_t = self.model[t](rep)
                    loss_t = self.Bloss(out_t, y[:, :, t].to(
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(torch.float32))
                    loss_data[t] = loss_t
                    if i > 0:
                        loss = loss + scale[t] * loss_t
                    else:
                        loss = scale[t] * loss_t
                loss.backward()
                self.optimizer.step()

                # writer.add_scalar('training_loss', loss.data.item(), n_iter)
                # for t in range(all_tasks):
                #     writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)
                # print('GLobal Round {} Epoch {}/{} iter {} training_loss {}'.format(glob_iter, epoch, self.local_epochs, n_iter, loss.data.item()))
                logging.debug('GLobal Round {} Task_id {} Epoch {}/{} iter {} training_loss {}'.format(glob_iter, self.id, epoch, self.local_epochs, n_iter, loss.data.item()))

        if lr_decay:
            self.lr_scheduler.step(glob_iter)
