import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from utils.model_utils import get_dataset_name, get_parameters
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False):
        if isinstance(model, dict):
            self.model = copy.deepcopy(model)
            self.model_name = model.keys()
        else:
            self.model = copy.deepcopy(model[0])
            self.model_name = model[1]
        # self.model = copy.deepcopy(model)
        # self.model_name = model.keys()
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.trainloader = DataLoader(train_data, self.batch_size, drop_last=False)
        self.testloader =  DataLoader(test_data, self.batch_size, drop_last=False)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        dataset_name = get_dataset_name(self.dataset)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        # those parameters are for personalized federated learning.
        # self.local_model = copy.deepcopy(list(get_parameters(self.model)))
        # self.personalized_model_bar = copy.deepcopy(list(get_parameters(self.model)))
        # if len(model) == 1:
        if isinstance(model, dict):
            # self.local_model = []
            # self.personalized_model_bar = []
            model_temp = get_parameters(self.model)
            self.local_model = [[] for _ in range(len(model_temp))]
            self.personalized_model_bar = [[] for _ in range(len(model_temp))]
            for i in range(len(model_temp)):
                self.local_model[i] = copy.deepcopy(model_temp[i])
                self.personalized_model_bar[i] = copy.deepcopy(model_temp[i])
                # self.local_model = copy.deepcopy(get_parameters(self.model))
            # self.personalized_model_bar = copy.deepcopy(list(get_parameters(self.model)))
        else:
            self.local_model = copy.deepcopy(list(self.model.parameters()))
            self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        self.init_loss_fn()
        if use_adam:
            if isinstance(model, dict):
                self.optimizer = torch.optim.Adam(
                    # params=self.model.parameters(),
                    params=get_parameters(self.model),
                    lr=self.learning_rate, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=1e-2, amsgrad=False)
            else:
                self.optimizer = torch.optim.Adam(
                    params=self.model.parameters(),
                    # params=get_parameters(self.model),
                    lr=self.learning_rate, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            if isinstance(model, dict):
                model_params = []
                for m in self.model:
                    model_params += self.model[m].parameters()
                self.optimizer = pFedIBOptimizer(model_params, lr=self.learning_rate)
                # self.optimizer = torch.optim.SGD(get_parameters(self.model), lr=self.learning_rate)
            else:
                self.optimizer = pFedIBOptimizer(self.model.parameters(), lr=self.learning_rate)
                # self.optimizer = torch.optim.SGD(self.model, lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}


    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.Bloss=nn.BCELoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def set_parameters(self, model,beta=1):
        # for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
        # if len(model) == 1:
        if isinstance(model, dict) or isinstance(model, list):
            for old_param, new_param, local_param in zip(get_parameters(self.model), get_parameters(model), self.local_model):
            # for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
                if beta == 1:
                    for i in range(len(new_param)):
                        old_param[i] = new_param[i].data.clone()
                        local_param[i] = new_param[i].data.clone()
                    # old_param.data = new_param.data.clone()
                    # local_param.data = new_param.data.clone()
                else:
                    old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()
                    local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()
        else:
            # for old_param, new_param, local_param in zip(get_parameters(self.model), get_parameters(model), self.local_model):
            for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
                if beta == 1:
                    old_param.data = new_param.data.clone()
                    local_param.data = new_param.data.clone()
                else:
                    old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()
                    local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()

    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()


    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()


    def clone_model_paramenter(self, param, clone_param):
        if isinstance(param, torch.Tensor):
            with torch.no_grad():
                for param, clone_param in zip(param, clone_param):
                    clone_param.data = param.data.clone()
        else:
            with torch.no_grad():
                for param, clone_param in zip(param, clone_param):
                    for i in range(len(param)):
                        clone_param[i].data = param[i].data.clone()
                    # clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params, keyword='all'):
        # for param , new_param in zip(self.model.parameters(), new_params):
        if isinstance(self.model, dict):
            for param, new_param in zip(get_parameters(self.model), new_params):
                for i in range(len(param)):
                    param[i].data = new_param[i].data.clone()
        else:
            for param, new_param in zip(self.model.parameters(), new_params):
                param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        # for param in self.model.parameters():
        if len(self.model) == 1:
            for param in get_parameters(self.model):
                if param.grad is None:
                    grads.append(torch.zeros_like(param.data))
                else:
                    grads.append(param.grad.data)
        else:
            for param in self.model.parameters():
                if param.grad is None:
                    grads.append(torch.zeros_like(param.data))
                else:
                    grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            # output = self.model(x)['output']
            output = self.model(x)
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, loss, y.shape[0]



    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0], loss


    def get_next_train_batch(self, count_labels=True):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        if count_labels:
            unique_y, counts=torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))