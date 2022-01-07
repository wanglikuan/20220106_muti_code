# from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.users.userpFedC import UserpFedC
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, read_user_multidata, get_parameters
import numpy as np
import copy
import torch
import logging
# Implementation for FedAvg Server
import time
import torch.nn.functional as F

class pFedC(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        total_users = len(data[0])
        self.use_adam = 'adam' in self.algorithm.lower()
        self.model = copy.deepcopy(model)

        self.client_ws = [model for i in range(args.num_users)]
        self.client_us = [model for i in range(args.num_users)]

        self.model_name = model.keys()
        self.device = args.device

        self.labels_index = {}
        for i in range(20):
            if i < 4:  # i =0,1,2,3
                self.labels_index[i] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif 4 <= i < 8:  # i =4,5,6,7
                self.labels_index[i] = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
            elif 8 <= i < 12:  # i =8,9,10,11
                self.labels_index[i] = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
            elif 12 <= i < 16:  # i =12,13,14,15
                self.labels_index[i] = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
            else:
                self.labels_index[i] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

        print("Users in total: {}".format(total_users))

        for i in range(total_users):
            id, train_data , test_data = read_user_multidata(i, data, dataset=args.dataset)
            user = UserpFedC(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        #初始化-输入tensor P
        self.input_P = {}
        for i in range(args.num_classes):
            #self.input_P[i] = torch.rand(args.num_classes, 2, requires_grad=True)
        #    
            self.input_P[i] = torch.ones(total_users, 2, requires_grad=True)
            self.input_P[i] = self.input_P[i].float()/2
            self.input_P[i] = self.input_P[i].cuda()
        #
        #初始化-输出mask Y
        self.output_Y = {}
        for i in range(args.num_classes):
            self.output_Y[i] = torch.zeros(total_users, requires_grad=True)
        #
            
        print("Number of users / total users:",args.num_users, " / " ,total_users)
        print("Finished creating pFedC server.")

    #定义计算gumbel softmax
    def my_gumbel_softmax(self):
        for i in range(args.num_classes):
            for index in range(total_users):
                tmp_tensor = F.gumbel_softmax(self.input_P[i][index],tau=0.1,hard=True)
                self.output_Y[i][index] = tmp_tensor[0]

    #定义更新 mask; 
    def update_masks(self):
        for mu_index, mu in enumerate(self.selected_users):
            grad_lr = 0.01
            #当前client原始参数
            origin_state_list=[]
            #当前client变化参数
            delta_attention_theta_list = []

            for id, (param, local_param) in enumerate(zip(get_parameters(mu.personalized_model), get_parameters(mu.model))):
                if id == 0: #'rep'
                    continue
                else :
                    for j in range(len(param)): #当前client 当前id的分类器 所有model param
                        origin_state_list.append(param[j].data)
                        delta_attention_theta_list.append(param[j].data - local_param[j].data)

                    tmp_i_grads = torch.autograd.grad(origin_state_list, self.input_P[id-1], grad_outputs=delta_attention_theta_list, retain_graph=True, allow_unused=True)                    
                    self.input_P[id-1].sub_(tmp_i_grads*grad_lr)
                    self.input_P[id-1].grad.zero_()




    def evaluate(self, save=True, selected=False):
        all_tasks = 10
        # override evaluate function to multi-task log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)
        glob_acc = []
        glob_loss = []
        for i in range(self.num_classes):
            global_acc_t = 0
            global_loss_t = 0
            for u in range(self.num_users):
                # global_acc_t += test_accs[u][i].cpu().detach().numpy().tolist()
                global_acc_t += test_samples[u] * test_accs[u][i]
                global_loss_t += test_samples[u] * test_losses[u][i].cpu().detach().numpy().tolist()
            glob_acc.append(global_acc_t / np.sum(test_samples))
            glob_loss.append(global_loss_t / np.sum(test_samples))
        # glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        # glob_loss = np.sum([x * y.detach() for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        # print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
        # for i in range(all_tasks):
        # print("Average Global Accuracy and Loss for Each Task:")
        # print("Accuracy = {}".format(glob_acc))
        # print("Loss = {}".format(glob_loss))
        logging.info('Average Global Accuracy and Loss for Each Task:')
        logging.info("Accuracy = {}".format(glob_acc))
        logging.info("Loss = {}".format(glob_loss))

    def test(self, selected=False):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []

        # all_tasks = 10  # only for MNIST
        # tot_correct = {}
        # tot_losses = {}
        # tot_num_samples = {}
        # for t in range(all_tasks):
        #     tot_correct[t] = []
        #     tot_losses[t] = []
        #     tot_num_samples[t] = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test()
            tot_correct.append(ct)
            # tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses


    def test_personalized_model(self, selected=True):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, ns, loss = c.test_personalized_model()
            tot_correct.append(ct)
            num_samples.append(ns)
            losses.append(loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized_model(self, selected=True, save=True):
        stats = self.test_personalized_model(selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]

        glob_acc = []
        test_loss = []
        for i in range(self.num_classes):
            global_acc_t = 0
            test_loss_t = 0
            for u in range(self.num_users):
                global_acc_t += test_tot_correct[u][i] * test_num_samples[u]
                test_loss_t += test_num_samples[u] * test_losses[u][i].detach()
            glob_acc.append(global_acc_t / np.sum(test_num_samples))
            test_loss.append(test_loss_t / np.sum(test_num_samples))
        # glob_acc = np.sum(test_tot_correct)*1.0/np.sum(test_num_samples)
        # test_loss = np.sum([x * y for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        # print("Average Personalized Accuracy = {0}, Loss = {1}.".format(glob_acc, test_loss))
        print("Average Personalized Accuracy and Loss for Each Task:")
        print("Accuracy = {}".format(glob_acc))
        print("Loss = {}".format(test_loss))

    def personalized_aggregate_parameters(self, partial=False):  # personalized aggregation
        assert (self.selected_users is not None and len(self.selected_users) > 0)

        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples

        # each client will aggregate a personalized global model on the server
        for mu_index, mu in enumerate(self.selected_users):
            for param in get_parameters(mu.personalized_model):
                for i in range(len(param)):
                    param[i].data = torch.zeros_like(param[i].data)
            # for i, param in enumerate(get_parameters(mu.personalized_model)):
            #     flag = 0
            #     if i == 0:  #'rep'
            #         for mw_index, mw in enumerate(self.selected_users):
            #             local_param = get_parameters(mw.model)
            #             for j in range(len(local_param[0])):
            #                 param[j].data += local_param[0][j].data.clone() * (mw.train_samples / total_train)
            #
            #     elif i > 0 and self.labels_index[mu_index][i-1] == 1:
            #         for mw_index, mw in enumerate(self.selected_users):
            #             local_param = get_parameters(mw.model)
            #             if self.labels_index[mw_index][i-1] == 0:
            #                 continue
            #             else:
            #                 for j in range(len(local_param[i])):
            #                     param[j].data += self.labels_index[mw_index][i-1] * local_param[i][j].data.clone() * (mw.train_samples / total_train)
            #
            #     elif i > 0 and self.labels_index[mu_index][i-1] == 0:
            #         original_params = get_parameters(mu.model)
            #         for j in range(len(param)):
            #             param[j].data += original_params[i][j].data.clone()

            #-----------------origin----------------------------------------------------------------------------------------------------------------------
            # original_params = get_parameters(mu.model)
            # for mw_index, mw in enumerate(self.selected_users):
            #     for id, (param, local_param) in enumerate(zip(get_parameters(mu.personalized_model), get_parameters(mw.model))):
            #         if id == 0: #'rep'
            #             for j in range(len(param)):
            #                 param[j].data += local_param[j].data.clone() * (mw.train_samples / total_train)
            #         elif id > 0 and self.labels_index[mu_index][id-1] == 1:
            #             if self.labels_index[mw_index][id-1] == 0:
            #                 continue
            #             for j in range(len(param)):
            #                 param[j].data += self.labels_index[mw_index][id-1] * local_param[j].data.clone() * (mw.train_samples / total_train)
            #         elif id > 0 and self.labels_index[mu_index][id-1] == 0:
            #             for j in range(len(param)):
            #                 param[j].data += original_params[id][j].data.clone()
            #---------------------------------------------------------------------------------------------------------------------------------------------

            #--------learnable mask--output_Y为(class,client)---------------------------------------------------------------------------------------------------------------------
            original_params = get_parameters(mu.model)
            for mw_index, mw in enumerate(self.selected_users):
                for id, (param, local_param) in enumerate(zip(get_parameters(mu.personalized_model), get_parameters(mw.model))):
                    if id == 0: #'rep'
                        for j in range(len(param)):
                            param[j].data += local_param[j].data.clone() * (mw.train_samples / total_train)
                    elif id > 0 and self.output_Y[id-1][mu_index] == 1:
                        if self.output_Y[id-1][mw_index] == 0:
                            continue
                        for j in range(len(param)):
                            param[j].data += self.output_Y[id-1][mw_index] * local_param[j].data.clone() * (mw.train_samples / total_train)
                    elif id > 0 and self.output_Y[id-1][mu_index] == 0:
                        for j in range(len(param)):
                            param[j].data += original_params[id][j].data.clone()
            #---------------------------------------------------------------------------------------------------------------------------------------------

            # for mw_index, mw in enumerate(self.selected_users):
            #     flag = 0
            #     if mw == mu:
            #         ll = get_parameters((mu.personalized_model))
            #         for param, local_param in zip(get_parameters(mu.personalized_model), get_parameters(mw.model)):
            #             if len(param) == len(ll[0]): # rep
            #                 for j in range(len(param)):
            #                     param[j].data += local_param[j].data.clone() * (mw.train_samples / total_train)
            #             else:
            #                 if self.labels_index[mu_index][flag] == 0:
            #                     continue
            #                 # else:
            #
            #     else:
            #         ll = get_parameters(mu.personalized_model)
            #         for param, local_param in zip(get_parameters(mu.personalized_model), get_parameters(mw.model)):
            #             if len(param) == len(ll[0]): # rep
            #                 for j in range(len(param)):
            #                     param[j].data += local_param[j].data.clone() * (mw.train_samples / total_train)
            #                 #flag += 1
            #             else:  # task-specific parameters
            #                 if self.labels_index[mw_index][flag] == 0:
            #                     continue
            #                 for j in range(len(param)):
            #                     param[j].data += self.labels_index[mw_index][flag] \
            #                                      * local_param[j].data.clone() \
            #                                      * (mw.train_samples / total_train)
            #                 flag += 1
                #else:
                #    flag += 1

    def send_models(self):
        # users = self.users
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        users = self.selected_users
        for user in users:
            user.set_parameters(copy.deepcopy(user.personalized_model), beta=1)
            # for new_param, old_param in zip(user.local_model, user.personalized_model_bar):
            #     for i in range(len(old_param)):
            #         old_param[i] = new_param[i].data.clone()

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            # print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            logging.info("-------------Round number: {}-------------".format(glob_iter))
            # self.selected_users, self.user_idxs = self.select_users(glob_iter, self.num_users, return_idx=True)
            self.selected_users = self.select_users(glob_iter, self.num_users)
            # broadcast averaged prediction model
            logging.info("-------------Sending models-------------")
            self.send_models()
            logging.info("-------------Evaluating models-------------")
            self.evaluate()
            # self.evaluate_personalized_model()
            self.timestamp = time.time() # log user-training start time
            logging.info("-------------Training local models-------------")
            for user_index, user in enumerate(self.selected_users): # allow selected users to train
                print("client No.:",user_index)
                user.train(glob_iter, personalized=self.personalized) #* user.train_samples
            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)

            self.timestamp = time.time() # log server-agg start time
            
            #-----------update learnable mask----------------------------------------------------------
            logging.info("-------------updating local masks-------------")
            # if glob_iter != 0:
            #     self.update_masks()
            self.update_masks()
            self.my_gumbel_softmax()
            #--------------------------------------------------------------------------------

            logging.info("-------------Aggregating local models-------------")
            self.personalized_aggregate_parameters()  #!!!!!!!!!!!!
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            # self.metrics['server_agg_time'].append(agg_time)
        self.save_results(args)
        self.save_model()