import torch
import torch.nn as nn
import random
import copy
from models.base_federated import BaseServerModel, BaseClientModel


class RAPBase(nn.Module):
    def __init__(self):
        super(RAPBase, self).__init__()
        self.refactored = False

    def forward(self, *args):
        if self.refactored:
            return self.forward_after_refactor(*args)
        else:
            return self.forward_before_refactor(*args)

    def forward_before_refactor(self, item_indices):
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)
        logits = self.affine_output(item_personality + item_commonality)
        rating = self.logistic(logits)
        return rating.view(-1), item_personality, item_commonality

    def forward_after_refactor(self, client_id, item_indices):
        item_personality_embedding = self.client_embeddings[client_id[0]](item_indices)
        item_commonality_embedding = self.item_commonality(item_indices)
        logits = self.affine_output(item_personality_embedding + item_commonality_embedding)
        rating = self.logistic(logits)
        return rating.view(-1), item_personality_embedding, item_commonality_embedding

    def get_embedding_dim(self):
        return self.item_commonality.weight.sum(dim=1).view(-1).shape[0]

    # is for adv
    def get_user_embedding(self, users):
        return self.item_personality.weight.sum(dim=1).view(-1)

    # is for attack
    def get_user_embedding_weight(self):
        return self.item_personality.weight.sum(dim=1).view(-1)

    def get_user_embedding_matrix(self):
        return self.client_embeddings

    def get_user_embedding_grad(self):
        return [self.item_personality.weight.grad]

class FedRAPServer(RAPBase, BaseServerModel):
    def __init__(self, num_items, latent_dim=32, global_lr=0.001, local_lr=0.001):
        RAPBase.__init__(self)
        BaseServerModel.__init__(self, global_lr)
        self.item_commonality = nn.Embedding(num_items, latent_dim)
        self.affine_output = nn.Linear(latent_dim, 1)
        self.serve_optimizer = torch.optim.Adam(self.affine_output.parameters(), lr=global_lr)
        self.item_optimizer = torch.optim.Adam(self.item_commonality.parameters(), lr=local_lr)
        self.init_serve_weights()

    def init_serve_weights(self):
        nn.init.normal_(self.item_commonality.weight, std=0.01)
        nn.init.kaiming_uniform_(self.affine_output.weight, a=1, nonlinearity='sigmoid')
        if self.affine_output.bias is not None:
            self.affine_output.bias.data.zero_()

    def get_server_modules(self):
        return nn.ModuleList([self.item_commonality, self.affine_output])

    def recieve_server_modules_grad(self, modules_grad_list, select_user_num):
        for serve_param, *client_grads in zip(self.get_server_modules().parameters(), *modules_grad_list):
            if serve_param.grad is None:
                serve_param.grad = torch.zeros_like(serve_param)
            serve_param.grad += sum(client_grads) / select_user_num

    def fit_serve_epoch(self):
        self.serve_optimizer.step()
        self.item_optimizer.step()
        self.serve_optimizer.zero_grad()
        self.item_optimizer.zero_grad()
        

class FedRAPClient(RAPBase, BaseClientModel):
    def __init__(self, user_interaction, local_epochs, local_lr, num_items, 
                 latent_dim=32, uid=torch.tensor(0), attr=None):
        RAPBase.__init__(self)
        BaseClientModel.__init__(self, local_epochs, local_lr)
        self.item_personality = nn.Embedding(num_items, latent_dim)
        self.logistic = nn.Sigmoid()
        self.client_optimizer = torch.optim.Adam(self.get_client_modules().parameters(), lr=self.local_lr)
        self.register_buffer("uid", uid)
        self.register_buffer("interaction", user_interaction)
        self.register_buffer("attr", attr)
        self.init_client_weights()

    def init_client_weights(self):
        nn.init.normal_(self.item_personality.weight, std=0.01)

    def sample_user_pos_neg(self, num_negatives=4):
        device = self.interaction.device
        user_interaction = self.interaction.bool().squeeze()
        pos_items = torch.nonzero(user_interaction).flatten().tolist()
        neg_items = torch.nonzero(~user_interaction).flatten().tolist()

        random.shuffle(pos_items)

        users, items, labels = [], [], []
        for pos_item in pos_items:
            users.append(0)
            items.append(pos_item)
            labels.append(1)
            for neg_item in random.sample(neg_items, num_negatives):
                users.append(0)
                items.append(neg_item)
                labels.append(0)

        return (
            torch.tensor(users).to(device),
            torch.tensor(items).to(device),
            torch.tensor(labels).float().to(device)
        )

    def calculate_loss(self, loss_list):
        loss_fn = nn.BCELoss()
        loss_independency = torch.nn.MSELoss()
        loss_reg = torch.nn.L1Loss()
        _, items, labels = self.sample_user_pos_neg()
        preds, item_personality, item_commonality = self.forward(items)
        dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
        third = loss_reg(item_commonality, dummy_target)
        loss_f = loss_fn(preds, labels)
        loss_indep = 0 * loss_independency(item_personality, item_commonality)
        loss_third = 0.1 * third
        total_loss = (loss_f - loss_indep + loss_third)

        if 'rap_total_loss' not in loss_list:
            loss_list['rap_total_loss'] = []
        if 'rec_loss' not in loss_list:
            loss_list['rec_loss'] = []
        if 'indep_loss' not in loss_list:
            loss_list['indep_loss'] = []
        if 'reg_loss' not in loss_list:
            loss_list['reg_loss'] = []

        loss_list['rap_total_loss'].append(total_loss.item())
        loss_list['rec_loss'].append(loss_f.item())
        loss_list['indep_loss'].append(loss_indep.item())
        loss_list['reg_loss'].append(loss_third.item())
        return total_loss

    def fit_client_epochs(self, loss_list):
        self.train()
        server_opt = torch.optim.Adam(self.get_server_modules().parameters(), lr=self.local_lr)

        for _ in range(self.local_epochs):
            self.client_optimizer.zero_grad()
            server_opt.zero_grad()
            loss = self.calculate_loss(loss_list)
            loss.backward()
            self.client_optimizer.step()
            server_opt.step()

        modules_grad_list = [p.grad.clone().detach() for p in self.get_server_modules().parameters()]
        self.del_server_modules()
        return modules_grad_list

    def recieve_server_modules(self, server_modules):
        self.item_commonality = copy.deepcopy(server_modules[0])
        self.affine_output = copy.deepcopy(server_modules[1])

    def get_server_modules(self):
        return nn.ModuleList([self.item_commonality, self.affine_output])

    def get_client_modules(self):
        return nn.ModuleList([self.item_personality])

    def refactoring_user_embeding(self, clients_list):
        weights = []

        for client in clients_list:
            weights.append(client.item_personality.weight.sum(dim=1).view(-1).unsqueeze(0))

        embedding_matrix = torch.cat(weights, dim=0)

        num_clients = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        user_embedding = nn.Embedding(num_clients, embedding_dim)
        user_embedding.weight.data = embedding_matrix.clone()
        return user_embedding

    def refactoring(self, serve_model: FedRAPServer, clients_list):
        self.refactored = True
        self.client_embeddings = nn.ModuleList()
        for client in clients_list:
            self.client_embeddings.append(client.item_personality)

        self.user_embedding = self.refactoring_user_embeding(clients_list)
        self.item_commonality = serve_model.item_commonality
        self.affine_output = serve_model.affine_output
        self.logistic = nn.Sigmoid()

    def del_server_modules(self):
        del self.item_commonality
        del self.affine_output
