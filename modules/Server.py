import random

from modules.ClientCluster import ClientCluster


class Server():
    def __init__(self, serve_model, client_cluster: ClientCluster, fraction=0.1):
        self.serve_model = serve_model
        self.serve_model.init_serve_weights()
        self.client_cluster = client_cluster
        self.serve_modules = serve_model.get_server_modules()
        self.fraction = fraction
        self.progress = None

    def select_clients(self, clients, fraction=0.1):
        if fraction == 0:
            idx = random.sample(range(len(clients)), 1)
        else:
            idx = random.sample(range(len(clients)), int(fraction * len(clients)))
        return idx

    def send_serve_modules(self, c_list):
        self.client_cluster.receive_server_modules(self.serve_modules, c_list)

    def recieve_serve_modules_grad(self, c_list, select_user_num, loss_list):
        grad_moudules_list = self.client_cluster.send_serve_modules_grad(c_list, loss_list)
        self.serve_model.recieve_server_modules_grad(grad_moudules_list, select_user_num)

    def train_model(self, device, loss_list):
        clients_list = self.client_cluster.clients_list
        self.client_cluster.clients_list = clients_list.to(device)
        self.serve_model = self.serve_model.to(device)
        c_list = self.select_clients(clients_list, self.fraction)
        for i in c_list:
            self.send_serve_modules([i])
            self.recieve_serve_modules_grad([i], len(c_list), loss_list)
        self.serve_model.fit_serve_epoch()
