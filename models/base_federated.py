from abc import ABC, abstractmethod


class BaseServerModel(ABC):
    def __init__(self, global_lr):
        self.global_lr = global_lr
        self.serve_optimizer = None

    @abstractmethod
    def get_server_modules(self):
        pass

    @abstractmethod
    def init_serve_weights(self):
        pass

    @abstractmethod
    def recieve_server_modules_grad(self, modules_grad_list):
        pass

    @abstractmethod
    def fit_serve_epoch(self):
        pass


class BaseClientModel(ABC):
    def __init__(self, local_epochs, local_lr):
        self.local_epochs = local_epochs
        self.local_lr = local_lr

    @abstractmethod
    def recieve_server_modules(self, serve_modules):
        pass

    @abstractmethod
    def get_server_modules(self):
        pass

    @abstractmethod
    def calculate_loss(self):
        pass

    @abstractmethod
    def fit_client_epochs(self):
        pass

    @abstractmethod
    def del_server_modules(self):
        pass

    def refactoring(self, serve_model, clients_list):
        pass
