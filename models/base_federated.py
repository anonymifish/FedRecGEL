from abc import ABC, abstractmethod

class BaseServerModel(ABC):
    def __init__(self, global_lr):
        self.global_lr = global_lr
        self.serve_optimizer = None

    @abstractmethod
    def get_server_modules(self):
        """返回需要训练的模块列表"""
        pass

    @abstractmethod
    def init_serve_weights(self):
        """初始化服务器模块的权重"""
        pass

    @abstractmethod
    def recieve_server_modules_grad(self, modules_grad_list):
        """从客户端接收梯度"""
        pass

    @abstractmethod
    def fit_serve_epoch(self):
        """标准训练流程（不需要子类重复写）"""
        pass

class BaseClientModel(ABC):
    def __init__(self, local_epochs, local_lr):
        self.local_epochs = local_epochs
        self.local_lr = local_lr

    @abstractmethod
    def recieve_server_modules(self, serve_modules):
        """接收服务器下发的模型模块"""
        pass

    @abstractmethod
    def get_server_modules(self):
        """返回本地需要训练的模块"""
        pass

    @abstractmethod
    def calculate_loss(self):
        """计算本地训练时的 loss"""
        pass

    @abstractmethod
    def fit_client_epochs(self):
        """执行本地训练，并返回梯度信息"""
        pass

    @abstractmethod
    def del_server_modules(self):
        """清除临时 server 模块"""
        pass

    def refactoring(self, serve_model, clients_list):
        """可选：重构模型或合并交互数据等"""
        pass