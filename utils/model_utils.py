from collections import OrderedDict
import torch
import torch.nn as nn

class Precode(nn.Module):
    def __init__(self, dim, out_dim=2, beta=0.001, mu_std=0.1):
        super().__init__()
        self.encoder = nn.Linear(dim, out_dim * 2, bias=False)  # outputs μ and logσ
        # self.decoder = nn.Linear(bottleneck_dim, 2)
        self.beta = beta
        self.mu_std = mu_std

    def forward(self, z):
        # encode to μ and logσ
        mu_logsigma = self.encoder(z)
        mu, log_sigma = torch.chunk(mu_logsigma, 2, dim=-1)
        sigma = torch.exp(log_sigma)

        # reparameterization trick
        eps1 = torch.randn_like(sigma)
        eps2 = torch.normal(1, self.mu_std, size=mu.size(), device=mu.device)
        b = mu * eps2 + sigma * eps1
        # kl_loss = 0.5 * torch.sum((1 + self.mu_std**2) * mu**2 + sigma**2 - 1 -
        #                             torch.log(sigma**2 + self.mu_std**2 * mu**2), dim=-1).mean()
        kl_loss = 0
        # KL divergence loss (stored as attribute for later use)

        self.kl_loss = self.beta * kl_loss

        return b


class PolyLinearWithPrecode(nn.Module):
    def __init__(self, layer_config: list, activation_fn, output_fn=None, input_dropout=None,
                 precode_beta=0.001, mu_std=0.1):
        super().__init__()

        assert len(layer_config) > 2, "To insert PRECODE, network must have at least 3 layers"

        self.layer_config = layer_config
        self.activation_fn = activation_fn
        self.output_fn = output_fn
        self.n_layers = len(layer_config) - 1
        self.precode = None  # will be inserted before last layer
        self.kl_loss = 0.0  # used to hold PRECODE loss

        layer_dict = OrderedDict()

        if input_dropout is not None:
            layer_dict["input_dropout"] = nn.Dropout(p=input_dropout)

        for i, (d1, d2) in enumerate(zip(layer_config[:-1], layer_config[1:])):
            if i == self.n_layers - 2:
                # 倒数第二层（插入前）
                layer_dict[f"linear_{i}"] = nn.Linear(d1, d2)
                layer_dict[f"{activation_fn.__class__.__name__.lower()}_{i}"] = activation_fn
                # Add normalization layer
                # layer_dict[f"norm_{i}"] = nn.LayerNorm(d2)

            elif i == self.n_layers - 1:
                self.precode = Precode(d1, out_dim=d2, beta=precode_beta, mu_std=mu_std)
                layer_dict[f"precode"] = self.precode

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        x = self.layers(x)
        # 将 PRECODE 的 KL loss 作为属性暴露出来，方便外部使用
        self.kl_loss = self.precode.kl_loss if self.precode is not None else 0.0
        return x


class PolyLinear(nn.Module):
    def __init__(self, layer_config: list, activation_fn, output_fn=None, input_dropout=None):

        super().__init__()

        assert len(layer_config) > 1, "For a linear network, we at least need one " \
                                      "input and one output dimension"

        self.layer_config = layer_config
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        self.n_layers = len(layer_config) - 1

        layer_dict = OrderedDict()

        if input_dropout is not None:
            layer_dict["input_dropout"] = nn.Dropout(p=input_dropout)

        for i, (d1, d2) in enumerate(zip(layer_config[:-1], layer_config[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            layer_dict[f"linear_{i}"] = layer
            if i < self.n_layers - 1:
                # only add activation functions in intermediate layers
                layer_dict[f"{activation_fn.__class__.__name__.lower()}_{i}"] = activation_fn

        if self.output_fn is not None:
            layer_dict[f"{output_fn.__class__.__name__.lower()}"] = self.output_fn

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        x = self.layers(x)
        return x