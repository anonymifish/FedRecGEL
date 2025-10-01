import torch
import torch.nn as nn
import json
import os
import random
from data.data_utils import get_label_dict
from utils.save_utils import construct_logger, construct_weight_path_wocheck, construct_weight_path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score


def initialize_settings(path):
    class Args:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)

        def add_attribute(self, k, v):
            setattr(self, k, v)

        def record_config(self, logger):
            for attr in self.__dict__:
                logger.info(f'{attr}: {getattr(self, attr)}')

    with open(path, 'r') as f:
        configuration = json.load(f)
    return Args(configuration)

class WeakClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_classes=2):
        super(WeakClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class EnsembleClassifier(nn.Module):
    def __init__(self, embed_dim, num_items, num_classes=2):
        super(EnsembleClassifier, self).__init__()
        self.weak_classifiers = nn.ModuleList([
            WeakClassifier(input_dim=embed_dim, num_classes=num_classes)
            for _ in range(num_items)
        ])

    def forward(self, user_mat):
        """
        user_mat: (batch_size, num_items, embed_dim)
        - 把 user_mat 的每一列 (即每个item的embedding) 单独输入到对应的 weak_classifier。
        - 最后返回每个 item 的预测结果。
        """
        batch_size, num_items, embed_dim = user_mat.size()

        all_item_logits = []
        for i in range(num_items):
            item_emb = user_mat[:, i, :]
            item_logits = self.weak_classifiers[i](item_emb)
            all_item_logits.append(item_logits.unsqueeze(1))

        all_item_logits = torch.cat(all_item_logits, dim=1)
        
        final_logits = all_item_logits.mean(dim=1)

        return final_logits

def sample_attack_sample(label_dict, num_user, seed, logger):
    random.seed(seed)
    label_number_dict = {label: idx for idx, label in enumerate(sorted(set(label_dict.values())))}
    label_user_dict = {idx: [] for idx in label_number_dict.values()}
    for i in range(num_user):
        label = label_dict[i]
        label_user_dict[label_number_dict[label]].append(i)

    min_len = min(len(u) for u in label_user_dict.values())
    for k in label_user_dict:
        if len(label_user_dict[k]) > min_len:
            logger.info(f"Trimming label {k} from {len(label_user_dict[k])} to {min_len}")
            label_user_dict[k] = random.sample(label_user_dict[k], min_len)

    return label_user_dict

def train(model, train_user_mat, train_labels, optimizer, loss_fn, test_user_mat, test_labels, epochs=50, logger=None):
    model.train()
    prev_loss = None
    stop_count = 0

    from sklearn.metrics import f1_score, balanced_accuracy_score

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_user_mat)
        loss = loss_fn(outputs.squeeze(), train_labels)
        loss.backward()
        optimizer.step()

        # 计算在测试集上的BAcc和F1
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_user_mat)
            if test_outputs.dim() > 1 and test_outputs.size(-1) > 1:
                preds = torch.argmax(test_outputs, dim=-1)
            else:
                preds = (test_outputs > 0.5).long().squeeze()
            preds = preds.cpu().numpy()
            test_labels_np = test_labels.cpu().numpy()
            bacc = balanced_accuracy_score(test_labels_np, preds)
            f1 = f1_score(test_labels_np, preds, average='macro')
        model.train()

        logger.info(f"[Train] Epoch {epoch}, Loss: {loss.item():.4f}, BAcc: {bacc:.4f}, F1: {f1:.4f}")


if __name__ == '__main__':
    config_path = 'configs/attack_config.json'
    args = initialize_settings(config_path)
    save_path = f'./exp_results/{args.model}/{args.dataset}/{args.method}/'
    device = torch.device(args.device)
    if not os.path.exists(save_path):
        print('attack model not exist')
        exit()

    args.add_attribute("save_path", save_path)
    weight_folder_path = construct_weight_path(args)
    logger = construct_logger(args)
    args.record_config(logger)
    args.add_attribute("logger", logger)
    model_path = construct_weight_path_wocheck(args)

    if not os.path.exists(model_path):
        args.logger.info(f'Target model path {model_path} does not exist')
        exit()

    args.add_attribute("target_model_path", model_path)
    args.logger.info('loading user matrix...')
    model = torch.load(args.target_model_path, map_location='cpu', weights_only=False)
    user_mat = model.get_user_embedding_matrix()
    args.logger.info('loading label user dict...')
    label_dict = get_label_dict(args)
    label_user_dict = sample_attack_sample(label_dict, len(user_mat), 4, args.logger)

    all_indices = []
    all_labels = []
    for label, users in label_user_dict.items():
        all_indices.extend(users)
        all_labels.extend([label] * len(users))  # label 应该是编号后的

    all_labels = torch.tensor(all_labels)
    all_indices = torch.tensor(all_indices)

    num_rounds = 3
    f1_list = []
    bacc_list = []

    for round_idx in range(num_rounds):
        args.logger.info(f"\n=== Round {round_idx + 1} ===")
        train_indices, test_indices, train_labels, test_labels = train_test_split(
            all_indices, all_labels, test_size=0.8, stratify=all_labels)

        num_classes = len(torch.unique(all_labels))
        train_labels = train_labels.to(device)
        test_labels = test_labels.to(device)

        train_user_mat = torch.stack([user_mat[i].weight for i in train_indices]).to(device)
        test_user_mat = torch.stack([user_mat[i].weight for i in test_indices]).to(device)

        embed_dim = user_mat[0].weight.shape[1]
        num_items = user_mat[0].weight.shape[0]
        model = EnsembleClassifier(embed_dim=embed_dim, num_items=num_items, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        train(model, train_user_mat, train_labels, optimizer, loss_fn, test_user_mat, test_labels, epochs=80, logger=logger)

        # 评估本轮的Bacc和F1
        model.eval()
        with torch.no_grad():
            outputs = model(test_user_mat)
            preds = torch.argmax(outputs, dim=1)

            bacc = balanced_accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())
            f1 = f1_score(test_labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            args.logger.info(f"[Round {round_idx+1}] Bacc: {bacc:.4f}, F1: {f1:.4f}")
            bacc_list.append(bacc)
            f1_list.append(f1)
        model.train()

    avg_bacc = sum(bacc_list) / num_rounds
    avg_f1 = sum(f1_list) / num_rounds
    args.logger.info(f"\n=== {num_rounds}轮平均结果 ===")
    args.logger.info(f"平均Bacc: {avg_bacc:.4f}, 平均F1: {avg_f1:.4f}")

