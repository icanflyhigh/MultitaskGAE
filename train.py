import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from No49.model import MultiTaskGNN
import KYXL
import time
import numpy as np
import No49.util as util
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from No49.util import load_data, mask_test_edges, preprocess_graph, get_roc_score, loss_function
import scipy.sparse as sp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=22, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=128, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--l1_loss_weight', type=float, default=0.5, help='Initial VGAE loss weight to classifier.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, labels = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape
    n_nodes = torch.tensor(n_nodes)
    class_num = len(np.unique(labels.numpy()))
    train_mask, test_mask, val_mask = util.one_time_mask(int(n_nodes), [0.10, 0.85, 0.05])
    l1_loss_weight = args.l1_loss_weight

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, labels)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    data = util.get_data(adj_norm, features)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))

    model = MultiTaskGNN(feat_dim, args.hidden1, args.hidden2, class_num, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # local platform
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("cuda is available", end=' ')
        device = torch.device('cuda')
        model.to(device)
        # features = features.to(device)
        # adj_norm = adj_norm.to(device)
        data = data.to(device)
        adj_label = adj_label.to(device)
        n_nodes = n_nodes.to(device)
        norm = norm.to(device)
        pos_weight = pos_weight.to(device)
        labels = labels.to(device).squeeze()
        print("using cuda...")
    # print(labels.shape)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar, out = model(data)
        loss1 = loss_function(preds=recovered, labels=adj_label,
                              mu=mu, logvar=logvar, n_nodes=n_nodes,
                              norm=norm, pos_weight=pos_weight)
        loss2 = F.cross_entropy(out[train_mask], labels[train_mask])
        loss = l1_loss_weight * loss1 + (1 - l1_loss_weight) * loss2
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.cpu().detach().data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        _, out = out.max(dim=1)
        correct = int(out[val_mask].eq
                      (labels[val_mask]).sum().item())
        val_acc = correct / int(len(val_mask))
        correct = int(out[train_mask].eq
                      (labels[train_mask]).sum().item())
        train_acc = correct / int(len(train_mask))
        if (epoch + 1) % 10 >= 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t),
                  "train acc:{:.5f}".format(train_acc),
                  "val acc:{:.5f}".format(val_acc),
                  )
        if (epoch + 1) % 10 == 0:
            model.eval()
            _, _, _, out = model(data)
            _, out = out.max(dim=1)
            correct = int(out[test_mask].eq
                          (labels[test_mask]).sum().item())
            acc = correct / int(len(test_mask))
            out = out.cpu().detach().numpy()
            pred = out[test_mask]
            lab = labels[test_mask].cpu().detach().numpy().astype(int)
            roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)

            print("Test ROC score: {:.5f}".format(roc_score))
            print("Test AP score : {:.5f}".format(ap_score))
            print("Test Acc      : {:.5f}".format(acc))
            print("micro: {:.4f}".format(f1_score(lab, pred, average="micro")))
            print("macro: {:.4f}".format(f1_score(lab, pred, average="macro")))

    print("Optimization Finished!")
    model.eval()
    _, _, _, out = model(data)
    _, out = out.max(dim=1)
    correct = int(out[test_mask].eq
                  (labels[test_mask]).sum().item())
    acc = correct / int(len(test_mask))
    out = out.cpu().detach().numpy()
    pred = out[test_mask]
    lab = labels[test_mask].cpu().detach().numpy().astype(int)
    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)

    print("Test ROC score: {:.5f}".format(roc_score))
    print("Test AP score : {:.5f}".format(ap_score))
    print("Test Acc      : {:.5f}".format(acc))
    print("micro: {:.4f}".format(f1_score(lab, pred, average="micro")))
    print("macro: {:.4f}".format(f1_score(lab, pred, average="macro")))

if __name__ == '__main__':
    gae_for(args)
#
# # file_path = r"D:\pycharm_project\datasets\mat/"
# # file_name = 'cora'
# # dataset, adj = util.load_data(file_path, file_name)
# # dense_adj = adj.toarray()
# # feature_num = dataset.x.shape[1]
# # class_num = int(dataset.y.max() + 1)
# # hidden_num1 = 128
# # hidden_num2 = 64
# # device = torch.device('cuda')
# #
# # # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# # # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
# #
# #
# # train_ratio = 0.9
# # dataset = dataset.to(device)
# # model = MultiTaskGNN(feature_num, hidden_num1, hidden_num2, class_num, dropout=False).to(device)
# # optimizer = optim.Adam(model.parameters(), lr=0.01)
# # epoch_num = 200
# # micro_f1 = 0
# # macro_f1 = 0
# # fold_num = 0
# # l1_loss_weight = 0.5
# # for test_mask, train_mask in util.k_fold(dataset.x.shape[0], 10):
# #     n_nodes = dataset.x.shape[0]
# #     # adj_label = torch.from_numpy(adj[train_mask, :][:, train_mask].toarray()).float().to(device)
# #     adj_label = torch.from_numpy(dense_adj).float().to(device)
# #     pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()).float().to(device)
# #     norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)).float().to(device)
# #     # print("pos_weight{}, norm{}".format(pos_weight, norm))
# #     for epoch in range(epoch_num):
# #         t = time.time()
# #         model.train()
# #         optimizer.zero_grad()
# #         recovered, mu, logvar, out = model(dataset)
# #         pred_adj = recovered
# #         loss1 = util.loss_function(preds=pred_adj, labels=adj_label,
# #                                    mu=mu, logvar=logvar, n_nodes=n_nodes,
# #                                    norm=norm, pos_weight=pos_weight)
# #         loss2 = F.cross_entropy(out[train_mask], dataset.y[train_mask])
# #         loss = l1_loss_weight * loss1 + (1 - l1_loss_weight) * loss2
# #         loss.backward()
# #         cur_loss = loss.item()
# #         optimizer.step()
# #
# #         hidden_emb = mu.cpu().detach().data.numpy()
# #         roc_curr, ap_curr = util.get_roc_score(hidden_emb, adj)
# #
# #         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
# #               "val_ap=", "{:.5f}".format(ap_curr),
# #               "time=", "{:.5f}".format(time.time() - t)
# #               )
# #     model.eval()
# #     recovered, mu, logvar, pred = model(dataset)
# #     _, pred = pred.max(dim=1)
# #
# #     correct = int(pred[test_mask].eq
# #                   (dataset.y[test_mask]).sum().item())
# #
# #     acc = correct / int(np.sum(test_mask))
# #     print(correct)
# #     _, _, _, pred = model(dataset)
# #     pred = pred[test_mask].cpu().detach().numpy()
# #     pred = np.argmax(pred, axis=1)
# #     lab = dataset.y[test_mask].cpu().detach().numpy().astype(int)
# #     macro_f1 += f1_score(lab, pred, average="macro")
# #     micro_f1 += f1_score(lab, pred, average="micro")
# #     # print('Accuracy: {:.4f}'.format(acc))
# #     print("train ratio: {:.2f}".format(train_ratio))
# #     print("micro: {:.4f}".format(f1_score(lab, pred, average="micro")))
# #     print("macro: {:.4f}".format(f1_score(lab, pred, average="macro")))
# #     print(classification_report(pred, lab, digits=4))
# #     fold_num += 1
# #     break
# # print("micro: {:.4f}".format(micro_f1 / fold_num))
# # print("macro: {:.4f}".format(macro_f1 / fold_num))
# # print("Optimization Finished!")
