from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from model import AGNN


def train():
    t_total = time.time()
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=True,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of attention layers.')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # Model and optimizer
    model = AGNN(nfeat=features.shape[1],
                 nhid=args.hidden,
                 nclass=labels.max() + 1,
                 nlayers=args.layers,
                 dropout_rate=args.dropout_rate)
    # print(model)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    train()
    test()



"""
Epoch: 0930 loss_train: 0.8499 acc_train: 0.5714 loss_val: 1.2415 acc_val: 0.4967 time: 0.0165s
Epoch: 0931 loss_train: 0.7878 acc_train: 0.6714 loss_val: 1.1716 acc_val: 0.5233 time: 0.0166s
Epoch: 0932 loss_train: 0.8722 acc_train: 0.6143 loss_val: 1.1497 acc_val: 0.5533 time: 0.0165s
Epoch: 0933 loss_train: 0.8424 acc_train: 0.6286 loss_val: 1.1204 acc_val: 0.5567 time: 0.0166s
Epoch: 0934 loss_train: 0.8440 acc_train: 0.6500 loss_val: 1.1260 acc_val: 0.5433 time: 0.0164s
Epoch: 0935 loss_train: 0.7924 acc_train: 0.6429 loss_val: 1.1277 acc_val: 0.5500 time: 0.0166s
Epoch: 0936 loss_train: 0.8434 acc_train: 0.6143 loss_val: 1.1859 acc_val: 0.5367 time: 0.0163s
Epoch: 0937 loss_train: 0.8321 acc_train: 0.6500 loss_val: 1.2349 acc_val: 0.5033 time: 0.0166s
Epoch: 0938 loss_train: 0.8619 acc_train: 0.5929 loss_val: 1.2063 acc_val: 0.5167 time: 0.0163s
Epoch: 0939 loss_train: 0.8854 acc_train: 0.6214 loss_val: 1.1820 acc_val: 0.5200 time: 0.0167s
Epoch: 0940 loss_train: 0.8175 acc_train: 0.6714 loss_val: 1.1642 acc_val: 0.5833 time: 0.0165s
Epoch: 0941 loss_train: 0.7979 acc_train: 0.6786 loss_val: 1.1809 acc_val: 0.5300 time: 0.0176s
Epoch: 0942 loss_train: 0.8745 acc_train: 0.6714 loss_val: 1.2185 acc_val: 0.4700 time: 0.0239s
Epoch: 0943 loss_train: 0.9356 acc_train: 0.5857 loss_val: 1.1631 acc_val: 0.5100 time: 0.0245s
Epoch: 0944 loss_train: 0.7580 acc_train: 0.6714 loss_val: 1.1893 acc_val: 0.4867 time: 0.0283s
Epoch: 0945 loss_train: 0.8043 acc_train: 0.6357 loss_val: 1.2182 acc_val: 0.4933 time: 0.0272s
Epoch: 0946 loss_train: 0.9400 acc_train: 0.5643 loss_val: 1.2206 acc_val: 0.4967 time: 0.0279s
Epoch: 0947 loss_train: 0.9084 acc_train: 0.6357 loss_val: 1.1772 acc_val: 0.5333 time: 0.0256s
Epoch: 0948 loss_train: 0.8625 acc_train: 0.5929 loss_val: 1.1185 acc_val: 0.5433 time: 0.0228s
Epoch: 0949 loss_train: 0.8884 acc_train: 0.6214 loss_val: 1.0731 acc_val: 0.5600 time: 0.0252s
Epoch: 0950 loss_train: 0.9325 acc_train: 0.6357 loss_val: 1.1647 acc_val: 0.5233 time: 0.0259s
Epoch: 0951 loss_train: 0.8079 acc_train: 0.6857 loss_val: 1.2200 acc_val: 0.5067 time: 0.0267s
Epoch: 0952 loss_train: 0.8873 acc_train: 0.6429 loss_val: 1.2604 acc_val: 0.4967 time: 0.0259s
Epoch: 0953 loss_train: 0.7947 acc_train: 0.6500 loss_val: 1.1636 acc_val: 0.5067 time: 0.0262s
Epoch: 0954 loss_train: 0.8874 acc_train: 0.6071 loss_val: 1.0987 acc_val: 0.5533 time: 0.0259s
Epoch: 0955 loss_train: 0.7976 acc_train: 0.6286 loss_val: 1.1749 acc_val: 0.5433 time: 0.0254s
Epoch: 0956 loss_train: 0.8287 acc_train: 0.7000 loss_val: 1.1717 acc_val: 0.5333 time: 0.0471s
Epoch: 0957 loss_train: 0.7877 acc_train: 0.6857 loss_val: 1.2114 acc_val: 0.5333 time: 0.0458s
Epoch: 0958 loss_train: 0.8089 acc_train: 0.6357 loss_val: 1.2859 acc_val: 0.4867 time: 0.0455s
Epoch: 0959 loss_train: 0.7537 acc_train: 0.6857 loss_val: 1.1830 acc_val: 0.5200 time: 0.0454s
Epoch: 0960 loss_train: 0.8031 acc_train: 0.6786 loss_val: 1.1257 acc_val: 0.5700 time: 0.0454s
Epoch: 0961 loss_train: 0.7798 acc_train: 0.6429 loss_val: 1.1224 acc_val: 0.5367 time: 0.0455s
Epoch: 0962 loss_train: 0.9215 acc_train: 0.6357 loss_val: 1.2498 acc_val: 0.5233 time: 0.0452s
Epoch: 0963 loss_train: 0.8086 acc_train: 0.6786 loss_val: 1.2122 acc_val: 0.5167 time: 0.0463s
Epoch: 0964 loss_train: 0.8570 acc_train: 0.6429 loss_val: 1.2230 acc_val: 0.5167 time: 0.0466s
Epoch: 0965 loss_train: 0.7860 acc_train: 0.6286 loss_val: 1.1754 acc_val: 0.5233 time: 0.0385s
Epoch: 0966 loss_train: 0.8311 acc_train: 0.6214 loss_val: 1.1870 acc_val: 0.5467 time: 0.0217s
Epoch: 0967 loss_train: 0.7661 acc_train: 0.6571 loss_val: 1.1990 acc_val: 0.5300 time: 0.0157s
Epoch: 0968 loss_train: 0.9326 acc_train: 0.6000 loss_val: 1.1721 acc_val: 0.5300 time: 0.0167s
Epoch: 0969 loss_train: 0.8361 acc_train: 0.6857 loss_val: 1.1725 acc_val: 0.4967 time: 0.0167s
Epoch: 0970 loss_train: 0.7964 acc_train: 0.7071 loss_val: 1.1766 acc_val: 0.5167 time: 0.0167s
Epoch: 0971 loss_train: 0.8367 acc_train: 0.6714 loss_val: 1.1240 acc_val: 0.5067 time: 0.0169s
Epoch: 0972 loss_train: 0.9318 acc_train: 0.6071 loss_val: 1.1170 acc_val: 0.5500 time: 0.0172s
Epoch: 0973 loss_train: 0.8629 acc_train: 0.6143 loss_val: 1.1285 acc_val: 0.5067 time: 0.0171s
Epoch: 0974 loss_train: 0.8361 acc_train: 0.6286 loss_val: 1.2190 acc_val: 0.5033 time: 0.0181s
Epoch: 0975 loss_train: 0.7361 acc_train: 0.7357 loss_val: 1.1640 acc_val: 0.5400 time: 0.0175s
Epoch: 0976 loss_train: 0.7646 acc_train: 0.6857 loss_val: 1.1716 acc_val: 0.4867 time: 0.0177s
Epoch: 0977 loss_train: 0.8757 acc_train: 0.6286 loss_val: 1.2189 acc_val: 0.5033 time: 0.0174s
Epoch: 0978 loss_train: 0.8393 acc_train: 0.6286 loss_val: 1.1784 acc_val: 0.4900 time: 0.0172s
Epoch: 0979 loss_train: 0.9142 acc_train: 0.6429 loss_val: 1.1484 acc_val: 0.5167 time: 0.0178s
Epoch: 0980 loss_train: 0.8412 acc_train: 0.6571 loss_val: 1.1137 acc_val: 0.5467 time: 0.0178s
Epoch: 0981 loss_train: 0.8278 acc_train: 0.6714 loss_val: 1.2166 acc_val: 0.4900 time: 0.0173s
Epoch: 0982 loss_train: 0.7961 acc_train: 0.6571 loss_val: 1.2050 acc_val: 0.5367 time: 0.0174s
Epoch: 0983 loss_train: 0.8683 acc_train: 0.6286 loss_val: 1.1412 acc_val: 0.5633 time: 0.0167s
Epoch: 0984 loss_train: 0.8877 acc_train: 0.5357 loss_val: 1.2729 acc_val: 0.5200 time: 0.0170s
Epoch: 0985 loss_train: 1.0161 acc_train: 0.5786 loss_val: 1.1996 acc_val: 0.5533 time: 0.0171s
Epoch: 0986 loss_train: 0.8860 acc_train: 0.6143 loss_val: 1.1826 acc_val: 0.5200 time: 0.0172s
Epoch: 0987 loss_train: 0.7464 acc_train: 0.7429 loss_val: 1.2093 acc_val: 0.4933 time: 0.0173s
Epoch: 0988 loss_train: 0.7609 acc_train: 0.6357 loss_val: 1.2505 acc_val: 0.4900 time: 0.0175s
Epoch: 0989 loss_train: 0.8499 acc_train: 0.5929 loss_val: 1.1615 acc_val: 0.5500 time: 0.0174s
Epoch: 0990 loss_train: 0.7747 acc_train: 0.6714 loss_val: 1.1045 acc_val: 0.5467 time: 0.0175s
Epoch: 0991 loss_train: 0.9087 acc_train: 0.5786 loss_val: 1.2008 acc_val: 0.5367 time: 0.0176s
Epoch: 0992 loss_train: 0.7646 acc_train: 0.6429 loss_val: 1.1710 acc_val: 0.5667 time: 0.0237s
Epoch: 0993 loss_train: 0.8515 acc_train: 0.6714 loss_val: 1.1833 acc_val: 0.5400 time: 0.0249s
Epoch: 0994 loss_train: 0.8330 acc_train: 0.6500 loss_val: 1.1621 acc_val: 0.5567 time: 0.0268s
Epoch: 0995 loss_train: 0.9080 acc_train: 0.5929 loss_val: 1.2255 acc_val: 0.5133 time: 0.0255s
Epoch: 0996 loss_train: 0.8601 acc_train: 0.6500 loss_val: 1.1151 acc_val: 0.5700 time: 0.0269s
Epoch: 0997 loss_train: 0.7653 acc_train: 0.6643 loss_val: 1.1136 acc_val: 0.5333 time: 0.0255s
Epoch: 0998 loss_train: 0.8709 acc_train: 0.6214 loss_val: 1.2863 acc_val: 0.5133 time: 0.0266s
Epoch: 0999 loss_train: 0.8826 acc_train: 0.6286 loss_val: 1.1636 acc_val: 0.5433 time: 0.0171s
Epoch: 1000 loss_train: 0.8133 acc_train: 0.6643 loss_val: 1.1758 acc_val: 0.5367 time: 0.0257s
Optimization Finished!
Total time elapsed: 26.7623s
Test set results: loss= 0.7828 accuracy= 0.8370
"""