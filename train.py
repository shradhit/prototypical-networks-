import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diabeticretinopathy import DiabeticRetinopathy
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric


if __name__ == '__main__':
    save_epoch = 10
    train_way = 4
    test_way  = 4
    shot = 1
    query = 3
    gpu = 0
    save_path = './save/proto-1'
    max_epoch  = 60

    ensure_path(save_path)

    trainset = DiabeticRetinopathy('train')
    train_sampler_ = PrototypicalBatchSampler(trainset.label, 3, train_way, shot + query)
    train_loader = DataLoader(dataset=trainset,batch_sampler=train_sampler_, num_workers=8)

    valset = DiabeticRetinopathy('val')
    val_sampler = PrototypicalBatchSampler(valset.label, 4, test_way, shot + query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    model = Convnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_ for _ in batch]
            p = shot * train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(shot, train_way, -1).mean(dim=0)

            label = torch.arange(train_way).repeat(2)
            label = label.type(torch.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}' .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_ for _ in batch]
            p = shot * test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(shot, test_way, -1).mean(dim=0)

            label = torch.arange(test_way).repeat(2)
            label = label.type(torch.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)

            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / max_epoch)))
