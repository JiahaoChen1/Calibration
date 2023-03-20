import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from calibration_metric import *
import os
from matplotlib.collections import LineCollection
import numpy as np


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def Wasserstein(mu1, sigma1, mu2, sigma2):
    p1 = np.sum(np.power((mu1 - mu2),2))
    p2 = np.sum(np.power(np.power(sigma1,1/2) - np.power(sigma2, 1/2),2))
    return p1 + p2


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0]


def generate(data_loader, model, return_acc=False):
    t_s, t_l, t_f = [], [], []
    top1 = AverageMeter('acc', ':.2f')
    with torch.no_grad():
        for _, (image, label) in enumerate(data_loader):
            image, label = image.cuda(), label.cuda()
            pre_fea, pre_cls = model(image, return_feature=True)
            t_s.append(pre_cls)
            t_l.append(label)
            t_f.append(pre_fea)
            acc = accuracy(pre_cls, label)
            top1.update(acc.item(), image.shape[0])

        print('cls is {:.2f}'.format(float(top1.avg)))
        t_s = torch.cat(t_s, dim=0)
        t_l = torch.cat(t_l, dim=0)
        t_f = torch.cat(t_f, dim=0)
    if return_acc:
        return float(top1.avg), t_f, t_s, t_l
    return t_f, t_s, t_l


def make_model_diagrams(outputs, labels, save_path='', n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    if type(outputs) == np.ndarray:
        outputs = torch.tensor(outputs)
    
    if type(labels) == np.ndarray:
        labels = torch.tensor(labels)

    softmaxes = torch.nn.functional.softmax(outputs, 1)
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)
    # overall_accuracy = (predictions==labels).sum().item()/len(labels)
    
    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)
    plt.clf()
    plt.figure(0, figsize=(8, 8))
    gap = np.array(bin_scores - bin_corrects)
    
    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
    bin_corrects = np.nan_to_num(np.array([bin_correct for bin_correct in bin_corrects]))
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')

    ece_metric = ECELoss()
    ece = ece_metric.loss(outputs.numpy(), labels.numpy())

    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    plt.text(0.17, 0.82, "ECE: {:.2f}".format(ece * 100), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    plt.title("Reliability Diagram", size=22)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(save_path)
    return ece


def make_model_diagrams_sce(outputs, labels, save_path='', n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    if type(outputs) == np.ndarray:
        outputs = torch.tensor(outputs)
    
    if type(labels) == np.ndarray:
        labels = torch.tensor(labels)

    for i in range(outputs.shape[-1]):
        plt.clf()
        softmaxes = torch.nn.functional.softmax(outputs, 1)
        # confidences, predictions = softmaxes.max(1)
        confidences = softmaxes[:, i]
        accuracies = (labels == i)
        # overall_accuracy = (predictions==labels).sum().item()/len(labels)
        
        # Reliability diagram
        bins = torch.linspace(0, 1, n_bins + 1)
        width = 1.0 / n_bins
        bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
        bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
        
        bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
        bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
        bin_corrects = np.nan_to_num(bin_corrects)
        bin_scores = np.nan_to_num(bin_scores)
        
        plt.figure(0, figsize=(8, 8))
        gap = np.array(bin_scores - bin_corrects)
        
        confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
        bin_corrects = np.nan_to_num(np.array([bin_correct for bin_correct in bin_corrects]))
        gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
        
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')

        sce_metric = SCELoss()
        sce,_ = sce_metric.loss(outputs.numpy(), labels.numpy())

        # Clean up
        bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
        plt.text(0.17, 0.82, "ECE: {:.4f}".format(sce[i]), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

        plt.title("Reliability Diagram", size=22)
        plt.ylabel("Accuracy",  size=18)
        plt.xlabel("Confidence",  size=18)
        plt.xlim(0,1)
        plt.ylim(0,1)
        # print(os.path.join(save_path, str(i) + '.jpg'))
        plt.savefig(os.path.join(save_path, str(i) + '.jpg'))


def label2onehot(logits, labels):
    label_onehot = torch.zeros_like(torch.tensor(logits))
    label_onehot.scatter_(1, torch.tensor(labels).long().view(-1, 1), 1)
    return label_onehot.numpy()


def class_instance_number(labels, cls):
    instance_number = []
    for i in range(cls):
        instance_number.append(np.sum(np.equal(labels, i).astype(np.float64)))
    return instance_number


def ir_draw(x, y, p, path):
    plt.clf()
    ids = np.argsort(x)
    segments = [[[x[ids[i]], y[ids[i]]], [x[ids[i]], p[ids[i]]]] for i in range(x.shape[0])]
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(y.shape[0]))
    lc.set_linewidths(0.1 * np.ones(x.shape[0]))

    plt.gca().add_collection(lc)
    plt.plot(x,y,"r.",markersize=5)
    # plt.plot(x,p,"g.-",markersize=5)

    # print(x.shape, y.shape, p.shape)
    # plt.scatter(x,y,"r",markersize=5)
    plt.scatter(x,p)

    plt.title('Isotonic Regression')
    plt.savefig(path)
    print('save OK')