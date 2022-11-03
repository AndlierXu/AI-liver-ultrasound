import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,f1_score,precision_score, roc_curve, auc
# from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
import csv
import cv2
import segmentation_models_pytorch as smp
import torch.nn as nn

from reader.reader import init_dataset
from model.loss import get_loss
from utils.util import gen_result, print_info, time_to_str#, gen_IOU
from model.dice_loss import dice_coeff
from model.dice_loss import dice_coef_multilabel,dice_coef_new
from model.loss import cross_entropy_loss
from model.loss import BinaryDiceLoss
from model.loss import SoftDiceLoss
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

def iou(dice):
    return dice*2/(1+dice)

def cross_entropy2d(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']



def seg_net(net, valid_dataset, use_gpu, config, epoch, writer=None, test_name=''):

    net.eval()

    task_loss_type = config.get("train", "type_of_loss")
    train_type = config.get("train","train_type")
    data_type = test_name
    criterion = get_loss(task_loss_type)


    running_acc = 0
    running_loss = 0
    cnt = 0

    for cnt, data in enumerate(valid_dataset):
        accu = 0
        if data is None:
            break

        for key in data.keys():
            if key == "image_num"  or key == "mean" or key == "std" or key == "l_size" or key  == "r_size" or key == "origin_image":
                continue
            if isinstance(data[key], torch.Tensor):
                if torch.cuda.is_available() and use_gpu:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        logits = net(data["image"])
        pred = torch.sigmoid(logits)
        
        label = data["label"]

        loss = criterion(logits, label) 
        accu = dice_coeff(pred, label)

        #auc
        accu = accu/len(data["image_num"])
        running_loss += loss.item()
        running_acc += accu
        # print("per accu",accu)
    cnt += 1 


    net.train()

    print(running_acc / cnt)
    return running_loss / cnt, running_acc / cnt 


def predict_net(net, valid_dataset, use_gpu, config, epoch, writer=None, test_name=''):

    net.eval()

    task_loss_type = config.get("train", "type_of_loss")
    train_type = config.get("train","train_type")
    data_type = test_name
    criterion = get_loss(task_loss_type)
    softmax = nn.Softmax()


    running_acc = 0
    running_loss = 0
    cnt = 0

    for cnt, data in enumerate(valid_dataset):
        if data is None:
            break

        for key in data.keys():
            if key == "image_num" :
                        continue
            if isinstance(data[key], torch.Tensor):
                if torch.cuda.is_available() and use_gpu:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        outputs = net(data)
        label = data["label"]
        
        _, y_pred = outputs.max(1)
        y_pred = y_pred.cpu().data.numpy()
        prob = softmax(outputs)[:].cpu().data.numpy()

        loss = criterion(outputs, label)
        accu = accuracy_score(y_pred,label.cpu().data.numpy())
        
        running_loss += loss.item()
        running_acc += accu.item()
        # print("per accu",accu)
    cnt += 1
    if writer is None:
        pass
    else:
        writer.add_scalar(config.get("output", "model_name") + " valid loss", running_loss / cnt, epoch)
        writer.add_scalar(config.get("output", "model_name") + " valid accuracy", running_acc / cnt, epoch)

    net.train()

    print(running_acc / cnt)
    return running_loss / cnt, running_acc / cnt 


def valid_net(net, valid_dataset, use_gpu, config, epoch, writer=None):
    net.eval()

    dice_loss = SoftDiceLoss()
    task_loss_type = config.get("train", "type_of_loss")
    train_type = config.get("train","train_type")
    criterion = get_loss(task_loss_type)

    running_acc = 0
    running_loss = 0
    running_dex = 0
    cnt = 0
    acc_result = []
    y_trues = []
    y_preds = []

    # doc_list = []
    for cnt, data in enumerate(valid_dataset):
        if data is None:
            break

        for key in data.keys():
            if key == "image_num" or key == "mean" or key == "std" or key == "l_size" or key  == "r_size" or key == "origin_image":
                        continue
            if isinstance(data[key], torch.Tensor):
                if torch.cuda.is_available() and use_gpu:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        
        if train_type == "seg":

            logits = net(data["image"])
            pred = torch.sigmoid(logits)

            label = data["label"]


            loss = criterion(logits, label) + dice_loss(pred, label)
            accu = dice_coeff(pred, label)
        else:
            outputs = net(data)
            label = data["label"]

            _, y_pred = outputs.max(1)
            y_pred = y_pred.cpu().data.numpy()

            loss = criterion(outputs, label)
            accu = accuracy_score(y_pred,label.cpu().data.numpy())

        running_loss += loss.item()
        running_acc += accu.item()

    cnt += 1
    if writer is None:
        pass
    else:
        writer.add_scalar(config.get("output", "model_name") + " valid loss", running_loss / cnt, epoch)
        writer.add_scalar(config.get("output", "model_name") + " valid accuracy", running_acc / cnt, epoch)

    net.train()
    return running_loss / cnt, running_acc / cnt



def train_net(net, train_dataset, valid_dataset, test_dataset, use_gpu, config):
    # CE = CrossEntropyLoss2d()
    dice_loss = SoftDiceLoss()
    epoch = config.getint("train", "epoch")
    learning_rate = config.getfloat("train", "learning_rate")
    task_loss_type = config.get("train", "type_of_loss")
    train_type = config.get("train","train_type")

    model_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))

    criterion = get_loss(task_loss_type)

    optimizer_type = config.get("train", "optimizer")
    if optimizer_type == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                               weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=config.getfloat("train", "momentum"),
                              weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "rms":
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    else:
        raise NotImplementedError

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "gamma")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if optimizer_type == "rms":
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    print('** start training here! **')
    print('----------------|----------TRAIN-----------|----------VALID----------|----------------|')
    print('  lr    epoch   |   loss           evalu   |   loss           evalu  | Forward num')
    print('----------------|--------------------------|-------------------------|----------------|')
    start = timer()

    best_loss = 100
    best_epoch = 0

    for epoch_num in range(trained_epoch, epoch):
        net.train()
        total = 0

        train_cnt = 0
        train_loss = 0
        train_acc = 0
        train_label_acc = 0

        lr = 0
        
        for g in optimizer.param_groups:
            lr = float(g['lr'])
            break

        for cnt, data in enumerate(train_dataset):
            if data is None:
                break
            
            for key in data.keys():
                if key == "image_num" or key == "mean" or key == "std" or key == "l_size" or key  == "r_size" or key == "origin_image":
                    continue
                if isinstance(data[key], torch.Tensor):
                    if torch.cuda.is_available() and use_gpu:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            optimizer.zero_grad()



            if train_type == "seg":
                logits = net(data["image"])
                pred = torch.sigmoid(logits)

                label = data["label"]

                loss = criterion(logits, label) + dice_loss(pred, label)

                accu = dice_coeff(pred, label)
            else:
                outputs = net(data)
                label = data["label"]

                _, y_pred = outputs.max(1)
                y_pred = y_pred.cpu().data.numpy()

                loss = criterion(outputs, label)
                accu = accuracy_score(y_pred,label.cpu().data.numpy())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accu.item()
            train_cnt += 1

            del loss
            del accu
            del data

            total += config.getint("train", "batch_size")

            print('\r', end='', flush=True)
            print('%.4f   % 3d    |  %.4f         % 2.2f   |   ????           ?????     |  %2.2f  | %d' % (
            lr, epoch_num + 1, train_loss / train_cnt, train_acc / train_cnt,
            train_label_acc / train_cnt, total), end='',
                flush=True)

        train_loss /= train_cnt
        train_acc /= train_cnt
        train_label_acc /= train_cnt

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

        with torch.no_grad():
            # pass

            valid_loss, valid_accu = valid_net(net, valid_dataset, use_gpu, config, epoch_num + 1, writer=None)
        print('\r', end='', flush=True)

        print('%.4f   % 3d    |  %.4f          %.2f   |  %.4f         % 2.2f        |  %s  | %d' % (
            lr, epoch_num + 1, train_loss, train_acc , valid_loss, valid_accu , 
            1, total))
        if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch_num + 1
        if optimizer_type == "rms":
            exp_lr_scheduler.step(valid_accu)
        else:
            exp_lr_scheduler.step()
    
    print("predict test by model:",best_epoch)


    net.load_state_dict(torch.load(os.path.join(model_path, "model-%d.pkl" % best_epoch)))
    if train_type == "seg":
        with torch.no_grad():
            seg_net(net, test_dataset, use_gpu, config, best_epoch)
            torch.cuda.empty_cache()
    else:
        with torch.no_grad():
            predict_net(net, test_dataset, use_gpu, config, best_epoch)
            torch.cuda.empty_cache()


