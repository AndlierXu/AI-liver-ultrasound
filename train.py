import argparse
import os
import torch
from torch import nn

from config_reader.parser import ConfigParser
from model.get_model import get_model
from reader.reader import init_dataset
from model.work import train_net
# from model.work import multiseg_train_net
from utils.util import print_info

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--gpu', '-g')
args = parser.parse_args()

configFilePath = args.config
print(configFilePath)
if configFilePath is None:
    print("python *.py\t--config/-c\tconfigfile")
use_gpu = True

if args.gpu is None:
    use_gpu = False
else:
    use_gpu = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("gpu:",args.gpu)

config = ConfigParser(configFilePath)

print_info("Start to build Net")

model_name = config.get("model", "name")
net = get_model(model_name, config)

device_id = []

print_info("CUDA:%s" % str(torch.cuda.is_available()))
# device = torch.device('cuda: 0')

if torch.cuda.is_available() and use_gpu:
    device_list = args.gpu.split(",")
    for a in range(0, len(device_list)):
        device_id.append(int(a))

    net = net.cuda()
    if len(device_id)>1:
        try:
            net = torch.nn.DataParallel(net, device_ids=device_id)
        except Exception as e:
            print_info(str(e))

pretrain = config.get("train", "pre_train")
if pretrain:
    try:
        basic_model = torch.load(config.get("model","pretrain_path"))
        model_dict = net.state_dict()
        pretrained_dict = {}
        for k, v in basic_model.items():
                pretrained_dict[k] = v
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    except Exception as e:
        print_info(str(e))



print_info("Net build done")

print_info("Start to prepare Data")

train_dataset, valid_dataset, test_dataset = init_dataset(config)

print_info("Data preparation Done")

train_net(net, train_dataset, valid_dataset, test_dataset, use_gpu, config)

for a in range(0, len(train_dataset.read_process)):
    train_dataset.read_process[a].terminate()
for a in range(0, len(valid_dataset.read_process)):
    valid_dataset.read_process[a].terminate()
for a in range(0, len(test_dataset.read_process)):
    valid_dataset.read_process[a].terminate()
