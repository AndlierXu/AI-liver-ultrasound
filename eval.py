import argparse
import os
import torch
from torch import nn

from config_reader.parser import ConfigParser
from model.get_model import get_model
from reader.reader import init_dataset
from model.work import predict_net
from utils.util import print_info

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--gpu', '-g')
parser.add_argument('--model', '-m')
args = parser.parse_args()

configFilePath = args.config
if configFilePath is None:
    print("python *.py\t--config/-c\tconfigfile")
use_gpu = True

if args.gpu is None:
    use_gpu = False
else:
    use_gpu = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

config = ConfigParser(configFilePath)

print_info("Start to build Net")

model_name = config.get("model", "name")
net = get_model(model_name, config)

device = []
if torch.cuda.is_available() and use_gpu:
    device_list = args.gpu.split(",")
    for a in range(0, len(device_list)):
        device.append(int(a))

    net = net.cuda()

    try:
        net.init_multi_gpu(device)
    except Exception as e:
        print_info(str(e))

net.load_state_dict(torch.load(args.model))

print_info("Net build done")

print_info("Start to prepare Data")

train_dataset, valid_dataset, test_dataset = init_dataset(config)

print_info("Data preparation Done")

print(predict_net(net, test_dataset, use_gpu, config, 0))

for a in range(0, len(train_dataset.read_process)):
    train_dataset.read_process[a].terminate()
for a in range(0, len(valid_dataset.read_process)):
    valid_dataset.read_process[a].terminate()
