# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--at_type', type=str, default="Base", help='choose a AT type: Base, FGSM, FGM, PGD, Free, default is Base')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained, default is random')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = "TextCNN"  # Fix baseline is 'TextRCNN'
    x = import_module('models.' + model_name)

    at_type = args.at_type
    y = import_module("models." + at_type + "Model")


    config = x.Config(dataset, embedding, at_type)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    atModel = y.ATModel(model)

    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter, atModel)

    time_dif = get_time_dif(start_time)
    print("All time usage:", time_dif)
