import numpy as np
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math
import copy

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)  # cpu
torch.cuda.manual_seed(RANDOM_SEED)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

from utils import utils
from utils.readers import InHospitalMortalityReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils

def parse_arguments(parser):
    parser.add_argument('--test_mode', type=int, default=0, help='Test SA-CRNN on MIMIC-III dataset')
    parser.add_argument('--data_path', type=str, metavar='<data_path>', help='The path to the MIMIC-III data directory',default='data/')
    parser.add_argument('--file_name', type=str, metavar='<data_path>', help='File name to save model',default='model/concare')
    parser.add_argument('--small_part', type=int, default=0, help='Use part of training data')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learing rate')

    parser.add_argument('--input_dim', type=int, default=76, help='Dimension of visit record data')
    parser.add_argument('--rnn_dim', type=int, default=384, help='Dimension of hidden units in RNN')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--r_visit', type=int, default=4, help='Compress ration r for visit features')
    parser.add_argument('--r_conv', type=int, default=4, help='Compress ration r for convolutional features')
    parser.add_argument('--kernel_size', type=int, default=2, help='Convolutional kernel size')
    parser.add_argument('--kernel_num', type=int, default=64, help='Number of convolutional filters')
    parser.add_argument('--activation_func', type=str, default='sigmoid', help='Activation function for feature recalibration (sigmoid / sparsemax)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    arg_timestep = 1.0

    # Build readers, discretizers, normalizers
    # train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'train'),
    #                                          listfile=os.path.join(args.data_path, 'train_listfile.csv'),
    #                                          period_length=48.0)
    #
    # val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'train'),
    #                                        listfile=os.path.join(args.data_path, 'val_listfile.csv'),
    #                                        period_length=48.0)

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'train'),
                                             listfile=os.path.join(args.data_path, 'train', 'listfile.csv'),
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'test'),
                                           listfile=os.path.join(args.data_path, 'test', 'listfile.csv'),
                                           period_length=48.0)

    discretizer = Discretizer(timestep=arg_timestep,
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    # normalizer_state = 'ihm_normalizer'
    normalizer_state = 'decomp_normalizer'
    normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
    normalizer.load_params(normalizer_state)

    # %%

    n_trained_chunks = 0
    train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part, return_names=True)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part, return_names=True)

    # %%

    demographic_data = []
    diagnosis_data = []
    idx_list = []

    demo_path = args.data_path + 'demographic/'
    for cur_name in os.listdir(demo_path):
        cur_id, cur_episode = cur_name.split('_', 1)
        cur_episode = cur_episode[:-4]
        cur_file = demo_path + cur_name

        with open(cur_file, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            if header[0] != "Icustay":
                continue
            cur_data = tsfile.readline().strip().split(',')

        if len(cur_data) == 1:
            cur_demo = np.zeros(12)
            cur_diag = np.zeros(128)
        else:
            if cur_data[3] == '':
                cur_data[3] = 60.0
            if cur_data[4] == '':
                cur_data[4] = 160
            if cur_data[5] == '':
                cur_data[5] = 60

            cur_demo = np.zeros(12)
            cur_demo[int(cur_data[1])] = 1
            cur_demo[5 + int(cur_data[2])] = 1
            cur_demo[9:] = cur_data[3:6]
            cur_diag = np.array(cur_data[8:], dtype=np.int)

        demographic_data.append(cur_demo)
        diagnosis_data.append(cur_diag)
        idx_list.append(cur_id + '_' + cur_episode)

    for each_idx in range(9, 12):
        cur_val = []
        for i in range(len(demographic_data)):
            cur_val.append(demographic_data[i][each_idx])
        cur_val = np.array(cur_val)
        _mean = np.mean(cur_val)
        _std = np.std(cur_val)
        _std = _std if _std > 1e-7 else 1e-7
        for i in range(len(demographic_data)):
            demographic_data[i][each_idx] = (demographic_data[i][each_idx] - _mean) / _std

    # %%

    device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
    # device = torch.device('cpu')
    print("available device: {}".format(device))

    # %% md

    ### model

    # %%

    import torch
    from torch import nn
    import torch.nn.utils.rnn as rnn_utils
    from torch.utils import data
    from torch.autograd import Variable
    import torch.nn.functional as F
    from model_ours import AdaCare

    RANDOM_SEED = 12345
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # %%

    def get_loss(y_pred, y_true):
        loss = torch.nn.BCELoss()
        return loss(y_pred, y_true)


    # %%

    model = AdaCare(args.rnn_dim, args.kernel_size, args.kernel_num, args.input_dim, args.output_dim, args.dropout_rate,
                    args.r_visit, args.r_conv, args.activation_func, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # %%

    class Dataset(data.Dataset):
        def __init__(self, x, y, name):
            self.x = x
            self.y = y
            self.name = name

        def __getitem__(self, index):  # 返回的是tensor
            return self.x[index], self.y[index], self.name[index]

        def __len__(self):
            return len(self.x)


    # %%

    train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # %% md

    ### Run

    # %%

    max_roc = 0
    max_prc = 0
    train_loss = []
    train_model_loss = []
    train_decov_loss = []
    valid_loss = []
    valid_model_loss = []
    valid_decov_loss = []
    history = []
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    for each_epoch in range(args.epochs * 2):
        batch_loss = []
        model_batch_loss = []
        decov_batch_loss = []

        model.train()

        for step, (batch_x, batch_y, batch_name) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_demo = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                batch_demo.append(cur_demo)

            batch_demo = torch.stack(batch_demo).to(device)
            output, decov_loss = model(batch_x, batch_demo)

            model_loss = get_loss(output, batch_y.unsqueeze(-1))
            loss = model_loss + 1000 * decov_loss

            batch_loss.append(loss.cpu().detach().numpy())
            model_batch_loss.append(model_loss.cpu().detach().numpy())
            decov_batch_loss.append(decov_loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            if step % 30 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f' % (each_epoch, step, np.mean(np.array(batch_loss))))
                print('Model Loss = %.4f, Decov Loss = %.4f' % (
                np.mean(np.array(model_batch_loss)), np.mean(np.array(decov_batch_loss))))
        train_loss.append(np.mean(np.array(batch_loss)))
        train_model_loss.append(np.mean(np.array(model_batch_loss)))
        train_decov_loss.append(np.mean(np.array(decov_batch_loss)))

        batch_loss = []
        model_batch_loss = []
        decov_batch_loss = []

        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_demo = []
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_demo.append(cur_demo)

                batch_demo = torch.stack(batch_demo).to(device)
                output, decov_loss = model(batch_x, batch_demo)

                model_loss = get_loss(output, batch_y.unsqueeze(-1))

                loss = model_loss + 1000 * decov_loss
                batch_loss.append(loss.cpu().detach().numpy())
                model_batch_loss.append(model_loss.cpu().detach().numpy())
                decov_batch_loss.append(decov_loss.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())

        valid_loss.append(np.mean(np.array(batch_loss)))
        valid_model_loss.append(np.mean(np.array(model_batch_loss)))
        valid_decov_loss.append(np.mean(np.array(decov_batch_loss)))

        print("\n==>Predicting on validation")
        print('Valid Loss = %.4f' % (valid_loss[-1]))
        print('valid_model Loss = %.4f' % (valid_model_loss[-1]))
        print('valid_decov Loss = %.4f' % (valid_decov_loss[-1]))
        y_pred = np.array(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        ret = metrics.print_metrics_binary(y_true, y_pred)
        history.append(ret)
        print()

        cur_auroc = ret['auroc']

        if cur_auroc > max_roc:
            max_roc = cur_auroc
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': each_epoch
            }
            torch.save(state, args.file_name)
            print('\n------------ Save best model ------------\n')

    # %% md

    ### Run for test

    # %%

    checkpoint = torch.load(args.file_name)
    save_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'test'),
                                            listfile=os.path.join(args.data_path, 'test_listfile.csv'),
                                            period_length=48.0)
    test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part, return_names=True)
    test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # %%

    batch_loss = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for step, (batch_x, batch_y, batch_name) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_demo = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                batch_demo.append(cur_demo)

            batch_demo = torch.stack(batch_demo).to(device)
            output = model(batch_x, batch_demo)[0]

            loss = get_loss(output, batch_y.unsqueeze(-1))
            batch_loss.append(loss.cpu().detach().numpy())
            y_pred += list(output.cpu().detach().numpy().flatten())
            y_true += list(batch_y.cpu().numpy().flatten())

    print("\n==>Predicting on test")
    print('Test Loss = %.4f' % (np.mean(np.array(batch_loss))))
    y_pred = np.array(y_pred)
    y_pred = np.stack([1 - y_pred, y_pred], axis=1)
    test_res = metrics.print_metrics_binary(y_true, y_pred)

