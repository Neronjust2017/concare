import numpy as np
import os
import pandas as pd

path = '/home/weiyuhua/Code/concare/in-hospital-mortality'

train_pth = os.path.join(path, "train")
test_pth = os.path.join(path, "test")

train_listfile = os.path.join(train_pth, "listfile.csv")
test_listfile = os.path.join(test_pth, "listfile.csv")

with open(train_listfile, "r") as lfile:
    train_list = lfile.readlines()[1:]

with open(test_listfile, "r") as lfile:
    test_list = lfile.readlines()[1:]

train_data = os.listdir(train_pth)
test_data = os.listdir(test_pth)

for k in range(5):
    # train_val proportion: 0.85:0.15
    N = len(train_data)
    np.random.seed(k)
    ind = np.random.permutation(N)
    train_ind = list(ind[:int(0.85*N)])
    val_ind = list(ind[int(0.85*N):])

    train_file = []
    val_file = []
    for i in train_ind:
        train_file.append(train_data[i])
    for i in val_ind:
        val_file.append(train_data[i])

    train_list_new = []
    val_list_new = []

    print(len(train_list))
    for i in range(len(train_list)):
    # for i in range(10000):
        print(i)
        csv, y = train_list[i].split(',')
        y = int(y[0])
        if csv in val_file:
            val_list_new.append([csv, y])
        else:
            train_list_new.append([csv, y])

    name = ['stay','y_true']
    train_list_df = pd.DataFrame(columns=name, data=train_list_new)
    val_list_df = pd.DataFrame(columns=name, data=val_list_new)

    train_list_df.to_csv('in-hospital-mortality/train_listfile_{}.csv'.format(k), index=False)
    val_list_df.to_csv('in-hospital-mortality/val_listfile_{}.csv'.format(k), index=False)










