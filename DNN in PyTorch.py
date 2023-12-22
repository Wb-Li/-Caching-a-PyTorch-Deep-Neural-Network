import pandas as pd
import numpy as np
import math
import warnings
import datetime
 
from zipline.finance.commission import PerOrder
from zipline.api import get_open_orders
from zipline.api import symbol
 
from bigtrader.sdk import *
from bigtrader.utils.my_collections import NumPyDeque
from bigtrader.constant import OrderType
from bigtrader.constant import Direction

from bigdatasource.api import DataSource
from bigdata.api.datareader import D
from biglearning.api import M
from biglearning.api import tools as T
from biglearning.module2.common.data import Outputs


# @param(id="m8", name="run")
# Python code entry function, input_1/2/3 correspond to three input ports, data_1/2/3 correspond to three output ports
def m8_run_bigquant_run(input_1, input_2, input_3):
    # Example code below. Write your code here
    df = input_1.read()
    df[df['list_days_0'] > 365]
    df[df['pe_ttm_0'] > 0]
    df[df['rank_turn_0'] < 0.5]
    
    # Handling missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    data_1 = DataSource.write_df(df)
    return Outputs(data_1=data_1, data_2=None, data_3=None)

# @param(id="m8", name="post_run")
# Post-processing function, optional. The input is the output of the main function. You can process data here or return a more friendly outputs data format. The output of this function will not be cached.
def m8_post_run_bigquant_run(outputs):
    return outputs

# @param(id="m13", name="run")
# Python code entry function, input_1/2/3 correspond to three input ports, data_1/2/3 correspond to three output ports
def m13_run_bigquant_run(input_1, input_2, input_3):
    # Example code below. Write your code here
    data = input_1.read()
    df = input_2.read()
    columns = list(input_3.read())
    data = pd.merge(data, df, on=['date', 'instrument'], how='inner')
    data = data[columns + ['label']]
    data_1 = DataSource.write_df(data)
    return Outputs(data_1=data_1, data_2=None, data_3=None)

# @param(id="m13", name="post_run")
# Post-processing function, optional. The input is the output of the main function. You can process data here or return a more friendly outputs data format. The output of this function will not be cached.
def m13_post_run_bigquant_run(outputs):
    return outputs

# @param(id="m9", name="run")
# Python code entry function, input_1/2/3 correspond to three input ports, data_1/2/3 correspond to three output ports
def m9_run_bigquant_run(input_1, input_2, input_3):
    # Example code below. Write your code here
    data = input_1.read()
    label_count = data['label'].value_counts().to_dict()       # Count the number of labels
    if label_count[0] > label_count[1]:
        count = label_count[1]
        label_0_df = data[data['label'] == 0]
        label_0_df = label_0_df.sample(n=count)
        df = pd.concat([label_0_df, data[data['label'] == 1]], axis=0)
    else:
        count = label_count[0]
        label_1_df = data[data['label'] == 1]
        label_1_df = label_1_df.sample(n=count)
        df = pd.concat([label_1_df, data[data['label'] == 0]], axis=0)
    data_1 = DataSource.write_df(df)
    return Outputs(data_1=data_1, data_2=None, data_3=None)

# @param(id="m9", name="post_run")
# Post-processing function, optional. The input is the output of the main function. You can process data here or return a more friendly outputs data format. The output of this function will not be cached.
def m9_post_run_bigquant_run(outputs):
    return outputs

# @param(id="m10", name="run")
# Python code entry function, input_1/2/3 correspond to three input ports, data_1/2/3 correspond to three output ports
def m10_run_bigquant_run(input_1, input_2, input_3):
    import os
    import torch
    import torch.nn.init as init
    import numpy as np
    import torch.nn as nn
    import torch.optim as op
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    
    #=================User-defined parameters========================
    path = '/home/aiuser/work/userlib/model.pth'             # Model save path, save model parameters as model.pth, fill None if not saving
    epoch_num = 10                                           # Number of model training epochs
    lr = 1e-2                                                # Learning rate of the model
    class_num = 2                                            # Number of classes for classification
    #=====================================================
    
    # Load data
    data = input_1.read()
    x = np.array(data.drop('label', axis=1))
    y = np.array(data['label'])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    ytrain = np.eye(class_num)[ytrain]      # One-hot encoding, 0 and 1
    ytest = np.eye(class_num)[ytest]
    xtrain = torch.Tensor(xtrain)
    xtest = torch.Tensor(xtest)
    ytrain = torch.Tensor(ytrain)
    ytest = torch.Tensor(ytest)
    
    # Package training set
    train_data = TensorDataset(xtrain, ytrain)
    data_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
    
    # Package test set
    test_data = TensorDataset(xtest, ytest)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=True)
    
    # Define loss function (maximum likelihood negative log loss)
    def loss_fn(ypre, ytrue):
        out = ypre * ytrue
        out = torch.sum(out, axis=1)
        out = -torch.mean(torch.log(out))
        return out
    
    # Model construction
    class net(nn.Module):
        def __init__(self):
            super(net, self).__init__()
            self.fc1 = nn.Linear(xtrain.shape[1], 256)
            self.sig = nn.Sigmoid()
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 2)
            self.soft = nn.Softmax(dim=1)
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.soft(x)
            return x
    
    # Model instance
    model = net()
    
    # Model training
    optimizer = op.SGD(model.parameters(), lr=lr)
    for epoch in range(epoch_num):
        model.train()
        for x, y in data_loader:
            ypre = model(x)
            loss = loss_fn(ypre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Model test
    model.eval()
    with torch.no_grad():
        correct = 0
        for x, y in test_loader:
            ypre = model(x)
            correct += ((torch.argmax(ypre, axis=1) - torch.argmax(y, axis=1)) == 0).sum()
        acc = correct.item() / ytest.shape[0]
        print(f'Test accuracy:{acc}')
    
    # Save the model
    if path != None:
        torch.save(model.state_dict(), path)
    # Turn the model parameters into a tensor format, which will be passed to the next node
    w = []
    b = []
    for p in model.parameters():
        if len(p.data.shape) == 2:
            w.append(p.data)
        elif len(p.data.shape) == 1:
            b.append(p.data)
    w = torch.cat(w, axis=0)
    b = torch.cat(b, axis=0)
    
    # The output of the last node will be passed to the next node, and the shape of the output will be used as the shape parameter of the last node output
    data_1 = DataSource.write_df(data)
    data_2 = DataSource.write_df(pd.DataFrame(w.detach().numpy()))
    data_3 = DataSource.write_df(pd.DataFrame(b.detach().numpy()))
    return Outputs(data_1=data_1, data_2=data_2, data_3=data_3)

# @param(id="m10", name="post_run")
# Post-processing function, optional. The input is the output of the main function. You can process data here or return a more friendly outputs data format. The output of this function will not be cached.
def m10_post_run_bigquant_run(outputs):
    return outputs
