import torch
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset, T_co
import numpy as np
import pandas as pd
from torch.autograd import Variable


def standard_transform(data:np.array, train_index):
    means, stds=[],[]
    train_data=data[:, :train_index[-1][0]+1, :]
    for index in range(data.shape[0]):
        means.append(train_data[index].mean())
        stds.append(train_data[index].std())
        data[index]=(data[index]-means[index])/stds[index]

    return data, means, stds

def inspect_miss_value(data):
    data[0]=np.nan_to_num(data[0], nan=np.nanmean(data[0],dtype=float))
    return data

def get_train_val_test_residential(file_paths='../data/residential',
                                   files=['all_elect.csv','all_weather.csv','all_other_information.csv'],
                                   train_ratio=0.6,valid_ratio=0.2,test_ratio=0.2,history_seq_len=168,
                                   future_seq_len=12):
    data=[]
    for index in range(len(files)):
        tmp=pd.read_csv(file_paths+'/'+files[index])
        tmp=np.array(tmp.iloc[:,2:])
        data.append(tmp)
    print(data)

    # 按窗口提取特征
    L,N = data[0].shape
    num_samples = L-(history_seq_len+future_seq_len)+1
    train_num=round(num_samples*train_ratio)
    valid_num=round(num_samples*valid_ratio)
    test_num=round(num_samples*test_ratio)

    index_list=[]
    for i in range(history_seq_len,num_samples+history_seq_len):
        index=(i-history_seq_len, i, i+future_seq_len)
        index_list.append(index)

    train_index=index_list[:train_num]
    valid_index=index_list[train_num:train_num+valid_num]
    test_index=index_list[train_num+valid_num:train_num+valid_num+test_num]

    all_other_info=np.expand_dims(data[2],axis=1)
    all_data = np.stack((data[0], data[1]))
    #额外信息
    # for index in range(data[2].shape[0]):
    #     all_data=np.concatenate([all_data,
    #                              np.expand_dims(
    #                                  np.tile(all_other_info[index][0],
    #                                          (data[0].shape[0],1)),
    #                                  axis=0)],axis=0)
    print(all_data)
    # 缺失值处理
    # all_data[0]=all_data[0].astype(float).copy()
    all_data=all_data.astype(float)
    all_data=inspect_miss_value(all_data)
    # 训练集部分进行归一化处理
    all_data, means, stds=standard_transform(all_data, train_index)

    # 加入时间周期性信息
    time_in_day = [(i % 24) / 24 for i in range(all_data.shape[1])]
    time_in_day = np.array(time_in_day)
    time_in_day = np.tile(time_in_day, [1, N, 1]).transpose((0, 2, 1))
    all_data = np.concatenate([all_data,time_in_day], axis=0)

    day_in_week = [(i // 24) % 7 for i in range(all_data.shape[1])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((0, 2, 1))
    all_data = np.concatenate([all_data,day_in_week], axis=0)

    return all_data, train_index, valid_index, test_index

class Dataloader_residential(object):
    # 数据集按比例划分为训练集，测试集，验证集
    def __init__(self,user=737,file_paths='/data/residential',
                                       files=['all_elect.csv', 'all_weather.csv', 'all_other_information.csv'],
                                       train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,device='cuda:1',horizon=1,window=168,normalize=2,node_num=936,args=None) -> None:
        super().__init__()
        self.history = window
        self.future = horizon
        self.norm_type = 'max2' if args == None else args.norm_type  # max standard
        self.maxs=None
        self.means=None
        self.stds=None
        self.data, self.all_data_without_miss, self.train_index, self.valid_index, self.test_index = self._get_train_val_test_residential(user=user,future_seq_len=self.future)
        self.data=self.data[args.feature_select]
        self.all_data_without_miss=self.all_data_without_miss[args.feature_select]
        self.node = self.data.shape[2]
        self.node_without_miss=self.all_data_without_miss.shape[2]
        self.dim_info=self.data.shape[0]
        self.device=device
        # self.train_data=self._batchify(self.train_index, self.data, self.node)
        # self.test_data = self._batchify(self.test_index, self.data, self.node)
        # self.valid_data = self._batchify(self.valid_index, self.data, self.node)

        #self.train_data_without_miss = self._batchify(self.train_index, self.all_data_without_miss, self.node_without_miss)
        # self.test_data_without_miss = self._batchify(self.test_index, self.all_data_without_miss, self.node_without_miss)
        # self.valid_data_without_miss = self._batchify(self.valid_index, self.all_data_without_miss, self.node_without_miss)

        calc_tmp=(self._batchify(self.test_index, self.all_data_without_miss, self.node_without_miss))[1]*self.maxs[[0]].squeeze()
        self.rse=self.normal_std(calc_tmp)
        self.rae=torch.mean(torch.abs(calc_tmp-torch.mean(calc_tmp)))

    def normal_std(self,x):
        return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

    def get_batches(self, data, batch_size, shuffle=True):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        i=0
        while i+batch_size<length: #表明不使用i<length
            end_idx=min(length, i+batch_size)
            X=input[index[i:end_idx]]
            Y=target[index[i:end_idx]]
            X=X.to(self.device)
            Y=Y.to(self.device)
            yield Variable(X), Variable(Y)
            i+=batch_size

    def get_batches_index(self, data, batch_size,index_list, shuffle=True, model_type_is_student=True):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        index_list.append(index)
        i=0
        while i+batch_size<=length:
            end_idx=min(length, i+batch_size)
            X=input[index[i:end_idx]]
            Y=target[index[i:end_idx]]
            X=X.to(self.device)
            Y=Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i+=batch_size

    def get_batches_index_without_batchify(self, data_index, batch_size,index_list, shuffle=True, model_type_is_student=True):
        length = len(data_index)

        tmp_x = torch.zeros(batch_size, self.dim_info, self.history, self.node_without_miss)
        tmp_y = torch.zeros(batch_size, 1, self.node_without_miss)
        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        index_list.append(index)
        i=0
        while i+batch_size<=length:
            end_idx=min(length, i+batch_size)
            for d_i in range(i,end_idx):
                tmp_x[d_i-i]=torch.from_numpy(self.all_data_without_miss[:, data_index[index[d_i]][0]:data_index[index[d_i]][1], :])
                tmp_y[d_i-i]=torch.from_numpy(self.all_data_without_miss[0, data_index[index[d_i]][2] - 1, :])
            X=tmp_x
            Y=tmp_y
            X=X.to(self.device)
            Y=Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i+=batch_size

    def _max2_transform(self, data: np.array, train_index):
        maxs = []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        max=np.expand_dims(np.amax(train_data,axis=1),axis=1)
        data /= max
        return data, torch.from_numpy(max)

    def _standard_transform(self,data: np.array, train_index):
        means, stds = [], []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        for index in range(st_length):
            means.append(train_data[index].mean())
            stds.append(train_data[index].std())
            data[index] = (data[index] - means[index]) / stds[index]

        return data, means, stds

    def _inspect_miss_value(self, data):
        data[0] = np.nan_to_num(data[0], nan=np.nanmean(data[0], dtype=float))
        return data

    def _get_train_val_test_residential(self,user=737, root_paths='./data/residential',
                                       files=['all_elect.csv', 'all_temperature.csv', 'all_humidity.csv', 'all_other_information_digitization.csv'],
                                       train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, history_seq_len=168,
                                       future_seq_len=1):
        data = []
        for index in range(len(files)):
            tmp = pd.read_csv(root_paths + '/' + files[index])
            tmp = np.array(tmp.iloc[:, 2:])
            data.append(tmp)
        # print(data ) # elect temp humidity other_information(state climate location)

        # 按窗口提取特征
        L, N = data[0].shape
        num_samples = L - (history_seq_len + future_seq_len) + 1
        train_num = round(num_samples * train_ratio)
        valid_num = round(num_samples * valid_ratio)
        test_num = round(num_samples * test_ratio)

        index_list = []
        for i in range(history_seq_len, num_samples + history_seq_len):
            index = (i - history_seq_len, i, i + future_seq_len)
            index_list.append(index)

        train_index = index_list[:train_num]
        valid_index = index_list[train_num:train_num + valid_num]
        test_index = index_list[train_num + valid_num:train_num + valid_num + test_num]

        all_other_info = np.expand_dims(data[-1], axis=1)
        # [用电，温度，湿度]
        all_data = np.stack((data[0], data[1], data[2]))
        # 额外信息
        # for index in range(data[-1].shape[0]):
        #     all_data=np.concatenate([all_data,
        #                              np.expand_dims(
        #                                  np.tile(all_other_info[index][0],
        #                                          (data[0].shape[0],1)),
        #                                  axis=0)],axis=0)
        print('all_data_shape:{}'.format(all_data.shape))
        # 缺失值处理
        # all_data[0]=all_data[0].astype(float).copy()
        all_data = all_data.astype(float)
        all_data = self._inspect_miss_value(all_data)


        # 加入时间周期性信息
        time_in_day = [(i % 24) for i in range(all_data.shape[1])] # 需要注意？
        time_in_day = np.array(time_in_day)
        time_in_day = np.tile(time_in_day, [1, N, 1]).transpose((0, 2, 1))
        all_data = np.concatenate([all_data, time_in_day], axis=0)

        day_in_week = [(i // 24) % 7 for i in range(all_data.shape[1])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((0, 2, 1))
        all_data = np.concatenate([all_data, day_in_week], axis=0)

        # 训练集部分进行归一化处理
        if self.norm_type=='max':
            all_data, self.maxs = self._max_transform(all_data, train_index)  # 最大值归一化
            self.maxs = self.maxs[..., 0:user]
        elif self.norm_type=='max_2':
            all_data,self.max = self._max2_transform(all_data, train_index)
            self.max=self.max[...,0:user]
            self.maxs=self.max
        else:
            all_data, self.means, self.stds = self._standard_transform(all_data, train_index) # z-score 归一化

        # 做用户选取，选择无气候，区域缺失的用户信息。 条件选择
        all_data_without_miss = all_data[:,:,(all_data[-3]!=-1)[0]]

        # other_information(state climate location) 只使用state信息
        all_data_without_miss=all_data_without_miss[:,:,0:user]
        all_data=all_data[:,:,:]

        return all_data, all_data_without_miss, train_index, valid_index, test_index

    def _max_transform(self,data: np.array, train_index):
        maxs = []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        for index in range(data.shape[2]):
            maxs.append(train_data[:,:,index].max())
            data[:,:,index] = (data[:,:,index]) / maxs[index]
        return data, torch.from_numpy(np.array(maxs))

    def _batchify(self,index, data,node,single_predict=True):
        if single_predict:
            length=1
        else:
            length=self.future
        X=torch.zeros(len(index),self.dim_info,self.history, node)
        Y=torch.zeros(len(index),length,node)
        for i in range(len(index)):
            X[i,:,:,:]=torch.from_numpy(data[:,index[i][0]:index[i][1],:])
            if single_predict:
                Y[i,:,:] = torch.from_numpy(data[0,[index[i][2]-1],:]) #只对一个变量进行预测
            else:
                Y[i, :, :] = torch.from_numpy(data[0, index[i][1]:index[i][2], :])
        return [X,Y]

class Dataloader_hidden_feature(object):
    def __init__(self,data,features,originDataLoader,device='cuda:1',horizon=1,window=168,args=None,normalize=2,node_num=936) -> None:
        super().__init__()
        self.history = window
        self.future = originDataLoader.future
        self.data, self.features, self.train_index,  self.valid_index, self.test_index = self._get_train_val_test_hidden_feature(data,features,future_seq_len=self.future)
        self.node = self.data.shape[2]
        self.node_without_miss = self.features.shape[2]
        self.dim_info = self.data.shape[0]
        self.rep_dim=64
        self.device = device
        self.norm_type=originDataLoader.norm_type
        if (originDataLoader.norm_type=='max'):
            self.maxs=originDataLoader.maxs
        elif(originDataLoader.norm_type=='max_2'):
            self.max=originDataLoader.max
        else:
            self.means=originDataLoader.means
            self.stds=originDataLoader.stds
        # self.train_data=self._batchify(self.train_index, self.data, self.node)
        # self.test_data = self._batchify(self.test_index, self.data, self.node)
        # self.valid_data = self._batchify(self.valid_index, self.data, self.node)

        # self.train_data_without_miss = self._batchify(self.train_index, self.features,self.data,
        #                                               self.node_without_miss)
        # self.test_data_without_miss = self._batchify(self.test_index, self.features,self.data,
        #                                              self.node_without_miss)
        # self.valid_data_without_miss = self._batchify(self.valid_index, self.features,self.data,
        #                                               self.node_without_miss)

    def _get_train_val_test_hidden_feature(self,data, features, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, history_seq_len=168,
                                  future_seq_len=1):

        data = np.transpose(data,(2,1,0))
        features = np.transpose(features,(2,1,0))
        # 按窗口提取特征
        L, N = data[0].shape
        num_samples = L - (history_seq_len + future_seq_len) + 1
        train_num = round(num_samples * train_ratio)
        valid_num = round(num_samples * valid_ratio)
        test_num = round(num_samples * test_ratio)

        index_list = []
        for i in range(history_seq_len, num_samples + history_seq_len):
            index = (i - history_seq_len, i, i + future_seq_len)
            index_list.append(index)

        train_index = index_list[:train_num]
        valid_index = index_list[train_num:train_num + valid_num]
        test_index = index_list[train_num + valid_num:train_num + valid_num + test_num]

        # 做用户选取，选择无气候，区域缺失的用户信息。 条件选择
        all_data_without_miss = data
        return data, features, train_index, valid_index, test_index

    def get_batches_index(self, data, batch_size, index_list, shuffle=True, model_type_is_student=False):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        index_list.append(index)
        i = 0
        while i + batch_size <= length:
            end_idx = min(length, i + batch_size)
            X = input[index[i:end_idx]]
            Y = target[index[i:end_idx]]
            X = X.to(self.device)
            Y = Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i += batch_size

    def get_batches_index_without_batchify(self, data_index, batch_size,index_list, shuffle=True, model_type_is_student=True):
        length = len(data_index)

        tmp_x = torch.zeros(batch_size, self.rep_dim, 1, self.node_without_miss)
        tmp_y = torch.zeros(batch_size, 1, self.node_without_miss)
        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        index_list.append(index)
        i=0
        while i+batch_size<=length:
            end_idx=min(length, i+batch_size)
            for d_i in range(i,end_idx):
                tmp_x[d_i-i]=torch.from_numpy(self.features[:, data_index[index[d_i]][1]-1:data_index[index[d_i]][1], :])
                tmp_y[d_i-i]=torch.from_numpy(self.data[0, data_index[index[d_i]][2] - 1, :])
            X=tmp_x
            Y=tmp_y
            X=X.to(self.device)
            Y=Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i+=batch_size

    def _batchify(self, index,features, data, node,single_predict=True):
        if single_predict:
            length=1
        else:
            length=self.future
        X = torch.zeros(len(index), self.rep_dim,1, node)
        Y = torch.zeros(len(index), length, node)
        for i in range(len(index)):
            X[i, :, :] = torch.from_numpy(features[:, index[i][1]-1:index[i][1], :])
            if single_predict:
                Y[i,:,:] = torch.from_numpy(data[0,[index[i][2]-1],:]) # 只对一个变量进行预测
            else:
                Y[i, :, :] = torch.from_numpy(data[0, index[i][1]:index[i][2], :])
        return [X, Y]

class Dataloader_ideal(object):
    # 数据集按比例划分为训练集，测试集，验证集
    def __init__(self,user=55,file_paths='/data/residential',
                                       files=['all_elect.csv', 'all_weather.csv', 'all_other_information.csv'],
                                       train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,device='cuda:1',horizon=1,window=168,normalize=2,node_num=936,args=None) -> None:
        super().__init__()
        self.history = window
        self.future = horizon
        self.norm_type = 'max_2' if args == None else args.norm_type  # max standard
        self.data, self.all_data_without_miss, self.train_index, self.valid_index, self.test_index = self._get_train_val_test_ideal(norm_type=self.norm_type,future_seq_len=self.future)
        if (args!=None):
            self.data=self.data[args.feature_select]
            self.all_data_without_miss=self.all_data_without_miss[args.feature_select]
        self.node = self.data.shape[2]
        self.node_without_miss=self.all_data_without_miss.shape[2]
        self.dim_info=self.data.shape[0]
        self.device=device
        # self.train_data=self._batchify(self.train_index, self.data, self.node)
        # self.test_data = self._batchify(self.test_index, self.data, self.node)
        # self.valid_data = self._batchify(self.valid_index, self.data, self.node)

        self.train_data_without_miss = self._batchify(self.train_index, self.all_data_without_miss, self.node_without_miss)
        self.test_data_without_miss = self._batchify(self.test_index, self.all_data_without_miss, self.node_without_miss)
        self.valid_data_without_miss = self._batchify(self.valid_index, self.all_data_without_miss, self.node_without_miss)

        calc_tmp=self.test_data_without_miss[1]*self.maxs[[0]].squeeze()
        self.rse=self.normal_std(calc_tmp)
        self.rae=torch.mean(torch.abs(calc_tmp-torch.mean(calc_tmp)))

    def normal_std(self,x):
        return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

    def _max_transform(self,data: np.array, train_index):
        maxs = []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        for index in range(data.shape[2]):
            maxs.append(train_data[:,:,index].max())
            data[:,:,index] = (data[:,:,index]) / maxs[index]
        return data, torch.from_numpy(np.array(maxs))

    def get_batches(self, data, batch_size, shuffle=True):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        i=0
        while i+batch_size<length: # 表明不使用i<length
            end_idx=min(length, i+batch_size)
            X=input[index[i:end_idx]]
            Y=target[index[i:end_idx]]
            X=X.to(self.device)
            Y=Y.to(self.device)
            yield Variable(X), Variable(Y)
            i+=batch_size

    def get_batches_index(self, data, batch_size,index_list, shuffle=True, model_type_is_student=True):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        index_list.append(index)
        i=0
        while i+batch_size<=length:
            end_idx=min(length, i+batch_size)
            X=input[index[i:end_idx]]
            Y=target[index[i:end_idx]]
            X=X.to(self.device)
            Y=Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i+=batch_size

    def _standard_transform(self,data: np.array, train_index):
        means, stds = [], []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        for index in range(st_length):
            means.append(train_data[index].mean())
            stds.append(train_data[index].std())
            data[index] = (data[index] - means[index]) / stds[index]

        return data, means, stds

    def _max2_transform(self, data: np.array, train_index):
        maxs = []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        max=np.expand_dims(np.amax(train_data,axis=1),axis=1)
        data /= max
        return data, torch.from_numpy(max)

    def _inspect_miss_value(self, data):
        for i in range(data.shape[0]):
            data[i] = np.nan_to_num(data[i], nan=np.nanmean(data[i], dtype=float))
        return data



    def _get_train_val_test_ideal(self, norm_type='max',root_paths='data/ideal_handle',
                                   # files=['all_elect.csv', 'all_temperature.csv', 'all_humidity.csv', 'all_location.csv', 'all_datetime.csv'],_3039_34
                                  files=['all_elect_3039_34.csv', 'all_temperature_3039_34.csv', 'all_humidity_3039_34.csv', 'all_location_3039_34.csv',
                                         'all_datetime_3039_34.csv'],
                                   train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, history_seq_len=168,
                                   future_seq_len=1):
        data = []
        for index in range(len(files)):
            tmp = pd.read_csv(root_paths + '/' + files[index])
            tmp = np.array(tmp.iloc[:, 1:])
            data.append(tmp)
        # print(data)  # elect temp humidity other_information(state climate location)


        # 按窗口提取特征
        L, N = data[0].shape
        num_samples = L - (history_seq_len + future_seq_len) + 1
        train_num = round(num_samples * train_ratio)
        valid_num = round(num_samples * valid_ratio)
        test_num = round(num_samples * test_ratio)

        index_list = []
        for i in range(history_seq_len, num_samples + history_seq_len):
            index = (i - history_seq_len, i, i + future_seq_len)
            index_list.append(index)

        train_index = index_list[:train_num]
        valid_index = index_list[train_num:train_num + valid_num]
        test_index = index_list[train_num + valid_num:train_num + valid_num + test_num]

        all_other_info = np.expand_dims(data[-1], axis=1)
        # [用电，温度，湿度]
        all_data = np.stack((data[0], data[1], data[2]))
        # 额外信息
        # all_data = np.concatenate([all_data,
        #                            np.expand_dims(
        #                                np.tile(np.transpose(data[3]), (data[0].shape[0], 1)), axis=0)], axis=0)

        # print(all_data.shape)
        # 缺失值处理
        # all_data[0]=all_data[0].astype(float).copy()
        all_data[0:4] = all_data[0:4].astype(float)
        all_data = self._inspect_miss_value(all_data)

        # 加入时间周期性信息
        time_in_day = [(i % 24) for i in range(all_data.shape[1])]  # 需要注意？
        time_in_day = np.array(time_in_day)
        time_in_day = np.tile(time_in_day, [1, N, 1]).transpose((0, 2, 1))
        all_data = np.concatenate([all_data, time_in_day], axis=0)

        day_in_week = [(i // 24) % 7 for i in range(all_data.shape[1])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((0, 2, 1))
        all_data = np.concatenate([all_data, day_in_week], axis=0)

        # 训练集部分进行归一化处理
        if norm_type=='max':
            all_data, self.maxs = self._max_transform(all_data, train_index)  # 最大值归一化
        elif norm_type=='max_2':
            all_data,self.max = self._max2_transform(all_data, train_index)
            self.maxs=self.max
        else:
            all_data, self.means, self.stds = self._standard_transform(all_data, train_index) # z-score 归一化

        # all_data, self.means, self.stds = self._standard_transform(all_data, train_index)

        # 做用户选取，选择无气候，区域缺失的用户信息。 条件选择
        all_data_without_miss = all_data[:, :, (all_data[-3] != -1)[0]]

        return all_data, all_data_without_miss, train_index, valid_index, test_index


    def _batchify(self,index, data,node,single_predict=True):
        if single_predict:
            length=1
        else:
            length=self.future
        X=torch.zeros(len(index),self.dim_info,self.history, node)
        Y=torch.zeros(len(index),length,node)
        for i in range(len(index)):
            X[i,:,:,:]=torch.from_numpy(data[:,index[i][0]:index[i][1],:])
            if single_predict:
                Y[i,:,:] = torch.from_numpy(data[0,[index[i][2]-1],:]) # 只对一个变量进行预测
            else:
                Y[i, :, :] = torch.from_numpy(data[0, index[i][1]:index[i][2], :])
        return [X,Y]

def get_train_val_test_ideal(root_paths='../data/ideal_handle',
                                   files=['all_elect.csv', 'all_temperature.csv', 'all_humidity.csv', 'all_location.csv', 'all_datetime.csv'],
                                   train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, history_seq_len=168,
                                   future_seq_len=1):
    data = []
    for index in range(len(files)):
        tmp = pd.read_csv(root_paths + '/' + files[index])
        tmp = np.array(tmp.iloc[:, 1:])
        data.append(tmp)
    print(data) # elect temp humidity other_information(state climate location)

    # 按窗口提取特征
    L, N = data[0].shape
    num_samples = L - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = round(num_samples * test_ratio)

    index_list = []
    for i in range(history_seq_len, num_samples + history_seq_len):
        index = (i - history_seq_len, i, i + future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num:train_num + valid_num]
    test_index = index_list[train_num + valid_num:train_num + valid_num + test_num]

    all_other_info = np.expand_dims(data[-1], axis=1)
    # [用电，温度，湿度]
    all_data = np.stack((data[0], data[1], data[2]))
    # 额外信息
    all_data=np.concatenate([all_data,
                    np.expand_dims(
                        np.tile(np.transpose(data[3]), (data[0].shape[0], 1)), axis=0)], axis=0)

    print(all_data.shape)
    # 缺失值处理
    # all_data[0]=all_data[0].astype(float).copy()
    all_data = all_data.astype(float)
    all_data = self._inspect_miss_value(all_data)
    # 训练集部分进行归一化处理
    all_data, self.means, self.stds = self._standard_transform(all_data, train_index)

    # 加入时间周期性信息
    time_in_day = [(i % 24) / 24 for i in range(all_data.shape[1])] # 需要注意？
    time_in_day = np.array(time_in_day)
    time_in_day = np.tile(time_in_day, [1, N, 1]).transpose((0, 2, 1))
    all_data = np.concatenate([all_data, time_in_day], axis=0)

    day_in_week = [(i // 24) % 7 for i in range(all_data.shape[1])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((0, 2, 1))
    all_data = np.concatenate([all_data, day_in_week], axis=0)

    # 做用户选取，选择无气候，区域缺失的用户信息。 条件选择
    all_data_without_miss = all_data[:,:,(all_data[-3]!=-1)[0]]

    return all_data, all_data_without_miss, train_index, valid_index, test_index

class Dataloader_HDFB(object):
    # 数据集按比例划分为训练集，测试集，验证集
    def __init__(self,file_paths='/data/residential',
                                       files=['all_elect.csv', 'all_weather.csv', 'all_other_information.csv'],
                                       train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,device='cuda:1',horizon=1,window=168,normalize=2,node_num=936,args=None) -> None:
        super().__init__()
        self.history = window
        self.future = horizon
        self.norm_type = 'max_2' if args == None else args.norm_type  # max standard
        self.maxs=None
        self.means=None
        self.stds=None
        self.data, self.all_data_without_miss, self.train_index, self.valid_index, self.test_index = self._get_train_val_test_hdfb(norm_type=self.norm_type,future_seq_len=self.future)
        self.data=self.data[args.feature_select]
        self.all_data_without_miss=self.all_data_without_miss[args.feature_select]
        self.node = self.data.shape[2]
        self.node_without_miss=self.all_data_without_miss.shape[2]
        self.dim_info=self.data.shape[0]
        self.device=device
        # self.train_data=self._batchify(self.train_index, self.data, self.node)
        # self.test_data = self._batchify(self.test_index, self.data, self.node)
        # self.valid_data = self._batchify(self.valid_index, self.data, self.node)

        self.train_data_without_miss = self._batchify(self.train_index, self.all_data_without_miss, self.node_without_miss)
        self.test_data_without_miss = self._batchify(self.test_index, self.all_data_without_miss, self.node_without_miss)
        self.valid_data_without_miss = self._batchify(self.valid_index, self.all_data_without_miss, self.node_without_miss)

        calc_tmp=self.test_data_without_miss[1]*self.maxs[[0]].squeeze()
        self.rse=self.normal_std(calc_tmp)
        self.rae=torch.mean(torch.abs(calc_tmp-torch.mean(calc_tmp)))

    def normal_std(self,x):
        return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

    def get_batches(self, data, batch_size, shuffle=True):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        i=0
        while i+batch_size<length: # 表明不使用i<length
            end_idx=min(length, i+batch_size)
            X=input[index[i:end_idx]]
            Y=target[index[i:end_idx]]
            X=X.to(self.device)
            Y=Y.to(self.device)
            yield Variable(X), Variable(Y)
            i+=batch_size

    def get_batches_index(self, data, batch_size,index_list, shuffle=True, model_type_is_student=True):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        index_list.append(index)
        i=0
        while i+batch_size<=length:
            end_idx=min(length, i+batch_size)
            X=input[index[i:end_idx]]
            Y=target[index[i:end_idx]]
            # if single_predict:
            #     Y=Y[:,-1,:]
            X=X.to(self.device)
            Y=Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i+=batch_size

    def get_batches_index_without_batchify(self, data_index, batch_size,index_list, shuffle=True, model_type_is_student=True):
        length = len(data_index)

        tmp_x = torch.zeros(batch_size, self.dim_info, self.history, self.node_without_miss)
        tmp_y = torch.zeros(batch_size, 1, self.node_without_miss)
        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        index_list.append(index)
        i=0
        while i+batch_size<=length:
            end_idx=min(length, i+batch_size)
            for d_i in range(i,end_idx):
                tmp_x[d_i-i]=torch.from_numpy(self.all_data_without_miss[:, data_index[index[d_i]][0]:data_index[index[d_i]][1], :])
                tmp_y[d_i-i]=torch.from_numpy(self.all_data_without_miss[0, data_index[index[d_i]][2] - 1, :])
            X=tmp_x
            Y=tmp_y
            X=X.to(self.device)
            Y=Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i+=batch_size

    def _max_transform(self,data: np.array, train_index):
        maxs = []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        for index in range(data.shape[2]):
            maxs.append(train_data[:,:,index].max())
            data[:,:,index] = (data[:,:,index]) / maxs[index]
        return data, torch.from_numpy(np.array(maxs))

    def _max2_transform(self, data: np.array, train_index):
        maxs = []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        max=np.expand_dims(np.amax(train_data,axis=1),axis=1)
        data /= max
        return data, torch.from_numpy(max)

    def _standard_transform(self,data: np.array, train_index):
        means, stds = [], []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        for index in range(st_length):
            means.append(train_data[index].mean())
            stds.append(train_data[index].std())
            data[index] = (data[index] - means[index]) / stds[index]

        return data, means, stds

    def _inspect_miss_value(self, data):
        for i in range(data.shape[0]):
            data[i] = np.nan_to_num(data[i], nan=np.nanmean(data[i], dtype=float))
        return data

    def _get_train_val_test_hdfb(self, root_paths='data/HDFB',norm_type='max',
                                   files=['all_elect.csv', 'all_temperature.csv', 'all_humidity.csv',],
                                   train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, history_seq_len=168,
                                   future_seq_len=1):
        data = []
        for index in range(len(files)):
            tmp = pd.read_csv(root_paths + '/' + files[index])
            tmp = np.array(tmp.iloc[:, 1:])
            data.append(tmp)
        # print(data)  # elect temp humidity other_information(state climate location)


        # 按窗口提取特征
        L, N = data[0].shape
        num_samples = L - (history_seq_len + future_seq_len) + 1
        train_num = round(num_samples * train_ratio)
        valid_num = round(num_samples * valid_ratio)
        test_num = round(num_samples * test_ratio)

        index_list = []
        for i in range(history_seq_len, num_samples + history_seq_len):
            index = (i - history_seq_len, i, i + future_seq_len)
            index_list.append(index)

        train_index = index_list[:train_num]
        valid_index = index_list[train_num:train_num + valid_num]
        test_index = index_list[train_num + valid_num:train_num + valid_num + test_num]

        all_other_info = np.expand_dims(data[-1], axis=1)
        # [用电，温度，湿度]
        all_data = np.stack((data[0], data[1], data[2]))
        # 额外信息
        # all_data = np.concatenate([all_data,
        #                            np.expand_dims(
        #                                np.tile(np.transpose(data[3]), (data[0].shape[0], 1)), axis=0)], axis=0)

        # print(all_data.shape)
        # 缺失值处理
        # all_data[0]=all_data[0].astype(float).copy()
        all_data[0:4] = all_data[0:4].astype(float)
        all_data = self._inspect_miss_value(all_data)
        # if norm_type=='max':
        #     all_data, self.maxs = self._max_transform(all_data, train_index)  # 最大值归一化
        # elif norm_type=='max_2':
        #     all_data,self.max = self._max2_transform(all_data, train_index)
        # else:
        #     all_data, self.means, self.stds = self._standard_transform(all_data, train_index) # z-score 归一化

        # 加入时间周期性信息
        time_in_day = [(i % 24 )for i in range(all_data.shape[1])]  # 需要注意？
        time_in_day = np.array(time_in_day)
        time_in_day = np.tile(time_in_day, [1, N, 1]).transpose((0, 2, 1))
        all_data = np.concatenate([all_data, time_in_day], axis=0)

        day_in_week = [(i // 24) % 7 for i in range(all_data.shape[1])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((0, 2, 1))
        all_data = np.concatenate([all_data, day_in_week], axis=0)

        if norm_type=='max':
            all_data, self.maxs = self._max_transform(all_data, train_index)  # 最大值归一化
        elif norm_type=='max_2':
            all_data,self.max = self._max2_transform(all_data, train_index)
            self.maxs=self.max
        else:
            all_data, self.means, self.stds = self._standard_transform(all_data, train_index) # z-score 归一化

        # 做用户选取，选择无气候，区域缺失的用户信息。 条件选择
        all_data_without_miss = all_data

        return all_data, all_data_without_miss, train_index, valid_index, test_index


    def _batchify(self,index, data,node,single_predict=True):
        if single_predict:
            length=1
        else:
            length=self.future
        X=torch.zeros(len(index),self.dim_info,self.history, node)
        Y=torch.zeros(len(index),length,node)
        for i in range(len(index)):
            X[i,:,:,:]=torch.from_numpy(data[:,index[i][0]:index[i][1],:])
            if single_predict:
                Y[i,:,:] = torch.from_numpy(data[0,[index[i][2]-1],:]) # 只对一个变量进行预测
            else:
                Y[i, :, :] = torch.from_numpy(data[0, index[i][1]:index[i][2], :])
        return [X,Y]

class Dataloader_electricity(object):
    # 数据集按比例划分为训练集，测试集，验证集
    def __init__(self,user=321,file_paths='/data',
                                       files=['all_elect.csv', 'all_weather.csv', 'all_other_information.csv'],
                                       train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,device='cuda:1',horizon=1,window=168,normalize=2,node_num=936,args=None) -> None:
        super().__init__()
        self.history = window
        self.future = horizon
        self.norm_type = 'max_2' if args == None else args.norm_type  # max standard
        self.maxs=None
        self.means=None
        self.stds=None
        self.data, self.all_data_without_miss, self.train_index, self.valid_index, self.test_index = self._get_train_val_test_electricity(user=user,norm_type=self.norm_type,future_seq_len=self.future)
        if (args!=None):
            self.data=self.data[args.feature_select]
            self.all_data_without_miss=self.all_data_without_miss[args.feature_select]
        self.node = self.data.shape[2]
        self.node_without_miss=self.all_data_without_miss.shape[2]
        self.dim_info=self.data.shape[0]
        self.device=device
        # self.train_data=self._batchify(self.train_index, self.data, self.node)
        # self.test_data = self._batchify(self.test_index, self.data, self.node)
        # self.valid_data = self._batchify(self.valid_index, self.data, self.node)

        # self.train_data_without_miss = self._batchify(self.train_index, self.all_data_without_miss, self.node_without_miss)
        # self.test_data_without_miss = self._batchify(self.test_index, self.all_data_without_miss, self.node_without_miss)
        # self.valid_data_without_miss = self._batchify(self.valid_index, self.all_data_without_miss, self.node_without_miss)

        calc_tmp=(self._batchify(self.test_index, self.all_data_without_miss, self.node_without_miss))[1]*self.maxs[[0]].squeeze()
        self.rse=self.normal_std(calc_tmp)
        self.rae=torch.mean(torch.abs(calc_tmp-torch.mean(calc_tmp)))

    def normal_std(self,x):
        return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

    def get_batches(self, data, batch_size, shuffle=True):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        i=0
        while i+batch_size<length: #表明不使用i<length
            end_idx=min(length, i+batch_size)
            X=input[index[i:end_idx]]
            Y=target[index[i:end_idx]]
            X=X.to(self.device)
            Y=Y.to(self.device)
            yield Variable(X), Variable(Y)
            i+=batch_size

    def get_batches_index(self, data, batch_size,index_list, shuffle=True, model_type_is_student=True):
        length = data[0].shape[0]
        input = data[0]
        target = data[1]

        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        index_list.append(index)
        i=0
        while i+batch_size<=length:
            end_idx=min(length, i+batch_size)
            X=input[index[i:end_idx]]
            Y=target[index[i:end_idx]]
            X=X.to(self.device)
            Y=Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i+=batch_size

    def get_batches_index_without_batchify(self, data_index, batch_size,index_list, shuffle=True, model_type_is_student=True):
        length = len(data_index)

        tmp_x = torch.zeros(batch_size, self.dim_info, self.history, self.node_without_miss)
        tmp_y = torch.zeros(batch_size, 1, self.node_without_miss)
        if shuffle:
            index=torch.randperm(length)
        else:
            index=torch.LongTensor(range(length))
        index_list.append(index)
        i=0
        while i+batch_size<=length:
            end_idx=min(length, i+batch_size)
            for d_i in range(i,end_idx):
                tmp_x[d_i-i]=torch.from_numpy(self.all_data_without_miss[:, data_index[index[d_i]][0]:data_index[index[d_i]][1], :])
                tmp_y[d_i-i]=torch.from_numpy(self.all_data_without_miss[0, data_index[index[d_i]][2] - 1, :])
            X=tmp_x
            Y=tmp_y
            X=X.to(self.device)
            Y=Y.to(self.device)
            if (model_type_is_student):
                yield Variable(X), Variable(Y), i, end_idx
            else:
                yield Variable(X), Variable(Y)
            i+=batch_size

    def _standard_transform(self,data: np.array, train_index):
        means, stds = [], []
        # 对用电 温度， 湿度 进行归一化
        st_length=1
        train_data = data[:, :train_index[-1][0] + 1, :]
        for index in range(st_length):
            means.append(train_data[index].mean())
            stds.append(train_data[index].std())
            data[index] = (data[index] - means[index]) / stds[index]
        return data, means, stds

    def _max_transform(self,data: np.array, train_index):
        maxs = []
        # 对用电 温度， 湿度 进行归一化
        st_length=1
        train_data = data[:, :train_index[-1][0] + 1, :]
        for index in range(data.shape[2]):
            maxs.append(train_data[:,:,index].max())
            data[:,:,index] = (data[:,:,index]) / maxs[index]
        return data, torch.from_numpy(np.array(maxs))

    def _max2_transform(self, data: np.array, train_index):
        maxs = []
        # 对用电 温度， 湿度 进行归一化
        st_length=3
        train_data = data[:, :train_index[-1][0] + 1, :]
        max=np.expand_dims(np.amax(train_data,axis=1),axis=1)
        data /= max
        return data, torch.from_numpy(max)

    def _inspect_miss_value(self, data):
        data[0] = np.nan_to_num(data[0], nan=np.nanmean(data[0], dtype=float))
        return data

    def _get_train_val_test_electricity(self,user=737, norm_type='max',root_paths='./data/',
                                       files=['electricity.csv'],
                                       train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, history_seq_len=168,
                                       future_seq_len=1):
        data = []
        for index in range(len(files)):
            tmp = pd.read_csv(root_paths + '/' + files[index],header=None)
            tmp = np.array(tmp.iloc[:, :])
            data.append(tmp)
        # print(data ) # elect temp humidity other_information(state climate location)

        # 按窗口提取特征
        L, N = data[0].shape
        num_samples = L - (history_seq_len + future_seq_len) + 1
        train_num = round(num_samples * train_ratio)
        valid_num = round(num_samples * valid_ratio)
        test_num = round(num_samples * test_ratio)

        index_list = []
        for i in range(history_seq_len, num_samples + history_seq_len):
            index = (i - history_seq_len, i, i + future_seq_len)
            index_list.append(index)

        train_index = index_list[:train_num]
        valid_index = index_list[train_num:train_num + valid_num]
        test_index = index_list[train_num + valid_num:train_num + valid_num + test_num]

        all_other_info = np.expand_dims(data[-1], axis=1)
        # [用电，温度，湿度]
        all_data = np.stack((data[0]))
        all_data = np.expand_dims(all_data, axis=0)
        print('all_data_shape:{}'.format(all_data.shape))
        # 缺失值处理
        # all_data[0]=all_data[0].astype(float).copy()
        all_data = all_data.astype(float)
        all_data = self._inspect_miss_value(all_data)

        # 加入时间周期性信息
        time_in_day = [(i % 24) for i in range(all_data.shape[1])] # 需要注意？
        time_in_day = np.array(time_in_day)
        time_in_day = np.tile(time_in_day, [1, N, 1]).transpose((0, 2, 1))
        all_data = np.concatenate([all_data, time_in_day], axis=0)

        day_in_week = [(i // 24) % 7 for i in range(all_data.shape[1])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((0, 2, 1))
        all_data = np.concatenate([all_data, day_in_week], axis=0)

        if norm_type=='max':
            all_data, self.maxs = self._max_transform(all_data, train_index)  # 最大值归一化
        elif norm_type=='max_2':
            all_data,self.max = self._max2_transform(all_data, train_index)
            self.maxs=self.max
        else:
            all_data, self.means, self.stds = self._standard_transform(all_data, train_index) # z-score 归一化

        # 做用户选取，选择无气候，区域缺失的用户信息。 条件选择
        all_data_without_miss = all_data[:,:,(all_data[-3]!=-1)[0]]

        # other_information(state climate location) 只使用state信息
        all_data_without_miss=all_data_without_miss[:,:,0:user]

        return all_data, all_data_without_miss, train_index, valid_index, test_index


    def _batchify(self,index, data,node,single_predict=True):
        if single_predict:
            length=1
        else:
            length=self.future
        X=torch.zeros(len(index),self.dim_info,self.history, node)
        Y=torch.zeros(len(index),length,node)
        for i in range(len(index)):
            X[i,:,:,:]=torch.from_numpy(data[:,index[i][0]:index[i][1],:])
            if single_predict:
                Y[i,:,:] = torch.from_numpy(data[0,[index[i][2]-1],:]) # 只对一个变量进行预测
            else:
                Y[i, :, :] = torch.from_numpy(data[0, index[i][1]:index[i][2], :])
        return [X,Y]
# dataloader=Dataloader_ideal()
# dataloader.get_train_val_test_ideal()
