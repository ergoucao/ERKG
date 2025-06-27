import os

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils.timefeatures import time_features

from util import StandardScaler
from sklearn.preprocessing import StandardScaler as standScaler


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode, electricity_path, electricity_type_path):
        super(BaseDataset, self).__init__()
        self.data, self.status_type, self.date_features = self.load_data(electricity_path = electricity_path, electrcity_type_path = electricity_type_path) # './data/AMPds2/load_type_1h_dif_total_status_windows=1.csv'
        # self.data = np.delete(self.data,12,axis=1)
        # self.status_type = np.delete(self.status_type,12,axis=1)
        # self.date_features = np.delete(self.date_features,12,axis=1)
        self.status_num = list((torch.max(self.status_type, dim=0)[0].numpy()) + 1)
        self.window_size = config.seq_len
        self.horizon = config.horizon
        self.pred_len = config.pred_len
        self.future = self.horizon + self.pred_len - 1
        # default train set if mode == 'train':
        start = 0
        end = int(self.data.shape[0] * 0.6)
        self.max, self.max_idx = torch.max(self.data[start:end], dim=0)
        self.min, self.min_idx = torch.min(self.data[start:end], dim=0)
        self.max_date, self.max_date_idx = torch.max(self.date_features[start:end], dim=0)
        self.min_date, self.min_date_idx = torch.min(self.date_features[start:end], dim=0)

        self.std = torch.std(self.data[start:end], dim=0)
        self.mean = torch.mean(self.data[start:end], dim=0)
        self.std_date = torch.std(self.date_features[start:end], dim=0)
        self.mean_date = torch.mean(self.date_features[start:end], dim=0)

        if mode == 'val':
            start = int(self.data.shape[0] * 0.6)
            end = int(self.data.shape[0] * 0.8)
        if mode == 'test':
            start = int(self.data.shape[0] * 0.8)
            end = self.data.shape[0]
        self.input_size = config.INPUT_SIZE
        self.status_type = self.status_type[start:end]
        # self.data = (self.data[start:end]-self.min)/(self.max-self.min)
        # self.date_features = (self.date_features[start:end]-self.min_date)/(self.max_date-self.min_date)
        self.data = ((self.data[start:end] - self.mean) / (self.std))
        self.date_features = ((self.date_features[start:end] - self.mean_date) / (self.std_date))
        self.length = end - start
        print("****************{}".format(config.DEVICE))
        self.std = self.std.to(config.DEVICE)
        self.mean = self.mean.to(config.DEVICE)

    def __len__(self):
        return self.length - self.window_size - self.future + 1

    def __getitem__(self, index):
        # print("index{}".format(index))
        try:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(index)
        except:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(0)
        return history, future, future_status, date_features, date_future_features, history_status

    def load_item(self, index):  # 0-167 167 168
        future_index = index + self.window_size + self.future
        history = self.data[index:index + self.window_size]
        history_status = self.status_type[index:index + self.window_size]
        date_features = self.date_features[index:index + self.window_size]
        future = self.data[index + self.window_size + self.horizon - 1:future_index]
        future_status = self.status_type[index + self.window_size + self.horizon - 1:future_index]
        return history, future, future_status, date_features, self.date_features[
                                                              index + self.window_size + self.horizon - 1:future_index], history_status

    def reverse_norm(self, x, device='cuda'):
        # return x.to(device)*(self.max.to(device)-self.min.to(device))+self.min.to(device)
        return x.to(device) * (self.std.to(device)) + self.mean.to(device)

    def load_data(self, electricity_path, electrcity_type_path):
        
        print(electricity_path)
        all_elect = pd.read_csv(electricity_path, index_col=0)
        all_elect_status = np.loadtxt(electrcity_type_path, delimiter=' ' or ',')

        sum_load = np.array(all_elect).sum(axis=0)
        is_marjor = sum_load / sum_load[0] > 0.01

        # date feature
        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(all_elect.index)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = data_stamp.drop(['date'], 1).values

        # all_elect = np.array(all_elect)[:, is_marjor]
        # all_elect_status = all_elect_status[:, is_marjor]
        return torch.tensor(np.array(all_elect), dtype=torch.float), torch.tensor(all_elect_status, dtype=torch.int), \
               torch.tensor(np.array(data_stamp), dtype=torch.float)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class UmassDataset(BaseDataset):
    def __init__(self, config, mode, house_num):
        base_path='./data/umass'
        house=['HomeA', 'HomeB', 'HomeC', 'HomeD', 'HomeF', 'HomeH']
        super().__init__(config, mode,
                         electricity_path=f"{base_path}/{house[int(house_num)-1]}_2016.csv",
                         electricity_type_path=f"{base_path}/load_type_{house[int(house_num)-1]}.csv")

class UmassDatasetWithClusterK(BaseDataset):
    def __init__(self, config, mode, house_num, cluster_num):
        base_path='./data/umass'
        house=['HomeA', 'HomeB', 'HomeC', 'HomeD', 'HomeF', 'HomeH']
        super().__init__(config, mode,
                         electricity_path=f"{base_path}/{house[int(house_num)-1]}_2016.csv",
                         electricity_type_path=f"{base_path}/load_type_{house[int(house_num)-1]}{cluster_num}.csv")

class AMPds2Dataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(AMPds2Dataset, self).__init__()
        self.data, self.status_type, self.date_features = self.load_data(
            electrcity_type_path='./data/AMPds2/load_type_1h_dif_total_status_windows=1.csv')
        # self.data = np.delete(self.data,12,axis=1)
        # self.status_type = np.delete(self.status_type,12,axis=1)
        # self.date_features = np.delete(self.date_features,12,axis=1)
        self.status_num = list((torch.max(self.status_type, dim=0)[0].numpy()) + 1)
        self.window_size = 336
        self.horizon = config.horizon
        self.pred_len = config.pred_len
        self.future = self.horizon + self.pred_len - 1
        # default train set if mode == 'train':
        start = 0
        end = int(self.data.shape[0] * 0.6)
        self.max, self.max_idx = torch.max(self.data[start:end], dim=0)
        self.min, self.min_idx = torch.min(self.data[start:end], dim=0)
        self.max_date, self.max_date_idx = torch.max(self.date_features[start:end], dim=0)
        self.min_date, self.min_date_idx = torch.min(self.date_features[start:end], dim=0)

        self.std = torch.std(self.data[start:end], dim=0)
        self.mean = torch.mean(self.data[start:end], dim=0)
        self.std_date = torch.std(self.date_features[start:end], dim=0)
        self.mean_date = torch.mean(self.date_features[start:end], dim=0)

        if mode == 'val':
            start = int(self.data.shape[0] * 0.6)
            end = int(self.data.shape[0] * 0.8)
        if mode == 'test':
            start = int(self.data.shape[0] * 0.8)
            end = self.data.shape[0]
        self.input_size = config.INPUT_SIZE
        self.status_type = self.status_type[start:end]
        # self.data = (self.data[start:end]-self.min)/(self.max-self.min)
        # self.date_features = (self.date_features[start:end]-self.min_date)/(self.max_date-self.min_date)
        self.data = ((self.data[start:end] - self.mean) / (self.std))
        self.date_features = ((self.date_features[start:end] - self.mean_date) / (self.std_date))
        self.length = end - start
        self.std = self.std.to(config.DEVICE)
        self.mean = self.mean.to(config.DEVICE)

    def __len__(self):
        return self.length - self.window_size - self.future + 1

    def __getitem__(self, index):
        # print("index{}".format(index))
        try:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(index)
        except:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(0)
        return history, future, future_status, date_features, date_future_features, history_status

    def load_item(self, index):  # 0-167 167 168
        future_index = index + self.window_size + self.future
        history = self.data[index:index + self.window_size]
        history_status = self.status_type[index:index + self.window_size]
        date_features = self.date_features[index:index + self.window_size]
        future = self.data[index + self.window_size + self.horizon - 1:future_index]
        future_status = self.status_type[index + self.window_size + self.horizon - 1:future_index]
        return history, future, future_status, date_features, self.date_features[
                                                              index + self.window_size + self.horizon - 1:future_index], history_status

    def reverse_norm(self, x, device='cuda'):
        # return x.to(device)*(self.max.to(device)-self.min.to(device))+self.min.to(device)
        return x.to(device) * (self.std.to(device)) + self.mean.to(device)

    def load_data(self, electricity_path: str = "./data/AMPds2/Electricity_P_1h.csv",
                  electrcity_type_path: str = "./data/AMPds2/load_type_1h.csv"):

        all_elect = pd.read_csv(electricity_path, index_col=0)
        all_elect_status = np.loadtxt(electrcity_type_path, delimiter=',' or ' ')

        sum_load = np.array(all_elect).sum(axis=0)
        is_marjor = sum_load / sum_load[0] > 0.01

        # date feature
        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(all_elect.index)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = data_stamp.drop(['date'], 1).values

        # all_elect = np.array(all_elect)[:, is_marjor]
        # all_elect_status = all_elect_status[:, is_marjor]
        return torch.tensor(np.array(all_elect), dtype=torch.float), torch.tensor(all_elect_status, dtype=torch.int), \
               torch.tensor(np.array(data_stamp), dtype=torch.float)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class HDFBDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(HDFBDataset, self).__init__()
        self.data, self.status_type, self.date_features = self.load_data(
            electrcity_type_path='./data/HDFB/load_type_1h_dif_total_status_windows=1_v2.csv')
        # self.data = np.delete(self.data,12,axis=1)
        # self.status_type = np.delete(self.status_type,12,axis=1)
        # self.date_features = np.delete(self.date_features,12,axis=1)
        self.status_num = list((torch.max(self.status_type, dim=0)[0].numpy()) + 1)
        self.window_size = 336
        self.horizon = config.horizon
        self.pred_len = config.pred_len
        self.future = self.horizon + self.pred_len - 1
        # default train set if mode == 'train':
        start = 0
        end = int(self.data.shape[0] * 0.6)
        self.max, self.max_idx = torch.max(self.data[start:end], dim=0)
        self.min, self.min_idx = torch.min(self.data[start:end], dim=0)
        self.max_date, self.max_date_idx = torch.max(self.date_features[start:end], dim=0)
        self.min_date, self.min_date_idx = torch.min(self.date_features[start:end], dim=0)

        self.std = torch.std(self.data[start:end], dim=0)
        self.mean = torch.mean(self.data[start:end], dim=0)
        self.std_date = torch.std(self.date_features[start:end], dim=0)
        self.mean_date = torch.mean(self.date_features[start:end], dim=0)

        if mode == 'val':
            start = int(self.data.shape[0] * 0.6)
            end = int(self.data.shape[0] * 0.8)
        if mode == 'test':
            start = int(self.data.shape[0] * 0.8)
            end = self.data.shape[0]
        self.input_size = config.INPUT_SIZE
        self.status_type = self.status_type[start:end]
        # self.data = (self.data[start:end]-self.min)/(self.max-self.min)
        # self.date_features = (self.date_features[start:end]-self.min_date)/(self.max_date-self.min_date)
        self.data = ((self.data[start:end] - self.mean) / (self.std))
        self.date_features = ((self.date_features[start:end] - self.mean_date) / (self.std_date))
        self.length = end - start
        self.mean = self.mean.to(config.DEVICE)
        self.std = self.std.to(config.DEVICE)

    def __len__(self):
        return self.length - self.window_size - self.future + 1

    def __getitem__(self, index):
        # print("index{}".format(index))
        try:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(index)
        except:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(0)
        return history, future, future_status, date_features, date_future_features, history_status

    def load_item(self, index):  # 0-167 167 168
        future_index = index + self.window_size + self.future
        history = self.data[index:index + self.window_size]
        history_status = self.status_type[index:index + self.window_size]
        date_features = self.date_features[index:index + self.window_size]
        future = self.data[index + self.window_size + self.horizon - 1:future_index]
        future_status = self.status_type[index + self.window_size + self.horizon - 1:future_index]
        return history, future, future_status, date_features, self.date_features[
                                                              index + self.window_size + self.horizon - 1:future_index], history_status

    def reverse_norm(self, x, device='cuda'):
        # return x.to(device)*(self.max.to(device)-self.min.to(device))+self.min.to(device)
        return x.to(device) * (self.std.to(device)) + self.mean.to(device)

    def load_data(self, electricity_path: str = "./data/HDFB/all_load.csv",
                  electrcity_type_path: str = "./data/HDFB/load_type_1h.csv"):

        all_elect = pd.read_csv(electricity_path, index_col=0)
        all_elect_status = np.loadtxt(electrcity_type_path)

        # sum_load = np.array(all_elect).sum(axis=0)
        # is_marjor = sum_load / sum_load[0] > 0.01

        # date feature
        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(all_elect.index)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = data_stamp.drop(['date'], 1).values

        # all_elect = np.array(all_elect)[:, is_marjor]
        # all_elect_status = all_elect_status[:, is_marjor]
        return torch.tensor(np.array(all_elect), dtype=torch.float), torch.tensor(all_elect_status, dtype=torch.int), \
               torch.tensor(np.array(data_stamp), dtype=torch.float)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class UK_DALEDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode, house_num):
        super(UK_DALEDataset, self).__init__()
        self.data, self.status_type, self.date_features = self.load_data(
            electricity_path='./data/UK-DALE/load_building{}.csv'.format(house_num),
            electrcity_type_path='./data/UK-DALE/load_type_building{}.csv'.format(house_num))
        # self.data = np.delete(self.data,12,axis=1)
        # self.status_type = np.delete(self.status_type,12,axis=1)
        # self.date_features = np.delete(self.date_features,12,axis=1)
        self.status_num = list((torch.max(self.status_type, dim=0)[0].numpy()) + 1)
        self.window_size = 336
        # if house_num == '3':
        #     self.window_size=int(self.window_size)
        self.horizon = config.horizon
        self.pred_len = config.pred_len
        self.future = self.horizon + self.pred_len - 1
        # default train set if mode == 'train':
        start = 0
        end = int(self.data.shape[0] * 0.6)
        self.max, self.max_idx = torch.max(self.data[start:end], dim=0)
        self.min, self.min_idx = torch.min(self.data[start:end], dim=0)
        self.max_date, self.max_date_idx = torch.max(self.date_features[start:end], dim=0)
        self.min_date, self.min_date_idx = torch.min(self.date_features[start:end], dim=0)

        self.std = torch.std(self.data[start:end], dim=0)
        self.mean = torch.mean(self.data[start:end], dim=0)
        self.std_date = torch.std(self.date_features[start:end], dim=0)
        self.mean_date = torch.mean(self.date_features[start:end], dim=0)

        if mode == 'val':
            start = int(self.data.shape[0] * 0.6) - self.window_size
            end = int(self.data.shape[0] * 0.8)
        if mode == 'test':
            start = int(self.data.shape[0] * 0.8) - self.window_size
            end = self.data.shape[0]
        self.input_size = config.INPUT_SIZE
        self.status_type = self.status_type[start:end]
        # self.data = (self.data[start:end]-self.min)/(self.max-self.min)
        # self.date_features = (self.date_features[start:end]-self.min_date)/(self.max_date-self.min_date)
        self.data = ((self.data[start:end] - self.mean) / (self.std))
        self.date_features = ((self.date_features[start:end] - self.mean_date) / (self.std_date))
        self.length = end - start
        self.mean = self.mean.to(config.DEVICE)
        self.std = self.std.to(config.DEVICE)

    def __len__(self):
        return self.length - self.window_size - self.future + 1

    def __getitem__(self, index):
        # print("index{}".format(index))
        try:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(index)
        except:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(0)
        return history, future, future_status, date_features, date_future_features, history_status

    def load_item(self, index):  # 0-167 167 168
        future_index = index + self.window_size + self.future
        history = self.data[index:index + self.window_size]
        history_status = self.status_type[index:index + self.window_size]
        date_features = self.date_features[index:index + self.window_size]
        future = self.data[index + self.window_size + self.horizon - 1:future_index]
        future_status = self.status_type[index + self.window_size + self.horizon - 1:future_index]
        return history, future, future_status, date_features, self.date_features[
                                                              index + self.window_size + self.horizon - 1:future_index], history_status

    def reverse_norm(self, x, device='cuda'):
        # return x.to(device)*(self.max.to(device)-self.min.to(device))+self.min.to(device)
        return x.to(device) * (self.std.to(device)) + self.mean.to(device)

    def load_data(self, electricity_path: str = "./data/UK-DALE/load_building1.csv",
                  electrcity_type_path: str = "./data/UK-DALE/load_type_building0.csv"):

        all_elect = pd.read_csv(electricity_path, delimiter=' ')
        all_elect_status = np.loadtxt(electrcity_type_path, delimiter=' ' or ',')

        # sum_load = np.array(all_elect).sum(axis=0)
        # is_marjor = sum_load / sum_load[0] > 0.01

        # date feature
        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(all_elect.index)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = data_stamp.drop(['date'], 1).values

        # all_elect = np.array(all_elect)[:, is_marjor]
        # all_elect_status = all_elect_status[:, is_marjor]
        return torch.tensor(np.array(all_elect), dtype=torch.float), torch.tensor(all_elect_status, dtype=torch.int), \
               torch.tensor(np.array(data_stamp), dtype=torch.float)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class ElectricityDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(ElectricityDataset, self).__init__()
        self.data, self.status_type, self.date_features = self.load_data(
            electrcity_type_path='./data/electricity/all_load.csv')
        # self.data = np.delete(self.data,12,axis=1)
        # self.status_type = np.delete(self.status_type,12,axis=1)
        # self.date_features = np.delete(self.date_features,12,axis=1)
        self.status_num = list((torch.max(self.status_type, dim=0)[0].numpy()) + 1)
        self.window_size = 336
        self.horizon = config.horizon
        self.pred_len = config.pred_len
        self.future = self.horizon + self.pred_len - 1
        # default train set if mode == 'train':
        start = 0
        end = int(self.data.shape[0] * 0.6)
        self.max, self.max_idx = torch.max(self.data[start:end], dim=0)
        self.min, self.min_idx = torch.min(self.data[start:end], dim=0)
        self.max_date, self.max_date_idx = torch.max(self.date_features[start:end], dim=0)
        self.min_date, self.min_date_idx = torch.min(self.date_features[start:end], dim=0)

        self.std = torch.std(self.data[start:end], dim=0)
        self.mean = torch.mean(self.data[start:end], dim=0)
        self.std_date = torch.std(self.date_features[start:end], dim=0)
        self.mean_date = torch.mean(self.date_features[start:end], dim=0)

        if mode == 'val':
            start = int(self.data.shape[0] * 0.6)
            end = int(self.data.shape[0] * 0.8)
        if mode == 'test':
            start = int(self.data.shape[0] * 0.8)
            end = self.data.shape[0]
        self.input_size = config.INPUT_SIZE
        self.status_type = self.status_type[start:end]
        # self.data = (self.data[start:end]-self.min)/(self.max-self.min)
        # self.date_features = (self.date_features[start:end]-self.min_date)/(self.max_date-self.min_date)
        self.data = ((self.data[start:end] - self.mean) / (self.std))
        self.date_features = ((self.date_features[start:end] - self.mean_date) / (self.std_date))
        self.length = end - start

    def __len__(self):
        return self.length - self.window_size - self.future + 1

    def __getitem__(self, index):
        # print("index{}".format(index))
        try:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(index)
        except:
            history, future, future_status, date_features, date_future_features, history_status = self.load_item(0)
        return history, future, future_status, date_features, date_future_features, history_status

    def load_item(self, index):  # 0-167 167 168
        future_index = index + self.window_size + self.future
        history = self.data[index:index + self.window_size]
        history_status = self.status_type[index:index + self.window_size]
        date_features = self.date_features[index:index + self.window_size]
        future = self.data[index + self.window_size + self.horizon - 1:future_index]
        future_status = self.status_type[index + self.window_size + self.horizon - 1:future_index]
        return history, future, future_status, date_features, self.date_features[
                                                              index + self.window_size + self.horizon - 1:future_index], history_status

    def reverse_norm(self, x, device='cuda'):
        # return x.to(device)*(self.max.to(device)-self.min.to(device))+self.min.to(device)
        return x.to(device) * (self.std.to(device)) + self.mean.to(device)

    def load_data(self, electricity_path: str = "./data/electricity/electricity.csv",
                  electrcity_type_path: str = "./data/electricity/load_type_1h.csv"):

        all_elect = pd.read_csv(electricity_path, index_col=0)
        all_elect_status = np.loadtxt(electrcity_type_path, delimiter=' ')

        # sum_load = np.array(all_elect).sum(axis=0)
        # is_marjor = sum_load / sum_load[0] > 0.01

        # date feature
        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(all_elect.index)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = data_stamp.drop(['date'], 1).values

        # all_elect = np.array(all_elect)[:, is_marjor]
        # all_elect_status = all_elect_status[:, is_marjor]
        return torch.tensor(np.array(all_elect), dtype=torch.float), torch.tensor(all_elect_status, dtype=torch.int), \
               torch.tensor(np.array(data_stamp), dtype=torch.float)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv', status_path='load_type.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.status_path = status_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = standScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        load_type = np.loadtxt(os.path.join(self.root_path,
                                            self.status_path), delimiter=' ' or ',')
        self.status_num = list((np.max(load_type, axis=0) + 1).astype(int))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.mean = torch.tensor(self.scaler.mean_)
            self.std = torch.tensor(self.scaler.var_)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2].astype(np.float32)
        self.data_y = data[border1:border2].astype(np.float32)
        self.status = load_type[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        status_y = self.status[r_begin:r_end]
        status_x = self.status[s_begin:s_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, status_y, seq_x_mark, seq_y_mark, status_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def reverse_norm(self, x, device='cuda'):
        # return x.to(device)*(self.max.to(device)-self.min.to(device))+self.min.to(device)
        return x.to(device) * (self.std.to(device)) + self.mean.to(device)


class AMPds2DatasetWithBalance(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(AMPds2DatasetWithBalance, self).__init__()
        self.data, self.status_type, self.date_features = self.load_data(
            electrcity_type_path='../data/AMPds2/load_type_1h_dif_total_status_with_mean_filter.csv')
        self.data = np.delete(self.data, 12, axis=1)
        self.status_type = np.delete(self.status_type, 12, axis=1)
        self.date_features = np.delete(self.date_features, 12, axis=1)
        self.status_num = list((torch.max(self.status_type, dim=0)[0].numpy()) + 1)
        self.window_size = 336
        self.horizon = config["horizon"]
        self.pred_len = config["pred_len"]
        self.future = self.horizon + self.pred_len - 1
        # default train set if mode == 'train':
        start = 0
        end = int(self.data.shape[0] * 0.6)
        self.max, self.max_idx = torch.max(self.data[start:end], dim=0)
        self.min, self.min_idx = torch.min(self.data[start:end], dim=0)
        self.max_date, self.max_date_idx = torch.max(self.date_features[start:end], dim=0)
        self.min_date, self.min_date_idx = torch.min(self.date_features[start:end], dim=0)

        self.std = torch.std(self.data[start:end], dim=0)
        self.mean = torch.mean(self.data[start:end], dim=0)
        self.std_date = torch.std(self.date_features[start:end], dim=0)
        self.mean_date = torch.mean(self.date_features[start:end], dim=0)

        if mode == 'val':
            start = int(self.data.shape[0] * 0.6)
            end = int(self.data.shape[0] * 0.8)
        if mode == 'test':
            start = int(self.data.shape[0] * 0.8)
            end = self.data.shape[0]
        self.input_size = config["INPUT_SIZE"]
        self.status_type = self.status_type[start:end]
        # self.data = (self.data[start:end]-self.min)/(self.max-self.min)
        # self.date_features = (self.date_features[start:end]-self.min_date)/(self.max_date-self.min_date)
        self.data = (self.data[start:end] - self.mean) / (self.std)
        self.date_features = (self.date_features[start:end] - self.mean_date) / (self.std_date)
        self.length = end - start

    def __len__(self):
        return self.length - self.window_size - self.future + 1

    def __getitem__(self, index):
        # print("index{}".format(index))
        try:
            history, future, future_status, date_features, date_future_features = self.load_item(index)
        except:
            history, future, future_status, date_features, date_future_features = self.load_item(0)
        return history, future, future_status, date_features, date_future_features

    def load_item(self, index):  # 0-167 167 168
        future_index = index + self.window_size + self.future
        history = self.data[index:index + self.window_size]
        date_features = self.date_features[index:index + self.window_size]
        future = self.data[index + self.window_size + self.horizon - 1:future_index]
        future_status = self.status_type[index + self.window_size + self.horizon - 1:future_index]
        return history, future, future_status, date_features, self.date_features[
                                                              index + self.window_size + self.horizon - 1:future_index]

    def reverse_norm(self, x, device='cuda'):
        # return x.to(device)*(self.max.to(device)-self.min.to(device))+self.min.to(device)
        return x.to(device) * (self.std.to(device)) + self.mean.to(device)

    def load_data(self, electricity_path: str = "../data/AMPds2/Electricity_P_1h.csv",
                  electrcity_type_path: str = "../data/AMPds2/load_type_1h.csv"):

        all_elect = pd.read_csv(electricity_path, index_col=0)
        all_elect_status = np.loadtxt(electrcity_type_path)

        sum_load = np.array(all_elect).sum(axis=0)
        is_marjor = sum_load / sum_load[0] > 0.01

        # date feature
        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(all_elect.index)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = data_stamp.drop(['date'], 1).values

        all_elect = np.array(all_elect)[:, is_marjor]
        all_elect_status = all_elect_status[:, is_marjor]
        return torch.tensor(np.array(all_elect), dtype=torch.float), torch.tensor(all_elect_status, dtype=torch.int), \
               torch.tensor(np.array(data_stamp), dtype=torch.float)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class AMPds2DatasetWithPartDataset(Dataset):
    def __init__(self,
                 path: str = "Electricity_P.csv",
                 resample_freq: str = "1H",
                 history_len: int = 336,
                 horizon: int = 1,
                 pred_len: int = 1,
                 mode: str = "train",
                 device: str = "cpu"
                 ):
        super().__init__()
        self.status_num=[2]
        self.window_size = history_len
        self.horizon = horizon
        self.pred_len = pred_len
        self.future = self.horizon + self.pred_len - 1
        self.mode = mode
        self.device = device
        
        self.data, self.date_features = self._load_data(load_path = path, resample_freq = resample_freq)
        self.length = self.end - self.start

        """
        Constructor for the MyDataset class.

        Parameters:
        - path: str,  dataset path.
        - history_len: int, default=1, the history length
        - horizon: int, default=1, the prediction horizon.
        - pred_len: int, default=336, the prediction length.
        - mode: str, specifies the mode ('train', 'val', or 'test').
        - device: specify the device for tensor operations.
        
        Attributes:
        - data: torch.Tensor, the loaded data.
        - date_features: torch.Tensor, date-related features.
        - window_size: int, default=336, size of the time window.
        - horizon: int, the prediction horizon.
        - pred_len: int, the prediction length.
        - future: int, the future time steps.
        - max, min, max_date, min_date: torch.Tensor, statistical information for normalization.
        - max_idx, min_idx, max_date_idx, min_date_idx: torch.Tensor, corresponding indices for statistical information.
        - std, mean, std_date, mean_date: torch.Tensor, standard deviation and mean for normalization.
        - length: int, the length of the dataset.

        Functions:
        - __len__(): return the sample numbers
        - __getitem__(index): return the input and label of the sample by the index
        - _load_data(load_path, resample_freq): load, preprocess and return the data by the data path
        - _load_item(index): the implement of self.__getitem__()
        - data_iterator(batch_size): the dataloader
        - data_get_all(): return all samples
        - normalization(mdoe): normalize the data by Z-Score or Max-Min method
        """
        
        
    def __len__(self):
        return self.length - self.window_size - self.future + 1
    
    def __getitem__(self, index):
        try:
            return self._load_item(index)
        except:
            return self._load_item(0)

    def _load_data(self, load_path: str, resample_freq: str="1H"):
        data = pd.read_csv(load_path, index_col=0)
        data.index = pd.to_datetime(data.index, unit="s")
        data = data.drop(columns=["WHE", "RSE", "MHE", "GRE", "BME", "CWE", "DWE", "CDE"])
        data = data.resample(resample_freq).sum()
        data["OT"] = data.iloc[:,:15].sum(axis=1)

        date_features = pd.DataFrame()
        date_features["date"] = pd.to_datetime(data.index)
        date_features["month"] = date_features.date.apply(lambda row: row.month)
        date_features["day"] = date_features.date.apply(lambda row: row.day)
        date_features["weekday"] = date_features.date.apply(lambda row: row.weekday())
        date_features["hour"] = date_features.date.apply(lambda row: row.hour)
        date_features = date_features.drop(labels=["date"], axis=1)

        if self.mode == "train":
            self.start = 0
            self.end = int(data.shape[0] * 0.6)
        elif self.mode == "val":
            self.start = int(data.shape[0] * 0.6)
            self.end = int(data.shape[0] * 0.8)
        elif self.mode == "test":
            self.start = int(data.shape[0] * 0.8)
            self.end = int(data.shape[0])

        data = torch.tensor(np.array(data), dtype=torch.float)
        date_features = torch.tensor(np.array(date_features), dtype=torch.float)

        self.std = torch.std(data, dim=0)
        self.mean = torch.mean(data, dim=0)
        self.std_date = torch.std(date_features, dim=0)
        self.mean_date = torch.mean(date_features, dim=0)

        data = data[self.start: self.end]
        date_features = date_features[self.start: self.end]

        return data, date_features 

    def _load_item(self, index):
        future_index = index + self.window_size + self.future
        history_data = self.data[index : index + self.window_size]
        #history_date_features = self.date_features[index : index + self.window_size]
        future_data = self.data[index + self.window_size + self.horizon - 1 : future_index, -1]
        #future_date_features = self.date_features[index + self.window_size +self.horizon - 1: future_index]
        return history_data, future_data,future_data, future_data, future_data, future_data#, history_date_features, future_date_features
    
    def data_iterator(self, batch_size):
        if self.mode == "train":
            shuffle_tag = True
        else:
            shuffle_tag = False
        
        # while True:
        sample_loader = DataLoader(dataset=self,
                                       batch_size=batch_size,
                                       shuffle=shuffle_tag,
                                       drop_last=shuffle_tag
                                       )
        for item in sample_loader:
            yield item
    
    def data_get_all(self):
        history_data, future_data = [], []
        for i in range(self.__len__()):
            history_data.append(self.data[i: i+self.window_size])
            future_data.append(self.data[i+self.window_size+self.horizon-1: i+self.window_size+self.future])
        return torch.stack(history_data), torch.stack(future_data)
    
    def normalization(self, mode: str="ZS", criterion: str="all"):
        if criterion == "all":
            std_data = self.std
            mean_data = self.mean
            std_date = self.std_date
            mean_date = self.mean_date
        else:
            std_data = torch.std(self.data, dim=0)
            mean_data = torch.mean(self.data, dim=0)
            std_date = torch.std(self.date_features, dim=0)
            mean_date = torch.mean(self.date_features, dim=0)
        
        if mode == "ZS":
            self.data = (self.data - mean_data) / (std_data)
            self.date_features = (self.date_features - mean_date) / (std_date)
        elif mode == "MM":
            self.max, self.max_idx = torch.max(self.data, dim=0)
            self.min, self.min_idx = torch.min(self.data, dim=0)
            self.max_date, self.max_date_idx = torch.max(self.date_features, dim=0)
            self.min_date, self.min_date_idx = torch.min(self.date_features, dim=0)
            
            self.data = (self.data - self.min) / (self.max - self.min)
            self.date_features = (self.date_features - self.min_date) / (self.max_date - self.min_date)
    
    def get_reverse_norm(self, x):
        return x * self.std + self.mean
    
    def reverse_norm(self, x, device='cuda'):
        return x.to(device)
        # return x * self.std + self.mean
if __name__ == "__main__":
    config = {"horizon": 1, "INPUT_SIZE": 336, "pred_len": 1}
    a = []
    b = []
    c = []

    dataset = HDFBDataset(config=config, mode='train')

    dataset = AMPds2DatasetWithBalance(config=config, mode='val')
    # 0 tensor([2854, 2840, 2924, 2670, 3316, 3328, 3209, 3216, 3208, 3466, 3275, 3312,
    #          918, 1056, 2822, 2822])
    # 1 tensor([ 650,  664,  580,  271,   97,  176,   28,   93,   91,   19,   74,   63,
    #         2586, 2448,  682,  682])
    # 2 tensor([  0,   0,   0, 250,  91,   0,  94,  98,  81,  19,  76,  65,   0,   0,
    #           0,   0])
    # tensor([  0,   0,   0, 237,   0,   0,  84,  97,  36,   0,  79,  64,   0,
    0,
    #           0,   0])
    # tensor([ 0,  0,  0, 76,  0,  0, 89,  0, 88,  0,  0,  0,  0,  0,  0,  0])
    dataset = AMPds2DatasetWithBalance(config=config, mode='train')
    # tensor([ 8102,  8055,  7586,  7368,  9885,  9982,  7575,  7788,  7707, 10404,
    #    9787,  9879,  4768,  5663,  8023,  8024])
    # tensor([2410, 2457, 2926,  945,  301,  530,  362, 1130,  751,   51,  232,  210,
    #         5744, 4849, 2489, 2488])
    # tensor([  0,   0,   0, 912, 326,   0, 750, 766, 832,  57, 248, 218,   0,   0,
    #           0,   0])
    # tensor([  0,   0,   0, 671,   0,   0, 740, 828, 473,   0, 245, 205,   0,   0,
    #           0,   0])
    dataset = AMPds2DatasetWithBalance(config=config, mode='test')
    # tensor([2854, 2840, 2924, 2670, 3316, 3328, 3209, 3216, 3208, 3466, 3275, 3312,
    #          918, 1056, 2822, 2822])
    # tensor([ 650,  664,  580,  271,   97,  176,   28,   93,   91,   19,   74,   63,
    #         2586, 2448,  682,  682])
    # tensor([  0,   0,   0, 250,  91,   0,  94,  98,  81,  19,  76,  65,   0,   0,
    #      0,   0])
    # tensor([  0,   0,   0, 237,   0,   0,  84,  97,  36,   0,  79,  64,   0,   0,
    #           0,   0])
    # tensor([ 0,  0,  0, 76,  0,  0, 89,  0, 88,  0,  0,  0,  0,  0,  0,  0])
