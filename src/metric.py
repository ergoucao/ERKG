import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np


class StatusClassClassificationAccuracy(nn.Module):
    def __init__(self):
        super(StatusClassClassificationAccuracy, self).__init__()

    def __call__(self, outputs, labels, debug_print=True):
        pred_res = []
        for output in outputs:
            pred_res.append(torch.argmax(output, dim=2))
        pred_res = torch.stack(pred_res, dim=2)
        pred_res = torch.flatten(pred_res, start_dim=0, end_dim=1)
        labels = torch.flatten(labels, start_dim=0, end_dim=1)

        accuracy, precision, f1, recall = (0, 0, 0, 0)
        for i in range(labels.shape[1]):
            label = labels[:, i].to("cpu").detach()
            pred = pred_res[:, i].to("cpu").detach()
            acc_score = accuracy_score(label, pred)
            prec_score = precision_score(label, pred, average="macro")
            rec_score = recall_score(label, pred, average="macro")
            f1sc = f1_score(label, pred, average="macro")
            accuracy += acc_score
            precision += prec_score
            recall += rec_score
            f1 += f1sc
            if debug_print:
                print(
                    "part:{},accuracy:{},precision:{},f1:{}".format(
                        i,
                        accuracy_score(label, pred),
                        precision_score(label, pred, average="macro"),
                        recall_score(label, pred, average="macro"),
                        f1_score(label, pred, average="macro"),
                    )
                )
                if accuracy_score(label, pred) < 0.1:
                    print("poor !")

            # confusion_mat = confusion_matrix(pred_res, labels)
        accuracy /= labels.shape[1]
        precision /= labels.shape[1]
        recall /= labels.shape[1]
        f1 /= labels.shape[1]
        # print(accuracy)
        # print(precision)
        # print(recall)
        # print(f1)

        # if relevant == 0 and selected == 0:
        #     return torch.tensor(1), torch.tensor(1)
        #
        # true_positive = ((outputs == labels) * labels).float()
        # recall = torch.sum(true_positive) / (relevant + 1e-8)
        # precision = torch.sum(true_positive) / (selected + 1e-8)

        return accuracy, precision, recall, f1

class TSF_metric_torch():
    def RSE(self, pred, true):
        return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))

    def CORR(self, pred, true):
        u = ((true - true.mean(dim=0)) * (pred - pred.mean(dim=0))).sum(dim=0)
        d = torch.sqrt(((true - true.mean(dim=0)) ** 2).sum(dim=0) * ((pred - pred.mean(dim=0)) ** 2).sum(dim=0))
        return (u / d).mean()

    def MAE(self, pred, true):
        return torch.mean(torch.abs(pred - true))

    def MSE(self, pred, true):
        return torch.mean((pred - true) ** 2)

    def MSE_individual(self, pred, true):
        return torch.mean(torch.mean((pred - true) ** 2, dim=0), dim=0)

    def MAE_individual(self, pred, true):
        return torch.mean(torch.mean(torch.abs(pred - true), dim=0), dim=0)

    def RMSE(self, pred, true):
        return torch.sqrt(self.MSE(pred, true))

    def MAPE(self, pred, true):
        abs_error = torch.abs((pred - true) / true)
        return torch.mean(abs_error[~torch.isinf(abs_error)])

    def SMAPE(self, pred, true):
        abs_error = torch.abs((pred - true) / ((torch.abs(true) + torch.abs(pred)) / 2))
        return torch.mean(abs_error[~torch.isinf(abs_error)])

    def MSPE(self, pred, true):
        sq_error = torch.square((pred - true) / true)
        return torch.mean(sq_error[~torch.isinf(sq_error)])

class TSF_metric(nn.Module):
    def __init__(self):
        super(TSF_metric, self).__init__()

    def __call__(self, outputs, labels):
        return self.metric(outputs, labels)

    def get_individual(self, outputs, labels):
        return self.metric_individual(outputs, labels)

    def RSE(self, pred, true):
        return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
            np.sum((true - true.mean()) ** 2)
        )

    def CORR(self, pred, true):
        u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
        d = np.sqrt((((true - true.mean(0)) ** 2).sum(0) * (pred - pred.mean(0)) ** 2).sum(0))
        return (u / d).mean()

    def MAE(self, pred, true):
        return np.mean(np.abs(pred - true))

    def MSE(self, pred, true):
        return np.mean((pred - true) ** 2)

    def MSE_individual(self, pred, true):
        return np.mean(np.mean((pred - true) ** 2, axis=0), axis=0)

    def MAE_individual(self, pred, true):
        return np.mean(np.mean(np.abs(pred - true), axis=0), axis=0)

    def RMSE(self, pred, true):
        return np.sqrt(self.MSE(pred, true))

    def MAPE(self, pred, true):
        abs = np.abs((pred - true) / true)
        return np.mean(abs[~np.isinf(abs)])

    def SMAPE(self, pred, true):
        abs = np.abs((pred - true) / ((np.abs(true) + np.abs(pred))/2))
        return np.mean(abs[~np.isinf(abs)])

    def MSPE(self, pred, true):
        sq = np.square((pred - true) / true)
        return np.mean(sq[~np.isinf(sq)])
    

    def metric_individual(self, pred, true):
        # mae = self.MAE(pred, true)
        # mse = self.MSE(pred, true)
        # rse = self.RSE(pred, true)
        # mae_i = self.MSE_individual(pred, true)
        # mse_i = self.MAE_individual(pred, true)
        # rmse = self.RMSE(pred, true)
        # mape = self.SMAPE(pred, true)
        # mspe = self.MSPE(pred, true)
        # corr = self.CORR(pred,true)
        torch_metric = TSF_metric_torch()
        mae = torch_metric.MAE(pred, true).item()
        mse = torch_metric.MSE(pred, true).item()
        rse = torch_metric.RSE(pred, true).item()
        mae_i = np.array(torch_metric.MSE_individual(pred, true).tolist())
        mse_i = np.array(torch_metric.MAE_individual(pred, true).tolist())
        rmse = torch_metric.RMSE(pred, true).item()
        mape = torch_metric.SMAPE(pred, true).item()
        mspe = torch_metric.MSPE(pred, true).item()
        corr = torch_metric.CORR(pred,true).item()
        return mae, mae_i, mse, mse_i, rmse, mape, mspe, rse, corr

    def metric(self, pred, true):
        mae = self.MAE(pred, true)
        mse = self.MSE(pred, true)
        rmse = self.RMSE(pred, true)
        mape = self.MAPE(pred, true)
        mspe = self.MSPE(pred, true)
        return mae, mse, rmse, mape, mspe


class PainterAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """

    def __init__(self, threshold=0.5):
        super(PainterAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = inputs > self.threshold
        outputs = outputs > self.threshold

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall


if __name__ == "__main__":
    pred = [torch.randn((32, 3)), torch.randn((32, 3))]
    label = torch.randint(0, 2, (32, 2))
    sc = StatusClassClassificationAccuracy()
    accuracy, precision, recall, f1 = sc(pred, label)
