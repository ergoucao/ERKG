import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import (
    AMPds2Dataset,
    HDFBDataset,
    ElectricityDataset,
    UK_DALEDataset,
    Dataset_Custom,
    UmassDataset,
    AMPds2DatasetWithPartDataset,
    UmassDatasetWithClusterK
)
from .models import StatusModel, TSFModel
from .utils import Progbar, create_dir, stitch_images, imsave
from src.metric import StatusClassClassificationAccuracy, TSF_metric
from datetime import datetime
import warnings
# import line_profiler
import time
# from torchstat import stat
# from torchsummary import summary

warnings.filterwarnings("ignore")

from .etsformer import ETSformer

np.set_printoptions(precision=4, suppress=True)
import wandb

os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_MODE"] = "disabled"
DEBUG = 1

wandb.init(
    settings=wandb.Settings(start_method="fork"),
    # set the wandb project where this run will be logged
    project="p4tsf",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-4,
        "architecture": "share-cnn",
        "dataset": "AMPds2",
        "epochs": 40,
    },
)


class StatusTSFNet:
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = "status_model"
        elif config.MODEL == 2:
            model_name = "TSF_model"
        elif config.MODEL == 3:
            model_name = "status_TSF"
        elif config.MODEL == 4:
            model_name = "joint"

        # self.psnr = PSNR(255.0).to(config.DEVICE)
        # self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        #
        # self.test_dataset = ElectricityDataset(config, mode='test')
        # self.train_dataset = ElectricityDataset(config, mode='train')
        # self.val_dataset = ElectricityDataset(config, mode='val')

        data_dict = {
            "hdfb": HDFBDataset,
            "uk_dale": UK_DALEDataset,
            "ampds2": AMPds2Dataset,
            "traffic": Dataset_Custom,
        }

        if config.data == "hdfb":
            self.test_dataset = HDFBDataset(config, mode="test")
            self.train_dataset = HDFBDataset(config, mode="train")
            self.val_dataset = HDFBDataset(config, mode="val")
        elif config.data[:-1] == "uk_dale":
            self.test_dataset = UK_DALEDataset(
                config, mode="test", house_num=config.data[-1]
            )
            self.train_dataset = UK_DALEDataset(
                config, mode="train", house_num=config.data[-1]
            )
            self.val_dataset = UK_DALEDataset(
                config, mode="val", house_num=config.data[-1]
            )
        elif config.data[:-1] == "umass":
            self.test_dataset = UmassDataset(
                config, mode="test", house_num=config.data[-1]
            )
            self.train_dataset = UmassDataset(
                config, mode="train", house_num=config.data[-1]
            )
            self.val_dataset = UmassDataset(
                config, mode="val", house_num=config.data[-1]
            )
        elif config.data[:-2]=="umass": # ablation cluster k
            self.test_dataset = UmassDatasetWithClusterK(
                config, mode="test", house_num=config.data[-1], cluster_num=config.data[-2]
            )
            self.train_dataset = UmassDatasetWithClusterK(
                config, mode="train", house_num=config.data[-1], cluster_num=config.data[-2]
            )
            self.val_dataset = UmassDatasetWithClusterK(
                config, mode="val", house_num=config.data[-1], cluster_num=config.data[-2]
            )
        elif config.data == "ampds2":
            self.test_dataset = AMPds2Dataset(config, mode="test")
            self.train_dataset = AMPds2Dataset(config, mode="train")
            self.val_dataset = AMPds2Dataset(config, mode="val")
        elif config.data == "traffic":
            self.train_dataset = Dataset_Custom(
                root_path="./data/traffic/",
                data_path="traffic.csv",
                status_path="load_type_traffic.csv",
                flag="train",
                size=[config.seq_len, 0, config.pred_len],
            )
            self.test_dataset = Dataset_Custom(
                root_path="./data/traffic/",
                data_path="traffic.csv",
                status_path="load_type_traffic.csv",
                flag="test",
                size=[config.seq_len, 0, config.pred_len],
            )
            self.val_dataset = Dataset_Custom(
                root_path="./data/traffic/",
                data_path="traffic.csv",
                status_path="load_type_traffic.csv",
                flag="val",
                size=[config.seq_len, 0, config.pred_len],
            )
        # ampds2部分数据测试
        elif config.data == "amds2_part":
            self.test_dataset = AMPds2DatasetWithPartDataset(path='./data/AMPds2/Electricity_P.csv',device='cuda:0', mode="test")
            self.test_dataset.normalization()
            self.train_dataset = AMPds2DatasetWithPartDataset(path='./data/AMPds2/Electricity_P.csv',device='cuda:0', mode="train")
            self.train_dataset.normalization()
            self.val_dataset = AMPds2DatasetWithPartDataset(path='./data/AMPds2/Electricity_P.csv',device='cuda:0', mode="val")
            self.val_dataset.normalization()

        self.debug = False
        self.model_name = model_name
        self.status_model = StatusModel(config, self.train_dataset.status_num).to(
            config.DEVICE
        )
        # print("*********"+config.DEVICE)
        self.TSF_model = TSFModel(
            config=config, status_num=self.train_dataset.status_num
        ).to(config.DEVICE)
        self.classification_metric = StatusClassClassificationAccuracy()
        self.TSF_metric = TSF_metric()

        # self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
        self.sample_iterator = None

        self.samples_path = os.path.join(config.PATH, "samples")
        self.results_path = os.path.join(config.PATH, "results")

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, "log_" + model_name + ".dat")

    def load(self, mode=1):
        if self.config.MODEL == 3:
            self.status_model.load()

        # elif self.config.MODEL == 2 and mode == 2 :
        #     print("*********** load tsf_model")
        #     self.TSF_model.load()
        # elif mode == 3:
        #     print("*********** load sttaus_model")
        #     self.status_model.load()
        #     # self.TSF_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.status_model.save()

        elif self.config.MODEL == 2:
            self.TSF_model.save()
        else:
            # self.status_model.save()
            self.TSF_model.save()

    def print_loss_metric(self, outputs, groundtruth, loss, logs, all_task_loss=None):
        logs.append(("loss", loss.cpu().detach().numpy()))
        logs.append(("groundtruth_status", groundtruth.cpu().detach().numpy()))
        wandb.log({"loss": loss})
        if all_task_loss:
            for idx, l in enumerate(all_task_loss):
                wandb.log({"all_task_loss:{}".format(idx): l})

    # @line_profiler.profile
    def print_tsf_loss_metric(
        self, outputs, future, a_logs, metric=None, all_variates=None
    ):
        # mae, mae_i, mse, mse_i, rmse, mape, mspe, rse, corr = self.TSF_metric.get_individual(
        #     outputs.detach().to("cpu").numpy(), future.detach().to("cpu").numpy()
        # )
        mae, mae_i, mse, mse_i, rmse, mape, mspe, rse, corr = self.TSF_metric.get_individual(
            outputs, future        )
        # self.val_dataset.reverse_norm(outputs).detach().to('cpu').numpy(),
        # self.val_dataset.reverse_norm(future).detach().to('cpu').numpy())
        wandb.log({"mae": mae})
        if metric != None:
            metric[0] += mae
            metric[1] += mse
            metric[2] += rmse
            metric[3] += mape
            metric[4] += mspe
            metric[5] += rse
            metric[6] += corr
        if all_variates != None:
            all_variates[0] += mae_i
            all_variates[1] += mse_i
        # print("\n***** mae:{:.6f},mse:{:.6f},rmse:{:.6f},mape{:.6f}, mspe{:.6f} *****\n".format(mae, mse, rmse, mape,
        #                                                                                         mspe))

        a_logs.append(("mae", mae))
        a_logs.append(("mse", mse))
        a_logs.append(("rmse", rmse))
        a_logs.append(("mape", mape))
        a_logs.append(("mspe", mspe))
        a_logs.append(("rse", rse))
        a_logs.append(("rse", corr))
    class EarlyStopping:
        def __init__(self, patience=10, verbose=False, delta=0, warn_patience=5):
            self.patience = patience
            self.warn_patience = warn_patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.warn_early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta

        def __call__(self, val_loss, model, path, filename):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path, filename)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
                if self.counter >= self.warn_patience:
                    self.warn_early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path, filename)
                self.counter = 0

        def save_checkpoint(self, val_loss, model, path, filename):
            if self.verbose:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model:{filename} ..."
                )
            torch.save(model.state_dict(), path + "/" + filename)
            self.val_loss_min = val_loss

    # @line_profiler.profile
    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True,
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_epoch = int(float((self.config.MAX_EPOCH)))
        total = len(self.train_dataset)

        if total == 0:
            print(
                "No training data was provided! Check 'TRAIN_FLIST' value in the configuration file."
            )
            return

        early_stopping = self.EarlyStopping(verbose=True, patience=10)

        model_filename = (
            self.model_name 
            # + "ablation_wo_msp"
            + f"{model}"
            + datetime.now().strftime("%m:%d")
            + "_predL_"
            + str(self.config.pred_len)
            + "_horizon_"
            + str(self.config.horizon)
            + "_topk_"
            + str(self.config.topk)
            +"_"
            + str(self.config.data)
            + ".pth"
        )
        print("traing status model:{}".format(model_filename))
        T1=time.time()
        count=0

        for epoch in range(max_epoch):
            count+=1
            print("\nTraining epoch: %d" % epoch)
            if epoch == 38:
                print("DeBug check")
            progbar = Progbar(total, width=20, stateful_metrics=["epoch", "iter"])
            total_loss = 0
            metric = [0, 0, 0, 0, 0, 0, 0]
            # val_loss = self.test(self.val_dataset, is_shuffle=True, criterion=nn.MSELoss())
            for (
                history,
                future,
                status,
                date_features,
                date_future_features,
                history_status,
            ) in train_loader:
                history = history.cuda(self.config.DEVICE)
                future = future.cuda(self.config.DEVICE)
                status = status.cuda(self.config.DEVICE).type(torch.long)
                history_status = history_status.cuda(self.config.DEVICE).type(
                    torch.float
                )
                # self.inpaint_model.train()

                # status classifier model
                if model == 1:
                    self.status_model.train()
                    # train
                    (
                        outputs,
                        loss,
                        logs,
                        all_task_loss,
                        _,
                        _,
                    ) = self.status_model.process(
                        history, status, future, history_status
                    )
                    # logs.append(('outputs', outputs[0].item())) # 需要修改为值
                    self.print_loss_metric(
                        outputs=outputs,
                        groundtruth=status,
                        loss=loss,
                        logs=logs,
                        all_task_loss=all_task_loss,
                    )
                    # train metric
                    # accuracy, precision, recall, f1 = self.classification_metric(outputs, status)
                    # wandb.log({'train acc': accuracy, 'train precision': precision, 'train recall': recall, 'train f1': f1})
                    # backward
                    self.status_model.backward(loss)
                    total_loss += loss
                    iteration = self.status_model.iteration

                # TSF model
                elif model == 2:
                    self.status_model.eval()
                    self.TSF_model.train()

                    # test

                    # train
                    outputs, TSF_loss, logs = self.TSF_model.process(
                        history,
                        future,
                        date_features,
                        date_future_features,
                        status_truth=None,
                        mean=self.train_dataset.mean,
                        std=self.test_dataset.std,
                    )  # status
                    total_loss += TSF_loss
                    # logs.append(('outputs', outputs[0].item())) # 需要修改为值
                    self.print_loss_metric(
                        outputs=outputs, groundtruth=future, loss=TSF_loss, logs=logs
                    )
                    logs.append(("TSF_loss", TSF_loss.cpu().detach().numpy()))
                    logs.append(("groundtruth", future.cpu().detach().numpy()))
                    self.print_tsf_loss_metric(
                        outputs=outputs, future=future, a_logs=logs, metric=metric
                    )
                    iteration = self.TSF_model.iteration
                # TSF with attr model
                elif model == 3:
                    self.TSF_model.train()
                    self.status_model.eval()

                    if early_stopping.warn_early_stop or True:
                        # get the event info
                        # summary(self.status_model,  history, history_status)
                        # print("*********input "+str(history.device)+str(history_status.device))
                        status_soft, hidden, _ = self.status_model(
                            history, history_status
                        )  # 23 , hidden : 256, 512, 2
                        # train 
                        outputs, TSF_loss, logs = self.TSF_model.process(
                            history, 
                            future, 
                            date_features, 
                            date_future_features, 
                            status_info_hidden=hidden, 
                            status=status_soft, 
                            status_truth=status, 
                            mean=self.train_dataset.mean, 
                            std=self.test_dataset.std, 
                            alpha=self.config.alpha, 
                            history_satus_truth=history_status, 
                            topk=self.config.topk, 
                        )
                    else:
                        outputs, TSF_loss, logs = self.TSF_model.process(
                            history,
                            future,
                            date_features,
                            date_future_features,
                            status_truth=None,
                        )  # status

                    total_loss += TSF_loss
                    # logs.append(('outputs', outputs[0].item())) # 需要修改为值
                    self.print_loss_metric(
                        outputs=outputs, groundtruth=future, loss=TSF_loss, logs=logs
                    )
                    self.print_tsf_loss_metric(
                        outputs=outputs, future=future, a_logs=logs, metric=metric
                    )
                    logs.append(("TSF_loss", TSF_loss.cpu().detach().numpy()))
                    logs.append(("groundtruth", future.cpu().detach().numpy()))

                    iteration = self.TSF_model.iteration
                    
                # joint model
                else:
                    pass

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(
                    len(history),
                    values=logs
                    if self.config.VERBOSE
                    else [x for x in logs if not x[0].startswith("l_")],
                )

                # log model at checkpoints
                if (
                    self.config.LOG_INTERVAL
                    and iteration % self.config.LOG_INTERVAL == 0
                ):
                    self.log(logs)

                # sample model at checkpoints
                if (
                    self.config.SAMPLE_INTERVAL
                    and iteration % self.config.SAMPLE_INTERVAL == 0
                ):
                    # 只在预测模型时查看：
                    if self.config.MODEL == 2:
                        mae, mse, rmse, mape, mspe = self.TSF_metric(
                            self.val_dataset.reverse_norm(outputs)
                            .detach()
                            .to("cpu")
                            .numpy(),
                            self.val_dataset.reverse_norm(future)
                            .detach()
                            .to("cpu")
                            .numpy(),
                        )

                        print(
                            "\ntrain : mae:{},mse:{},rmse:{},mape{}, mspe{} \n".format(
                                mae, mse, rmse, mape, mspe
                            )
                        )
                    # self.sample()
            # self.TSF_model.dynamic_alpha()
            wandb.log({"epoch_loss": total_loss})
            if (epoch + 1) % 1 == 0:
                print("\n validSet test:")
                val_loss,_ = self.test(
                    self.val_dataset, is_shuffle=True, criterion=nn.MSELoss()
                )
                print("\n testSet test:")
                test_loss,_ = self.test(self.test_dataset)
                wandb.log({"val_loss": val_loss})
                wandb.log({"test_loss": test_loss})
                if model == 3 or model == 2:
                    early_stopping(
                        val_loss, self.TSF_model, self.config.PATH, model_filename
                    )
                elif model == 1:
                    early_stopping(
                        val_loss, self.status_model, self.config.PATH, model_filename
                    )
            if early_stopping.early_stop:
                break
            
        T2=time.time()
        print(f"total traing time {(T2-T1)*1000}, epoch num {count}")

        print("\nEnd training....")
        print("\n Start final testing :")
        load_path = self.config.PATH + "/" + model_filename
        print("final choose test model{}".format(load_path))
        if model == 1:
            self.status_model.load_state_dict(
                torch.load(self.config.PATH + "/" + model_filename)
            )
        if model == 3:
            self.TSF_model.load_state_dict(
                torch.load(self.config.PATH + "/" + model_filename)
            )
        if model == 2:
            self.TSF_model.load_state_dict(
                torch.load(self.config.PATH + "/" + model_filename)
            )
        # 绘制训练集分布
        # test_loss = self.test(self.train_dataset, is_visual=True)

        test_loss, metric = self.test(self.test_dataset, is_visual=False)
        return metric

    def test(
        self, dataset=None, criterion=nn.L1Loss(), is_shuffle=False, compare_model=None, is_visual=False
    ):
        self.status_model.eval()
        self.TSF_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=dataset, batch_size=self.config.BATCH_SIZE, shuffle=is_shuffle
        )

        loss = 0
        index = 0
        metric = [0, 0, 0, 0, 0, 0, 0]
        all_variates=[[0 for i in range(self.config.data_dim)] for j in range(2)]
        prediction = None
        ground_truth = None
        pred_list = []
        groud_list = []
        with torch.no_grad():
            for iter, (
                history,
                future,
                status,
                date_features,
                date_future_features,
                history_status,
            ) in enumerate(test_loader):
                history = history.cuda(self.config.DEVICE)
                future = future.cuda(self.config.DEVICE)
                status = status.cuda(self.config.DEVICE).type(torch.long)
                history_status = history_status.cuda(self.config.DEVICE).type(
                    torch.float
                )
                # self.inpaint_model.train()

                # status classifier model
                if model == 1:
                    self.status_model.eval()
                    # train
                    (
                        outputs,
                        total_loss,
                        logs,
                        _,
                        _,
                        status_loss,
                    ) = self.status_model.process(
                        history, status, future, history_status
                    )
                    # logs.append(('outputs', outputs[0].item())) # 需要修改为值
                    logs.append(("status_loss", status_loss.cpu().detach().numpy()))
                    logs.append(("groundtruth_status", status.cpu().detach().numpy()))
                    accuracy, precision, recall, f1 = self.classification_metric(
                        outputs, status, debug_print=False
                    )
                    loss += status_loss
                    metric[0] += accuracy
                    metric[1] += precision
                    metric[2] += recall
                    metric[3] += f1
                    # print("\ntest :  accuracy:{}, precision:{}, recall:{}, f1{} *****\n".format(accuracy, precision,
                    #                                                                          recall, f1))
                    # backward
                    # self.status_model.backward(status_loss)
                    iteration = self.status_model.iteration

                # TSF model
                elif model == 2:
                    self.TSF_model.eval()
                    # train
                    outputs, tsf_status = self.TSF_model.specific_model_process(
                        history, future, date_features, date_future_features
                    )
                    # logs.append(('outputs', outputs[0].item())) # 需要修改为值
                    logs = []
                    loss += criterion(outputs, future)

                    self.print_tsf_loss_metric(
                        outputs=outputs,
                        future=future,
                        a_logs=logs,
                        metric=metric,
                        all_variates=all_variates,
                    )
                # TSF model
                elif model == 3:
                    self.TSF_model.eval()
                    self.status_model.eval()
                    status_soft, hidden, _ = self.status_model(
                        history, history_status
                    )  # 23 , hidden : 256, 512, 2
                    # train
                    outputs, tsf_status = self.TSF_model.specific_model_process(
                        history,
                        future,
                        date_features,
                        date_future_features,
                        status=status_soft,
                    )
                    if compare_model:
                        outputs2, tsf_status2 = compare_model.specific_model_process(
                            history,
                            future,
                            date_features,
                            date_future_features,
                            status=status_soft,
                        )
                    # logs.append(('outputs', outputs[0].item())) # 需要修改为值
                    logs = []
                    loss += criterion(outputs, future)
                    self.print_tsf_loss_metric(
                        outputs=outputs,
                        future=future,
                        a_logs=logs,
                        metric=metric,
                        all_variates=all_variates,
                    )
                else:
                    pass
                if model==2 or model==3 :
                    # pred_list.append(torch.flatten(outputs,start_dim=0,end_dim=1))
                    # groud_list.append(torch.flatten(future,start_dim=0,end_dim=1))
                    pred_list.append(outputs)
                    groud_list.append(future)
            if model == 1:
                print(
                    "accuracy:{:.6f},precision:{:.6f},recall:{:.6f},f1:{:.6f} *****\n".format(
                        metric[0] / len(test_loader),
                        metric[1] / len(test_loader),
                        metric[2] / len(test_loader),
                        metric[3] / len(test_loader),
                    )
                )
            else:
                if is_visual:
                    from src.utils import draw_plt
                    # torch.save(torch.cat(pred_list,dim=0),'robust_forecast_model2.npy')
                    torch.save(torch.cat(pred_list,dim=0).squeeze(),f'./result/visual_load_curve/pred_list{model}_{self.config.TSF_MODEL}_{self.config.data}_{self.config.pred_len}_{self.config.alpha}.csv')
                    torch.save(torch.cat(groud_list,dim=0).squeeze(),f'./result/visual_load_curve/groud_list{model}_{self.config.TSF_MODEL}_{self.config.data}_{self.config.pred_len}_{self.config.alpha}.csv')
                    print("success save result......")
                    # np.savetxt(f'pred_list{model}_crossformer_umass4.csv',torch.cat(pred_list,dim=0).squeeze().to('cpu').detach().numpy())
                    # np.savetxt(f'groud_list{model}_crossformer_umass4.csv',torch.cat(groud_list,dim=0).squeeze().to('cpu').detach().numpy())
                    # draw_plt(None,torch.cat(pred_list,dim=0).squeeze().unsqueeze(dim=0).to('cpu').numpy(),
                    #          torch.cat(groud_list,dim=0).squeeze().unsqueeze(dim=0).to('cpu').numpy(),model=model)
                    # draw_plt(None,torch.cat(pred_list,dim=0)[:168].squeeze().unsqueeze(dim=0).to('cpu').numpy(),
                    #          torch.cat(groud_list,dim=0)[:168].squeeze().unsqueeze(dim=0).to('cpu').numpy(),model=f"{model}0_168")                        
                    # draw_plt(None,torch.cat(pred_list,dim=0)[:48].squeeze().unsqueeze(dim=0).to('cpu').numpy(),
                    #          torch.cat(groud_list,dim=0)[:48].squeeze().unsqueeze(dim=0).to('cpu').numpy(),model=f"{model}0_48")                    
                print(
                    "mae:{:.6f},mse:{:.6f},rmse:{:.6f},mape:{:.6f},mspe:{:.6f},rse:{:.6f},corr:{:.6f}\n".format(
                        metric[0] / len(test_loader),
                        metric[1] / len(test_loader),
                        metric[2] / len(test_loader),
                        metric[3] / len(test_loader),
                        metric[4] / len(test_loader),
                        metric[5] / len(test_loader),
                        metric[6] / len(test_loader),
                    )
                )
                print("mae:")
                print(all_variates[0] / len(test_loader))
                print("mse:")
                print(all_variates[1] / len(test_loader))
            return loss, np.array(metric)/ len(test_loader)

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.status_model.eval()
        # self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        (
            history,
            future,
            status,
            date_features,
            date_future_features,
            history_status,
        ) = self.cuda(*items)
        status = status.type(torch.long)
        # attr model
        if model == 1:
            iteration = self.status_model.iteration
            inputs = history
            outputs, _, a_logs, _, _ = self.status_model.process(inputs, status)
            accuracy, precision, recall, f1 = self.classification_metric(
                outputs, status
            )

            print(
                "\nvalid : ***** accuracy:{},precision:{},recall:{},f1{} *****\n".format(
                    accuracy, precision, recall, f1
                )
            )

            a_logs.append(("accuracy", accuracy))
            a_logs.append(("precision", precision))
            a_logs.append(("recall", recall))
            a_logs.append(("f1", f1))
            wandb.log(
                {
                    "val acc": accuracy,
                    "val precision": precision,
                    "val recall": recall,
                    "val f1": f1,
                }
            )

        # TSF model
        elif model == 2 or model == 3:
            iteration = self.TSF_model.iteration
            outputs, TSF_loss, a_logs = self.TSF_model.process(
                history, future, date_features, date_future_features
            )
            self.print_tsf_loss_metric(outputs=outputs, future=future, a_logs=a_logs)
            # mae, mse, rmse, mape, mspe = self.TSF_metric(
            #     self.val_dataset.reverse_norm(outputs).detach().to('cpu').numpy(),
            #     self.val_dataset.reverse_norm(future).detach().to('cpu').numpy())
            #
            # print("\n***** mae:{},mse:{},rmse:{},mape{}, mspe{} *****\n".format(mae, mse, rmse, mape, mspe))
            #
            # a_logs.append(('mae', mae))
            # a_logs.append(('mse', mse))
            # a_logs.append(('rmse', rmse))
            # a_logs.append(('mape', mape))
            # a_logs.append(('mspe', mspe))
        # inpaint with edge model / joint model
        # inpaint with edge model / joint model
        # else:
        #     iteration = self.inpaint_model.iteration
        #     inputs = (images * (1 - masks)) + masks
        #     a_outputs, _, a_logs = self.attr_model.process(images, masks, attr)
        #     outputs = self.inpaint_model(images, masks, a_outputs)
        #     outputs_merged = (outputs * masks) + (images * (1 - masks))
        #
        #     a_logs.append(('mask_percent', mask_percent.item()))
        #     # metrics
        #     psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
        #     mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
        #     a_logs.append(('psnr', psnr.item()))
        #     a_logs.append(('mae', mae.item()))
        #
        #     for i in range(a_outputs.shape[0]):
        #         a_logs.append(('predict_attr', a_outputs[i].cpu().detach().numpy()))
        #         a_logs.append(('groundtruth_attr', attr[i].cpu().detach().numpy()))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        path = os.path.join(self.samples_path, self.model_name)
        create_dir(path)

        if model != 2:
            attr_file = os.path.join(path, "log_" + "attr" + ".dat")
            a_logs = [
                ("it", iteration),
            ] + a_logs
            with open(attr_file, "a") as f:
                f.write("%s\n" % " ".join([str(item[1]) for item in a_logs]))

    def log(self, logs):
        with open(self.log_file, "a") as f:
            f.write("%s\n" % " ".join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def mask_percent(self, mask):
        holes = torch.sum((mask > 0).float())
        pixel_num = torch.sum((mask >= 0).float())
        percent = holes / pixel_num
        return percent
