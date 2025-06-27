import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.network import StatusPredictor
from src.loss import StatusLoss

from STID_arch.STID_arch import STID
from src.cross_models.cross_former import Crossformer
from src.etsformer.model import ETSformer
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
from src.vae.model import VAE
from src.mtl_gru.model import MTL_GRU
from einops import reduce
from .etsformer.modules import ETSEmbedding
from .etsformer.encoder import EncoderLayer, Encoder
from .etsformer.decoder import DecoderLayer, Decoder
from .metric import StatusClassClassificationAccuracy
from RevIN.RevIN import RevIN
import torch.nn.functional as F
# import line_profiler
# from torchsummary import summary

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        # final HK_DALE
        # final HDFB :status_model120:40:51_predL_336_horizon_1.pth  status_model120:04:11_predL_168_horizon_1.pth status_model120:45:54_predL_72_horizon_1.pth
        # final status_model121:34:16_predL_336_horizon_1.pth status_model118:53:31_predL_168_horizon_1.pth  'status_model117:02:37_predL_72_horizon_1.pth'
        self.status_weights_path = config.PATH
        if config.status_file:
            self.status_weights_path = os.path.join(config.PATH,
                                                    config.status_file)  # status_model109:41:22_predL_72_horizon_1.pth
        self.tsf_weights_path = os.path.join(config.PATH,
                                             'status_model111:17:39.pth')  # 336: status_model115:47:36_predL_336.pth   192: status_model115:47:13_predL_192.pth   168 : status_model114:49:26.pth  96: status_model111:17:39.pth
        # stid horizon = 1 : status_model121:27:56_predL_1_horizon_1.pth horizon =1 dataset win=1: status_model120:48:24_predL_1_horizon_1.pth
        # stid horizon = 24 : status_model122:59:29_predL_1_horizon_24.pth . horizon = 72 : status_model122:59:10_predL_1_horizon_72.pth
        # stid horizon = 192 : status_model122:31:46_predL_1_horizon_192.pth
        # stid mean_filter horizon = 192 : status_model110:40:04_predL_1_horizon_192.pth horizon = 72 : status_model110:39:21_predL_1_horizon_72.pth
        # self.dis_weights_path = os.path.join(config.PATH, nae + '_dis.pth')
        print("*** the model path ****"+self.status_weights_path)

    def load(self):
        if os.path.exists(self.status_weights_path):
            print('Loading %s StatusModel...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.status_weights_path, map_location=self.config.DEVICE)
            else:
                data = torch.load(self.status_weights_path, map_location=lambda storage, loc: storage)
            print("***** load status_model *****")
            self.load_state_dict(data)
            self.iteration = 0

        # load discriminator only when training
        # if self.config.MODE == 1 and os.path.exists(self.dis_weights_path) and self.config.MODEL != 1:
        #     print('Loading %s discriminator...' % self.name)
        #
        #     if torch.cuda.is_available():
        #         data = torch.load(self.dis_weights_path)
        #     else:
        #         data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)
        #
        #     self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        if (self.name == 'TSFModel'):
            torch.save({
                'iteration': self.iteration,
                'TSFModel': self.TSFModel.state_dict()
            }, self.tsf_weights_path)
        else:
            torch.save({
                'iteration': self.iteration,
                'StatusModel': self.status_predictor.state_dict()
            }, self.status_weights_path)
        # if self.name != 'StatusModel':
        #     torch.save({
        #         'discriminator': self.discriminator.state_dict()
        #     }, self.dis_weights_path)


class StatusModel(BaseModel):
    def __init__(self, config, status_num=None):
        super(StatusModel, self).__init__('StatusModel', config)
        self.status_predictor = StatusPredictor(status=status_num, pred_length=config.pred_len).to(config.DEVICE)
        # if len(config.GPU) > 1:
        #     status_predictor = nn.DataParallel(status_predictor, device_ids=[0, 1])
        ce_loss = StatusLoss()
        self.mse_loss = torch.nn.MSELoss()
        # self.add_module('status_predictor', status_predictor)
        self.add_module('ce_loss', ce_loss)
        self.iteration = 0
        self.status_optimizer = optim.Adam(
            params=self.status_predictor.parameters(),
            lr=float(config.STATUS_LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.classification_metric = StatusClassClassificationAccuracy()

    def process(self, history: torch.Tensor, status: torch.Tensor, future: torch.Tensor, history_status: torch.Tensor):
        self.iteration += 1

        self.status_optimizer.zero_grad()
        outputs, hidden, regression_output = self(history, history_status)
        loss = 0
        status_loss, all_task_loss = self.ce_loss(outputs, status, history_status)
        regression_loss = self.mse_loss(regression_output, future)
        self.iteration = self.iteration + 1
        if (self.iteration % 100 == 0):
            accuracy, precision, recall, f1 = self.classification_metric(outputs, status, False)
            print(
                "\ntrain :  accuracy:{}, precision:{}, recall:{}, f1{} ,status_loss{}, regression_loss{} *****\n".format(
                    accuracy, precision,
                    recall, f1, status_loss.item(), regression_loss.item()))
        loss += status_loss
        # loss += regression_loss
        # create logs
        logs = [
            ("loss", loss.item()),
            ("status_loss", status_loss.item()),
            ("regression_loss", regression_loss.item()),
        ]
        return outputs, loss, logs, all_task_loss, hidden, status_loss

    def forward(self, history, history_status):
        # 计算存储花费
        # summary(self.status_predictor, history, history_status)
        outputs, hidden, regression_output = self.status_predictor(history, history_status)
        return outputs, hidden, regression_output

    def backward(self, loss=None):
        if loss is not None:
            loss.backward()
        self.status_optimizer.step()


class TSFModel(BaseModel):
    def __init__(self, config, status_num):
        super(TSFModel, self).__init__('TSFModel', config)
        self.status_num = status_num
        self.tsf_model = config.TSF_MODEL
        if self.tsf_model == 'etsformer':
            ts_predictor = ETSformer(configs=config, status=status_num).to(config.DEVICE)
        elif self.tsf_model == 'stid':
            ts_predictor = STID(node=config.enc_in, input_dim=1, input_len=config.seq_len, status=status_num).to(
                config.DEVICE)
        elif self.tsf_model == 'patchTsMixer':
            ts_predictor = PatchTSMixerForPrediction(PatchTSMixerConfig(context_length = config.seq_len, prediction_length = config.pred_len)) # num_targets=1,prediction_channel_indices=[-1]
        elif self.tsf_model == 'crossformer':
            # cross_d_model: 64
            # cross_d_ff: 128  
            # cross_n_heads: 2 
            # cross_e_layers: 3
            ts_predictor = Crossformer(
                config.data_dim,
                config.in_len,
                config.pred_len,
                config.seg_len,
                config.win_size,
                config.factor,
                config.cross_d_model,
                config.cross_d_ff,
                config.cross_n_heads,
                config.cross_e_layers,
                config.dropout,
                config.baseline,
                config.DEVICE,
                status_num
            ).float()
        elif self.tsf_model == 'vae':
            ts_predictor = VAE(input_dim=config.data_dim, output_length=config.pred_len).to(config.DEVICE)
        elif self.tsf_model == 'mtl_gru': 
            ts_predictor = MTL_GRU(data_dim=config.data_dim, output_size=config.data_dim, forecast_steps=config.pred_len).to(config.DEVICE)
        # if len(config.GPU) > 1 and False:
        #     ts_predictor = nn.DataParallel(ts_predictor, device_ids=[0, 1])
        criterion = nn.MSELoss(reduction='none')
        ce_criterion = StatusLoss()
        kd_criterion = StatusLoss(type='KLLoss')
        self.alpha = 1  # config.alpha_kd
        self.alpha_class = config.alpha_class
        self.add_module('TSFModel', ts_predictor)
        self.add_module('criterion', criterion)
        self.add_module('ce_criterion', ce_criterion)
        self.add_module('kd_criterion', kd_criterion)

        self.TSF_optimizer = optim.Adam(
            params=ts_predictor.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

    # @line_profiler.profile
    def specific_model_process(self, history: torch.Tensor, future: torch.Tensor, date_features, date_future_features,
                               status_info_hidden=None, status=None, status_truth=None):
        print(self.tsf_model)
        if self.tsf_model == 'etsformer':
            batch_x = history.float()
            batch_x_mark = date_features.float()
            batch_y = future.float()
            batch_y_mark = date_future_features.float()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float()

            # 统计存储花费
            # summary(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, status_info_hidden, status)
            # encoder -> decoder
            outputs, tsf_status = self(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                       status_info_hidden=status_info_hidden,
                                       status=status)
            tsf_status = torch.split(tsf_status, self.status_num, dim=-1)
        elif self.tsf_model == 'stid':
            outputs, tsf_status = self(history.unsqueeze(dim=-1))
            tsf_status = torch.split(tsf_status, self.status_num, dim=-1)
        elif self.tsf_model == 'crossformer':
            # summary(self, history.float())
            outputs, tsf_status = self(history.float())
        elif self.tsf_model == 'patchTsMixer':
            # summary(self, history)
            outputs = self(history).prediction_outputs
            tsf_status = None
        elif self.tsf_model == 'vae':
            outputs, _, _ = self(history)
            tsf_status = None
        elif self.tsf_model == 'mtl_gru':
            outputs = self(history)
            tsf_status = None
        return outputs, tsf_status

    def dynamic_alpha(self):
        if self.alpha<5:
            self.alpha*=2
        else:
            self.alpha = max(self.alpha/2, 1e-3)
        return self.alpha
    
    # @line_profiler.profile
    def process(self, history: torch.Tensor, future: torch.Tensor, date_features, date_future_features,
                status_info_hidden=None, status=None, status_truth=None, mean=0, std=1, alpha=0.5,
                history_satus_truth=None, topk=12):
        self.iteration += 1
        self.TSF_optimizer.zero_grad()
        tsf_status = None
        
        # revin_layer = RevIN(history.shape[2]).to(history.device)
        # x_in = revin_layer(history, 'norm')
        outputs, tsf_status = self.specific_model_process(history, future, date_features, date_future_features,
                                                          status_info_hidden, status, status_truth)
        # outputs = revin_layer(outputs, 'denorm')

        # import torch.nn.functional as F
        # max = 5
        # status_align = []
        # for s in status:
        #     status_align.append( F.pad(s,(0, max-s.shape[2])))
        # tmp = torch.stack(status_align, dim=2)
        # ouputs_reverse = outputs * std + mean
        # aggregate = torch.stack([torch.sum(ouputs_reverse[:, :, :24], dim=2),
        #                          torch.sum(ouputs_reverse[:, :, :12], dim=2),
        #                          torch.sum(ouputs_reverse[:, :, 12:24], dim=2),
        #                          torch.sum(ouputs_reverse[:, :, :6], dim=2),
        #                          torch.sum(ouputs_reverse[:, :, 6:12], dim=2),
        #                          torch.sum(ouputs_reverse[:, :, 12:18], dim=2),
        #                          torch.sum(ouputs_reverse[:, :, 18:24], dim=2),
        #                          ], dim=2)
        assert outputs.squeeze().shape == future.squeeze().shape , "The shapes of 'outputs' and 'future' are not the same." 
        prediction_loss = self.criterion(outputs.squeeze(), future.squeeze())
        # tmp = future*std+mean
        if status != None:
            max = 5
            status_align = []
            for s in status:
                # status_align.append(F.pad(F.softmax(s,dim=2), (0, max - s.shape[2]))) # F.softmax(s,dim=2)*2-1
                status_align.append(F.pad(s, (0, max - s.shape[2])))
            tmp = torch.stack(status_align, dim=2)
            # tmp[:, :, [0, 1, 3], :] = 0
            history_most_status = (torch.sum(history_satus_truth, axis=1) / history_satus_truth.shape[1]).type(torch.int64)
            # history_mean = torch.mean(history, axis=1)
            cluster_loss = abs((outputs*(torch.max(tmp, dim=3)[1] == history_most_status.unsqueeze(dim=1))).sum(axis=2)-(future*(torch.max(tmp, dim=3)[1] == history_most_status.unsqueeze(dim=1))).sum(axis=2))
            + abs((outputs*(torch.max(tmp, dim=3)[1] != history_most_status.unsqueeze(dim=1))).sum(axis=2)-(future*(torch.max(tmp, dim=3)[1] != history_most_status.unsqueeze(dim=1))).sum(axis=2))
            # history_loss = (torch.max(tmp, dim=3)[1] != history_most_status.unsqueeze(dim=1)) * self.criterion(outputs,
            #
            #                                                                                                      dim=1))
            profile = (torch.max(tmp, dim=3)[1] == status_truth) * torch.max(tmp, dim=3)[0]  # torch.max(tmp, dim=3)[0] (torch.max(tmp, dim=3)[1] == status_truth) * torch.max(tmp, dim=3)[0]
            profile_top = torch.topk(profile, k=topk)
            # # # # profile[profile_top[1]] = 0*profile[profile_top[1]]
            # # # # drop top
            # # profile = profile.scatter(2, profile_top[1], 0*profile_top[0])
            # # # keep top
            profile = 0 * profile
            profile = profile.scatter(2, profile_top[1], profile_top[0]).squeeze()
            # profile[:,:,17:] = 0*profile[:,:,17:]
            # profile[:,:,13:16] = 0*profile[:,:,13:16]
            # origin ********
            assert profile.shape == prediction_loss.shape, "The shapes of 'profile' and 'prediction_loss' are not the same." 
            prediction_loss = prediction_loss + alpha * torch.abs(
                profile * prediction_loss)   #- history_loss  # + profile * prediction_loss  # torch.gather(tmp,dim=2,index=torch.unsqueeze(status_truth,dim=-1)).squeeze()).mean()
            # prediction_loss[:, :, [7, 17, 20, 23]] = 0.01 * prediction_loss[:, :, [7, 17, 20, 23]]
            # if (prediction_loss.mean()<1.1):
            #     prediction_loss = - prediction_loss
            prediction_loss = prediction_loss.mean() # +0.1*torch.abs(outputs).mean() # + 0.1*cluster_loss.mean()

        else :
            # prediction_loss[:, :, [7, 17, 20, 23]] = 0
            # tmp = prediction_loss
            # tmp[:, :, [7, 17, 20, 23]] = 0 * tmp[:, :, [7, 17, 20, 23]]

            # tmp = tmp.to('cuda:1')
            # tmp_list = [7, 17, 20, 23]
            prediction_loss = prediction_loss.mean()  # + 10*prediction_loss[:,:,[4,6,7,15,17,20]].mean() # + 10*prediction_loss[:, :, [i for i  in range(prediction_loss.shape[-1]) if i not in tmp_list]].mean()# + 0.5*(self.criterion((aggregate - mean[-7:]) / std[-7:], future[:, :,
            # -7:])).mean()  # -(1*prediction_loss[:, :, ~[7, 17, 20, 23] ]).mean() #  0 7 13 21 17 20 23
        status_loss = torch.tensor(0)
        # if tsf_status !=None:n'n
        #     split_dim = len(tsf_status.shape) - 1
        #     # hard_labels
        #     if status_truth != None:
        #         status_loss, all_task_loss = self.ce_criterion(torch.split(tsf_status, self.status_num, dim=split_dim),
        #                                                        status_truth)
        #     # soft_labels
        #     if status != None:
        #         # knowledge distillation
        #         kd_status_loss, kd_all_task_loss = self.kd_criterion(
        #             torch.split(tsf_status, self.status_num, dim=split_dim),
        #             status)
        #         status_loss = self.alpha * status_loss + (1 - self.alpha) * kd_status_loss
        # hard_labels
        if status_truth != None and tsf_status !=None:
            status_loss, all_task_loss = self.ce_criterion(tsf_status, status_truth)
        # soft_labels
        if status != None and tsf_status !=None:
            # knowledge distillation
            kd_status_loss, kd_all_task_loss = self.kd_criterion(tsf_status, status)
            status_loss = self.alpha * status_loss + (1 - self.alpha) * kd_status_loss
        loss = prediction_loss  # + 0 * status_loss
        # create logs
        # create logs
        logs = [
            ("loss", loss.item()),
            ("prediction_loss", prediction_loss.item()),
            ("status_loss", status_loss.item())
        ]
        # backward
        self.backward(loss)

        return outputs, loss, logs

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, status_info_hidden=None,
                status=None):
        if self.tsf_model == 'etsformer':
            outputs = self.TSFModel(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                    status_info_hidden=status_info_hidden,
                                    status=status)
        elif self.tsf_model == 'stid':
            outputs = self.TSFModel(batch_x)
        elif self.tsf_model == 'crossformer':
            outputs = self.TSFModel(batch_x)
        elif self.tsf_model == 'patchTsMixer':
            outputs = self.TSFModel(batch_x)
        elif self.tsf_model == 'vae':
            outputs = self.TSFModel(batch_x)
        elif self.tsf_model == 'mtl_gru':
            outputs = self.TSFModel(batch_x)
        return outputs

    # @line_profiler.profile
    def backward(self, loss=None):
        if loss is not None:
            loss.backward()
        self.TSF_optimizer.step()

    # class InpaintingModel(BaseModel):
    #     def __init__(self, config):
    #         super(InpaintingModel, self).__init__('InpaintingModel', config)
    #
    #         generator = InpaintGenerator()
    #         discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
    #         if len(config.GPU) > 1:
    #             generator = nn.DataParallel(generator, device_ids=[0,1])
    #             discriminator = nn.DataParallel(discriminator, device_ids=[0,1])
    #
    #         l1_loss = nn.L1Loss()
    #         perceptual_loss = PerceptualLoss()
    #         style_loss = StyleLoss()
    #         adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
    #
    #         self.add_module('generator', generator)
    #         self.add_module('discriminator', discriminator)
    #
    #         self.add_module('l1_loss', l1_loss)
    #         self.add_module('perceptual_loss', perceptual_loss)
    #         self.add_module('style_loss', style_loss)
    #         self.add_module('adversarial_loss', adversarial_loss)
    #
    #         self.gen_optimizer = optim.Adam(
    #             params=generator.parameters(),
    #             lr=float(config.LR),
    #             betas=(config.BETA1, config.BETA2)
    #         )
    #
    #         self.dis_optimizer = optim.Adam(
    #             params=discriminator.parameters(),
    #             lr=float(config.LR) * float(config.D2G_LR),
    #             betas=(config.BETA1, config.BETA2)
    #         )
    #
    #     def process(self, images, masks, attr):
    #         self.iteration += 1
    #
    #         # zero optimizers
    #         self.gen_optimizer.zero_grad()
    #         self.dis_optimizer.zero_grad()
    #
    #         with torch.autograd.set_detect_anomaly(True):
    #         # process outputs
    #             outputs = self(images, masks, attr)
    #             gen_loss = 0
    #             dis_loss = 0
    #
    #
    #         # discriminator loss
    #             dis_input_real = images
    #             dis_input_fake = outputs.detach()
    #             dis_real, _ = self.discriminator(dis_input_real)
    #             dis_fake, _ = self.discriminator(dis_input_fake)
    #             dis_real_loss = self.adversarial_loss(dis_real, True, True)
    #             dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
    #             dis_loss = dis_loss + (dis_real_loss + dis_fake_loss) / 2
    #
    #
    #         # generator adversarial loss
    #             gen_input_fake = outputs
    #             gen_fake, _ = self.discriminator(gen_input_fake)
    #             gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
    #         #gen_loss = gen_loss + gen_gan_loss
    #             gen_loss1 = gen_loss + gen_gan_loss
    #
    #
    #         # generator l1 loss
    #             gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
    #         #gen_loss = gen_loss + gen_l1_loss
    #             gen_loss2 = gen_loss1 + gen_l1_loss
    #
    #
    #         # generator perceptual loss
    #             gen_content_loss = self.perceptual_loss(outputs, images)
    #             gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
    #         #gen_loss = gen_loss + gen_content_loss
    #             gen_loss3 = gen_loss2 + gen_content_loss
    #
    #
    #
    #         # generator style loss
    #             gen_style_loss = self.style_loss(outputs * masks, images * masks)
    #             gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
    #         #gen_loss = gen_loss + gen_style_loss
    #             gen_loss4 = gen_loss3 + gen_style_loss
    #
    #
    #         # create logs
    #         logs = [
    #             ("l_d2", dis_loss.item()),
    #             ("l_g2", gen_gan_loss.item()),
    #             ("l_l1", gen_l1_loss.item()),
    #             ("l_per", gen_content_loss.item()),
    #             ("l_sty", gen_style_loss.item()),
    #         ]
    #
    #         return outputs, gen_loss4, dis_loss, logs
    #
    #     def forward(self, images, masks, attr):
    #         images_masked = (images * (1 - masks).float()) + masks
    #         outputs = self.generator(images_masked, attr)
    #         return outputs
    #
    #     def backward(self, gen_loss=None, dis_loss=None):
    #         dis_loss.backward(retain_graph=True)
    #         gen_loss.backward(retain_graph=True)
    #         self.gen_optimizer.step()
    #         self.dis_optimizer.step()
