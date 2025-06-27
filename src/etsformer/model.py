import torch
import torch.nn as nn
from einops import reduce

from .modules import ETSEmbedding
from .modules import StatusEmbedding
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder
from RevIN.RevIN import RevIN
# import line_profiler

class Transform:
    def __init__(self, sigma):
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape, device=x.device) * self.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1), device=x.device) * self.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1), device=x.device) * self.sigma)


class ETSformer(nn.Module):

    def __init__(self, status, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # total status num
        self.status_num = sum(status)

        self.configs = configs

        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = ETSEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout).to(configs.DEVICE)

        # Embedding growths, seasons
        # self.status_embedding = StatusEmbedding(self.status_num, self.seq_len, dropout=configs.dropout).to(
        #     configs.DEVICE)

        # Embedding
        self.status_growths_embedding = StatusEmbedding(self.status_num, self.seq_len + 1, self.pred_len,
                                                        dropout=configs.dropout).to(
            configs.DEVICE)
        self.status_seasons_embedding = StatusEmbedding(self.status_num, self.seq_len + 1, self.pred_len,
                                                        dropout=configs.dropout).to(
            configs.DEVICE)

        # Encoder
        self.encoder = Encoder(
            layers=
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.seq_len, configs.pred_len, configs.K,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    output_attention=configs.output_attention,
                ) for _ in range(configs.e_layers)
            ],
            statusLayers=
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.seq_len, configs.pred_len, configs.K,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    output_attention=configs.output_attention,
                ) for _ in range(configs.e_layers)
            ],
        )

        # Decoder
        self.decoder = Decoder(
            layers=
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.pred_len,
                    dropout=configs.dropout,
                    output_attention=configs.output_attention,
                ) for _ in range(configs.d_layers)
            ],
            statusLayers=
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.pred_len,
                    dropout=configs.dropout,
                    output_attention=configs.output_attention,
                ) for _ in range(configs.d_layers)
            ], status=status
        )

        self.transform = Transform(sigma=self.configs.std)

        # align and fuse
        self.align_layer = nn.Conv1d(in_channels=21, out_channels=337, kernel_size=1)
        self.fuse_layer = nn.Sequential(
            nn.Conv1d(in_channels=337 * 2, out_channels=337, kernel_size=1))  # ,
        # nn.Dropout(p=configs.dropout),
        # nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
        # nn.ReLU(),
        # nn.Conv1d(in_channels=512, out_channels=337, kernel_size=1))
        self.revin_layer=RevIN(num_features=configs.enc_in)

    # @line_profiler.profile
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                decomposed=False, attention=False, status_info_hidden=None, status=None, fuse_type='logit_direct', fuse_in_decoder=False):
        x_enc=self.revin_layer(x_enc, mode='norm')
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)  # data enhanced (scale,shift,jt)
        res = self.enc_embedding(x_enc)  # 256x336x23, 256x336x512
        status_emb = None
        status_growths_emb = None
        status_seasons_emb = None
        if status != None and fuse_in_decoder:
        #     # status_emb = self.status_embedding(status)
            status_growths_emb = self.status_growths_embedding(status)
            status_seasons_emb = self.status_seasons_embedding(status)
        level, growths, seasons, season_attns, growth_attns = self.encoder(res, x_enc, attn_mask=enc_self_mask,
                                                                           status_emb=status_emb)

        # if status_info_hidden != None and len(status)>0:
        #     if fuse_type == 'logit_direct':
        #         status_emb = self.status_embedding(status)
        
        #         for idx in range(len(growths)):
        #             growths[idx] = self.fuse_layer(torch.cat([status_emb, growths[idx]], dim=1))
        #             seasons[idx] = self.fuse_layer(torch.cat([status_emb, seasons[idx]], dim=1))
        #     else :
        #         status_info_hidden = self.align_layer(status_info_hidden.transpose(dim0=1, dim1=2))
        
        #         for idx in range(len(growths)):
        #             growths[idx] = self.fuse_layer(torch.cat([status_info_hidden, growths[idx]], dim=1))
        #             seasons[idx] = self.fuse_layer(torch.cat([status_info_hidden, seasons[idx]], dim=1))

        growth, season, growth_dampings, tsf_status = self.decoder(growths, seasons, level[:, -1:],
                                                                   status_growths_emb=status_growths_emb,
                                                                   status_seasons_emb=status_seasons_emb)

        if decomposed:
            return level[:, -1:], growth, season

        preds = level[:, -1:] + growth + season  # + x_enc[:,[-1],:]

        if attention:
            decoder_growth_attns = []
            for growth_attn, growth_damping in zip(growth_attns, growth_dampings):
                decoder_growth_attns.append(torch.einsum('bth,oh->bhot', [growth_attn.squeeze(-1), growth_damping]))

            season_attns = torch.stack(season_attns, dim=0)[:, :, -self.pred_len:]
            season_attns = reduce(season_attns, 'l b d o t -> b o t', reduction='mean')
            decoder_growth_attns = torch.stack(decoder_growth_attns, dim=0)[:, :, -self.pred_len:]
            decoder_growth_attns = reduce(decoder_growth_attns, 'l b d o t -> b o t', reduction='mean')
            preds = self.revin_layer(preds, mode='denorm')
            return preds, season_attns, decoder_growth_attns
        preds = self.revin_layer(preds, mode='denorm')
        return preds, tsf_status
