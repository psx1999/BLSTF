import torch
from torch import nn

from .mlp import MultiLayerPerceptron, GraphMLP
from torch.nn import functional as F
from .transformer import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, \
    TransformerDecoderLayer

device = 'cuda'


class BLSTF(nn.Module):

    def __init__(self, adj_mx, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.nhead = model_args["nhead"]
        self.dim_feedforward = model_args["dim_feedforward"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_spatial"]
        self.if_two_way = model_args["if_two_way"]
        self.if_decouple = model_args["if_decouple"]

        self.adj_mx = adj_mx
        self.hidden_dim = self.input_len + self.if_spatial * self.node_dim + \
                          int(self.if_time_in_day) * self.temp_dim_tid + \
                          int(self.if_day_in_week) * self.temp_dim_diw + \
                          int(self.if_decouple) * self.input_len

        if self.if_decouple:
            self.transformer = Transformer(d_model=self.input_len, nhead=self.nhead,
                                           num_encoder_layers=self.num_layer, num_decoder_layers=self.num_layer,
                                           dim_feedforward=self.dim_feedforward, batch_first=True)

        if self.if_spatial:
            self.adj_mx_encoder_1_1 = nn.Sequential(
                *[GraphMLP(self.num_nodes, self.node_dim) for _ in range(1)],
            )

            self.adj_mx_encoder_2_1 = nn.Sequential(
                *[GraphMLP(self.num_nodes, self.node_dim) for _ in range(1)],
            )

        if self.if_time_in_day:
            self.time_in_day_emb_1 = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb_1)

            self.time_in_day_emb_2 = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb_2)
        if self.if_day_in_week:
            self.day_in_week_emb_1 = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb_1)

            self.day_in_week_emb_2 = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb_2)

        self.graph_model_1_1 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(3)],
        )
        self.graph_model_1_2 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim + self.output_len, self.hidden_dim + self.output_len) for _ in
              range(3)],
        )

        if self.if_two_way:
            self.graph_model_2_1 = nn.Sequential(
                *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(3)],
            )
            self.graph_model_2_2 = nn.Sequential(
                *[MultiLayerPerceptron(self.hidden_dim + self.output_len, self.hidden_dim + self.output_len) for _ in
                  range(3)],
            )

        if self.if_two_way:
            self.regression_layer_1 = nn.Sequential(
                *[MultiLayerPerceptron(2 * self.hidden_dim, 2 * self.hidden_dim) for _ in range(3)],
                nn.Linear(in_features=2 * self.hidden_dim, out_features=self.output_len, bias=True),
            )

            self.regression_layer_2 = nn.Sequential(
                *[MultiLayerPerceptron(2 * (self.hidden_dim + self.output_len), 2 * (self.hidden_dim + self.output_len))
                  for _ in range(3)],
                nn.Linear(in_features=2 * (self.hidden_dim + self.output_len), out_features=self.output_len, bias=True),
            )
        else:
            self.regression_layer_1 = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=self.output_len, bias=True),
            )

            self.regression_layer_2 = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim + self.output_len, out_features=self.output_len, bias=True),
            )

        self.regression_layer = nn.Sequential(
            *[MultiLayerPerceptron(2 * self.output_len, 2 * self.output_len) for _ in range(3)],
            nn.Linear(in_features=2 * self.output_len, out_features=self.output_len, bias=True),
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:

        input_data = history_data[..., 0].transpose(1, 2)
        batch_size, num_nodes, _ = input_data.shape
        noise_emb = []
        if self.if_decouple:
            input_data_periodicity = self.transformer(input_data, input_data)
            input_data_noise = input_data - input_data_periodicity
            time_series_emb = input_data_periodicity
            noise_emb.append(input_data_noise)
        else:
            time_series_emb = input_data

        tem_emb_1 = []
        tem_emb_2 = []
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1] * self.time_of_day_size
            tem_emb_1.append(self.time_in_day_emb_1[(t_i_d_data[:, -1, :]).type(torch.LongTensor)])
            tem_emb_2.append(self.time_in_day_emb_2[(t_i_d_data[:, -1, :]).type(torch.LongTensor)])
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            tem_emb_1.append(self.day_in_week_emb_1[(d_i_w_data[:, -1, :]).type(torch.LongTensor)])
            tem_emb_2.append(self.day_in_week_emb_2[(d_i_w_data[:, -1, :]).type(torch.LongTensor)])

        node_emb_forward = []
        node_emb_backward = []
        if self.if_spatial:
            node_emb_1_1 = self.adj_mx[0].to(device)
            node_emb_2_1 = self.adj_mx[1].to(device)

            node_emb_1_1 = self.adj_mx_encoder_1_1(node_emb_1_1.unsqueeze(0)).expand(batch_size, -1, -1)
            node_emb_2_1 = self.adj_mx_encoder_2_1(node_emb_2_1.unsqueeze(0)).expand(batch_size, -1, -1)

            node_emb_forward.append(node_emb_1_1)
            node_emb_backward.append(node_emb_2_1)

        if self.if_spatial and self.if_two_way:
            hidden_1_1 = torch.cat([time_series_emb] + node_emb_forward + tem_emb_1 + noise_emb, dim=2)
            hidden_2_1 = torch.cat([time_series_emb] + node_emb_backward + tem_emb_1 + noise_emb, dim=2)
            res_1_1 = self.graph_model_1_1(hidden_1_1)
            res_2_1 = self.graph_model_2_1(hidden_2_1)

            all_hidden_1 = [res_1_1, res_2_1]
            prediction_1 = self.regression_layer_1(torch.cat(all_hidden_1, dim=2))

            hidden_1_2 = torch.cat([time_series_emb] + [prediction_1] + node_emb_forward + tem_emb_2 + noise_emb,
                                   dim=2)
            hidden_2_2 = torch.cat([time_series_emb] + [prediction_1] + node_emb_backward + tem_emb_2 + noise_emb,
                                   dim=2)
            res_1_2 = self.graph_model_1_2(hidden_1_2)
            res_2_2 = self.graph_model_2_2(hidden_2_2)

            all_hidden_2 = [res_1_2, res_2_2]
            prediction_2 = self.regression_layer_2(torch.cat(all_hidden_2, dim=2))

            prediction = [prediction_1, prediction_2]
            prediction = self.regression_layer(torch.cat(prediction, dim=2))
        else:
            hidden_1_1 = torch.cat([time_series_emb] + node_emb_forward + tem_emb_1 + noise_emb, dim=2)
            res_1_1 = self.graph_model_1_1(hidden_1_1)
            prediction_1 = self.regression_layer_1(res_1_1)

            hidden_1_2 = torch.cat([time_series_emb] + [prediction_1] + node_emb_forward + tem_emb_2 + noise_emb,
                                   dim=2)
            res_1_2 = self.graph_model_1_2(hidden_1_2)
            prediction_2 = self.regression_layer_2(res_1_2)

            prediction = [prediction_1, prediction_2]
            prediction = self.regression_layer(torch.cat(prediction, dim=2))

        return prediction.transpose(1, 2).unsqueeze(-1)
