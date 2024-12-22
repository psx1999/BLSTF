import torch
from torch import nn

from .mlp import MultiLayerPerceptron, GraphMLP, FusionMLP
from .transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, \
    Transformer

device = 'cuda'


class BLSTF(nn.Module):

    def __init__(self, adj_mx, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]
        self.if_decouple = model_args["if_decouple"]

        self.fusion_num_step = model_args["fusion_num_step"]
        self.fusion_num_layer = model_args["fusion_num_layer"]
        self.fusion_dim = model_args["fusion_dim"]
        self.fusion_out_dim = model_args["fusion_out_dim"]
        self.fusion_dropout = model_args["fusion_dropout"]

        self.if_forward = model_args["if_forward"]
        self.if_backward = model_args["if_backward"]
        self.if_ada = model_args["if_ada"]
        self.adj_mx = adj_mx
        self.node_dim = model_args["node_dim"]
        self.nhead = model_args["nhead"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]

        self.graph_num = 1 * self.if_forward + 1 * self.if_backward + 1 * self.if_ada

        self.st_dim = (self.graph_num > 0) * self.node_dim + \
                      self.if_time_in_day * self.temp_dim_tid + \
                      self.if_day_in_week * self.temp_dim_diw

        self.output_dim = self.fusion_num_step * self.fusion_out_dim

        if self.if_forward:
            self.adj_mx_forward_encoder = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
            )
        if self.if_backward:
            self.adj_mx_backward_encoder = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
            )
        if self.if_ada:
            self.adj_mx_ada = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.adj_mx_ada)

        self.fusion_layers = nn.ModuleList([
            FusionMLP(
                input_dim=self.st_dim + self.input_len + self.input_len * self.if_decouple,
                hidden_dim=self.st_dim + self.input_len + self.input_len * self.if_decouple,
                out_dim=self.fusion_out_dim,
                graph_num=self.graph_num,
                first=True, **model_args)
        ])
        for _ in range(self.fusion_num_step - 1):
            self.fusion_layers.append(
                FusionMLP(input_dim=self.st_dim + self.input_len + self.input_len * self.if_decouple + self.fusion_out_dim,
                          hidden_dim=self.st_dim + self.input_len + self.input_len * self.if_decouple + self.fusion_out_dim,
                          out_dim=self.fusion_out_dim,
                          graph_num=self.graph_num,
                          first=False, **model_args)
            )
        if self.fusion_num_step > 1:
            self.regression_layer = nn.Sequential(
                *[MultiLayerPerceptron(input_dim=self.output_dim,
                                       hidden_dim=self.output_dim,
                                       dropout=self.fusion_dropout)
                  for _ in range(self.fusion_num_layer)],
                nn.Linear(in_features=self.output_dim, out_features=self.output_len, bias=True),
            )

        if self.if_decouple:
            self.transformer = Transformer(d_model=self.input_len, nhead=self.nhead,
                                           num_encoder_layers=self.nhead, num_decoder_layers=self.nhead,
                                           dim_feedforward=4 * self.input_len, batch_first=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:

        input_data = history_data[..., 0].transpose(1, 2)
        batch_size, num_nodes, _ = input_data.shape

        periodicity = []
        noise = []
        if self.if_decouple:
            input_data_periodicity = self.transformer(input_data, input_data)
            input_data_noise = input_data - input_data_periodicity
            periodicity.append(input_data_periodicity)
            noise.append(input_data_noise)
        else:
            periodicity.append(input_data)

        time_series_emb = [torch.cat(periodicity + noise, dim=2)]

        node_forward_emb = []
        node_backward_emb = []
        node_ada_emb = []
        if self.if_forward:
            node_forward = self.adj_mx[0].to(device)
            node_forward = self.adj_mx_forward_encoder(node_forward.unsqueeze(0)).expand(batch_size, -1, -1)
            node_forward_emb.append(node_forward)

        if self.if_backward:
            node_backward = self.adj_mx[1].to(device)
            node_backward = self.adj_mx_backward_encoder(node_backward.unsqueeze(0)).expand(batch_size, -1, -1)
            node_backward_emb.append(node_backward)

        if self.if_ada:
            node_ada_emb.append(self.adj_mx_ada.unsqueeze(0).expand(batch_size, -1, -1))

        predicts = []
        predict_emb = []
        for index, layer in enumerate(self.fusion_layers):
            predict = layer(history_data, time_series_emb, predict_emb,
                            node_forward_emb, node_backward_emb, node_ada_emb)
            predicts.append(predict)
            predict_emb = [predict]

        predicts = torch.cat(predicts, dim=2)
        if self.fusion_num_step > 1:
            predicts = self.regression_layer(predicts)
        return predicts.transpose(1, 2).unsqueeze(-1)
