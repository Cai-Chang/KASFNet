from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATConv,  BatchNorm # noqa
import torch.nn as nn
from torch_geometric.nn import GINConv,  BatchNorm
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import GCNConv, SAGEConv,GraphConv,ChebConv
from torch_geometric.utils import *
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch import Tensor
from torch.nn import Parameter
import math
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# 尝试使用你项目里的 KANLinear（若不可用自动回退到 nn.Linear）
try:
    from GNNs import KANLinear as _MaybeKANLinear  # 如果本文件名也是 GNNs，可改成: from . import KANLinear
    _HAS_KAN = True
except Exception:
    _HAS_KAN = False
    _MaybeKANLinear = nn.Linear

class kanChebConv(MessagePassing):
    """
    K 阶 Chebyshev 图卷积（Graph-KAN 版）：
      T0 = x
      T1 = Â x
      T_{k+1} = 2 Â T_k - T_{k-1}
    其中 Â 采用 GCN 归一化（由 gcn_norm 计算），对 mini-batch 拼接图也安全。

    特色：
    - 权重按阶次独立：W_0..W_K（优先使用 KANLinear）
    - 可选 edge_attr：通过一个小 MLP 生成每条边的缩放系数 s_e ∈ (0,1)，用于调制 message
    - 批图兼容：norm 的长度与边数 E 严格一致

    参数
    ----
    in_channels : int
    out_channels: int
    K           : int，Chebyshev 多项式的最高阶（>=1）
    edge_attr_dim: int 或 None；若不为 None，则启用边特征调制
    improved    : bool，传给 gcn_norm（是否改进版 GCN）
    add_self_loops: bool，传给 gcn_norm（是否添加自环）
    bias        : bool，线性层是否带 bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
        edge_attr_dim: int = None,
        improved: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
    ):
        super().__init__(aggr='add', node_dim=0)  # PyG 推荐写法

        assert K >= 1, "K must be >= 1"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = int(K)
        self.improved = improved
        self.add_self_loops = add_self_loops

        Linear = _MaybeKANLinear if _HAS_KAN else nn.Linear
        # K+1 个按阶的线性层：W_0, W_1, ..., W_K
        self.lins = nn.ModuleList([Linear(in_channels, out_channels, bias=bias) for _ in range(self.K + 1)])

        # 可选：边特征 → 缩放系数 (0,1)，用于调制每条边的信息
        self.use_edge_attr = edge_attr_dim is not None
        if self.use_edge_attr:
            hidden = max(16, min(128, edge_attr_dim * 2))
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_attr_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),  # 输出 (0,1) 的缩放因子
            )

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
            else:
                # KANLinear 若无 reset，可略过
                pass

        if self.use_edge_attr:
            for m in self.edge_mlp:
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

    # -------- MessagePassing 需要的三个关键方法：forward / message / propagate(父类实现) --------

    # def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
    #     """
    #     x:           (N, Fin)     —— 批图时 N 是该 batch 所有分子的节点数之和
    #     edge_index:  (2, E)
    #     edge_attr:   (E, Fe) 或 None
    #     return:      (N, Fout)
    #     """
    #     # 1) 计算边级别归一化系数（和 edge_index 对齐），对 mini-batch 安全
    #     edge_index, norm = gcn_norm(
    #         edge_index,
    #         edge_weight=None,
    #         num_nodes=x.size(0),
    #         add_self_loops=self.add_self_loops,
    #         improved=self.improved,
    #         dtype=x.dtype,
    #     )
    #
    #     # 2) 如启用边特征调制，则把 edge_attr -> scale ∈ (0,1)，与每条边对齐
    #     edge_scale = None
    #     if self.use_edge_attr:
    #         if edge_attr is None:
    #             raise ValueError("edge_attr_dim specified but edge_attr=None was passed.")
    #         if edge_attr.dim() == 1:
    #             edge_attr = edge_attr.unsqueeze(-1)
    #         edge_scale = self.edge_mlp(edge_attr).view(-1)  # (E,)
    #
    #     # 3) Chebyshev 多项式递推并线性组合
    #     # T0 = x
    #     T0 = x
    #     out = self._W(0)(T0)
    #
    #     # T1 = Â x
    #     T1 = self.propagate(edge_index, x=x, norm=norm, edge_scale=edge_scale)  # (N, Fin)
    #     out = out + self._W(1)(T1)
    #
    #     # 后续阶（K>=2）
    #     Tk_1, Tk = T0, T1
    #     for k in range(2, self.K + 1):
    #         ATk = self.propagate(edge_index, x=Tk, norm=norm, edge_scale=edge_scale)
    #         Tk1 = 2.0 * ATk - Tk_1  # T_{k+1} = 2Â T_k - T_{k-1}
    #         out = out + self._W(k)(Tk1)
    #         Tk_1, Tk = Tk, Tk1
    #
    #     return out
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        # 1) 先规范化（可能会添加自环）
        edge_index, norm = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=x.size(0),
            add_self_loops=self.add_self_loops,
            improved=self.improved,
            dtype=x.dtype,
        )

        # 2) 由边特征生成“完整门控” edge_scale_full，与 edge_index 完全对齐
        edge_scale_full = None
        if self.use_edge_attr:
            if edge_attr is None:
                raise ValueError("edge_attr_dim specified but edge_attr=None was passed.")
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

            # 原始 E 条边的门控
            edge_scale = self.edge_mlp(edge_attr).view(-1).to(dtype=x.dtype, device=x.device)  # (E,)

            if self.add_self_loops:
                Eprime = edge_index.size(1)  # E' = E + N
                loop_mask = (edge_index[0] == edge_index[1])  # (E',) True 表示自环
                nonloop_mask = ~loop_mask
                # nonloop 数应与原始 E 相同
                assert int(nonloop_mask.sum()) == edge_scale.numel(), \
                    f"Non-loop edges {int(nonloop_mask.sum())} != edge_scale {edge_scale.numel()}"

                edge_scale_full = torch.ones(Eprime, dtype=x.dtype, device=x.device)
                edge_scale_full[nonloop_mask] = edge_scale  # 非自环用 MLP 输出
                # 自环保持 1.0（不抑制也不放大）
            else:
                edge_scale_full = edge_scale  # 不加自环时，E'==E

        # 3) Chebyshev 递推
        T0 = x
        out = self._W(0)(T0)

        T1 = self.propagate(edge_index, x=x, norm=norm, edge_scale=edge_scale_full)  # (N, Fin)
        out = out + self._W(1)(T1)

        Tk_1, Tk = T0, T1
        for k in range(2, self.K + 1):
            ATk = self.propagate(edge_index, x=Tk, norm=norm, edge_scale=edge_scale_full)
            Tk1 = 2.0 * ATk - Tk_1
            out = out + self._W(k)(Tk1)
            Tk_1, Tk = Tk, Tk1

        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor, edge_scale: torch.Tensor = None):
        """
        x_j:        (E, Fin)    —— 邻接消息（每条边一行）
        norm:       (E,)        —— 与 edge_index 对齐的边级归一化系数
        edge_scale: (E,) 或 None —— 可选的边特征缩放系数
        """
        m = norm.view(-1, 1) * x_j
        if edge_scale is not None:
            m = edge_scale.view(-1, 1) * m
        return m

    # ----------------------- helpers -----------------------

    def _W(self, k: int) -> nn.Module:
        """返回第 k 阶的线性层。"""
        if k < 0 or k > self.K:
            raise IndexError(f"W_{k} is out of range [0, {self.K}]")
        return self.lins[k]

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

class kanGCNNet(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.conv1 = kanChebConv(graph.num_features, 512, K=1)
        self.conv2 = kanChebConv(512, 256, K=2)
        self.conv3 = kanChebConv(256, 128, K=4)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        self.fc = KANLinear(128, 6)

    def forward(self,graph):
        x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_attr  # the Forward path of model
        x = F.relu(self.ln1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.ln2(self.conv2(x, edge_index, edge_weight)))
        x = self.ln3(self.conv3(x, edge_index, edge_weight))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GCNNet768(torch.nn.Module):
    def __init__(self,graph):
        super(GCNNet768, self).__init__()
        self.conv1 = ChebConv(graph.num_features, 512, K=1)
        self.conv2 = ChebConv(512, 256, K=2)
        self.conv3 = ChebConv(256, 128, K=3)
        self.fc = torch.nn.Linear(128, 6)
        # self.conv1 = ChebConv(graph.num_features, 384, K=1)
        self.bn1 = BatchNorm(512)
        # self.conv2 = ChebConv(384, 192, K=2)
        self.bn2 = BatchNorm(256)
        # self.conv3 = ChebConv(192, 96, K=3)
        self.bn3 = BatchNorm(128)
        # self.fc = torch.nn.Linear(96, 6)

    def forward(self,graph):
        x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_attr  # the Forward path of model
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight)))
        x = self.bn3(self.conv3(x, edge_index, edge_weight))
        x = self.fc(x)
        # x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.relu(self.conv2(x, edge_index, edge_weight))
        # x = self.conv3(x, edge_index, edge_weight)
        # x = self.fc(x)
        # x = torch.dropout(input=x,p=0.3,train=False)
        return F.log_softmax(x, dim=1)