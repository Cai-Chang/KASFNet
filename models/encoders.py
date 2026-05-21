import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .GNNs import kanChebConv, KANLinear  # type: ignore


class GraphKANEncoder(nn.Module):
    def __init__(self, in_dim, edge_attr_dim, hidden=128, layers=(1, 2, 2, 3), dropout=0.3):
        super().__init__()
        if kanChebConv is None:
            raise ImportError("kanChebConv not found. Please ensure GNNs.py is available in PYTHONPATH.")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 4 层需要 5 个维度节点：[in, h1, h2, h3, h4]
        dims = [in_dim, hidden, hidden, hidden, hidden]
        Ks = list(layers)  # 长度必须是 4

        for i in range(4):  # ← 从 3 改为 4
            # self.convs.append(kanChebConv(dims[i], dims[i + 1], K=Ks[i]))
            self.convs.append(
                kanChebConv(dims[i], dims[i + 1], K=Ks[i], edge_attr_dim=edge_attr_dim, add_self_loops=True)
            )

            self.norms.append(nn.LayerNorm(dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, ln in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(ln(x))
            x = self.dropout(x)
        from torch_geometric.nn import global_mean_pool
        g = global_mean_pool(x, batch)
        return g  # (B, hidden)


class FingerprintEncoder(nn.Module):
    def __init__(self, fp_dim: int, desc_dim: int,
                 hidden: int = 256, out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        D = int(fp_dim) + int(desc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        # 防呆：一旦又被默认值坑，这里会立刻报错
        assert self.mlp[0].in_features == D, \
            f"in_features={self.mlp[0].in_features} != D={D}"

    def forward(self, fp, desc):
        x = torch.cat([fp, desc], dim=-1)
        return self.mlp(x)


class MambaEncoder(nn.Module):
    """
    Encode per-graph atom sequences with Mamba (fallback to GRU).
    """
    def __init__(self, in_dim, hidden=128, depth=2, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            self.mamba = nn.ModuleList([Mamba(d_model=in_dim if i == 0 else hidden) for i in range(depth)])
            self.in_proj = nn.Identity() if in_dim == hidden else nn.Linear(in_dim, hidden)
            self.use_mamba = True
        except Exception:
            self.use_mamba = False
            self.rnn = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        """
        x: (N_nodes, in_dim)
        batch: (N_nodes,) graph ids
        returns: (B, hidden)
        """
        # pack each graph's node sequence (keep input order)
        B = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        outs = []
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=False).squeeze(-1)
            xb = x[idx]  # (n_b, in_dim)
            if xb.ndim == 1:
                xb = xb.unsqueeze(0)
            xb = xb.unsqueeze(0)  # (1, n_b, d)
            if self.use_mamba:
                h = self.in_proj(xb)
                for layer in self.mamba:
                    h, _ = layer(h) if hasattr(layer, 'forward') else (layer(h), None)
                hb = h.mean(dim=1)  # (1, hidden)
            else:
                hb, _ = self.rnn(xb)
                hb = hb.mean(dim=1)
            outs.append(hb.squeeze(0))
        if len(outs) == 0:
            return torch.zeros((0, self.hidden), device=x.device)
        out = torch.stack(outs, dim=0)
        return self.dropout(out)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSSEncoder(nn.Module):
    """
    轻量 Linear State-Space (LSS) 编码器（纯 PyTorch，无外部依赖）
    - 将每个分子的原子特征序列视作 (L, in_dim)
    - 先投影到 hidden，再做 depth 层 LSS block：
        * 可学习时间尺度 tau -> 指数核 k_t = exp(-t / tau)
        * 通道内 depthwise 1D 卷积实现“选择性扫描”近似
        * 通道间 pointwise mixing + 门控 + 残差
    - 对序列做平均池化得到 (B, hidden)
    """
    def __init__(self, in_dim, hidden=128, depth=3, kernel_len=256, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.kernel_len = kernel_len

        self.in_proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.blocks = nn.ModuleList([_LSSBlock(hidden, kernel_len=kernel_len, dropout=dropout) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        """
        x: (N_nodes, in_dim)
        batch: (N_nodes,) 取值[0..B-1]，表示每个节点属于哪个图
        return: (B, hidden)
        """
        B = int(batch.max().item()+1) if batch.numel()>0 else 0
        outs = []
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=False).squeeze(-1)
            xb = x[idx]  # (L, in_dim)
            if xb.ndim == 1:
                xb = xb.unsqueeze(0)
            xb = self.in_proj(xb)         # (L, hidden)
            xb = xb.transpose(0,1).unsqueeze(0)  # (1, hidden, L)

            for blk in self.blocks:
                xb = blk(xb)              # (1, hidden, L)

            # 序列池化
            hb = xb.mean(dim=-1).squeeze(0)  # (hidden,)
            outs.append(hb)

        if len(outs) == 0:
            return torch.zeros((0, self.hidden), device=x.device)
        out = torch.stack(outs, dim=0)       # (B, hidden)
        return self.dropout(out)

class _LSSBlock(nn.Module):
    """
    单层 LSS block（无任何 in-place 参数改写；使用函数式 conv1d）:
      - 通道内 depthwise 1D 卷积，核由可学习 tau 生成
      - 通道间 pointwise mixing + GLU
      - 残差 + LayerNorm
    """
    def __init__(self, hidden, kernel_len=256, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.kernel_len = kernel_len

        # 学习时间常数（正数），初始化为较大的衰减尺度
        self.log_tau = nn.Parameter(torch.log(torch.ones(hidden)*64.0))

        # 通道间 mixing + 门控
        self.pw_in = nn.Linear(hidden, 2*hidden)  # for GLU
        self.pw_out = nn.Linear(hidden, hidden)

        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def _build_kernel(self, L, device, dtype):
        """
        根据可学习的时间常数构造长度 = min(kernel_len, L) 的指数衰减核
        返回形状: (C, 1, K)
        """
        klen = int(min(self.kernel_len, L))
        t = torch.arange(klen, device=device, dtype=dtype)  # (K,)
        tau = torch.exp(self.log_tau).clamp_min(1e-3).unsqueeze(-1)  # (C,1)
        k = torch.exp(-t.unsqueeze(0)/tau)  # (C, K)
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-8)
        return k.unsqueeze(1)  # (C, 1, K)

    def forward(self, x):
        """
        x: (B=1, C=hidden, L)
        """
        B, C, L = x.shape
        device, dtype = x.device, x.dtype

        # 1) depthwise conv kernel by tau (no in-place on parameters)
        weight = self._build_kernel(L, device, dtype)  # (C,1,K)
        klen = weight.size(-1)

        # 因果卷积：左侧 pad klen-1
        y = F.pad(x, (klen-1, 0))
        # 函数式 depthwise 卷积（groups=C）
        y = F.conv1d(y, weight, bias=None, stride=1, padding=0, dilation=1, groups=C)  # (1,C,L)

        # 2) pointwise + GLU
        y_t = y.transpose(1, 2)                # (1, L, C)
        gates = self.pw_in(y_t)                 # (1, L, 2C)
        a, b = gates.chunk(2, dim=-1)
        y2 = a * torch.sigmoid(b)               # GLU
        y2 = self.pw_out(y2)                    # (1, L, C)
        y2 = self.dropout(y2)

        # 3) 残差 + LN（非原地）
        out = self.norm((y2 + y_t).squeeze(0))  # (L, C)
        out = out.unsqueeze(0).transpose(1, 2)  # 回到 (1, C, L)
        return out


import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from .mamba_kan import MambaKANLayer  # 假设 mamba_kan.py 在同一目录下


class MambaKANEncoder(nn.Module):
    """
    使用 MambaKANLayer 替换原有的 LSSEncoder 处理原子序列
    """

    def __init__(self, in_dim, hidden=128, depth=2, d_state=16, dropout=0.1):
        super().__init__()
        self.hidden = hidden

        # 1. 维度对齐：将原始原子特征维度 (atom_in_dim) 映射到 hidden
        self.in_proj = nn.Linear(in_dim, hidden)

        # 2. 堆叠 MambaKAN 层
        # MambaKANLayer 输入/输出都是 hidden 维度，内部会自动进行 KAN 投影和 SSM 处理
        self.layers = nn.ModuleList([
            MambaKANLayer(d_model=hidden, d_state=d_state, expand=2)
            for _ in range(depth)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        """
        x: (Total_Nodes, in_dim) - 扁平化的所有原子特征
        batch: (Total_Nodes,) - 每个原子属于哪个分子的索引
        """
        # 1. [核心步骤] 将图数据转换为 Dense Batch 序列数据
        # x_dense: [Batch_Size, Max_Len, in_dim]
        # mask: [Batch_Size, Max_Len] (True表示真实原子，False表示填充)
        x_dense, mask = to_dense_batch(x, batch)

        # 2. 线性投影
        x_dense = self.in_proj(x_dense)  # (B, L, hidden)

        # 3. 经过多层 MambaKAN 处理
        for layer in self.layers:
            # MambaKANLayer 内部包含了残差连接，直接串联即可
            x_dense = layer(x_dense)

        # 4. [核心步骤] 序列池化 (Pooling)
        # 我们需要把 (B, L, H) 变回 (B, H)，通常使用 Mean Pooling，但要忽略 padding

        mask = mask.unsqueeze(-1).float()  # (B, L, 1)

        # 只对真实原子求和
        sum_pooled = (x_dense * mask).sum(dim=1)  # (B, H)

        # 计算每个分子的真实原子数（避免除以0）
        num_nodes = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)

        # 得到平均特征
        out = sum_pooled / num_nodes

        return self.dropout(out)


# class GraphKANEncoderWithKD(nn.Module):
#     def __init__(self, in_dim, edge_attr_dim, hidden=128, layers=(1, 2, 2, 3), dropout=0.3):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#
#         dims = [in_dim, hidden, hidden, hidden, hidden]
#         Ks = list(layers)
#
#         for i in range(4):
#             self.convs.append(
#                 kanChebConv(dims[i], dims[i + 1], K=Ks[i], edge_attr_dim=edge_attr_dim, add_self_loops=True)
#             )
#             self.norms.append(nn.LayerNorm(dims[i + 1]))
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, edge_index, edge_attr, batch):
#         # 收集各层特征用于知识蒸馏
#         layer_features = []
#
#         for conv, ln in zip(self.convs, self.norms):
#             x = conv(x, edge_index, edge_attr)
#             layer_features.append(x.clone())  # 保存当前层特征
#             x = F.relu(ln(x))
#             x = self.dropout(x)
#
#         # 计算图结构内部的层间蒸馏损失
#         graph_kd_loss = self.compute_layer_kd_loss(layer_features)
#
#         g = global_mean_pool(x, batch)
#         return g, graph_kd_loss  # 返回特征和蒸馏损失
#
#     def compute_layer_kd_loss(self, layer_features):
#         """LKM风格的层间蒸馏损失"""
#         if len(layer_features) <= 1:
#             return torch.tensor(0.0, device=layer_features[0].device)
#
#         # 计算平均特征
#         avg_feature = torch.stack(layer_features).mean(dim=0)
#
#         # LKM风格的损失计算
#         kd_loss = 0.0
#         for feat in layer_features:
#             kd_loss += torch.mean((feat - avg_feature) ** 2)
#
#         return kd_loss / len(layer_features)
#
#
# class LSSEncoderWithKD(nn.Module):
#     def __init__(self, in_dim, hidden=128, depth=3, kernel_len=256, dropout=0.1):
#         super().__init__()
#         self.hidden = hidden
#         self.depth = depth
#
#         self.in_proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
#         self.blocks = nn.ModuleList([_LSSBlock(hidden, kernel_len=kernel_len, dropout=dropout) for _ in range(depth)])
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, batch):
#         B = int(batch.max().item() + 1) if batch.numel() > 0 else 0
#         outs = []
#         seq_kd_losses = []
#
#         for b in range(B):
#             idx = (batch == b).nonzero(as_tuple=False).squeeze(-1)
#             xb = x[idx]
#             if xb.ndim == 1:
#                 xb = xb.unsqueeze(0)
#             xb = self.in_proj(xb)
#             xb = xb.transpose(0, 1).unsqueeze(0)  # (1, hidden, L)
#
#             # 收集各block的特征
#             block_features = []
#             current_xb = xb
#
#             for blk in self.blocks:
#                 current_xb = blk(current_xb)
#                 block_features.append(current_xb.clone())
#
#             # 计算序列内部的层间蒸馏损失
#             if len(block_features) > 1:
#                 seq_kd_loss = self.compute_layer_kd_loss(block_features)
#             else:
#                 seq_kd_loss = torch.tensor(0.0, device=xb.device)
#
#             seq_kd_losses.append(seq_kd_loss)
#             hb = current_xb.mean(dim=-1).squeeze(0)
#             outs.append(hb)
#
#         if len(outs) == 0:
#             return torch.zeros((0, self.hidden), device=x.device), torch.tensor(0.0, device=x.device)
#
#         out = torch.stack(outs, dim=0)
#         avg_seq_kd_loss = torch.stack(seq_kd_losses).mean() if seq_kd_losses else torch.tensor(0.0, device=x.device)
#
#         return self.dropout(out), avg_seq_kd_loss
#
#     def compute_layer_kd_loss(self, block_features):
#         """LKM风格的层间蒸馏损失"""
#         avg_feature = torch.stack(block_features).mean(dim=0)
#         kd_loss = 0.0
#         for feat in block_features:
#             kd_loss += torch.mean((feat - avg_feature) ** 2)
#         return kd_loss / len(block_features)
