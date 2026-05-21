import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficient_kan import *


class MambaKANLayer(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        """
        d_model: 输入特征维度
        d_state: SSM 状态维度 (Latent State)
        d_conv: 局部卷积核大小
        expand: 扩展因子 (通常是 2)
        """
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = self.d_inner // 16 if self.d_inner // 16 > 1 else 1

        # 1. 输入投影 (Strategy A: 用 KAN 增强特征提取)
        self.in_proj = KANLinear(d_model, self.d_inner * 2)

        # 2. 局部卷积 (模拟 Token 间的小范围交互)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 3. 选择性机制 (Strategy B: 用 KAN 生成更智能的参数)
        # 输入是 x (d_inner), 输出是 dt, B, C
        # 这里的 KAN 决定了模型"如何遗忘"和"如何记忆"
        self.x_proj = KANLinear(self.d_inner, self.dt_rank + d_state * 2, grid_size=3)

        # dt_proj 依然保持线性，因为它只是一个缩放
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A 是时间尺度参数，通常是对数参数化
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 4. 输出投影 (Strategy A)
        self.out_proj = KANLinear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        """
        batch, seq_len, dim = x.shape
        residual = x
        x = self.norm(x)

        # 1. 投影与切分
        x_and_res = self.in_proj(x)  # (B, L, 2*D)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # 2. 卷积处理 (需要转置适应 Conv1d)
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.conv1d(x)[:, :, :seq_len]  # 裁剪 padding
        x = F.silu(x)
        x = x.permute(0, 2, 1)  # (B, L, D)

        # 3. SSM 核心过程
        y = self.ssm(x)

        # 4. 门控与输出
        y = y * F.silu(res)  # 门控机制
        out = self.out_proj(y)

        return out + residual

    def ssm(self, x):
        """
        纯 PyTorch 实现的 Selective Scan (串行版，仅用于演示/非GPU环境)
        真实 Mamba 会在这里调用 CUDA kernel (selective_scan_cuda)
        """
        (d_in, n) = self.A_log.shape

        # A 矩阵离散化参数
        A = -torch.exp(self.A_log.float())  # (D, N)
        D = self.D.float()

        # Step 1: 动态生成参数 delta, B, C
        # 这里使用了 KAN (self.x_proj) 来生成这些参数！
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(delta))  # (B, L, D)

        # 形状调整以进行广播计算
        # 如果追求速度，这里必须写成 CUDA kernel 或使用 mamba_ssm 库
        # 下面是一个简化的循环实现 (RNN mode)，虽然慢但逻辑正确

        y_list = []
        h = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device)  # Hidden State

        # 扫描循环 (Scan Loop)
        for t in range(x.shape[1]):
            # 当前时刻的参数
            dt = delta[:, t, :].unsqueeze(-1)  # (B, D, 1)
            dA = torch.exp(dt * A)  # (B, D, N) 离散化后的 A
            dB = dt * B[:, t, :].unsqueeze(1)  # (B, D, N) 离散化后的 B

            # 状态更新: h_t = A_bar * h_{t-1} + B_bar * x_t
            xt_val = x[:, t, :].unsqueeze(-1)  # (B, D, 1)
            h = dA * h + dB * xt_val

            # 输出计算: y_t = C * h_t
            Ct = C[:, t, :].unsqueeze(1)  # (B, 1, N)
            y_t = torch.sum(h * Ct, dim=-1)  # (B, D)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # (B, L, D)

        return y + x * self.D


# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 输入: Batch=2, Length=32, Dim=64
    x = torch.randn(2, 32, 64).to(device)

    # 实例化 Mamba-KAN
    model = MambaKANLayer(d_model=64).to(device)

    output = model(x)
    print("Output shape:", output.shape)  # 应该是 [2, 32, 64]

    # 计算参数量
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")