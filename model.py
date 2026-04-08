# import torch
# import torch.nn as nn
# from .encoders import GraphKANEncoder, FingerprintEncoder
# from .encoders import LSSEncoder, MambaKANEncoder  # ← 新增导入
#
# class MambaKANModel(nn.Module):
#     def __init__(self, atom_in_dim= int, edge_attr_dim= int,
#                  fp_dim= int, desc_dim=int,
#                  graph_hidden=128, fp_out=128, seq_hidden=128,
#                  num_tasks=1, task_type="binary", dropout=0.4,
#                  seq_encoder_type="lss",   # ← 新增：'lss' or 'mamba'
#                  lss_depth=3, lss_kernel=256):
#         super().__init__()
#         self.task_type = task_type
#         self.num_tasks = num_tasks
#
#         self.graph_enc = GraphKANEncoder(atom_in_dim, edge_attr_dim, hidden=graph_hidden, layers=(1, 2, 2,3), dropout=dropout)
#
#         if seq_encoder_type.lower() == "lss":
#             self.seq_enc = LSSEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth, kernel_len=lss_kernel, dropout=0.1)
#         else:
#             # 仍支持 mamba（若你未来在 WSL 装好 mamba-ssm）
#             from .encoders import MambaEncoder
#             self.seq_enc = MambaEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=2, dropout=0.1)
#
#         self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out, dropout=dropout)
#
#         fusion_dim = graph_hidden + seq_hidden + fp_out
#         self.fuse = nn.Sequential(
#             nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
#         )
#         self.head = nn.Linear(128, num_tasks)
#
#     def forward(self, data):
#         g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
#         s_repr = self.seq_enc(data.x, data.batch)
#         fp_repr = self.fp_enc(data.fp, data.desc)
#         h = torch.cat([g_repr, s_repr, fp_repr], dim=-1)
#         h = self.fuse(h)
#         logits = self.head(h)
#         return logits
#
#
# class AblationMambaKANModel(nn.Module):
#     def __init__(self, atom_in_dim, edge_attr_dim, fp_dim, desc_dim,
#                  graph_hidden=128, fp_out=128, seq_hidden=128,
#                  num_tasks=1, task_type="binary", dropout=0.4,
#                  seq_encoder_type="lss", lss_depth=4, lss_kernel=128,
#                  remove_graph_encoder=False, remove_seq_encoder=False, remove_fp_encoder=False):
#         super().__init__()
#
#         self.task_type = task_type
#         self.num_tasks = num_tasks
#
#         # 有条件地移除 GraphKANEncoder
#         if not remove_graph_encoder:
#             self.graph_enc = GraphKANEncoder(atom_in_dim, edge_attr_dim, hidden=graph_hidden, layers=(1, 2, 2, 3),
#                                              dropout=dropout)
#
#         # 有条件地移除 LSSEncoder 或 MambaEncoder
#         if not remove_seq_encoder:
#             if seq_encoder_type.lower() == "lss":
#                 self.seq_enc = LSSEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth, kernel_len=lss_kernel,
#                                           dropout=0.1)
#             else:
#                 from .encoders import MambaEncoder
#                 self.seq_enc = MambaEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=2, dropout=0.1)
#
#         # 有条件地移除 FingerprintEncoder
#         if not remove_fp_encoder:
#             self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out,
#                                              dropout=dropout)
#
#         # 特征融合
#         fusion_dim = 0
#         if not remove_graph_encoder:
#             fusion_dim += graph_hidden
#         if not remove_seq_encoder:
#             fusion_dim += seq_hidden
#         if not remove_fp_encoder:
#             fusion_dim += fp_out
#
#         self.fuse = nn.Sequential(
#             nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
#         )
#         self.head = nn.Linear(128, num_tasks)
#
#     def forward(self, data):
#         features = []
#
#         if hasattr(self, 'graph_enc'):
#             g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
#             features.append(g_repr)
#         else:
#             g_repr = None
#
#         if hasattr(self, 'seq_enc'):
#             s_repr = self.seq_enc(data.x, data.batch)
#             features.append(s_repr)
#         else:
#             s_repr = None
#
#         if hasattr(self, 'fp_enc'):
#             fp_repr = self.fp_enc(data.fp, data.desc)
#             features.append(fp_repr)
#         else:
#             fp_repr = None
#
#         # 融合所有有效的特征
#         h = torch.cat(features, dim=-1)
#         h = self.fuse(h)
#         logits = self.head(h)
#         return logits
#
#     @torch.no_grad()
#     def encode_modalities(self, data):
#         """
#         返回各模态的分子级表征：
#         g_repr（图）、s_repr（序列）、fp_repr（指纹/理化）、h_fuse（融合后）
#         用于 t-SNE 可视化。
#         """
#         self.eval()
#         features = []
#
#         if hasattr(self, 'graph_enc'):
#             g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
#             features.append(g_repr)
#         else:
#             g_repr = None
#
#         if hasattr(self, 'seq_enc'):
#             s_repr = self.seq_enc(data.x, data.batch)
#             features.append(s_repr)
#         else:
#             s_repr = None
#
#         if hasattr(self, 'fp_enc'):
#             fp_repr = self.fp_enc(data.fp, data.desc)
#             features.append(fp_repr)
#         else:
#             fp_repr = None
#
#         h = torch.cat(features, dim=-1)
#         h = self.fuse(h)
#         return g_repr, s_repr, fp_repr, h

# #加入自我知识蒸馏
# import torch
# import torch.nn as nn
# from .encoders import GraphKANEncoder, FingerprintEncoder,GraphKANEncoderWithKD,LSSEncoderWithKD
# from .encoders import LSSEncoder
#
#
# class MambaKANModel(nn.Module):
#     def __init__(self, atom_in_dim=int, edge_attr_dim=int,
#                  fp_dim=int, desc_dim=int,
#                  graph_hidden=128, fp_out=128, seq_hidden=128,
#                  num_tasks=1, task_type="binary", dropout=0.3,
#                  seq_encoder_type="lss",
#                  lss_depth=3, lss_kernel=256,
#                  kd_weight=0.1):
#         super().__init__()
#         self.task_type = task_type
#         self.num_tasks = num_tasks
#         self.kd_weight = kd_weight
#
#         # 使用带内部蒸馏的编码器
#         self.graph_enc = GraphKANEncoderWithKD(atom_in_dim, edge_attr_dim, hidden=graph_hidden,
#                                                layers=(1, 2, 2, 3), dropout=dropout)
#
#         if seq_encoder_type.lower() == "lss":
#             self.seq_enc = LSSEncoderWithKD(in_dim=atom_in_dim, hidden=seq_hidden,
#                                             depth=lss_depth, kernel_len=lss_kernel, dropout=0.1)
#         else:
#             from .encoders import MambaEncoderWithKD  # 需要实现
#             self.seq_enc = MambaEncoderWithKD(in_dim=atom_in_dim, hidden=seq_hidden, depth=2, dropout=0.1)
#
#         self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out, dropout=dropout)
#
#         fusion_dim = graph_hidden + seq_hidden + fp_out
#         self.fuse = nn.Sequential(
#             nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
#         )
#         self.head = nn.Linear(128, num_tasks)
#
#     def forward(self, data):
#         # 获取各模态特征及内部蒸馏损失
#         g_repr, graph_kd_loss = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
#         s_repr, seq_kd_loss = self.seq_enc(data.x, data.batch)
#         fp_repr = self.fp_enc(data.fp, data.desc)
#
#         # 多模态特征间的知识蒸馏
#         modality_features = [g_repr, s_repr, fp_repr]
#         modality_kd_loss = self.compute_modality_kd_loss(modality_features)
#
#         # 特征融合
#         h = torch.cat([g_repr, s_repr, fp_repr], dim=-1)
#         h = self.fuse(h)
#
#         # 融合层蒸馏损失设为0
#         fusion_kd_loss = torch.tensor(0.0, device=h.device)
#
#         # 最终输出
#         logits = self.head(h)
#
#         # 总蒸馏损失 = 模态内部蒸馏 + 模态间蒸馏
#         total_kd_loss = (graph_kd_loss + seq_kd_loss +
#                          modality_kd_loss + fusion_kd_loss)
#
#         return logits, total_kd_loss
#
#     def compute_kd_loss(self, features):
#         """通用的知识蒸馏损失计算"""
#         if len(features) <= 1:
#             return torch.tensor(0.0, device=features[0].device)
#
#         avg_feature = torch.stack(features).mean(dim=0)
#         kd_loss = 0.0
#         for feat in features:
#             kd_loss += torch.mean((feat - avg_feature) ** 2)
#
#         return kd_loss / len(features)
#
#     def compute_modality_kd_loss(self, modality_features):
#         """多模态特征间的知识蒸馏"""
#         kd_loss = 0.0
#         count = 0
#
#         for i in range(len(modality_features)):
#             for j in range(i + 1, len(modality_features)):
#                 kd_loss += torch.mean((modality_features[i] - modality_features[j]) ** 2)
#                 count += 1
#
#         return kd_loss / count if count > 0 else torch.tensor(0.0)
#
#     def forward_without_kd(self, data):
#         """不计算蒸馏损失的前向传播（用于测试）"""
#         # 只取特征部分，忽略蒸馏损失
#         g_repr, _ = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)  # ← 添加逗号
#         s_repr, _ = self.seq_enc(data.x, data.batch)  # ← 添加逗号
#         fp_repr = self.fp_enc(data.fp, data.desc)
#         h = torch.cat([g_repr, s_repr, fp_repr], dim=-1)
#         h = self.fuse(h)
#         logits = self.head(h)
#         return logits




#融合mambakan
import torch
import torch.nn as nn
from .encoders import GraphKANEncoder, FingerprintEncoder
from .encoders import LSSEncoder, MambaKANEncoder  # ← 新增导入

# class MambaKANModel(nn.Module):
#     def __init__(self, atom_in_dim= int, edge_attr_dim= int,
#                  fp_dim= int, desc_dim=int,
#                  graph_hidden=128, fp_out=128, seq_hidden=128,
#                  num_tasks=1, task_type="binary", dropout=0.4,
#                  seq_encoder_type="mamba_kan",   # ← 新增：'lss' or 'mamba'
#                  lss_depth=3, lss_kernel=256):
#         super().__init__()
#         self.task_type = task_type
#         self.num_tasks = num_tasks
#
#         self.graph_enc = GraphKANEncoder(atom_in_dim, edge_attr_dim, hidden=graph_hidden, layers=(1, 2, 2,3), dropout=dropout)
#
#         # --- 修改部分 ---
#         if seq_encoder_type.lower() == "lss":
#             self.seq_enc = LSSEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth, kernel_len=lss_kernel,
#                                       dropout=0.1)
#
#         elif seq_encoder_type.lower() == "mamba":
#             # 这里调用我们新写的 MambaKANEncoder
#             # 复用 lss_depth 作为层数参数，或者你可以新增参数
#             self.seq_enc = MambaKANEncoder(
#                 in_dim=atom_in_dim,
#                 hidden=seq_hidden,
#                 depth=lss_depth,
#                 d_state=16,  # SSM 状态维度，16/32/64 均可
#                 dropout=0.1
#             )
#
#         self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out, dropout=dropout)
#
#         fusion_dim = graph_hidden + seq_hidden + fp_out
#         self.fuse = nn.Sequential(
#             nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
#         )
#         self.head = nn.Linear(128, num_tasks)
#
#     def forward(self, data):
#         g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
#         s_repr = self.seq_enc(data.x, data.batch)
#         fp_repr = self.fp_enc(data.fp, data.desc)
#         h = torch.cat([g_repr, s_repr, fp_repr], dim=-1)
#         h = self.fuse(h)
#         logits = self.head(h)
#         return logits
#
#
# class AblationMambaKANModel(nn.Module):
#     def __init__(self, atom_in_dim, edge_attr_dim, fp_dim, desc_dim,
#                  graph_hidden=128, fp_out=128, seq_hidden=128,
#                  num_tasks=1, task_type="binary", dropout=0.4,
#                  seq_encoder_type="lss", lss_depth=4, lss_kernel=128,
#                  remove_graph_encoder=False, remove_seq_encoder=False, remove_fp_encoder=False):
#         super().__init__()
#
#         self.task_type = task_type
#         self.num_tasks = num_tasks
#
#         # 有条件地移除 GraphKANEncoder
#         if not remove_graph_encoder:
#             self.graph_enc = GraphKANEncoder(atom_in_dim, edge_attr_dim, hidden=graph_hidden, layers=(1, 2, 2, 3),
#                                              dropout=dropout)
#
#         # 有条件地移除 LSSEncoder 或 MambaEncoder
#         if not remove_seq_encoder:
#             if seq_encoder_type.lower() == "lss":
#                 self.seq_enc = LSSEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth, kernel_len=lss_kernel,
#                                           dropout=0.1)
#             else:
#                 from .encoders import MambaEncoder
#                 self.seq_enc = MambaEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=2, dropout=0.1)
#
#         # 有条件地移除 FingerprintEncoder
#         if not remove_fp_encoder:
#             self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out,
#                                              dropout=dropout)
#
#         # 特征融合
#         fusion_dim = 0
#         if not remove_graph_encoder:
#             fusion_dim += graph_hidden
#         if not remove_seq_encoder:
#             fusion_dim += seq_hidden
#         if not remove_fp_encoder:
#             fusion_dim += fp_out
#
#         self.fuse = nn.Sequential(
#             nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
#         )
#         self.head = nn.Linear(128, num_tasks)
#
#     def forward(self, data):
#         features = []
#
#         if hasattr(self, 'graph_enc'):
#             g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
#             features.append(g_repr)
#         else:
#             g_repr = None
#
#         if hasattr(self, 'seq_enc'):
#             s_repr = self.seq_enc(data.x, data.batch)
#             features.append(s_repr)
#         else:
#             s_repr = None
#
#         if hasattr(self, 'fp_enc'):
#             fp_repr = self.fp_enc(data.fp, data.desc)
#             features.append(fp_repr)
#         else:
#             fp_repr = None
#
#         # 融合所有有效的特征
#         h = torch.cat(features, dim=-1)
#         h = self.fuse(h)
#         logits = self.head(h)
#         return logits
#
#     @torch.no_grad()
#     def encode_modalities(self, data):
#         """
#         返回各模态的分子级表征：
#         g_repr（图）、s_repr（序列）、fp_repr（指纹/理化）、h_fuse（融合后）
#         用于 t-SNE 可视化。
#         """
#         self.eval()
#         features = []
#
#         if hasattr(self, 'graph_enc'):
#             g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
#             features.append(g_repr)
#         else:
#             g_repr = None
#
#         if hasattr(self, 'seq_enc'):
#             s_repr = self.seq_enc(data.x, data.batch)
#             features.append(s_repr)
#         else:
#             s_repr = None
#
#         if hasattr(self, 'fp_enc'):
#             fp_repr = self.fp_enc(data.fp, data.desc)
#             features.append(fp_repr)
#         else:
#             fp_repr = None
#
#         h = torch.cat(features, dim=-1)
#         h = self.fuse(h)
#         return g_repr, s_repr, fp_repr, h


class GatedStructFusion(nn.Module):
    """用于融合图(Graph)和序列(Sequence)的核心结构特征的门控网络"""

    def __init__(self, graph_dim, seq_dim):
        super().__init__()
        fuse_dim = graph_dim + seq_dim
        # 门控网络评估每个特征维度的重要性 (输出0-1)
        self.gate = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.SiLU(),
            nn.Linear(fuse_dim, fuse_dim),
            nn.Sigmoid()
        )

    def forward(self, g_repr, s_repr):
        # 拼接结构特征
        struct_feat = torch.cat([g_repr, s_repr], dim=-1)
        # 计算门控权重并进行逐元素相乘(过滤噪声)
        gate_weights = self.gate(struct_feat)
        return struct_feat * gate_weights


class MambaKANModel(nn.Module):
    def __init__(self, atom_in_dim=int, edge_attr_dim=int,
                 fp_dim=int, desc_dim=int,
                 graph_hidden=128, fp_out=128, seq_hidden=128,
                 num_tasks=1, task_type="binary", dropout=0.4,
                 seq_encoder_type="mamba_kan",
                 lss_depth=3, lss_kernel=256):
        super().__init__()
        self.task_type = task_type
        self.num_tasks = num_tasks

        self.graph_enc = GraphKANEncoder(atom_in_dim, edge_attr_dim, hidden=graph_hidden, layers=(1, 2, 2, 3),
                                         dropout=dropout)

        if seq_encoder_type.lower() == "lss":
            self.seq_enc = LSSEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth, kernel_len=lss_kernel,
                                      dropout=0.1)
        elif seq_encoder_type.lower() == "mamba":
            self.seq_enc = MambaKANEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth, d_state=16,
                                           dropout=0.1)

        self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out, dropout=dropout)

        # --- 新增：门控融合模块 ---
        self.struct_fusion = GatedStructFusion(graph_hidden, seq_hidden)

        # 融合后的维度 = (图维度 + 序列维度) + 指纹维度
        fusion_dim = graph_hidden + seq_hidden + fp_out

        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(128, num_tasks)

    def forward(self, data):
        g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
        s_repr = self.seq_enc(data.x, data.batch)
        fp_repr = self.fp_enc(data.fp, data.desc)

        # 1. 结构特征进行门控融合
        fused_struct = self.struct_fusion(g_repr, s_repr)

        # 2. 辅助特征进行末端拼接
        h = torch.cat([fused_struct, fp_repr], dim=-1)

        h = self.fuse(h)
        logits = self.head(h)
        return logits

class AblationMambaKANModel(nn.Module):
        def __init__(self, atom_in_dim, edge_attr_dim, fp_dim, desc_dim,
                     graph_hidden=128, fp_out=128, seq_hidden=128,
                     num_tasks=1, task_type="binary", dropout=0.4,
                     seq_encoder_type="lss", lss_depth=4, lss_kernel=128,
                     remove_graph_encoder=False, remove_seq_encoder=False, remove_fp_encoder=False):
            super().__init__()

            self.task_type = task_type
            self.num_tasks = num_tasks

            self.has_graph = not remove_graph_encoder
            self.has_seq = not remove_seq_encoder
            self.has_fp = not remove_fp_encoder

            # 初始化编码器
            if self.has_graph:
                self.graph_enc = GraphKANEncoder(atom_in_dim, edge_attr_dim, hidden=graph_hidden, layers=(1, 2, 2, 3),
                                                 dropout=dropout)

            if self.has_seq:
                if seq_encoder_type.lower() == "lss":
                    self.seq_enc = LSSEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth,
                                              kernel_len=lss_kernel, dropout=0.1)
                else:
                    from .encoders import MambaEncoder  # 注意这里的导入是否对应你的项目结构
                    self.seq_enc = MambaEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=2, dropout=0.1)

            if self.has_fp:
                self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out,
                                                 dropout=dropout)

            # --- 动态构建融合层和维度 ---
            struct_dim = 0
            if self.has_graph and self.has_seq:
                # 只有图和序列同时存在时，才启用门控融合
                self.struct_fusion = GatedStructFusion(graph_hidden, seq_hidden)
                struct_dim = graph_hidden + seq_hidden
            elif self.has_graph:
                struct_dim = graph_hidden
            elif self.has_seq:
                struct_dim = seq_hidden

            fusion_dim = struct_dim
            if self.has_fp:
                fusion_dim += fp_out

            self.fuse = nn.Sequential(
                nn.Linear(max(1, fusion_dim), 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            )
            self.head = nn.Linear(128, num_tasks)

        def _get_fused_features(self, data):
            """内部辅助函数，用于提取并融合特征，供 forward 和 encode_modalities 复用"""
            g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch) if self.has_graph else None
            s_repr = self.seq_enc(data.x, data.batch) if self.has_seq else None
            fp_repr = self.fp_enc(data.fp, data.desc) if self.has_fp else None

            # 1. 处理结构特征 (Graph + Sequence)
            if self.has_graph and self.has_seq:
                fused_struct = self.struct_fusion(g_repr, s_repr)
            elif self.has_graph:
                fused_struct = g_repr
            elif self.has_seq:
                fused_struct = s_repr
            else:
                fused_struct = None

            # 2. 拼接指纹特征 (Fingerprint)
            features_to_concat = []
            if fused_struct is not None:
                features_to_concat.append(fused_struct)
            if fp_repr is not None:
                features_to_concat.append(fp_repr)

            if len(features_to_concat) > 0:
                h = torch.cat(features_to_concat, dim=-1)
            else:
                # 如果极端情况所有 encoder 都被移除了，返回一个全 0 的 dummy tensor (防崩溃)
                h = torch.zeros(1, device=data.x.device)

            return g_repr, s_repr, fp_repr, h

        def forward(self, data):
            _, _, _, h = self._get_fused_features(data)
            h = self.fuse(h)
            logits = self.head(h)
            return logits

        @torch.no_grad()
        def encode_modalities(self, data):
            """返回各模态表征以及融合后的表征，用于 t-SNE"""
            self.eval()
            g_repr, s_repr, fp_repr, h_fused = self._get_fused_features(data)
            h_final = self.fuse(h_fused)
            return g_repr, s_repr, fp_repr, h_final

