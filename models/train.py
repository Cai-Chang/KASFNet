import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from utils.metrics import compute_classification_auc, compute_regression_rmse, masked_bce_loss
import pandas as pd

def _bin_label_and_mask(batch):
    """将 batch.y / batch.mask 规整到合法取值范围。"""
    y = batch.y
    m = getattr(batch, 'mask', torch.ones_like(y))
    # 转 float
    y = y.float()
    m = m.float()
    # 将可能出现的 {-1,1} / 其它实数 裁剪到 [0,1]
    # 如果你的数据集中确实是 {-1,1}，可改成 (y > 0).float()
    y = y.clamp(0.0, 1.0)
    m = (m > 0.5).float()  # 二值化 mask
    return y, m

def train_one_epoch(model, loader, optimizer, device, task_type):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)  # (B, T)

        if task_type in ["binary", "multilabel"]:
            y, m = _bin_label_and_mask(batch)
            loss = masked_bce_loss(logits, y, m)
        else:
            # 回归：MSE
            pred = logits.squeeze(-1)
            y = batch.y.squeeze(-1).float()
            loss = nn.functional.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        # （可选）梯度裁剪，避免偶发不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def evaluate(model, loader, device, task_type: str):
    model.eval()
    y_list, p_list = [], []
    losses = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)  # (B, T)

        if task_type in ["binary", "multilabel"]:
            # 统一使用 sigmoid -> 正类概率
            prob = torch.sigmoid(logits)  # (B, T)

            # 规整标签与 mask
            y, m = _bin_label_and_mask(batch)  # (B, T)

            # T==1 时压成一维，便于后面拼接/计算 AUC
            if prob.size(-1) == 1:
                prob = prob.squeeze(-1)  # (B,)
                y    = y.squeeze(-1)     # (B,)
                m    = m.squeeze(-1)     # (B,)

            # 只保留有效样本
            valid = (m > 0)
            y_list.append(y[valid].detach().cpu().numpy())
            p_list.append(prob[valid].detach().cpu().numpy())

            # 也计算一个 loss（带 mask 的 BCE）
            bce = masked_bce_loss(logits, y if y.ndim == 2 else y.unsqueeze(-1),
                                  m if m.ndim == 2 else m.unsqueeze(-1))
            losses.append(float(bce.detach().cpu().item()))

        else:
            # 回归：收集 MSE / RMSE
            pred = logits.squeeze(-1)
            y = batch.y.squeeze(-1).float()
            mse = nn.functional.mse_loss(pred, y)
            losses.append(float(mse.detach().cpu().item()))
            y_list.append(y.detach().cpu().numpy())
            p_list.append(pred.detach().cpu().numpy())

    if task_type in ["binary", "multilabel"]:
        if len(y_list) == 0:
            return 0.0, float('nan')

        y_true = np.concatenate(y_list, axis=0)
        y_prob = np.concatenate(p_list, axis=0)

        # 再保险：若仍是二维概率，按“取正类列”转一维
        if y_prob.ndim == 2:
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
            elif y_prob.shape[1] == 1:
                y_prob = y_prob.reshape(-1)
            else:
                y_prob = y_prob.reshape(y_prob.shape[0], -1)[:, -1]

        # AUC 计算（内部也会做兜底）
        metric = compute_classification_auc(y_true, y_prob, average="macro")
        return float(np.mean(losses)) if losses else 0.0, float(metric) if metric == metric else float('nan')

    else:
        if len(y_list) == 0:
            return 0.0, float('nan')
        y_true = np.concatenate(y_list, axis=0)
        y_pred = np.concatenate(p_list, axis=0)
        metric = compute_regression_rmse(y_true, y_pred)
        return float(np.mean(losses)) if losses else 0.0, float(metric)


@torch.no_grad()
def predict_dataframe(model, loader, device, task_type: str, num_tasks: int, split_name: str, epoch: int):
    """
    用于逐分子导出预测结果到 DataFrame。
    - 对分类任务：输出正类概率 prob_*（mask==0 的位置置 NaN）
    - 对回归任务：输出 pred（mask==0 的位置置 NaN，如果有 mask）
    统一包含: ['epoch','split','smiles', label列..., 预测列..., mask列...]
    """
    model.eval()
    rows = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)  # (B, T)

        # 取 SMILES（PyG 的 Batch 会把自定义属性聚合成 list）
        smiles_list = getattr(batch, "smiles", None)
        if smiles_list is None:
            # 若你的 Data 没存 smiles，可用占位
            smiles_list = [""] * logits.size(0)

        if task_type == "regression":
            # 回归
            pred = logits.squeeze(-1).detach().cpu().numpy()  # (B,)
            y = batch.y.squeeze(-1).detach().cpu().numpy()    # (B,)
            m = getattr(batch, "mask", torch.ones_like(batch.y)).squeeze(-1).detach().cpu().numpy()  # (B,)

            # mask==0 的位置置 NaN 便于后续分析
            pred = np.where(m > 0.5, pred, np.nan)
            y    = np.where(m > 0.5, y,   np.nan)

            for s, yt, pr, mk in zip(smiles_list, y, pred, m):
                rows.append({
                    "epoch": epoch,
                    "split": split_name,
                    "smiles": s,
                    "label": yt,
                    "pred": pr,
                    "mask": float(mk),
                })

        else:
            # 分类（binary 或 multilabel）—— 使用 sigmoid 得到正类概率
            prob = torch.sigmoid(logits).detach().cpu().numpy()  # (B, T)
            y    = batch.y.detach().cpu().numpy()                # (B, T)
            m    = getattr(batch, "mask", torch.ones_like(batch.y)).detach().cpu().numpy()  # (B, T)

            # 保留 NaN：把 mask==0 的标签置 NaN
            y = np.where(m > 0.5, y, np.nan)
            # 概率不置 NaN（保留模型输出），但可同时输出 mask 便于过滤
            T = prob.shape[1]
            for i in range(prob.shape[0]):
                row = {
                    "epoch": epoch,
                    "split": split_name,
                    "smiles": smiles_list[i],
                }
                # 标签/概率/掩码列
                if num_tasks == 1:
                    row["label"] = y[i, 0] if y.ndim == 2 else y[i]
                    row["prob"]  = prob[i, 0] if prob.ndim == 2 else prob[i]
                    row["mask"]  = float(m[i, 0] if m.ndim == 2 else m[i])
                else:
                    for t in range(T):
                        row[f"label_t{t}"] = y[i, t]
                        row[f"prob_t{t}"]  = prob[i, t]
                        row[f"mask_t{t}"]  = float(m[i, t])
                rows.append(row)

    df = pd.DataFrame(rows)
    return df

#加入自我知识蒸馏

# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# import numpy as np
# from utils.metrics import compute_classification_auc, compute_regression_rmse, masked_bce_loss
# import pandas as pd
#
#
# def _bin_label_and_mask(batch):
#     """将 batch.y / batch.mask 规整到合法取值范围。"""
#     y = batch.y
#     m = getattr(batch, 'mask', torch.ones_like(y))
#     # 转 float
#     y = y.float()
#     m = m.float()
#     # 将可能出现的 {-1,1} / 其它实数 裁剪到 [0,1]
#     # 如果你的数据集中确实是 {-1,1}，可改成 (y > 0).float()
#     y = y.clamp(0.0, 1.0)
#     m = (m > 0.5).float()  # 二值化 mask
#     return y, m
#
#
# def train_one_epoch(model, loader, optimizer, device, task_type):
#     model.train()
#     losses = []
#     kd_losses = []  # ← 新增：记录蒸馏损失
#
#     for batch in loader:
#         batch = batch.to(device)
#         # ← 修改：模型现在返回 (logits, kd_loss)
#         logits, kd_loss = model(batch)  # (B, T), scalar
#
#         if task_type in ["binary", "multilabel"]:
#             y, m = _bin_label_and_mask(batch)
#             main_loss = masked_bce_loss(logits, y, m)
#         else:
#             # 回归：MSE
#             pred = logits.squeeze(-1)
#             y = batch.y.squeeze(-1).float()
#             main_loss = nn.functional.mse_loss(pred, y)
#
#         # ← 新增：总损失 = 主任务损失 + 蒸馏损失
#         total_loss = main_loss + model.kd_weight * kd_loss
#
#         optimizer.zero_grad()
#         total_loss.backward()
#         # （可选）梯度裁剪，避免偶发不稳定
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#         optimizer.step()
#
#         losses.append(float(main_loss.detach().cpu().item()))
#         kd_losses.append(float(kd_loss.detach().cpu().item()))  # ← 记录蒸馏损失
#
#     avg_main_loss = float(np.mean(losses)) if losses else 0.0
#     avg_kd_loss = float(np.mean(kd_losses)) if kd_losses else 0.0
#
#     # 返回主损失和蒸馏损失（用于打印）
#     return avg_main_loss, avg_kd_loss
#
#
# @torch.no_grad()
# def evaluate(model, loader, device, task_type: str):
#     model.eval()
#     y_list, p_list = [], []
#     losses = []
#
#     for batch in loader:
#         batch = batch.to(device)
#         # ← 修改：评估时也使用新的前向传播
#         logits, kd_loss = model(batch)  # (B, T), scalar
#
#         if task_type in ["binary", "multilabel"]:
#             # 统一使用 sigmoid -> 正类概率
#             prob = torch.sigmoid(logits)  # (B, T)
#
#             # 规整标签与 mask
#             y, m = _bin_label_and_mask(batch)  # (B, T)
#
#             # T==1 时压成一维，便于后面拼接/计算 AUC
#             if prob.size(-1) == 1:
#                 prob = prob.squeeze(-1)  # (B,)
#                 y = y.squeeze(-1)  # (B,)
#                 m = m.squeeze(-1)  # (B,)
#
#             # 只保留有效样本
#             valid = (m > 0)
#             y_list.append(y[valid].detach().cpu().numpy())
#             p_list.append(prob[valid].detach().cpu().numpy())
#
#             # 也计算一个 loss（带 mask 的 BCE）
#             bce = masked_bce_loss(logits, y if y.ndim == 2 else y.unsqueeze(-1),
#                                   m if m.ndim == 2 else m.unsqueeze(-1))
#             losses.append(float(bce.detach().cpu().item()))
#
#         else:
#             # 回归：收集 MSE / RMSE
#             pred = logits.squeeze(-1)
#             y = batch.y.squeeze(-1).float()
#             mse = nn.functional.mse_loss(pred, y)
#             losses.append(float(mse.detach().cpu().item()))
#             y_list.append(y.detach().cpu().numpy())
#             p_list.append(pred.detach().cpu().numpy())
#
#     if task_type in ["binary", "multilabel"]:
#         if len(y_list) == 0:
#             return 0.0, float('nan')
#
#         y_true = np.concatenate(y_list, axis=0)
#         y_prob = np.concatenate(p_list, axis=0)
#
#         # 再保险：若仍是二维概率，按"取正类列"转一维
#         if y_prob.ndim == 2:
#             if y_prob.shape[1] == 2:
#                 y_prob = y_prob[:, 1]
#             elif y_prob.shape[1] == 1:
#                 y_prob = y_prob.reshape(-1)
#             else:
#                 y_prob = y_prob.reshape(y_prob.shape[0], -1)[:, -1]
#
#         # AUC 计算（内部也会做兜底）
#         metric = compute_classification_auc(y_true, y_prob, average="macro")
#         return float(np.mean(losses)) if losses else 0.0, float(metric) if metric == metric else float('nan')
#
#     else:
#         if len(y_list) == 0:
#             return 0.0, float('nan')
#         y_true = np.concatenate(y_list, axis=0)
#         y_pred = np.concatenate(p_list, axis=0)
#         metric = compute_regression_rmse(y_true, y_pred)
#         return float(np.mean(losses)) if losses else 0.0, float(metric)
#
#
# @torch.no_grad()
# def predict_dataframe(model, loader, device, task_type: str, num_tasks: int, split_name: str, epoch: int):
#     """
#     用于逐分子导出预测结果到 DataFrame。
#     - 对分类任务：输出正类概率 prob_*（mask==0 的位置置 NaN）
#     - 对回归任务：输出 pred（mask==0 的位置置 NaN，如果有 mask）
#     统一包含: ['epoch','split','smiles', label列..., 预测列..., mask列...]
#     """
#     model.eval()
#     rows = []
#
#     for batch in loader:
#         batch = batch.to(device)
#         # ← 修改：预测时使用不计算蒸馏损失的方法
#         if hasattr(model, 'forward_without_kd'):
#             logits = model.forward_without_kd(batch)  # 使用专门的方法
#         else:
#             logits, _ = model(batch)  # 忽略蒸馏损失
#
#         # 取 SMILES（PyG 的 Batch 会把自定义属性聚合成 list）
#         smiles_list = getattr(batch, "smiles", None)
#         if smiles_list is None:
#             # 若你的 Data 没存 smiles，可用占位
#             smiles_list = [""] * logits.size(0)
#
#         if task_type == "regression":
#             # 回归
#             pred = logits.squeeze(-1).detach().cpu().numpy()  # (B,)
#             y = batch.y.squeeze(-1).detach().cpu().numpy()  # (B,)
#             m = getattr(batch, "mask", torch.ones_like(batch.y)).squeeze(-1).detach().cpu().numpy()  # (B,)
#
#             # mask==0 的位置置 NaN 便于后续分析
#             pred = np.where(m > 0.5, pred, np.nan)
#             y = np.where(m > 0.5, y, np.nan)
#
#             for s, yt, pr, mk in zip(smiles_list, y, pred, m):
#                 rows.append({
#                     "epoch": epoch,
#                     "split": split_name,
#                     "smiles": s,
#                     "label": yt,
#                     "pred": pr,
#                     "mask": float(mk),
#                 })
#
#         else:
#             # 分类（binary 或 multilabel）—— 使用 sigmoid 得到正类概率
#             prob = torch.sigmoid(logits).detach().cpu().numpy()  # (B, T)
#             y = batch.y.detach().cpu().numpy()  # (B, T)
#             m = getattr(batch, "mask", torch.ones_like(batch.y)).detach().cpu().numpy()  # (B, T)
#
#             # 保留 NaN：把 mask==0 的标签置 NaN
#             y = np.where(m > 0.5, y, np.nan)
#             # 概率不置 NaN（保留模型输出），但可同时输出 mask 便于过滤
#             T = prob.shape[1]
#             for i in range(prob.shape[0]):
#                 row = {
#                     "epoch": epoch,
#                     "split": split_name,
#                     "smiles": smiles_list[i],
#                 }
#                 # 标签/概率/掩码列
#                 if num_tasks == 1:
#                     row["label"] = y[i, 0] if y.ndim == 2 else y[i]
#                     row["prob"] = prob[i, 0] if prob.ndim == 2 else prob[i]
#                     row["mask"] = float(m[i, 0] if m.ndim == 2 else m[i])
#                 else:
#                     for t in range(T):
#                         row[f"label_t{t}"] = y[i, t]
#                         row[f"prob_t{t}"] = prob[i, t]
#                         row[f"mask_t{t}"] = float(m[i, t])
#                 rows.append(row)
#
#     df = pd.DataFrame(rows)
#     return df