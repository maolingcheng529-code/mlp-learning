"""第15节 XOR MLP 补全实现。输出层偏置初值 -1 为补充设定。"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
# ① 张量：4 个样本，每个样本 2 个输入
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
target = torch.tensor([[0.], [1.], [1.], [0.]])
# ② 模型：2 → 2 sigmoid 隐藏单元 → 1 输出 logit
W = torch.nn.Parameter(torch.tensor([[2., -2.], [-2., 2.]]))
b_hidden = torch.nn.Parameter(torch.tensor([-1., -1.]))
v = torch.nn.Parameter(torch.tensor([[3.], [3.]]))
b_out = torch.nn.Parameter(torch.tensor([-1.]))  # 截图未展示，补充
params = [W, b_hidden, v, b_out]
optimizer = torch.optim.Adam(params, lr=0.03, eps=1e-8)

def forward():
    hidden = torch.sigmoid(x @ W + b_hidden)
    return hidden @ v + b_out

def snapshot(epoch):
    with torch.no_grad():
        logits = forward()
        return dict(epoch=epoch,
                    loss=F.binary_cross_entropy_with_logits(logits, target).item(),
                    pred=torch.sigmoid(logits).flatten().tolist(),
                    p=torch.cat([p.flatten() for p in params]).tolist())

history = [snapshot(0)]
for epoch in range(1, 301):
    optimizer.zero_grad()  # 清除上一轮梯度，避免累积
    logits = forward()
    # ③ 损失：数值稳定的二元交叉熵，四个样本取均值
    loss = F.binary_cross_entropy_with_logits(logits, target)
    # ④ 自动求梯度：此时还没有更新参数
    loss.backward()
    # ⑤ 更新参数：Adam 使用梯度更新权重与偏置
    optimizer.step()
    history.append(snapshot(epoch))  # 记录更新后的损失与预测

Path('history_torch.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
print('最终损失:', history[-1]['loss'])
for sample, prob in zip(x.tolist(), history[-1]['pred']):
    print(sample, '概率:', round(prob, 6), '分类:', int(prob >= 0.5))
