# 第二次作业：第 15 节 MLP 学习参数的实现与可视化

独立作业，与第一次 Agent 使用体验网站分开。

依据课程截图补全 TensorFlow 和 PyTorch 两套 XOR 实现：2 输入 → 2 个 sigmoid 隐藏单元 → 1 输出，平均二元交叉熵，Adam，学习率 0.03，训练 300 轮，分类阈值 0.5，float64。

截图可见初值：W=[[2,-2],[-2,2]]，b_hidden=[-1,-1]，v=[[3],[3]]。截图未展示输出层偏置与训练循环；这里将输出偏置初始化为 -1，补全训练、日志和展示代码。不是原始讲义完整代码的逐字转录。

## 打开可视化

直接双击 `index.html`，无需服务器或网络。可播放、单步、拖动训练轮次，改变学习率和当前样本；观察损失曲线、参数、梯度和四样本预测。

浏览器使用 JavaScript 手写反向传播与 Adam（epsilon 位于平方根外），并非在浏览器内运行 TensorFlow 或 PyTorch。Python 两版使用框架自动求导。框架 Adam 的 epsilon 处理等细节可能造成数值差异，不保证逐位一致。

## 运行 Python

建议 Python 3.11、CPU，两个版本可任选一个，不需要同时安装两个框架。

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install torch
python train_torch.py
```

TensorFlow 版本：将最后两行替换成 `python -m pip install tensorflow` 和 `python train_tensorflow.py`。macOS/Linux 激活命令为 `source .venv/bin/activate`。

每个脚本输出最终损失、概率和分类，并生成 `history_torch.json` 或 `history_tensorflow.json`，包含初始状态和每次更新后的状态（共 301 个）。参数顺序为 W 展平、b_hidden、v 展平、b_out。

## 学习重点

1. 张量：输入形状 4×2，标签形状 4×1。
2. 模型：h=sigmoid(xW+b_hidden)，z=hv+b_out，概率=sigmoid(z)。共 9 个参数。
3. 损失：平均 BCE，使用 logits 形式保证数值稳定。
4. 自动求梯度：`loss.backward()` / `tape.gradient()`，不负责更新参数。
5. 参数更新：`optimizer.step()` / `apply_gradients()`。

四个训练样本的准确率不代表泛化能力；300 轮也不保证所有初始化都收敛。

## 验证

`node test.cjs` 检查解析梯度与有限差分梯度、默认训练损失下降和 XOR 分类，以及页面本地资源。Python 框架运行情况见交付说明；不要将浏览器运行当作框架实测。

参考：[TensorFlow GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape)、[PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)。学习材料为用户提供的第 15 节截图，完整原始讲义未提供。

## 展示版升级

深色首屏、响应式实验面板及减少动态效果支持。新增 XOR 连续决策区域（点击/方向键探针）、参数历史曲线、四步训练讲解和 JSON 训练记录导出。探针不会参与训练；导出包含当前学习率下完整 0–300 轮记录，displayed_epoch 标明界面正在查看的轮次。

已在 Edge 检查桌面和 390px 手机布局，并验证训练轮次、重置、单步、学习率、样本、参数选择和 Python 代码切换。
