"""第15节 XOR MLP 补全实现；与 PyTorch 使用相同初值和数据。"""
import json
from pathlib import Path
import tensorflow as tf

tf.keras.backend.set_floatx('float64')
# ① 张量：4×2 输入、4×1 标签
x = tf.constant([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=tf.float64)
target = tf.constant([[0.], [1.], [1.], [0.]], dtype=tf.float64)
# ② 模型参数
W = tf.Variable([[2., -2.], [-2., 2.]], dtype=tf.float64)
b_hidden = tf.Variable([-1., -1.], dtype=tf.float64)
v = tf.Variable([[3.], [3.]], dtype=tf.float64)
b_out = tf.Variable([-1.], dtype=tf.float64)  # 截图未展示，补充
params = [W, b_hidden, v, b_out]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.03, epsilon=1e-8)

def forward():
    hidden = tf.sigmoid(x @ W + b_hidden)
    return hidden @ v + b_out

def snapshot(epoch):
    logits = forward()
    return dict(epoch=epoch,
                loss=float(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=target, logits=logits)).numpy()),
                pred=tf.reshape(tf.sigmoid(logits), [-1]).numpy().tolist(),
                p=tf.concat([tf.reshape(p, [-1]) for p in params], 0).numpy().tolist())

history = [snapshot(0)]
for epoch in range(1, 301):
    with tf.GradientTape() as tape:
        logits = forward()
        # ③ 损失：四个样本的平均二元交叉熵
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=logits))
    # ④ 自动求梯度；⑤ 更新参数，两步分开
    gradients = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(gradients, params))
    history.append(snapshot(epoch))

Path('history_tensorflow.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
print('最终损失:', history[-1]['loss'])
for sample, prob in zip(x.numpy().tolist(), history[-1]['pred']):
    print(sample, '概率:', round(prob, 6), '分类:', int(prob >= 0.5))
