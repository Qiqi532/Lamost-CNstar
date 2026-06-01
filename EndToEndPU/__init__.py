"""EndToEndPU: 端到端 PU 深度网络 — 1D ResNet + CN 注意力 + nnPU 损失。

将 CN 感知 AE 证明有效的"关注 CN 分子带"思想融入分类网络注意力机制，
直接对 700-pixel 光谱做端到端 PU 分类，测试深度学习在此筛选任务中的有效性。
"""
