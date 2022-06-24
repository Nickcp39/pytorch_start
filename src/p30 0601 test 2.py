
import torch

outputs = torch.tensor([[0.1,0.2],
                       [0.05,0.4]])

print(outputs.argmax(1))
preds =  outputs.argmax(1)
# 真实输入target
targets = torch.tensor([0.1])
print(preds == targets)

"""
2 * input 
Model ( 2 分类）
【0.1,0.2】
【0.3,0.4】
0 ， 1
predicts = [1]
            [1]

predicts = input targets
[false, true].sum() = 1



整体就是， 如果predict match target， accu 就加一 
如果不match， 就false 为0
这个在p29 完整模型训练套路中， 可以用于计算整体的预测正确性 的一种方法，就是个简单的match算法
"""





























