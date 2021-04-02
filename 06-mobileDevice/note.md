#如何在手机上部署深度学习模型
Dense Residual U-Net
Self-Guided Network
ref：Deploying Image Deblurring across Mobile Devices: A Perspective of Quality and Latency
#什么是geometric self-ensemble
测试阶段通过翻转和旋转得到7张增强的输入低分图像，然后使用对应的逆变换。最终8张输出图像取均值得到最终结果。换句话说以八张图的结果平均作为最终结果
#NTIRE 2020去模糊比赛
ref:NTIRE 2020 Challenge on Image and Video Deblurring

#padding = same 
output = input/stride

#accuracy和loss
acc和loss没有必然的联系
如分类问题，输出正确类的概率为0.3，交叉熵损失为log0.3,如果0.3是输出的最大值，则acc是1，不是则是0
#batch normalize
使当前层的输出均值为0，标准差为1，有效避免梯度消失，使得学习更加稳定
作用的维度,bacth的维度，即对一个节点不同batch做正则，因为同一个节点的值可以归为一类
均值和方差都是基于当前batch中的训练数据
引入lam和beta用于恢复数据本身的表达能力
#打印训练过程中的损失变换图

#损失先减小后增大

#batch大小的选择
batch size太小，算法很难收敛
batch size过大，模型泛化能力很差

#梯度求导算法
建议使用adam，结合惯性+学习率质数衰减累加,adagrad会很快衰减

#gan
1.KL散度和JS散度
KL散度非对称，容易产生模式崩塌或不准确
2.原始gan的建议推导
固定G,D的最优值D=Pd/(Pd+Pg)
Pd==Pg 最优值,JS散度-log4,G有最优解
3.代码如何描述KL散度
使用二值交叉熵损失
4.gan训练的难点
lipschitz通过weight clipping来保证的缺点
weight clipping是粗暴的限制最大值，导致模型建模能力弱化，以及梯度爆炸或者消失
gradient penalty会有更好的效果，直接对x求导，对大于1的导数增加惩罚，再反向求导时对w求导




梯度方差比较大
wgan的优势
为什么gan是分布的损失，原始gan看的是期望，如何代码实现呢？

通过G函数映射的特征，比较高斯距离，