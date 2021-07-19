# 模型压缩和加速层次

算法层/框架层/硬件层
https://zhuanlan.zhihu.com/p/138059904

# 为什么不使用知识蒸馏

知识蒸馏适用于大规模推理，对于百万级别参数的问题直接训练即可

# 量化数学基础

(Q3-Z3)S3=(Q1-Z1)S1*(Q2-Z2)S2 量化计算过程q3=f(q1,q2,s1,s2,z1,z2)
每一层都是求解q
https://zhuanlan.zhihu.com/p/149659607

# post_training_quantization 训练后量化

工业场景比较适用

# quantization aware training 量化感知训练

Q:如何解决量化梯度为零的问题 A:量化梯度为零是因为量化函数是阶跃函数，所以导数为零，因此前向是假量化，后向直接透传梯度为1 Q:前向计算的时候量化的S和Z如何计算 以卷积层为例，s是卷积参数的最大值-最小值除以int最大值-最小值

效果优于训练后量化，应用部署缺乏框架支持 伪量化 前向计算 float先量化成int，使用round函数，再反量化成float

训练后量化和量化感知训练的推理过程基本一致， pytorch推理前先freeze计算weight的量化

Q:tensorflow量化训练后的值为什么还是浮点数? A:训练过程如此，可以使用TOCO转换得到真正的量化模型

# NNAPI

能够为android设备提供加速，支持GPU,DSP,NPU

# ref

https://github.com/tensorflow/examples.git

# ref

https://www.tensorflow.org/lite/performance/gpu

# ref

https://arxiv.org/pdf/2004.12599.pdf

Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors
当模型包含动态大小的输出时，也不支持 NNAPI 加速。在这种情况下，您会收到如下所示的警告 模型如何兼容NNAPI 每一层的size都必须固定，包括batch size Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
quantize_layer (QuantizeLaye (1, 28, 28)               3
_________________________________________________________________
quant_reshape (QuantizeWrapp (1, 28, 28, 1)            1
_________________________________________________________________
quant_conv2d (QuantizeWrappe (1, 26, 26, 12)           147
_________________________________________________________________
quant_max_pooling2d (Quantiz (1, 13, 13, 12)           1
_________________________________________________________________
quant_flatten (QuantizeWrapp (1, 2028)                 1
_________________________________________________________________
quant_dense (QuantizeWrapper (1, 10)                   20295
=================================================================
Total params: 20,448 Trainable params: 20,410 Non-trainable params: 38
_________________________________________________________________


nnapi支持的操作 we replace TRANSPOSE_CONV_2D by DEPTH_TO_SPACE6 and RESIZE_BILINEAR

延迟推理是什么：前向计算的时间！ nnapi支持的操作有哪些？参考https://www.tensorflow.org/lite/performance/nnapi
mac(Memory Access Cost)

# 量化卷积和量化激活

https://zhuanlan.zhihu.com/p/132561405