#xaviar
ref:https://zhuanlan.zhihu.com/p/86602524  
理解的关键在于每一个新的元素都是由n个w*x得到  
如何理解扇入和扇出 m*n@n*1 得到m*1 扇入是n,扇出是m
m*n的图像经过卷积得到m*n的图像，扇入扇出是多少？ 
fan_in = num_input_feature_maps*kernel_height*kernel_width
fan_out = num_output_feature_maps*kernel_height*kernel_width/max_pool_area
前向计算每个点的乘法数量是fan_in，反向传播计算每个点的乘法数量是fan_out 
目标是前向输出层的方差和上一层一致  反向每一层的导数值的方差和上一层一致
var_forward = 1/fan_in var_backward = 1/fan_out
均匀分布的方差是(b-a)^2/12
#不同激活函数的增益
Sigmoid 1
tanh 5/3
relu 2^0.5
#kaiming初始化
针对relu的增益进行缩放