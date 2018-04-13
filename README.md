# TensorFlow-MNIST-<br>
>> ### AleNet:<br>
>>> 1) 加载数据<br>
2）定义网络的参数<br>
3）定义卷积、池化、规范化操作<br>
4）定义所有的网络参数<br>
5）定义 AlexNet 的网络模型：<br>
[conv2d -> maxpool2d -> norm(规范化)] * 3 -> conv2d * 2 -> maxpool2d -> norm -> [全连接层 -> Droupt] * 2 -> out<br>
6）构建模型,定义损失函数和优化器<br>
7) 训练模型和评估模型<br>
>> ### CNN:<br>
>>> 1) 输入数据并预处理数据:把上述 trX 和 teX 的形状变为[-1,28,28,1],-1 表示不格虑输入图片的数量,28×28 是图片的长和宽的像素数,
1 是通道(channel)数量,因为 MNIST 的图片是黑白的,所以通道是 1,如果是 RGB 彩色图像,通道是 3<br>
2) 初始化权重与定义格络结构:设置卷积核的大小为 3×3:<br>
3) 定义一个模型函数:<br>
[conv2d -> max_pool -> dropout] * 3-> 全链接层 -> droupt -> out<br>
4) 定义损失函数: tf.nn.softmax_cross_entropy_with_logits<br>
优化器:tf.train.RMSPropOptimizer<br>
5) 训练模型和评估模型<br>
>> ### RNN:<br>
>>> 1) 加载数据<br>
2) 为了使用 RNN 来分类图片,我们把每张图片的行看成是一个像素序列(sequence)。因为MNIST 图片的大小是 28×28 像素,
所以我们把每一个图像样本看成一行行的序列。因此,共有(28 个元素的序列)×(28 行),然后每一步输入的序列长度是 28,输入的步数是 28 步<br>
3）定义超参数、神经网络参数、输入数据及权重<br>
4） 构建  RNN 模型：<br>
reshape -> matmul -> reshape ->  采用基本的 LSTM 循环网络单元 tf.contrib.rnn.BasicLSTMCell ->
初始化为零值, lstm 单元由两个部分组成: (c_state, h_state) lstm_cell.zero_state -> 
dynamic_rnn 接收张量 (batch, steps, inputs) 或者 (steps, batch, inputs) 作为 X_in -> tf.nn.dynamic_rnn -> tf.matmul<br>
5) 定义损失函数和优化器,优化器采用 AdamOptimizer<br>
6) 训练数据及评估模型<br>
>> ### 回归:<br>
>>> 1)加载数据<br>
2）优化器：tf.train.GradientDescentOptimizer<br>
3）训练模型： 让模型循环训练 1000 次,在每次循环中我们都随机抓取训练数据中 100 个数据点,来替换之前的占位符<br>
4） 评估训练好的模型<br>
>> ### 自编码网络的实现:<br>
>>> 1)加载数据<br>
2)设置训练超格数<br>
3）设置其他参数变量,表示从测试集中选择 10 张图片去验证自动编码器的结果:xamples_to_show = 10<br>
4）然后定义输入数据,这里是无监督学习,所以只需要输入图片数据,不需要标记数据<br>
5）随后初始化权重与定义格络结构。我们设计这个自动编码格络含有两个隐藏层,第一个隐藏层神经元为 256 个,第二个隐藏层神经元为 128 个<br>
6）初始化每一层的权重和偏置<br>
7）定义自动编码模型的格络结构,包括压缩和解压两个过程<br>
8）构建模型<br>
9）构建损失函数和优化器。这里的损失函数用“最小二乘法”对原始数据集和输出的数据集进行平格差并取均值运算;优化器采用 RMSPropOptimizer<br>
9）训练数据及评估模型<br>
