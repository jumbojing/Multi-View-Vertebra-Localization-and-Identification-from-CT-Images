# Heatmap Regression

对于heatmap regression,即使用热力图进行回归,主要有以下几种方法:

- CNN回归
可以使用卷积神经网络,输入是图像,输出是对应的热力图,经过训练预测每个像素的数值。

- 基于关键点回归
先检测图像中的关键点,然后生成每个关键点周围的高斯热力图,再叠加形成完整热力图。

- 基于语义分割回归
使用语义分割网络预测每个像素属于哪个类别,然后为每个类别指定不同的热度值,生成热力图。

- 基于距离的回归
计算每个像素与目标的距离,距离越近热度值越高,根据距离生成热力图。

- 基于坐标回归
直接预测图像中目标的坐标位置,然后根据坐标生成高斯热力图。

- GAN for heatmap
使用对抗生成网络生成对应的热力图,可以用于数据增强。