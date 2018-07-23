# PointCloudClassification_keras
tag: Point cloud classification, deep learning, keras
provider.py 对H5点云文件的数据进行读取，进行随机打乱，抖动处理
Point_cla_keras.py 读取训练集数据，定义训练集的generator函数，构建神经网络，使用优化函数，tenorboard可视化训练过程，评估训练效果并保存模型到Model文件夹下
predict_test.py 用测试集来对训练的模型进行测试，计算识别准确率
input_predict.py 使用flask框架构建网站后台，主要功能是设置更改模型，上传点云文件进行预测，分页，结果写入excel，绘制点云三维图像等功能
Matplotlib_draw.py 使用类似MATLAB的方法绘制点云的散点图，以可视化点云】
writeH5.py 拆分点云H5文件
H5toPcd.py 将H5格式的点云转换为PCD格式的点云文件
