## 基于MxNet Gluon开发的用于比赛的框架
* data模块：主要包括数据预处操作，1）图像指定大小、重叠度的裁剪；2）图像随机旋转、翻转、指定角度旋转等数据增强操作；3）获得训练所需的图像路径索引列表，支持指定比例划分以及K折划分操作。
* learning_rate模块：实现最佳初始学习率的查找。
* loss模块：主要包括dice loss、softmax_celoss_with_ignore、softmax_celoss_with_weight、softmax_celoss_with_weight，后续会加入更多loss实现。
* metrics模块：主要包括confusion matrix、accuracy、f1 score等计算，后续会继续完善。
* tta模块：模型测试数据增强模块，主要包括测试图像的旋转、翻转操作，从而获得更加鲁棒的预测结果，后续会继续完善。
* models模块：包含网络实现，待完善。
* train_model模块：将训练代码封装为函数，直接调用。
