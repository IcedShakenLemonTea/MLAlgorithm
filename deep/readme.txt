环境
python 3.6.3
pytorch 0.4

数据
cifar-10

运行
python train.py --gpu=-1 --data=./ --load_model=./models/net_params_pretrain.pkl --save_model=./models/net_params_train.pkl

运行参数
gpu：gpu编号(=-1默认cpu训练)
data：数据路径(cifar-10-batches-py所在目录)
load_model：初始化模型(net_params_pretrain.pkl是预训练的参数, 为空时表示随机初始化)
save_model：保存模型(为空时表示不保存，只保存参数不保存网络结构)