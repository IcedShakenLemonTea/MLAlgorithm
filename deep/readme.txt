����
python 3.6.3
pytorch 0.4

����
cifar-10

����
python train.py --gpu=-1 --data=./ --load_model=./models/net_params_pretrain.pkl --save_model=./models/net_params_train.pkl

���в���
gpu��gpu���(=-1Ĭ��cpuѵ��)
data������·��(cifar-10-batches-py����Ŀ¼)
load_model����ʼ��ģ��(net_params_pretrain.pkl��Ԥѵ���Ĳ���, Ϊ��ʱ��ʾ�����ʼ��)
save_model������ģ��(Ϊ��ʱ��ʾ�����棬ֻ�����������������ṹ)