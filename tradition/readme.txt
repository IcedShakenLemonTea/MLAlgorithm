����
python 3.6.3
scikit-learn v0.19.1

����
�β����iris
��д���֣�digits

����
python main.py --data=iris --standardization=zscore --decomposition=2 --algorithm=svm

���в���
data��ѵ������(iris��digits)
standardization=zscore�����ݱ�׼��(zscore��minmax)
decomposition��PCA��ά(1-max dim, ������ΧĬ��Ϊ����ά)
algorithm��ʹ���㷨(����svm��knn������k-means)