环境
python 3.6.3
scikit-learn v0.19.1

数据
鸢尾花：iris
手写数字：digits

运行
python main.py --data=iris --standardization=zscore --decomposition=2 --algorithm=svm

运行参数
data：训练数据(iris或digits)
standardization=zscore：数据标准化(zscore或minmax)
decomposition：PCA降维(1-max dim, 超出范围默认为不降维)
algorithm：使用算法(分类svm、knn，聚类k-means)