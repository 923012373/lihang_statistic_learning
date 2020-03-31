import numpy as np
import torchvision as tv
import heapq
from torchvision.transforms import transforms
from tqdm import tqdm

class Knn():
    '''
    线性扫描的knn算法
    '''
    def __init__(self,k,data,labels):
        '''
        :param k:k个最近的样本
        :param data:(n*m),n为训练样本数，m为特征数
        :param labels:训练集标签
        '''
        self.k = k
        self.data = data
        self.labels = labels

    def get_l1_distance(self,x1,x2):
        distance = np.sum(np.abs(x1 - x2))
        return distance

    def predict(self, x):
        distances = np.abs(self.data - x)
        distances = np.sum(distances,1)
        k_minest_dis =heapq.nsmallest(self.k,distances)
        k_minest_index = [np.argwhere(distances==dis)[0][0] for dis in k_minest_dis]
        # k_minest_index = np.array(list(map(distances.argwhere(),distances)))
        k_minest_labels = [self.labels[ind] for ind in k_minest_index]
        class_counts = np.bincount(k_minest_labels)
        res = np.argmax(class_counts)
        return res


transform = transforms.Compose([
    transforms.ToTensor(),
])
train_set = tv.datasets.CIFAR10(root='../data',train=True,download=True,transform=transform)
test_set = tv.datasets.CIFAR10(root='../data',train=False,download=True,transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#自定义训练集、测试集大小
train_size = 10000
test_size = 10000
train_data = train_set.data[:train_size,:,:,:].reshape(train_size,-1)
train_labels = train_set.targets[:train_size]
test_data = test_set.data[:test_size,:,:,:].reshape(test_size,-1)
test_labels = test_set.targets[:test_size]



correct = 0
model = Knn(1,train_data,train_labels)
for i,data in tqdm(enumerate(test_data)):
    pred = model.predict(data)
    if pred == test_labels[i]:
        correct += 1
print('测试正确率:%f %%' %(correct*100/test_size))




