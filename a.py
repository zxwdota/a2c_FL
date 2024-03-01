########手写数字数据集##########
###########加载模型参数方法############
########1层隐含层（全连接层）##########
#60000条训练数据和10000条测试数据，28x28像素的灰度图像
#隐含层激活函数：ReLU函数
#输出层激活函数：softmax函数（实现多分类）
#损失函数：稀疏交叉熵损失函数
#输入层有784个节点，隐含层有128个神经元，输出层有10个节点
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import time
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print(nowtime)


#初始化
plt.rcParams['font.sans-serif'] = ['SimHei']

#加载数据
mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape))

#数据预处理
#X_train = train_x.reshape((60000,28*28))
#Y_train = train_y.reshape((60000,28*28))       #后面采用tf.keras.layers.Flatten()改变数组形状
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)        #归一化
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

#建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     #添加Flatten层说明输入数据的形状
model.add(tf.keras.layers.Dense(128,activation='relu'))     #添加隐含层，为全连接层，128个节点，relu激活函数
model.add(tf.keras.layers.Dense(10,activation='softmax'))   #添加输出层，为全连接层，10个节点，softmax激活函数
print('\n',model.summary())     #查看网络结构和参数信息

#配置模型训练方法
#adam算法参数采用keras默认的公开参数，损失函数采用稀疏交叉熵损失函数，准确率采用稀疏分类准确率函数
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

#加载模型参数
history = model.load_weights('mnist_weights.h5') #路径根据文件实际位置修改，不然会报错


#评估模型
model.evaluate(X_test,y_test,verbose=2)     #每次迭代输出一条记录，来评价该模型是否有比较好的泛化能力

#使用模型
plt.figure()
for i in range(10):
    num = np.random.randint(1,10000)

    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.imshow(test_x[num],cmap='gray')
    demo = tf.reshape(X_test[num],(1,28,28))
    y_pred = np.argmax(model.predict(demo))
    plt.title('标签值：'+str(test_y[num])+'\n预测值：'+str(y_pred))
#y_pred = np.argmax(model.predict(X_test[0:5]),axis=1)
#print('X_test[0:5]: %s'%(X_test[0:5].shape))
#print('y_pred: %s'%(y_pred))

plt.ion()       #打开交互式操作模式
plt.show()
plt.pause(5)
plt.close()
