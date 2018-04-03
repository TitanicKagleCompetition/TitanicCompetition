import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt 
from pandas import DataFrame 
import csv

trainFile = pd.read_csv('C:/Users/rediron/Desktop/Titanic/data/train.csv')
testFile = pd.read_csv('C:/Users/rediron/Desktop/Titanic/data/test.csv')
answerFile = pd.read_csv('C:/Users/rediron/Desktop/Titanic/data/gender_submission.csv')

file_trainData = trainFile.as_matrix()

#Train Data set
#Survived label -y
train_Id_survived = file_trainData[:,[1]]
#Passenger Data that except the Survived data -x
train_PassengerData = np.delete(file_trainData,1,1) #delete survived 
train_PassengerData = np.delete(train_PassengerData ,2,1) #delete name
train_PassengerData = np.delete(train_PassengerData ,4,1) #delete sibSP
train_PassengerData = np.delete(train_PassengerData ,4,1) #delete parch
train_PassengerData = np.delete(train_PassengerData ,4,1) #delete ticket
train_PassengerData = np.delete(train_PassengerData ,6,1) #delete embark
train_PassengerData = np.delete(train_PassengerData ,5,1) #delete carbin


#Test Data set
#Passenger Id and Survived label -test y
test_Id_survived = answerFile.as_matrix()
test_Id_survived = np.delete(test_Id_survived,0,1)

#Passenger Data thar except the survived data -test x
test_PassengerData = testFile.as_matrix()
test_PassengerData = np.delete(test_PassengerData, 2,1)
test_PassengerData = np.delete(test_PassengerData, 4,1)
test_PassengerData = np.delete(test_PassengerData, 4,1)
test_PassengerData = np.delete(test_PassengerData, 4,1)
test_PassengerData = np.delete(test_PassengerData, 5,1)
test_PassengerData = np.delete(test_PassengerData, 5,1)
#print(test_PassengerData[2,:])

train_dataLenght = train_PassengerData.shape[0]
test_dataLenght = test_PassengerData.shape[0]

i=0
#sex convert to the binary
for i in range(train_dataLenght): 
  if train_PassengerData[i,2] == 'female':
        train_PassengerData[i,2] = 0.0
        i = i+1             
  else:
        train_PassengerData[i,2] = 1.0
        i = i+1
i=0

for i in range(test_dataLenght): 
  if test_PassengerData[i,2] == 'female':
        test_PassengerData[i,2] = 0
        i = i+1             
  else:
        test_PassengerData[i,2] = 1
        i = i+1
i=0
train_PassengerData.astype(np.float)
train_Id_survived.astype(np.float)
test_PassengerData.astype(np.float)
test_Id_survived.astype(np.float)

#print("data")
#print(train_PassengerData[2,:])
#print(test_PassengerData[2,:])
#print(train_Id_survived[2])
#print(test_Id_survived[2])

#print("lenght")
#print("train passenger : ", train_PassengerData.shape[0],train_PassengerData.shape[1])
#print("test passenger : ", test_PassengerData.shape[0],test_PassengerData.shape[1])
#print("train sur : ", train_Id_survived.shape[0])
#print("test sur : ", test_Id_survived.shape[0])
#print(train_PassengerData[1,0], " ",test_PassengerData[1,0])
#print(train_PassengerData[1,0]*test_PassengerData[1,0])
#print(train_Id_survived[1,0], " ",test_Id_survived[1,0])
#print(train_Id_survived[1,0]+test_Id_survived[1,0])


#print("\n")
#for i in range (5):     
#      print(train_PassengerData[2,i] * (1) )
      

#데이터 사용을 위한 플레이스 홀더 메모리 할당
x_data = tf.placeholder(tf.float32, [None, 5])
#x에 따른 y값 label 정답 데이터
label = tf.placeholder(tf.float32, [None, 1])

#dropout keep_prob
#keep_prob=tf.placeholder(tf.float32)

#업데이트를 해야하는 W,b variable 설정 zeros(): 0으로 초기화
#W,b를 알아야 (학습해야) 기능을 할 수 있다.
W_1= tf.get_variable("W_1", shape = [5, 5], initializer = tf.contrib.layers.xavier_initializer())
b_1 = tf.Variable(tf.random_normal([5]))
y_1 = tf.nn.relu(tf.matmul(x_data, W_1) + b_1)

W_2= tf.get_variable("W_2", shape = [5, 5], initializer = tf.contrib.layers.xavier_initializer())
b_2 = tf.Variable(tf.random_normal([5]))
y_2 = tf.nn.relu(tf.matmul(y_1, W_2) + b_2)

W1= tf.get_variable("W1", shape = [5, 1], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.nn.relu(tf.matmul(y_2, W1) + b1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = hypothesis, labels=label) )

#학습 방법은 그래디언트 디센트 방법을 사용. cross entropy를 줄여라      
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

predicted = tf.equal(tf.argmax(hypothesis,1),tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

#session생성
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    cost_val, ts = sess.run([cost, train_step], feed_dict={x_data: train_PassengerData, label: train_Id_survived}) 
    #print(i,"  ",cost,"  ",ts)


a= sess.run(accuracy, feed_dict={x_data: test_PassengerData, label: test_Id_survived})
#print("\nh: ",h,"\nc: ",c,"\nAccuracy:",a)
print("\nAccuracy:",a)
sess.close()


#plt.plot(train_PassengerData[:,[1]], train_PassengerData[:,[9]])
#plt.show()