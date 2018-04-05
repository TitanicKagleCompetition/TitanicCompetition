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
train_PassengerData = np.delete(train_PassengerData ,0,1) #delete id


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
test_PassengerData = np.delete(test_PassengerData, 0,1)
#print(test_PassengerData[2,:])

i=0
#sex convert to the binary
for i in range( train_PassengerData.shape[0]):      
      if np.isnan(train_PassengerData[i,0]):
            train_PassengerData[i,0] = 3
      if np.isnan(train_PassengerData[i,2]):
            train_PassengerData[i,2] = 20.0
      if np.isnan(train_PassengerData[i,3]):
            if train_PassengerData[i,0]==1:
                  train_PassengerData[i,3]=80.0
            elif train_PassengerData[i,0]==2:
                  train_PassengerData[i,3]=29.0
            else:
                  train_PassengerData[i,3]=7.0

      if train_PassengerData[i,2] <=20:
            train_PassengerData[i,2] = train_PassengerData[i,2] * 6
      elif train_PassengerData[i,2] >=60:
                  train_PassengerData[i,2] = train_PassengerData[i,2] * 2
      else:
            train_PassengerData[i,2] = train_PassengerData[i,2]/2

      if train_PassengerData[i,1] == 'female':
            train_PassengerData[i,1] = 1                      
      else:
            train_PassengerData[i,1] = 0

      if np.isnan(train_PassengerData[i,1]):
            train_PassengerData[i,1] = 0

      train_PassengerData[i,0] = train_PassengerData[i,0] * train_PassengerData[i,3]
      i=i+1     
i=0

for i in range(test_PassengerData.shape[0]): 
      #print(test_PassengerData[i,:])  
      if np.isnan(test_PassengerData[i,0] ):
            test_PassengerData[i,0] = 3     
      if np.isnan(test_PassengerData[i,2] ):
            test_PassengerData[i,2] = 20.0
      if np.isnan(test_PassengerData[i,3]):
            if test_PassengerData[i,0]==1:
                  test_PassengerData[i,3]=80.0
            elif test_PassengerData[i,0]==2:
                  test_PassengerData[i,3]=29.0
            else:
                  test_PassengerData[i,3]=7.0
      if test_PassengerData[i,2] <=20:
            test_PassengerData[i,2] = test_PassengerData[i,2] * 6
      elif test_PassengerData[i,2] >=60:
                  test_PassengerData[i,2] = test_PassengerData[i,2] * 2
      else:
            test_PassengerData[i,2] = test_PassengerData[i,2] /2

      if test_PassengerData[i,1] == 'female':
            test_PassengerData[i,1] = 1                     
      else:
            test_PassengerData[i,1] = 0

      if np.isnan(test_PassengerData[i,1]):
            test_PassengerData[i,1] = 0   
      i = i+1   

i=0
train_PassengerData[:,0] = train_PassengerData[:,0]/np.mean(train_PassengerData[:,3])
test_PassengerData[:,0] = test_PassengerData[:,0]/np.mean(test_PassengerData[:,3])
train_PassengerData = np.delete(train_PassengerData ,3,1)
test_PassengerData = np.delete(test_PassengerData, 3,1)

train_PassengerData[:,2] = train_PassengerData[:,2]/np.mean(train_PassengerData[:,2])
test_PassengerData[:,2] = test_PassengerData[:,2]/np.mean(test_PassengerData[:,2])


#print(train_Id_survived)


x_calss_data = np.zeros((train_PassengerData.shape[0],1))
x_sex_data = np.zeros((train_PassengerData.shape[0],1))
x_age_data = np.zeros((train_PassengerData.shape[0],1))

x_calss_Tdata = np.zeros((test_PassengerData.shape[0],1))
x_sex_Tdata = np.zeros((test_PassengerData.shape[0],1))
x_age_Tdata = np.zeros((test_PassengerData.shape[0],1))

for i in range(train_PassengerData.shape[0]):
      x_calss_data[i,0] = train_PassengerData[i,0]
      x_sex_data[i,0] = train_PassengerData[i,1]
      x_age_data[i,0] = train_PassengerData[i,2]

#데이터 사용을 위한 플레이스 홀더 메모리 할당
x_class = tf.placeholder(tf.float32, [None, 1])
x_sex = tf.placeholder(tf.float32, [None, 1])
x_age = tf.placeholder(tf.float32, [None, 1])
#x에 따른 y값 label 정답 데이터
label = tf.placeholder(tf.float32, [None, 1])
#dropout keep_prob
keep_prob=tf.placeholder(tf.float32)

#업데이트를 해야하는 W,b variable 설정 zeros(): 0으로 초기화
#W,b를 알아야 (학습해야) 기능을 할 수 있다.

#class node
W_1 = tf.get_variable(name = "W_1", shape = [1, 1], initializer = tf.contrib.layers.xavier_initializer())
W_1 = tf.nn.dropout(W_1, keep_prob = keep_prob)
b_1 = tf.Variable(tf.random_normal([1]))
y_1 = tf.nn.relu(tf.matmul(x_class, W_1) + b_1)

#sex node
W_2 = tf.get_variable(name = "W_2", shape = [1, 1], initializer = tf.contrib.layers.xavier_initializer())
W_2 = tf.nn.dropout(W_2, keep_prob = keep_prob)
b_2 = tf.Variable(tf.random_normal([1]))
y_2 = tf.nn.relu(tf.matmul(x_sex, W_2) + b_2)

#age node
W_3 = tf.get_variable(name = "W_3", shape = [1, 1], initializer = tf.contrib.layers.xavier_initializer())
W_3 = tf.nn.dropout(W_3, keep_prob = keep_prob)
b_3 = tf.Variable(tf.random_normal([1]))
y_3 = tf.nn.relu(tf.matmul(x_age, W_3) + b_3)
'''
y1 = y_1 * y_2
y2 = y_2 * y_3
y3 = y_3 * y_1

W_4 = tf.get_variable(name = "W_4", shape = [891, 1], initializer = tf.contrib.layers.xavier_initializer())
W_4 = tf.nn.dropout(W_4, keep_prob = keep_prob)
b_4 = tf.Variable(tf.random_normal([1]))
y_4 = tf.nn.sigmoid(tf.matmul(y1, W_4) + b_4)

W_5 = tf.get_variable(name = "W_5", shape = [891, 1], initializer = tf.contrib.layers.xavier_initializer())
W_5 = tf.nn.dropout(W_5, keep_prob = keep_prob)
b_5 = tf.Variable(tf.random_normal([1]))
y_5 = tf.nn.sigmoid(tf.matmul(y2, W_5) + b_5)

W_6 = tf.get_variable(name = "W_6", shape = [891, 1], initializer = tf.contrib.layers.xavier_initializer())
W_6 = tf.nn.dropout(W_6, keep_prob = keep_prob)
b_6 = tf.Variable(tf.random_normal([1]))
y_6 = tf.nn.sigmoid(tf.matmul(y3, W_6) + b_6)

'''
cost_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_1, labels=label))
cost_sex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_2, labels=label))
cost_age = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_3, labels=label))

#cost_calss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = hypothesis, labels=label))
#cost = tf.reduce_mean( tf.reduce_sum( (label * tf.log(hypothesis) + (1 - label) * tf.log(1 - hypothesis) )))
train_class = tf.train.GradientDescentOptimizer(0.00001).minimize(cost_class)
train_sex = tf.train.GradientDescentOptimizer(0.00001).minimize(cost_sex)
train_age = tf.train.GradientDescentOptimizer(0.00001).minimize(cost_age)

predicted_class = tf.equal(tf.argmax(y_1,1),tf.argmax(label,1))
accuracy_class = tf.reduce_mean(tf.cast(predicted_class, tf.float32))

predicted_sex = tf.equal(tf.argmax(y_2,1),tf.argmax(label,1))
accuracy_sex = tf.reduce_mean(tf.cast(predicted_sex, tf.float32))

predicted_age = tf.equal(tf.argmax(y_3,1),tf.argmax(label,1))
accuracy_age = tf.reduce_mean(tf.cast(predicted_age, tf.float32))

#predicted = tf.equal(tf.argmax(hypothesis,1),tf.argmax(label,1))
#accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))


#session생성
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print( sess.run(cost, feed_dict = { x_data : train_PassengerData, label:train_Id_survived }))


for i in range(3000):
    cost_val, ts = sess.run([cost_class, train_class] , feed_dict={ x_class: x_calss_data, label: train_Id_survived, keep_prob : 0.7}) 
    cost_val, ts = sess.run([cost_sex, train_sex], feed_dict={ x_sex:x_sex_data, label: train_Id_survived, keep_prob : 0.7}) 
    cost_val, ts = sess.run([cost_age, train_age], feed_dict={ x_age:x_age_data, label: train_Id_survived, keep_prob : 0.7}) 


a_class= sess.run(y_1, feed_dict={x_class: x_calss_Tdata, keep_prob : 1})
a_sex = sess.run(y_2, feed_dict={x_sex: x_sex_Tdata , keep_prob : 1})
a_age = sess.run(y_3, feed_dict={x_age: x_age_Tdata, keep_prob : 1})

for i in range (a_class.shape[0]):
      print("class : ",a_class[i],"sex : ",a_sex[i],"age : ",a_age[i])

#b = sess.run(accuracy, feed_dict={x_data: test_PassengerData , label:test_Id_survived, keep_prob : 1} )
#print("\nAccuracy:",b)

sess.close()


#plt.plot(train_PassengerData[:,[1]], train_PassengerData[:,[9]])
#plt.show()