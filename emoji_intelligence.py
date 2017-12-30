import cv2
import numpy as np
import glob
from keras.utils import np_utils
emojis = []
y=[]

#preparing dataset
def prepare_dataset(path,label):
    for img in glob.glob(path):
        n= cv2.imread(img,0)
        ret,n= cv2.threshold(n,127,1,cv2.THRESH_BINARY_INV) #making only the design pixels visible(1) rest set to zero
        n=np.ravel(n) #converting a 2-D vector to a 1-D vector 
        emojis.append(n)
        y.append(label)

emoji_1_path="C:/Users/sony/Desktop/emoji/1-expressionless/*.png"
prepare_dataset(emoji_1_path,1)

emoji_2_path="C:/Users/sony/Desktop/emoji/2-smile/*.png"
prepare_dataset(emoji_2_path,2)

emoji_3_path="C:/Users/sony/Desktop/emoji/3-worried/*.png"
prepare_dataset(emoji_3_path,3)

X_train=np.asarray(emojis)# list to numpy array

#For ex. y will look like as shown below
#y=[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3]

y_train=np_utils.to_categorical(y)#converting y to categorical data
y_train=y_train[:,1:]

from keras.models import Sequential
from keras.layers import Dense

#Training a neural network with 2 hidden layers

#Input is an array of 20x20 pixels 
#Hidden layer1 =25 units
#Hidden layer2 =16 units
#output layer is of 3 units for now detecting 3 emotions 


classifier=Sequential()

classifier.add(Dense(units=25,kernel_initializer="uniform",activation="relu",input_dim=400))

classifier.add(Dense(units=16,kernel_initializer="uniform",activation="relu")) 

classifier.add(Dense(units=3,kernel_initializer="uniform",activation="sigmoid"))

classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

classifier.fit(X_train,y_train,batch_size=2,epochs=1000)


#Testing

import matplotlib.pyplot as plt

test_set_location="C:/Users/sony/Desktop/emoji/test/*.png"

for img in glob.glob(test_set_location):
    
    input()
    X_test= cv2.imread(img,0)
    
    plt.axis("off")
    plt.imshow(X_test)
    plt.show()
    
    ret,X_test= cv2.threshold(X_test,127,1,cv2.THRESH_BINARY_INV)
    X_test=np.ravel(X_test)
    X_test=np.array([X_test])
    X_test.transpose()
    
    #predicting the probability of outcomes(emotions)
    y_pred=classifier.predict(X_test)
    
    #selecting one with highest probability
    pos=np.argmax(y_pred)
    
    classes=[":expressionless:",":smile:",":worried:"]
    
    emoji_cat=classes[pos]
    from pyemojify import emojify
    text = emojify("Life is "+emoji_cat)
    print(text)
    