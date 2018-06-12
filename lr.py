from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import skimage
#from skimage import viewer as v
from pathlib import Path
start_time = time.time()
#Decide a fixed size for all images
y=400
z=400

#Sigmoid function
def sigmoid(x):
    return(1.0/(1.0+np.exp(-x)))

p=Path('Images')

#Read images from given address
size=len([x for x in p.iterdir()])
k=0
#Allocate memory for mean,variance and histogram matrices
hist=np.zeros((256,size))
mean=np.zeros(size)
var=np.zeros(size)
#Finding the statistics values
for x in p.iterdir():
    try:  
        im=skimage.io.imread(x)
        im=skimage.color.rgb2gray(im)
        im=skimage.transform.resize(im,(y,z))
        for i in range(y):
             for j in range(z):
                 mean[k]=mean[k]+im[i][j]
                 var[k]=var[k]+im[i][j]*im[i][j]
                 hist[int(im[i][j])][k]=hist[int(im[i][j])][k]+1
    except IOError:
        print('Error') 
    mean[k]=mean[k]/(y*z)
    var[k]=var[k]-mean[k]*mean[k]
    var[k]=var[k]/(y*z)
    print('Mean of image',k+1,'is',mean[k],'and variance of image',k+1,'is',var[k])
    k=k+1
    
#Print the time it takes to do all this
print("--- %s seconds ---" % (time.time() - start_time))

#Histogram for means of all images
mid=(min(mean)+max(mean))/2
plt.hist(mean,10,histtype='bar')
plt.axvline(x=mid,color='red')
plt.show()
actual=np.zeros(size)
#Distinction between darker and brighter images
print('The darker images are: ')
for i in range(size):
    if mean[i]<mid:
        print(i+1,end=' ')
print('\nThe brighter images are: ')
for i in range(size):
    if mean[i]>=mid:
        actual[i]=1
        print(i+1,end=' ')

#Basic code for logistic regression
"""
weight_new = weight_old - a*dC/dy
where a is constant called learning rate
dC/dy = summation over all images...(y_i_expected-y_i_calculated)*x_i
where i is a pixel no.
"""
start_time = time.time()
y=300
z=300
gen=100
gener=list()
weight_l=np.random.uniform(0,1,(gen+1,y*z+1))
#LEARNING RATE
a=0.001
res=np.zeros((size,1))
k=0
error_l=np.zeros(gen)
derivative=np.zeros(gen)
image=[list() for x in range(size)]
for x in p.iterdir():
    try:  
        im=skimage.io.imread(x)
        im=skimage.transform.resize(im,(y,z))
        im=skimage.color.rgb2gray(im)
        #No need to reshape the 2d image array into 1d because np.append already does that
        #Add a 1 to the starting of the image array
        #Adding bias
        image[k]=(np.append(np.ones(1),im)).reshape(-1,1)
        
        res[k][0]=np.dot(weight_l[0],image[k])
    except IOError:
        print('Error') 
    scale=StandardScaler()
    image[k]=scale.fit_transform(image[k])
    k=k+1
scale=Normalizer()
#scale=StandardScaler()
res=scale.fit_transform(res)
for k in range(size):
    res[k][0]=sigmoid(res[k][0])
    error_l[0]=error_l[0]-(actual[k]*math.log(res[k][0])+(1-actual[k])*math.log(1-res[k][0]))
#Error in initial round
error_l[0]=error_l[0]/size       

for k in range(0,gen):
    gener.append(k)
    for i in range(y*z+1):
        derivative[k]=0
        for j in range(size):
            derivative[k]=derivative[k]-((actual[j]-res[j])*image[j][i][0])
        weight_l[k+1][i]=weight_l[k][i]-a*derivative[k]
        
    for l in range(size):
        res[l][0]=np.dot(weight_l[k+1],image[l])
    scale=StandardScaler()
    res=scale.fit_transform(res)
    for l in range(size):
        res[l][0]=sigmoid(res[l][0])
        error_l[k]=error_l[k]-(actual[l]*math.log(res[l][0])+(1-actual[l])*math.log(1-res[l][0]))
    #Error in each round
    error_l[k]=error_l[k]/size
    #print(error_lr[k])
plt.scatter(gener,error_l,color='red')
plt.plot(gener,error_l,color='blue')
plt.show()
for m in range(10):
    plt.plot(gener[1:21],weight_l[1:21,m:m+1])
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))

#Logistic regression with regularization
"""We have to reduce time being taken and complexity, therefore we have to reduce some number of weights to zero.
This will decrease the constraints on the model but accuracy will also be reduces by an insignificant amount.
We are adding a term L*summation over all the pixels(weight^2)
where L is a constant called regularization parameter
"""
L=2

start_time = time.time()
gen=100
weight_r=np.random.uniform(0,1,(gen+1,y*z+1))
#LEARNING RATE
a=0.01
res=np.zeros((size,1))
k=0
error_r=np.zeros(gen)
derivative=np.zeros(gen)
image=[list() for x in range(size)]
for x in p.iterdir():
    try:  
        im=skimage.io.imread(x)
        im=skimage.transform.resize(im,(y,z))
        im=skimage.color.rgb2gray(im)
        #No need to reshape the 2d image array into 1d because np.append already does that
        #Add a 1 to the starting of the image array
        #Adding bias
        image[k]=(np.append(np.ones(1),im)).reshape(-1,1)
        
        #res=np.matmul(image,weight)     #This won't work due to memory error, so take a for loop
        res[k][0]=np.dot(weight_r[0],image[k])
    except IOError:
        print('Error') 
    scale=StandardScaler()
    image[k]=scale.fit_transform(image[k])
    k=k+1
scale=Normalizer()
#scale=StandardScaler()
res=scale.fit_transform(res)
for k in range(size):
    res[k][0]=sigmoid(res[k][0])
    error_r[0]=error_r[0]-(actual[k]*math.log(res[k][0])+(1-actual[k])*math.log(1-res[k][0]))
#Error in initial round
error_r[0]=error_r[0]/size

for k in range(0,gen):
    w=0
    for i in range(y*z+1):
        derivative[k]=0
        for j in range(size):
            derivative[k]=derivative[k]-((actual[j]-res[j])*image[j][i][0])
        w=w+(weight_r[k][i]*weight_r[k][i])
        weight_r[k+1][i]=weight_r[k][i]-a*derivative[k]+L*weight_r[k+1][i]/(y*z+1)
        
    #Haath se multiplication
    for l in range(size):
        res[l][0]=np.dot(weight_r[k+1],image[l])
    scale=Normalizer()
    #scale=StandardScaler()
    res=scale.fit_transform(res)
    for l in range(size):
        res[l][0]=sigmoid(res[l][0])
        error_r[k]=error_r[k]-(actual[l]*math.log(res[l][0])+(1-actual[l])*math.log(1-res[l][0]))
    #Error in each round
    error_r[k]=(error_r[k]+L*w/(y*z+1))/size
    #print(error_lr[k])
plt.scatter(gener,error_r,color='red')
plt.plot(gener,error_r,color='blue')
plt.show()
m=0
for m in range(10):
    plt.plot(gener[0:21],weight_r[0:21,m:m+1])
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
