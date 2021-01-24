
import numpy as np
from numpy import exp
np.random.seed(0)

class Linearlayer():
    def __init__(self,n_inp,n_out):
        self.weights = np.random.randn(n_inp,n_out)
        self.bias = np.zeros((1,n_out))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias

class NeuralNet2():
    def __init__(self,n_inp,n_out,alpha):
        self.inp = n_inp
        self.out = n_out
        self.hidd_no1 = 60
        self.hidd_no2 = 80
        self.alpha = alpha
        self.error = 1
        self.layer1 = Linearlayer(n_inp,self.hidd_no1)
        self.layer2 = Linearlayer(self.hidd_no1,self.hidd_no2)
        self.layer3 = Linearlayer(self.hidd_no2,n_out)
        
    def act_fun(self,x):
        return 1/(1+exp(-x))
    def der_act_fun(self,x):
        x = self.act_fun(x)
        return x*(1-x)
    
    def forward(self,input_set):
        self.input_set = input_set
        self.layer1.forward(input_set)
        self.inp_hidden1 = self.act_fun(self.layer1.output)
        self.layer2.forward(self.inp_hidden1)
        self.inp_hidden2 = self.act_fun(self.layer2.output)
        self.layer3.forward(self.inp_hidden2)
        self.fout = self.act_fun(self.layer3.output)
        return self.fout
    def learn(self,input_set,output_set):
        nnout = self.forward(input_set)
#         print("output is - ",nnout)
        self.error = 0
        for i in range(len(output_set)):
            self.error+=(output_set[i]-nnout[0][i])**2
        self.error/=2
#         print("error - ",error)
        self.backpropgatel1(output_set)
        self.backpropgatel2()
        self.backpropgatel3()
    def backpropgatel1(self,output):
        self.errorhid1 = []
        xins = self.inp_hidden2[0]
        yout = self.fout[0]
        yins = self.layer3.output[0]
        for i in range(len(xins)):
            for j in range(len(yout)):
                diff = -1*(output[j]-yout[j])*self.der_act_fun(yins[j])
                chw = diff*xins[i]
#                 print("chw at ",i,j," is",chw)
                self.errorhid1.append(diff)
                self.layer3.weights[i][j]-= self.alpha*chw
        for j in range(len(yout)):
            self.layer3.bias[0][j]-=self.alpha*self.errorhid1[j]
    def backpropgatel2(self):
        self.errorhid2 = []
        for i in range(self.hidd_no1):
            for j in range(self.hidd_no2):
                cng = 0
                for k in range(len(self.layer3.weights[j])):
                    cng += self.errorhid1[k]*self.layer3.weights[j][k]
                diff = cng*self.der_act_fun(self.layer2.output[0][j])
                self.errorhid2.append(diff)
                cng=diff*self.inp_hidden1[0][i]
                self.layer2.weights[i][j]-=self.alpha*cng
        
        for j in range(self.hidd_no2):
            self.layer2.bias[0][j]-=self.alpha*self.errorhid2[j]
            
    def backpropgatel3(self):
        self.errorhid3 = []
        for i in range(self.inp):
            for j in range(self.hidd_no1):
                cng = 0
                for k in range(len(self.layer2.weights[j])):
                    cng+=self.errorhid2[k]*self.layer2.weights[j][k]
                diff = cng*self.der_act_fun(self.layer1.output[0][j])
                self.errorhid3.append(diff)
                cng = diff*self.input_set[i]
                self.layer1.weights[i][j]-=self.alpha*cng
            for j in range(self.hidd_no1):
                self.layer1.bias[0][j]-=self.alpha*self.errorhid3[j]
    def which_class(self):
        lst = list(self.fout[0])
        index = lst.index(max(lst))
        print(index+1)
    def percentage_classes(self):
      lst = list(self.fout[0])
      per = []
      smx = 0
      for e in lst:
        smx+=exp(-e)
      for e in lst:
        pro = exp(-e)/smx
        per.append(pro)
      print(pro)
      return pro


# For image preprocessing
from PIL import Image
from numpy import asarray

image = Image.open('url') # my url was different so ..
image.show()
print(image.size)
new_image = image.resize((100,100)).convert('L')
n_i = asarray(new_image)
print(n_i.shape)
new_image.save('url') # again my url was different so ..



# Prediction

cnn_img = NeuralNet2(100*100,3,0.1)
dataset = [] # Dataset
classes = [] # classes of datasets
for e in dataset:
    # For every image converted into 1-D i.e e is a list
    cnn_img.learn(e,classes)
    print(cnn_img.error)
