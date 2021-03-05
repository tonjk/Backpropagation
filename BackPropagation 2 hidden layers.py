import math
import numpy as np

def summation(input, weight):
    sum = 0
    for i in range(len(input)):
        sum += input[i]*weight[i]
    sum += weight[len(input)]
    return sum

def sigmoid(x):
    v = 1/(1+math.e**(-x))
    return v

# Three nodes
def y_hidden(x,wa,wb):
    ya = sigmoid(summation(x,wa))
    yb = sigmoid(summation(x,wb))
    yh = [ya, yb]
    return yh

# Two nodes
def y_hidden_2(yh,wa2,wb2):
    ya2 = sigmoid(summation(yh,wa2))
    yb2 = sigmoid(summation(yh,wb2))
    yh2 = [ya2, yb2]
    return yh2

def feed_forward(x,wh11,wh12,wh21,wh22,wo):
    yh = y_hidden(x,wh11,wh12)
    yh2 = y_hidden_2(yh,wh21,wh22)
    yo = sigmoid(summation(yh2,wo))
    return yo

# yp = y_predict
def error(desire,yp):
    error = np.array(desire) - np.array(yp)
    return error

def first_de(y):
    fd = y*(1-y)
    return fd

def gradient_output(error,y):
    go = error*first_de(y)
    return go
    
# two output nodes
# def gradient_output(error,y):
#     go = []
#     for i in range(len(error)):
#         g = error[i]*first_de(y[i])
#         go.append(g)
#     return go   

def gradient_hidden_2(yh):
    gh2 = []
    for i in range(len(yh)):
        g = first_de(yh[i])*go*w8[i]
        gh2.append(g)
    return gh2

def gradient_hidden(yh):
    gh = []
    for i in range(len(yh)):
        g = first_de(yh[i])*(gh2[0]*w6[i] + gh2[1]*w7[i])
        gh.append(g)
    return gh

def new_weight_output(last_w,LR,go,yh):
    new = []
    for i in range(len(yh)):
        x = last_w[i] + LR*go*yh[i]*(-1)
        new.append(x)
    y = last_w[len(yh)] + LR*go*(-1)
    new.append(y)
    return new

def new_weight_hidden(wa,wb,LR,gh,x_input):
    last_w = [wa,wb]
    new_all = []
    for i in range(len(gh)):
        new_hidden = []
        for j in range(len(x_input)):
            x = last_w[i][j] + LR*gh[i]*x_input[j]*(-1)
            new_hidden.append(x)
        y = last_w[i][len(x_input)] + LR*gh[i]*(-1)
        new_hidden.append(y)
        new_all.append(new_hidden)
    return new_all
'''--------------------------------------------------------------'''

#w = [w1,w2,w3,b]
x = [1,0,1]
w4 = [0.2,0.4,-0.5,-0.4]
w5 = [-0.3,0.1,0.2,0.2]
w6 = [-0.3,0.3,0.1]
w7 = [-0.2,-0.2,0.1]
w8 = [0.1,0.2,0.3]
LR = -0.9
desire = 1
print('Feed forward:',feed_forward(x,w4,w5,w6,w7,w8))


''' run loop'''
for i in range(10000):
    yh = y_hidden(x,w4,w5)
    yh2 = y_hidden_2(yh,w6,w7)
    go = gradient_output(error(desire,feed_forward(x,w4,w5,w6,w7,w8)),feed_forward(x,w4,w5,w6,w7,w8))
    gh2 = gradient_hidden_2(yh2)
    gh = gradient_hidden(y_hidden(x,w4,w5))
    new_weight_h2 = new_weight_hidden(w6,w7,LR,gh2,yh)
    new_weight_h1 = new_weight_hidden(w4,w5,LR,gh,x)
    
    # Update Weights Output Node
    w8 = new_weight_output(w8,LR,go,yh2)
    # Update weight hidden layers
    w6,w7 = new_weight_h2
    w4,w5 = new_weight_h1
    print('Iteration',i+1,'output :',feed_forward(x,w4,w5,w6,w7,w8))




''''Testing'''
# yh = y_hidden(x,w4,w5)
# yh2 = y_hidden_2(yh,w6,w7)
# go = gradient_output(error(1,feed_forward(x,w4,w5,w6,w7,w8)),feed_forward(x,w4,w5,w6,w7,w8))
# gh2 = gradient_hidden_2(yh2)
# gh = gradient_hidden(y_hidden(x,w4,w5))
# new_weight_h2 = new_weight_hidden(w6,w7,LR,gh2,yh)
# new_weight_h1 = new_weight_hidden(w4,w5,LR,gh,x)

# neww8 = new_weight_output(w8,LR,go,yh2) #update weight output first!!!
# print(neww8)
# neww6 = new_weight_h2[0]
# print(neww6)
# neww7 = new_weight_h2[1]
# print(neww7)
# neww4 = new_weight_h1[0]
# print(neww4)
# neww5 = new_weight_h1[1]
# print(neww5)
# print('Update weight:',feed_forward(x,neww4,neww5,neww6,neww7,neww8))

