from math import *
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

def y_hidden(x,wa,wb):
    ya = sigmoid(summation(x,wa))
    yb = sigmoid(summation(x,wb))
    yh = [ya, yb]
    return yh

def feed_forward(x,wa,wb,wo):
    yo = sigmoid(summation(y_hidden(x,wa,wb),wo))
    return yo
    # return '{0:.3f}'.format(yo)
# yp = y_predict
def error(desire,yp):
    error = desire - yp
    return error

def first_de(y):
    fd = y*(1-y)
    return fd

def gradient_output(error,y):
    go = error*first_de(y)
    return(go)

def gradient_hidden(yh):
    gh = []
    for i in range(len(yh)):
        g = first_de(yh[i])*go*w6[i]
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
w6 = [-0.3,-0.2,0.1]
LR = -0.9

test = feed_forward(x,w4,w5,w6)


go = gradient_output(error(1,test),feed_forward(x,w4,w5,w6))
gh = gradient_hidden(y_hidden(x,w4,w5))
gg = new_weight_hidden(w4,w5,LR,gh,x)




neww6 = new_weight_output(w6,LR,go,y_hidden(x,w4,w5))
neww4 = gg[0]
neww5 = gg[1]
print('Initialize weight:',feed_forward(x,neww4,neww5,neww6))

for i in range(2):
    go = gradient_output(error(1,feed_forward(x,w4,w5,w6)),feed_forward(x,w4,w5,w6))
    gh = gradient_hidden(y_hidden(x,w4,w5))
    gg = new_weight_hidden(w4,w5,LR,gh,x)
    w6 = new_weight_output(w6,LR,go,y_hidden(x,w4,w5))
    w4 = gg[0]
    w5 = gg[1]
    print('Iteration',i+1,'output :',feed_forward(x,w4,w5,w6))



