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

def y_hidden(x,wa,wb):
    ya = sigmoid(summation(x,wa))
    yb = sigmoid(summation(x,wb))
    yh = np.array([ya, yb])
    return yh

def feed_forward(x,wa,wb,wo1,wo2):
    yo1 = sigmoid(summation(y_hidden(x,wa,wb),wo1))
    yo2 = sigmoid(summation(y_hidden(x,wa,wb),wo2))
    yo = [yo1,yo2]
    return yo
    # return '{0:.3f}'.format(yo)
# yp = y_predict
def error(desire,yp):
    error = np.array(desire) - np.array(yp)
    return error
    # sum_error = 0
    # for i in range(len(error)):
    #     sum_error += error[i]*error[i]
    # sum_error = sum_error/2
    # return sum_error

def first_de(y):
    fd = y*(1-y)
    return fd

def gradient_output(error,y):
    go = []
    for i in range(len(error)):
        g = error[i]*first_de(y[i])
        go.append(g)
    return go    

def gradient_hidden(yh):
    gh = []
    for i in range(len(yh)):
        g = first_de(yh[i])*(go[0]*w6[i] + go[1]*w7[i])
        gh.append(g)
    return gh

def new_weight_output(last_w1,last_w2,LR,go,yh):
    last_w = [last_w1,last_w2]
    new = []
    for h in range(len(last_w)):
        new_w = []
        for i in range(len(go)):
            x = last_w[h][i] + LR*go[h]*yh[i]*(-1)
            new_w.append(x)
        y = last_w[h][len(go)] + LR*go[h]*(-1)
        new_w.append(y)
        new.append(new_w)
    return new[0],new[1]

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
w7 = [0.3,-0.2,0.1]
LR = -0.9
y_true = [1,0]

test = feed_forward(x,w4,w5,w6,w7)
print('feed',test)


# go = gradient_output(error(y_true,feed_forward(x,w4,w5,w6,w7)),feed_forward(x,w4,w5,w6,w7))
# gh = gradient_hidden(y_hidden(x,w4,w5))
# gg = new_weight_hidden(w4,w5,LR,gh,x)

# neww6,neww7 = new_weight_output(w6,w7,LR,go,y_hidden(x,w4,w5))
# neww4 = gg[0]
# neww5 = gg[1]
# print(neww4)
# print(neww5)
# print(neww6)
# print(neww7)
# print('First Iteration:',feed_forward(x,neww4,neww5,neww6,neww7))


for i in range(1):
    go = gradient_output(error(y_true,feed_forward(x,w4,w5,w6,w7)),feed_forward(x,w4,w5,w6,w7))
    gh = gradient_hidden(y_hidden(x,w4,w5))
    gg = new_weight_hidden(w4,w5,LR,gh,x)
    w6,w7 = new_weight_output(w6,w7,LR,go,y_hidden(x,w4,w5))  #update weight output first!!!
    w4,w5 = gg  
    print('Iteration',i+1,'output :',feed_forward(x,w4,w5,w6,w7))
    

