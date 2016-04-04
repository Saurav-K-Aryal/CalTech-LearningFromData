""" author: Saurva Keshari Aryal
	email: saurav.aryal@bison.howard.edu


	Pythonic solution for HW-1 of Learning from Data, Caltech MOOC
"""

from numpy import array

from random import uniform
from random import randint

from tools import build_training_set
from tools import data
from tools import randomline
from tools import sign
from tools import target_function

def build_misclassified_set(t_set,w):
    '''returns a tuple of index of t_set items
    such that t_set[index] is misclassified <=> yn != sign(w*point)'''
    res = tuple()
            
    for i in range(len(t_set)):
        point = t_set[i][0]
        s = h(w,point)
        yn = t_set[i][1]
        if s != yn:
            res = res + (i,)
    return res

def h(w,x):
    'Hypothesis function returns w0 x0 + w1 x1 ... + wn xn'
    res = 0
    for i in range(len(x)):
        res = res + w[i]*x[i]
    return sign(res)

def PLA(N_points,w,f,t_set):
    ''' 
    - t_set: item of t_set is: [[vector_x], y]
    - w: vector of same dimention as vector_x of weights
    - iteration: Number of iterations needed for convergence
    - f: target lambda function f
    
    Perceptron Algorithm:
    - pick a misclasified point from misclassified set
    - if there are no misclassified points break iteration weight are ok. break iteration.
    - if there is a misclassified point update weights
    '''
    N = N_points
    iteration = 0
    iterate = True

    while iterate:
        iteration = iteration + 1        
        misclassified_set = build_misclassified_set(t_set,w)
        if len(misclassified_set)==0 : break
        index = randint(0,len(misclassified_set)-1)
        j = misclassified_set[index]
        point = t_set[j][0]

        s = h(w,point)
        yn = t_set[j][1]

        if s != yn:
            xn = point
            w[0] = w[0] + yn*xn[0]
            w[1] = w[1] + yn*xn[1]
            w[2] = w[2] + yn*xn[2]
    return w,iteration

def evaluate_diff_f_g(f,w):
    'Returns the average of difference between f and g (g is equivalent as vector w )'
    count = 0
    limit = 10000
    diff = 0
    # generate random point as out of sample data
    # check result and count if there is a difference
    # between target function f and hypothesis function g
    while count < limit:
        count = count + 1
        x = uniform(-1,1)
        y = uniform(-1,1)
        vector = [1,x,y]

        sign_f = sign(f(x),y)
        sign_g = h(w,vector)

        if sign_f != sign_g: diff = diff + 1

    return diff/(count*1.0)
    
def run_PLA(N_samples,N_points):
    samples = []# vector of 1 clasified, 0 misclassified
    iterations = []#vector of iterations needed for each PLA
    b_misclassified = False
    diff = []#vector of difference average between f and g

    for i in range(N_samples):
        # run PLA in sample
        d = data(N_points)
        l = randomline()
        f = target_function(l)
        t_set = build_training_set(d,f)
        w = [0,0,0]

        w,iteration = PLA(N_points,w,f,t_set)

        iterations.append(iteration)
        # check if points are classified or not
        for i in range(len(t_set)):
            point = t_set[i][0]
            s = h(w,point)
            yn = t_set[i][1]
            if yn != s:
                samples.append(0)
                b_misclassified = True
                break

        # check difference between f and g
        diff.append(evaluate_diff_f_g(f,w))
        if not b_misclassified: samples.append(1)

        b_misclassified = False

    print 'number of samples misclassified: %s ' % samples.count(0)
    print 'number of classified samples: %s ' % samples.count(1)
    print 'number of iteration avg: %s ' % (str(sum(iterations)/len(iterations)*1.0))
    print 'average of difference in function g: %s' % ( sum(diff)/(len(diff)*1.0) )


# Calling functions
run_PLA(1000,10)
print "10 done\n"
run_PLA(1000,100)