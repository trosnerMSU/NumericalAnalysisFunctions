import sys
import requests
import json
import math
import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt

def SplitWord(word):
    return [char for char in word]

def function1(x):
        return math.cos(x) * x

def function2(x):
        return 1/x

def function3(x):
    return x**2 -2*x - 1

def linefunction(m,x,b):
    return m*x+b

def derivative(x):
    return math.sin(2*x) - math.sin(x) - x * math.cos(x) + x/2

def leastsq(A,b,degree):
    n = A.shape[0]
    if(degree > 1):
        A = np.append(A,np.zeros([len(A),degree-1]),1)
        i = 2
        j = 0
        while(i <= degree):
            while(j < n):
                A[j,i] = A[j,i-1] * A[j,i-1]
                j+=1
            i+=1
            j=0

    V = A.T.dot(A)  # V = A.T * A
    f = A.T.dot(b.T)  # f = A.T * b
    x = np.linalg.solve(V, f)  # V * x = f (Solve for x)

    out = (f"{x[0]} + {x[1]}x")
    if(degree > 1):
        i = 2
        while(i <= degree):
            new = (f" + {x[i]}x{i}")
            out = out + new
            i+=1

    degreeNum = (f"Degree {degree}: ")
    print(degreeNum + out)

    return x #End of def

def fibonacci(n):
    if n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)


def leastsquares(A,b):
    V = A.T.dot(A)                  #V = A.T * A
    print(V)
    f = A.T.dot(b.T)                #f = A.T * b
    print(f)
    x = np.linalg.solve(V,f)        #V * x = f (Solve for x)
    return x

def plot(m,x,b,y):
    n = x.shape[0]
    y1 = np.zeros(x.shape[0])
    x1 = np.zeros(x.shape[0])
    i=0
    while(i < n):
        x1[i] = x[i,1]
        y1[i] = linefunction(m,x1[i],b)
        i+=1
    x1 = np.array([x1])
    y1 = np.array([y1])
    plt.scatter(x1,y1)
    plt.scatter(x1,y)
    plt.show()
    return 0

def newtonmethod(funct3,derivative, x, a, b, p):
    initialx = x
    n = (p + math.log(b-a, 10))/(math.log(2, 10))
    i = 0
    error = 0

    while(i < n):
        x = x - (funct3(x))/(derivative(x))
        error = abs(x - initialx)
        print("x"+str(i+1)+": " + str(x) + "  error at x"+str(i + 1)+" is "+ str(error))
        i += 1


def bisect(funct1, a, b, p):
    n = (p + math.log(b - a, 10)) / math.log(2, 10)
    i = 0
    while (i < n):

        c = (a + b) / 2
        if (funct1(c) == 0): break
        if (funct1(a) * funct1(c) < 0):
            b = c

        else:
            a = c
        i += 1
        print(c)
    end

def lu_pivotFactor(A):

    #Overrides A with an LU matrix and returns.
    #This is done with partial pivoting

    n = A.shape[0]
    pivot = np.arange(0,n)
    for i in range(n-1):
        #pivot
        max_row_index = np.argmax(abs(A[i:n,i])) + i
        pivot[[i, max_row_index]] = pivot[[max_row_index,i]]
        A[[i, max_row_index]] = A[[max_row_index,i]]

        #LU
        for j in range(i+1,n):
            A[j,i] = A[j,i]/A[i,i]
            for k in range(i+1,n):
                A[j,k] -= A[j,i]*A[i,k]

    return [A, pivot]


def jacobi_method(A,b,tol,maxiters):
    num = A.shape[0]
    error = 0.0
    guess = np.ones(np.shape(b))

    iters = 0
    while iters < maxiters:
        for j in range(num):
            g = guess.copy()
            temp_j_row = A[j,:].copy()
            temp_j_row[j] = 0.0
            g[j] = -1 * (1/A[j][j]) * (np.dot(temp_j_row,g)+b[j])
        error = nla.norm(g - guess, np.inf) / nla.norm(g,np.inf)
        if error < tol:
            guess = g
            iters += 1
            break
        else:
            guess = g
            iters += 1

    return guess, iters

def gaus(A, b, tol, maxiters):
    guess = np.ones(np.shape(b))

    iters = 0
    find = 0
    r = len(A[0])

    while(iters<maxiters and find == 0):
        i = 0
        while(i < r):
            temp = b[i]
            j = 0
            while(j < r):
                if(i!=j):
                    temp -= A[i,j]*guess[j]
                j += 1
            k = temp/A[i,i]

            if(abs(x[i] - k)<=tol):
                find = 1
                break
            guess[i] = k
            j += 1
        iters += 1

    return guess, iters

def sor(A, b, omega, tol, maxiters):
    guess = np.ones(np.shape(b))
    n = A.shape[0]
    p = A.shape[1]
    phi = guess[:]
    iters = 0
    residual = np.linalg.norm(np.matmul(A, phi) - b)
    while(residual > maxiters):
        for i in range(n):
            sigma = 0
            for j in range(p):
                if j != i:
                    sigma += A[i][j] * phi[j]
            phi[i] = (1 - omega) * phi[i] + (omega / A[i][j]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A,phi) - b)
        iters += 1

    return phi, iters

def cholesky(A):
    n = A.shape[0]
    i = 0
    if(A[0,0] > 0 and n != 1):
        while i < n:
            j = i + 1
            while j < n:
                A[j, j:] = A[j, j:] - ((A[i, j:] * A[i, j]) / A[i, i])
                j += 1
            A[i,i + 1:] = A[i, i + 1:] / np.sqrt(A[i, i])
            A[i,i] = np.sqrt(A[i,i])
            i += 1

    return np.triu(A)

def forwardSolve(A, b):
    n = A.shape[0]
    p = b.shape[1]
    X = np.zeros(A.shape[0])
    i = 0
    j = 0

    if n != p:
        return print("Array sizes of A and b are incompatible")

    while i < n:
        total = b[0,i]
        while j < n:
            total = total - (A[i,j]*X[j])
            j += 1
        X[i] = total / A[i,i]
        i += 1
        j = 0
    return np.array([X])

def backwardSolve(A,b):
    n = A.shape[0]
    p = b.shape[1]
    X = np.zeros(A.shape[0])
    i = n - 1
    j = n - 1

    if n != p:
        return print("Array sizes of A and b are incompatible")

    while i >= 0:
        total = b[0,i]
        while j >= 0:
            total = total - (A[i,j]*X[j])
            j -= 1
        X[i] = total / A[i,i]
        i -= 1
        j = n - 1

    return np.array([X])

def backwardError(A,x,b):
    E = np.array([b.T - np.dot(A,x.T)])
    error = np.amax(E)
    return E,error

#R = cholesky(A.copy())
#np.set_printoptions(precision=2)
#print("Matrix A: \n", A)
#print("Cholesky Factorization: \n",R)

# Flip R Matrix into a lower triangle and then use forward solving to solve Ax = b
#L = R + R.T - np.diag(np.diag(R))
#L = np.tril(L)
#Y = forwardSolve(L,b)
#print("Y Matrix Using Forward Solve: \n", Y)
#Now use backward substitution by replacing X with B and solve Ax = b
#X = backwardSolve(R,Y)
#print("After Backward Solve: \n", X)

#Backward Error for Matrix
#E, error = backwardError(A,X,b)
#print("Backwards Error Matrix: \n", E)
#print("Backwards Error Norm Value: \n", error)

#C = np.array([7.0,8.0,8.0])
#n = A.shape[0]
#piv = np.arange(0,n)
#max_row_index = np.argmax(abs(A[2:n,2])) + 2
#print(max_row_index)
#print(A.dot(C))
#print(A + C)



# b = np.array([6.0,5.0,3.0])
#
# rbe = []
# itersTotal = []
#
#
#
# x, its = jacobi_method(A,b,0.5e-8, 100)
# rbe.append(max(abs(b-np.dot(A,x)))/max(abs(b)))
# itersTotal.append(its)
#
# x, its = gaus(A,b,0.5e-8, 100)
# rbe.append(max(abs(b-np.dot(A,x)))/max(abs(b)))
# itersTotal.append(its)
#
# x, its = sor(A,b,0.71725, 0.5e-8, 100)
# rbe.append(max(abs(b-np.dot(A,x)))/max(abs(b)))
# itersTotal.append(its)
#
# print (14*" "+" Iterations Rel. B.E.")
# print (40*" -")
# methods = [" Jacobi", "Gauss -Seidel", "SOR"]
# for m,k,e in zip(methods , iters , rbe):
#     print ("{m:12s} {k:^2d} {r:15.8e}". format(m=m, k=k, r=e))


#LU, P = lu_pivotFactor(A)

#print("LU: ", LU)
#print("Pivot: ", P)




#newtonmethod(function3, derivative, 10*3.14, 1, 2, 5)

#bisect(function3,1,3,3)
