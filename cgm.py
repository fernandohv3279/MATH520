import numpy as np

Q=np.array([[5,2],[2,1]])
def grad(vector):
    return np.matmul(Q,vector)-np.array([[3],[1]])

def cgm():
    xk=np.array([[0],[0]])
    gk=grad(xk)
    print(gk)
    dk=-1*gk
    tol=1e-15
    maxiter=100
    for k in range(maxiter):
        print("iter " + str(k))
        ak=-1*(np.matmul(np.transpose(gk),dk))/\
        (np.matmul(np.matmul(np.transpose(dk),Q),dk))
        print("ak")
        print(ak)
        xk1=xk+ak*dk
        print("xk1")
        print(xk1)
        gk1=grad(xk1)
        print("gk1")
        print(gk1)
        if(np.linalg.norm(gk1)<tol):
            print("Found it!")
            print(xk1)
            return
        bk=\
        (np.matmul(np.matmul(np.transpose(gk1),Q),dk))/\
        (np.matmul(np.matmul(np.transpose(dk),Q),dk))
        print("bk")
        print(bk)
        dk1=-1*gk1+bk*dk
        print("dk1")
        print(dk1)
        # update stuff
        xk=xk1
        gk=gk1
        dk=dk1
cgm()
