import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
import pyamg
import sys
from scipy.sparse import diags
from scipy.sparse.linalg import norm, inv
import copy


def print_time(time,name,N,it):
    with open('execution_time.log', 'a') as file:
        file.write(f"Tempo di calcolo: {time:.4f} secondi, Iterazioni: {it}, Metodo: {name}, Elementi: {N} \n")
    print(f"Tempo di calcolo: {time:.4f} secondi")

def run_func(function,N,tmax,l,x_obb,ml,mask1):
    start_time = time.time()
    xT,xA,xQ,func,it=function(N,tmax,l,x_obb,ml,mask1)
    end_time = time.time()
    execution_time = end_time - start_time
    print_time(execution_time,function.__name__,N,it)
    return xT,xA,xQ,func

def solver(N,dim):
    if (dim==2):
        A=np.zeros(((N+1)**2,(N+1)**2))
        np.fill_diagonal(A[:, :], -4/h**2)
        np.fill_diagonal(A[:, 1:], 1/h**2)
        np.fill_diagonal(A[1:, :], 1/h**2)
        np.fill_diagonal(A[:, N+1:], 1/h**2)
        np.fill_diagonal(A[N+1:, :], 1/h**2)

        A[0::N+1,:]=0
        A[N::N+1,:]=0

        np.fill_diagonal(A[0::N+1, 0::N+1], 1)
        np.fill_diagonal(A[N::N+1, N::N+1], 1)

        A[1:N,:]=0
        A[-N:-1,:]=0

        np.fill_diagonal(A[1:N, 1:N], -1/h**2)
        np.fill_diagonal(A[1:N, 1+N+1:N+N+1], 1/h**2)

        np.fill_diagonal(A[-N:-1, -N:-1], -1/h**2)
        np.fill_diagonal(A[-N:-1, -N-N-1:-N-1-1], 1/h**2)

        main_diag   = A.diagonal(0)
        side_diag_1 = A.diagonal(-1)
        side_diag_2 = A.diagonal(1)
        far_diag_1  = A.diagonal(-(N+1))
        far_diag_2  = A.diagonal(N+1)

        A = sp.diags(
            [far_diag_1, side_diag_1, main_diag, side_diag_2, far_diag_2],
            offsets=[-(N+1), -1, 0, 1, (N+1)],
            shape=((N+1)**2, (N+1)**2),
            format='csr'
        )


        # Creazione del solver AMG
        ml  = pyamg.smoothed_aggregation_solver(A, max_coarse=3, strength=('symmetric', {'theta': 0.25}))

        return ml

def mask(xmin,xmax,ymin,ymax):
    mask=np.zeros((N+1,N+1))
    indx_min=round(xmin/h)
    indx_max=round(xmax/h)+1
    indy_min=round(ymin/h)
    indy_max=round(ymax/h)+1
    mask[indy_min:indy_max,indx_min:indx_max]=1
    mask=mask.flatten()
    return mask

def plot(dim,N,plot_name,x,x_grid,y_grid):
    if (dim==2):
        sol=np.zeros((N+1,N+1))
        for i in range(N+1):
            sol[i,:]=x[i*(N+1):(i+1)*(N+1)]
        X, Y = np.meshgrid(x_grid, y_grid)
        plt.figure(figsize=(8,8))
        contourf_plot2= plt.contourf(X, Y, sol, levels=50, cmap='viridis')
        plt.colorbar(label="xT")
        file_name=f"{plot_name}_{N}"
        plt.savefig(file_name, format='pdf')
        plt.figure(figsize=(8,8))
        plt.plot(x_grid,sol[round((N+1)/2),:])
        file_name=f"{plot_name}_mid_{N}"
        plt.savefig(file_name, format='pdf')


    if (dim==0):
        plt.figure(figsize=(8,8))
        for j in range(len(plot_name)):
            plt.plot(x[j],label=f"{plot_name[j]}{N}")
        plt.yscale('log')
        plt.legend(loc='upper left')
        plt.savefig(f"funzionali{N}", format='pdf')

def functional(x,xobb,mask1,xQ):
    err=abs(mask1)*(x-xobb)**2 + l* xQ**2
    norm_1 = np.linalg.norm(err, ord=1)
    return norm_1/(N+1)**2

def steepest_gradient(N,tmax,l,x_obb,ml,mask1):
    xQ=np.zeros((N+1)**2)
    func=np.zeros(1)
    func[0]=1e10
    sol=np.zeros((N+1,N+1))
    err=1e10
    i=1
    beta1=0.9
    beta2=0.999
    eta = 1e4
    m=0
    v=0
    while(func[-1]>toll):
        xT=ml.solve(-xQ, tol=1e-10)
        func=np.append(func,functional(xT,x_obb*mask1,mask1,xQ))
        xA=ml.solve(-(xT-x_obb)*mask1, tol=1e-10)
        xQ=xQ-rho*(xA+xQ*l)
        xQ[0:N+1]=0
        xQ[-N-2:]=0
        xQ[0:N*(N+1):N+1]=0
        xQ[N:N*(N+2):N+1]=0

        err=(func[i]-func[i-1])/func[i]
        if (i%100==0):
            print('it: ',i,' func: ',func[-1])
        i=i+1
        if (abs((func[-1]-func[-2])/func[-1])<1e-6):
            print('convergence reached')
            break
    print('steepest conv at: ',i)
    return xT,xA,xQ,func[1:],i

def conjugate_gradient(N,tmax,l,x_obb,ml,mask1):
    func=np.zeros(1)
    xQ=np.zeros((N+1)**2)
    beta=np.zeros((N+1)**2)
    xT=ml.solve(-xQ, tol=1e-10)
    func[0]=functional(xT,x_obb*mask1,mask1,xQ)
    xA=ml.solve(-(xT-x_obb)*mask1, tol=1e-10)
    g_old=(xA+l*xQ)
    s=-g_old
    err=1e10
    i=1

    while(func[-1]>toll):

        xQ=xQ+1e6*s
        xQ[0:N+1]=0
        xQ[-N-2:]=0
        xQ[0:N*(N+1):N+1]=0
        xQ[N:N*(N+2):N+1]=0
        xT=ml.solve(-xQ, tol=1e-10)
        func=np.append(func,functional(xT,x_obb*mask1,mask1,xQ))
        xA=ml.solve(-(xT-x_obb)*mask1, tol=1e-10)
        g_new=(xA+l*xQ)
        beta=np.dot(g_new,g_new)/np.dot(g_old,g_old)
        g_old=g_new
        s=-g_new+beta*s
        if (i%100==0):
            print('it: ',i,' func: ',func[-1])
        i=i+1

    print('conjugate conv at: ',i)
    return xT,xA,xQ,func,i

def adam_method(N,tmax,l,x_obb,ml,mask1):
    xQ=np.zeros((N+1)**2)
    func=np.zeros(1)
    func[0]=1e10
    sol=np.zeros((N+1,N+1))
    err=1e10
    i=1
    beta1=0.9
    beta2=0.999
    eta = 1e4
    m=0
    v=0
    while(func[-1]>toll):
        xT=ml.solve(-xQ, tol=1e-10)
        func=np.append(func,functional(xT,x_obb*mask1,mask1,xQ))
        xA=ml.solve(-(xT-x_obb)*mask1, tol=1e-10)

        g = -(xA + l*xQ)
        beta1=beta1
        beta2=beta2
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        m_v = m/(1-beta1)
        v_v = v/(1-beta2)

        xQ = xQ + eta*(m_v)/(np.sqrt(v_v)+1e-8)

        xQ[0:N+1]=0
        xQ[-N-2:]=0
        xQ[0:N*(N+1):N+1]=0
        xQ[N:N*(N+2):N+1]=0

        err=(func[i]-func[i-1])/func[i]
        if (i%100==0):
            print('it: ',i,' func: ',func[-1])
        i=i+1
        if (abs((func[-1]-func[-2])/func[-1])<1e-6):
            print('convergence reached')
            break
    print('adam conv at: ',i)
    return xT,xA,xQ,func[1:],i

def newton_method(N,tmax,l,x_obb,ml,mask1):
    order=100000
    buff_order=4
    xA=np.zeros((N+1)**2)
    xQ=np.zeros((N+1)**2)
    record_y=np.zeros((buff_order,(N+1)**2))
    record_s=np.zeros((buff_order,(N+1)**2))
    func=np.ones(1)
    xQ_old=xQ
    xA_old=xA
    err=1e10
    i=0
    beta1=0.9
    beta2=0.999
    eta = 1e5
    m=0
    v=0
    while(func[-1]>toll):
        xT=ml.solve(-xQ, tol=1e-10)
        xA=ml.solve(-(xT-x_obb)*mask1, tol=1e-10)
        if (i<=buff_order-1):
            xQ=xQ-rho*(xA+l*xQ)
            record_s[i,:]=xQ-xQ_old
            record_y[i,:]=(xA-xA_old)+l*(xQ-xQ_old)
        elif (i<=order-1):
            z=newton_direction(record_s,record_y,i,xA,xQ)
            xQ=xQ+rho*z/abs(np.linalg.norm(z))
            record_s=np.vstack([record_s,(xQ-xQ_old)])
            record_y=np.vstack([record_y,(xA-xA_old)+l*(xQ-xQ_old)])
        else:
            z=newton_direction(record_s,record_y,order,xA,xQ)
            z=z/abs(np.linalg.norm(z))
            xQ=xQ+rho*z
            record_s=record_s[1:,:]
            record_y=record_y[1:,:]
            record_s=np.vstack([record_s,(xQ-xQ_old)])
            record_y=np.vstack([record_y,(xA-xA_old)+l*(xQ-xQ_old)])
        xQ_old=xQ
        xA_old=xA

        xQ[0:N+1]=0
        xQ[-N-2:]=0
        xQ[0:N*(N+1):N+1]=0
        xQ[N:N*(N+2):N+1]=0
        func=np.append(func,functional(xT,x_obb*mask1,mask1,xQ))
        if (i%10==0):
            print('it: ',i,' func: ',func[-1])
        i=i+1


    print('newton conv at: ',i)
    return xT,xA,xQ,func,i

def newton_direction(record_s,record_y,m,xA,xQ):
    q=xA+xQ*l
    rho=np.zeros(m)
    alpha=np.zeros(m)
    beta=np.zeros(m)
    for i in range(m-1,-1,-1):
        rho[i]=1/np.dot(record_y[i,:],record_s[i,:])
        s=record_s[i,:]
        alpha[i]=rho[i]*np.dot(s,q)
        q=q-alpha[i]*record_y[i,:]
    gamma=np.dot(record_s[-1,:],record_s[-1,:])/np.dot(record_s[-1,:],record_y[-1,:])
    H=np.eye((N+1)**2)*abs(gamma)
    z=np.dot(H,q)
    for i in range(0,m,1):
        beta[i]=rho[i]*np.dot(record_y[i,:],z)
        z=z+record_s[i,:]*(alpha[i]-beta[i])
    return -z

#user par
multigrid_order_max=5
multigrid_order_min=4
a=0.1
x_obb=1
dim=2
l=0.
toll=1e-6
rho=1e6
tmax=100
methods_tosolve=[conjugate_gradient,adam_method]

#run
for nn in range(multigrid_order_min,multigrid_order_max):
    N=48*nn
    h=a/N
    x=np.linspace(0,a,N+1)
    y=np.linspace(0,a,N+1)

    if(dim==2):
        xmin1=0.02
        xmax1=0.08
        ymin1=0.04
        ymax1=0.06
        mask1=mask(xmin1,xmax1,ymin1,ymax1)
        ml=solver(N,dim)


    number_ofeq=len(methods_tosolve)
    functionals = []
    plot_name=[""] * number_ofeq
    for i in range(number_ofeq):
        xT,xA,xQ,func=run_func(methods_tosolve[i],N,tmax,l,x_obb,ml,mask1)
        plot_name=methods_tosolve[i].__name__
        #plot(dim,N,plot_name,xQ,x,y)
        functionals.append((func, plot_name))

    plot(0,N,[f[1] for f in functionals], [f[0] for f in functionals],x,y)

plt.show()


