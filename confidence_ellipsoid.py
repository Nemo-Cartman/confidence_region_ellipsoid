# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:04:31 2019

@author: 39105
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import f
#show ellipsoid at initial position(no rotation, no shift)
def ellipsoid(beta,mean,axes,u,*kargs):
    a=kargs[0]
    b=kargs[1]
    c=kargs[2]
    rx=axes[a]
    ry=axes[b]
    rz=axes[c]
    center=(np.vstack([mean[a],mean[b],mean[c]])).reshape(3,)
    center=center-center#standard ellipsoid
    #print(center)
    rotation=np.vstack([np.hstack([u[a,a],u[a,b],u[a,c]]),np.hstack([u[b,a],u[b,b],u[b,c]]),np.hstack([u[c,a],u[c,b],u[c,c]])])
    r_square=(rotation*rotation).sum(axis=0)
    rotation=rotation/np.sqrt(r_square)
    #
    theta1 = np.linspace(0, 2 * np.pi, 100)
    theta2 = np.linspace(0, np.pi, 100)
    x=rx*np.outer(np.cos(theta1), np.sin(theta2))
    y=ry*np.outer(np.sin(theta1), np.sin(theta2))
    z=rz*np.outer(np.ones_like(theta1), np.cos(theta2))
    for i in range(len(x)):
        for j in range(len(x)):
            #[x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]],rotation)+center
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]],np.diag([1,1,1]))+center
    #show
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=4,cstride=4,color='white')
    ax.set_xlim3d(-beta+center[0], beta+center[0])
    ax.set_ylim3d(-beta+center[1], beta+center[1])
    ax.set_zlim3d(-beta+center[2], beta+center[2])
    return(rotation)
    
def SVD(a):
    u,sigma,vt=np.linalg.svd(a)
    return u,sigma,vt
    #u@sigma@v-a=0, sigma needed to be diaged
    #for variance, sigma has actually been squared
    
    
    
def prt_axp(i):
    w_end=axis_point(i)
    print('where the %d-th farthest w point lies:'%(i+1))
    print(((w_end@u.T)+mean).reshape(4,4))

#input: W(s); Data Structure: narray of numpy
#data=np.loadtxt(r'C:\Users\39105\Downloads\matrix_原始数据9人 - 2.txt')
with open(r'C:\Users\39105\Downloads\matrix_原始数据9人_3.txt') as file:
    content=file.read()
    content=str(content)
    content=content.replace(r'[','')
    content=content.replace(r']', '')
    content=content.replace(r',', '')
    #content=content.replace(r'\n', '')
with open(r'C:\Users\39105\Downloads\matrix_w.txt','w') as file:
    file.write(content)
data=np.loadtxt(r'C:\Users\39105\Downloads\matrix_w.txt')
data=data.reshape(37,4,4)
vectors=np.array([e.flatten('F') for e in data[:] ])
#Vec(W)
n=data.shape[0]#samples
mean=vectors.sum(axis=0)/n
func=lambda x,y:x.reshape(x.shape[0],1)*y.T#matrix alignment minus, 
variance=np.array([func(e,e) for e in (vectors-mean)[:]])
variance=variance.sum(axis=0)/n
p=vectors.shape[1]#dimension
alpha=0.05
#n=284
print('n:',n)
F=f.ppf(1-alpha,p,n-p)
k_square=p*(n-1)*F/(n*(n-p))
#SVD by linalg
u,sigma,vt=SVD(variance)
comparasion=data[-1].flatten('F')
comparasion_initial=(comparasion-mean)@u
axes=np.sqrt(k_square)*np.array([np.sqrt(sigma[i])*u[i] for i in range(sigma.shape[0])])
#mean as center, axes as major/minor axis（semi-）
length=np.sqrt(k_square)*np.sqrt(sigma)
print('sigma.max():',sigma.max())
print('length.max():',length.max())

#test

axis_point=lambda x:np.hstack([[0]*x,length[x],np.array([0]*(15-x))])
c_t=np.abs(comparasion_initial)
print('to reject apparant outside status(exist difference>0):',np.abs(comparasion_initial)-length)
w_end=np.hstack([length.max(),np.array([0]*15)])
print('where the farthest w point lies:')
print(((w_end@u.T)+mean).reshape(4,4))
w_end=np.hstack([-length.max(),np.array([0]*15)])
print('where the farthest w point lies:')
print(((w_end@u.T)+mean).reshape(4,4))
w_end=axis_point(1)
print('where the second farthest w point lies:')
print(((w_end@u.T)+mean).reshape(4,4))
w_end=axis_point(2)
print('where the %d-rd farthest w point lies:'%(2+1))
print(((w_end@u.T)+mean).reshape(4,4))
for i in range(3,16):
    prt_axp(i)



#
[i,j,k]=[0,1,2]
beta=length.max()
r=ellipsoid(beta,mean,length,u,*[i,j,k])
plt.title('3 major')

#[i,j,k]=[0,9,1]
#ellipsoid(mean,length,u,*[i,j,k])
#plt.title('1 major, 2 minor')
#
#[i,j,k]=[0,6,1]
#ellipsoid(mean,length,u,*[i,j,k])
#plt.title('1 major, 2 minor(pure inside)')
#
#[i,j,k]=[0,10,7]
#ellipsoid(mean,length,u,*[i,j,k])
#plt.title('1 major, 2 minor(pure outside)')
