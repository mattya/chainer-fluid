

import numpy as np
import chainer
from chainer import cuda, Function, FunctionSet, Variable
import chainer.functions as F

import matplotlib.pyplot as plt
from matplotlib import animation


from chainer.functions.nonparameterized_convolution_2d import convolution_2d

class Laplacian:
    def __init__(self):
        self.W = Variable(xp.asarray([[[[0,1,0],[1,-4,1],[0,1,0]]]],
                                dtype=np.float32), volatile=True)
        self.b = Variable(xp.zeros((1,), dtype=np.float32), volatile=True)
    def forward(self, x):
        return convolution_2d(x, self.W, self.b, pad=1)


class FTCS_X:
    def __init__(self):
        self.model0 = FunctionSet(l=F.Convolution2D(1, 1, 3, pad=1, nobias=True))
        self.model0.l.W[0,0,:,:] = np.array([[0,0,0],[-1,0,1],[0,0,0]]).astype(np.float32)/2
        self.model0.to_gpu()

    def forward(self, x_data):
        y0 = self.model0.l(x_data)
        return y0
class FTCS_Y:
    def __init__(self):
        self.model0 = FunctionSet(l=F.Convolution2D(1, 1, 3, pad=1, nobias=True))
        self.model0.l.W[0,0,:,:] = np.array([[0,-1,0],[0,0,0],[0,1,0]]).astype(np.float32)/2
        self.model0.to_gpu()

    def forward(self, x_data):
        y0 = self.model0.l(x_data)
        return y0

# Kawamura-Kuwahara method
class Kawamura_X:
    def __init__(self):
        self.model0 = FunctionSet(l=F.Convolution2D(1, 1, 5, stride=1, pad=2, nobias=True))
        self.model1 = FunctionSet(l=F.Convolution2D(1, 1, 5, stride=1, pad=2, nobias=True))
       # print self.model.l.W.shape
        self.model0.l.W[0,0,:,:] = np.array([[0,0,0,0,0],[0,0,0,0,0],[2,-12,6,4,0],[0,0,0,0,0],[0,0,0,0,0]]).astype(np.float32)/12.0
        self.model1.l.W[0,0,:,:] = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-4,-6,12,-2],[0,0,0,0,0],[0,0,0,0,0]]).astype(np.float32)/12.0

        #print self.model.l.W.shape
        self.model0.to_gpu()
        self.model1.to_gpu()

    def forward(self, x_data):
        y0 = self.model0.l(x_data)
        y1 = self.model1.l(x_data)
        return y0,y1
class Kawamura_Y:
    def __init__(self):
        self.model0 = FunctionSet(l=F.Convolution2D(1, 1, 5, stride=1, pad=2, nobias=True))
        self.model1 = FunctionSet(l=F.Convolution2D(1, 1, 5, stride=1, pad=2, nobias=True))
        #print self.model.l.W.shape

        self.model0.l.W[0,0,:,:] = np.array([[0,0,2,0,0],[0,0,-12,0,0],[0,0,6,0,0],[0,0,4,0,0],[0,0,0,0,0]]).astype(np.float32)/12.0
        self.model1.l.W[0,0,:,:] = np.array([[0,0,0,0,0],[0,0,-4,0,0],[0,0,-6,0,0],[0,0,12,0,0],[0,0,-2,0,0]]).astype(np.float32)/12.0
        #print self.model.l.W.shape
        self.model0.to_gpu()
        self.model1.to_gpu()

    def forward(self, x_data):
        y0 = self.model0.l(x_data)
        y1 = self.model1.l(x_data)
        return y0,y1

class Upwind(Function):
    def forward_gpu(self, x):
        df0 = x[0]
        df1 = x[1]
        v = x[2]
        ret = cuda.elementwise(
            'T df_plus, T df_minus, T v','T ret',
            '''
                ret = v>0?df_plus:df_minus;
            ''','upwind')(df0,df1,v)
        return ret,
    def backward_gpu(self, x, gy):
        df0 = x[0]
        df1 = x[1]
        v = x[2]
        gx0,gx1 = cuda.elementwise(
            'T df_plus, T df_minus, T v, T g','T gx0, T gx1',
            '''
                if(v>0){
                    gx0 = g;
                    gx1 = 0;
                }else{
                    gx0 = 0;
                    gx1 = g;
                }
            ''','upwind_b')(df0,df1,v, gy[0])
        return gx0,gx1,None

def upwind(df, v):
    return Upwind()(df[0], df[1], v)


def poisson_jacobi(b, x0=0, num_iter=15, alpha=1.0):
    if type(x0)==int:
        x = Variable(xp.zeros_like(b.data, dtype=np.float32), volatile=True)
    else:
        x = x0
    for i in range(num_iter):
        x -= alpha*(b - 0.25*lap.forward(x))
    return x



xp = cuda.cupy
cuda.get_device(0).use()

lap = Laplacian()

W=500
H=500
nu = 0.00001
dx=1.0
dt=0.001


rho0 = xp.zeros((1,1,H,W), dtype=np.float32)
for i in range(0, W, 20):
    rho0[0,0,i:i+10,:] = 1.0

fig = plt.figure(figsize=(10,10))
im = plt.imshow(rho0.get()[0,0,:,:], vmin=0, vmax=0.00001, cmap=plt.cm.gray)
plt.axis('off')

def sim():
    zeta0 = (np.random.uniform(-10.0, 10.0, (1,1,H,W)).astype(np.float32))
    zeta0 = Variable(chainer.cuda.to_gpu(zeta0.astype(np.float32)), volatile=True)
    for it in range(100):
        zeta0 += 0.1*lap.forward(zeta0)
    zeta = 0.0 + zeta0
    psi = poisson_jacobi(zeta, num_iter=1000)
    rho = Variable(rho0, volatile=True)

    for i in range(10000):
        psi = poisson_jacobi(zeta, x0=psi)

        dpdx = FTCS_X().forward(psi)  # -vy
        dpdy = FTCS_Y().forward(psi)  #  vx
        dzdx = upwind(Kawamura_X().forward(zeta), dpdy)
        dzdy = upwind(Kawamura_Y().forward(zeta), -dpdx)
        lapz = lap.forward(zeta)

        rho_ = rho-0.5*dt*(dpdy*upwind(Kawamura_X().forward(rho), dpdy)-dpdx*upwind(Kawamura_Y().forward(rho), -dpdx) - 0.000*lap.forward(rho))
        rho_.data[0,0,:,0] = rho_.data[0,0,:,499]
        sum_rho = chainer.functions.sum(rho_)
        rho_ = rho_/(xp.zeros_like(rho.data)+sum_rho)

        dzdt = dpdx*dzdy - dpdy*dzdx + nu*lapz
        zeta_ = zeta+0.5*dt * dzdt

        psi = poisson_jacobi(zeta_, x0=psi)

        dpdx = FTCS_X().forward(psi)  # -vy
        dpdy = FTCS_Y().forward(psi)  #  vx
        dzdx = upwind(Kawamura_X().forward(zeta_), dpdy)
        dzdy = upwind(Kawamura_Y().forward(zeta_), -dpdx)
        lapz = lap.forward(zeta_)

        rho = rho - dt*(dpdy*upwind(Kawamura_X().forward(rho_), dpdy)-dpdx*upwind(Kawamura_Y().forward(rho_), -dpdx) - 0.000*lap.forward(rho_))
        rho.data[0,0,:,0] = rho.data[0,0,:,499]
        sum_rho = chainer.functions.sum(rho)
        rho = rho/(xp.zeros_like(rho.data)+sum_rho)

        dzdt = dpdx*dzdy - dpdy*dzdx + nu*lapz
        zeta = zeta + dt * dzdt
        if i%10==0:
            yield zeta, psi, rho, i

def draw(data):
    zeta,psi,rho,i = data[0],data[1],data[2],data[3]
    print i
    im.set_array(rho.data.get()[0,0,:,:])
    return [im]


ani = animation.FuncAnimation(fig, draw, sim, blit=False, interval=1, repeat=False)
plt.show()
