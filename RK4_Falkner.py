print('FALKNER-SKAN ODE SHOOTING METHOD:')
print('Runge-Kutta 4th Order Approximation')
print('\nm= 0.05, beta= 2/3')
print('f\'"+(5/6)ff" -(2/3)(f\')^2+2/3=0')

from numpy import *
from numpy.linalg import inv
import matplotlib.pyplot as plt

F=zeros((10001,4),float)
F[0,1]=0.000
F[0,2]=0.000
F[0,3]=1.000

h=0.001
for i in range (1,10001):
    F[i,0]=h*i

for i in range (1,10001):
    g1=F[i-1,2]
    h1=F[i-1,3]
    j1=-5/6*F[i-1,1]*F[i-1,3]+2/3*(F[i-1,2])**2-2/3
    
    g2=F[i-1,2]+h/2*g1
    h2=F[i-1,3]+h/2*h1
    j2=-5/6*(F[i-1,1]+h/2*g1)*(F[i-1,3]+h/2*j1)+2/3*(F[i-1,2]+h/2*h1)**2-2/3

    g3=F[i-1,2]+h/2*g2
    h3=F[i-1,3]+h/2*h2
    j3=-5/6*(F[i-1,1]+h/2*g2)*(F[i-1,3]+h/2*j2)+2/3*(F[i-1,2]+h/2*h2)**2-2/3

    g4=F[i-1,2]+h*g3
    h4=F[i-1,3]+h*h3
    j4=-5/6*(F[i-1,1]+h*g3)*(F[i-1,3]+h*j3)+2/3*(F[i-1,2]+h*h3)**2-2/3

    F[i,1]=F[i-1,1]+h/6*(g1+2*g2+2*g3+g4)
    F[i,2]=F[i-1,2]+h/6*(h1+2*h2+2*h3+h4)
    F[i,3]=F[i-1,3]+h/6*(j1+2*j2+2*j3+j4)


G=zeros((10001,4),float)
G[0,1]=0.000
G[0,2]=0.000
G[0,3]=1.02204

h=0.001
for i in range (1,10001):
    G[i,0]=h*i

for i in range (1,10001):
    g1=G[i-1,2]
    h1=G[i-1,3]
    j1=-5/6*G[i-1,1]*G[i-1,3]+2/3*(G[i-1,2])**2-2/3
    
    g2=G[i-1,2]+h/2*g1
    h2=G[i-1,3]+h/2*h1
    j2=-5/6*(G[i-1,1]+h/2*g1)*(G[i-1,3]+h/2*j1)+2/3*(G[i-1,2]+h/2*h1)**2-2/3

    g3=G[i-1,2]+h/2*g2
    h3=G[i-1,3]+h/2*h2
    j3=-5/6*(G[i-1,1]+h/2*g2)*(G[i-1,3]+h/2*j2)+2/3*(G[i-1,2]+h/2*h2)**2-2/3

    g4=G[i-1,2]+h*g3
    h4=G[i-1,3]+h*h3
    j4=-5/6*(G[i-1,1]+h*g3)*(G[i-1,3]+h*j3)+2/3*(G[i-1,2]+h*h3)**2-2/3

    G[i,1]=G[i-1,1]+h/6*(g1+2*g2+2*g3+g4)
    G[i,2]=G[i-1,2]+h/6*(h1+2*h2+2*h3+h4)
    G[i,3]=G[i-1,3]+h/6*(j1+2*j2+2*j3+j4)


H=zeros((10001,4),float)
H[0,1]=0.000
H[0,2]=0.000
H[0,3]=1.050

h=0.001
for i in range (1,10001):
    H[i,0]=h*i

for i in range (1,10001):
    g1=H[i-1,2]
    h1=H[i-1,3]
    j1=-5/6*H[i-1,1]*H[i-1,3]+2/3*(H[i-1,2])**2-2/3
    
    g2=H[i-1,2]+h/2*g1
    h2=H[i-1,3]+h/2*h1
    j2=-5/6*(H[i-1,1]+h/2*g1)*(H[i-1,3]+h/2*j1)+2/3*(H[i-1,2]+h/2*h1)**2-2/3

    g3=H[i-1,2]+h/2*g2
    h3=H[i-1,3]+h/2*h2
    j3=-5/6*(H[i-1,1]+h/2*g2)*(H[i-1,3]+h/2*j2)+2/3*(H[i-1,2]+h/2*h2)**2-2/3

    g4=H[i-1,2]+h*g3
    h4=H[i-1,3]+h*h3
    j4=-5/6*(H[i-1,1]+h*g3)*(H[i-1,3]+h*j3)+2/3*(H[i-1,2]+h*h3)**2-2/3

    H[i,1]=H[i-1,1]+h/6*(g1+2*g2+2*g3+g4)
    H[i,2]=H[i-1,2]+h/6*(h1+2*h2+2*h3+h4)
    H[i,3]=H[i-1,3]+h/6*(j1+2*j2+2*j3+j4)

print('\nValues for eta= 1.02204: ')
print('eta        f(x)            f\'(x)      f"(x)')
for i in range (0,21):
    eeta=G[500*i,0]
    ffx=G[500*i,1]
    ffp=G[500*i,2]
    ffd=G[500*i,3]
    print('{:3.2f}        {:3.4f}         {:3.4f}       {:3.4f}'.format(eeta,ffx,ffp,ffd))


etaa=transpose(F[0:10000,0])
fpp1000=transpose(F[0:10000,2])
fpp1022=transpose(G[0:10000,2])
fpp1100=transpose(H[0:10000,2])
plt.plot(etaa, fpp1000,'g')
plt.plot(etaa, fpp1022,'b')
plt.plot(etaa, fpp1100,'r')
plt.xlabel('n')
plt.ylabel('df/dn')
plt.legend(('f"(0)=1.000','f"(0)=1.02204','f"(0)=1.100'),loc = 0)
plt.grid(True)
plt.show()
    
input('\nPress return to exit')
