import numpy as np


def weights(M, N, x0, grid):
    delta = np.zeros((M+1, N+1, N+1))
    delta[0, 0, 0] = 1
    c1 = 1
    for n in range(1, N+1):
        c2 = 1
        for nu in range(n):
            c3 = grid[n]-grid[nu]
            c2 *= c3
            m1 = min([n, M])+1
            for m in range(m1):
                delta[m, n, nu] = (
                    (grid[n]-x0)*delta[m, n-1, nu]-m*delta[m-1, n-1, nu])/c3
        for m in range(m1):
            delta[m, n, n] = c1/c2 * \
                (m*delta[m-1, n-1, n-1]-(grid[n-1]-x0)*delta[m, n-1, n-1])
        c1 = c2
    return delta


#delta = weights(4, 8, 0, [0, 1, -1, 2, -2, 3, -3, 4, -4])

#grid=np.array([-4,-3,-2,-1,0,1,2,3,4])

#delta=weights(4,8,0,np.take(grid,np.argsort(np.abs(grid))))

import fractions
np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

#print('alphas=', grid)

#for der_order in [1,2,3,4]:
#    print('m=',der_order)
#    for accorder in [2,4,6,8]:
#        print('n=', accorder,' ', np.take(delta[der_order, accorder, :],[7,5,3,1,0,2,4,6,8]))

grid=[-7/2,-5/2,-3/2,-1/2,1/2,3/2,5/2,7/2]

#sorted_grid=[-1/2 1/2 -3/2 3/2 -5/2 5/2 -7/2 7/2]



def sort_grid(x0,der_grid):
    x0vect=np.zeros_like(der_grid)+x0
    dist=(x0vect-der_grid)**2
    sorted_grid=np.take(der_grid,np.argsort(dist))
    return sorted_grid

print(np.take(grid,np.argsort(np.abs(grid))))

print(sort_grid(0,grid))

delta=weights(4,7,0,sort_grid(0,grid))


print('alphas=', grid)

for der_order in [0,1,2,3,4]:
    print('m=',der_order)
    for accorder in [1,3,5,7]:
        print('n=', accorder,' ', np.take(delta[der_order, accorder, :],[6,4,2,0,1,3,5,7]))




def take_der(func,grid,der_grid,der_order=1,acc_order=2):
    der_list=[]
    for x0 in grid:
        delta=weights(der_order,len(der_grid)-1,x0,sort_grid(x0,der_grid))
        #der_sum=sum(delta[der_order, acc_order, :]*func(der_grid))
        der_sum=sum(delta[der_order, acc_order, :]*func(sort_grid(x0,der_grid)))
        der_list.append(der_sum)
    return der_list


def func(x0):
    return x0**2

grid=np.linspace(-7/2,7/2,21)

der_grid=np.array([-7/2,-5/2,-3/2,-1/2,1/2,3/2,5/2,7/2])

der_grid=np.linspace(-7/2,7/2,21)

d1=take_der(func,grid,der_grid,der_order=1,acc_order=20)

import matplotlib.pyplot as plt

plt.plot(grid,d1)
plt.plot(grid, 2*grid)
plt.show()

def func(x0):
    return np.sin(x0)

grid=np.linspace(0,2*np.pi,20)
#der_grid=np.linspace(-1,1,41)
der_grid=np.linspace(0,2*np.pi,100)

import matplotlib.pyplot as plt



for acc_order in range(2,50,2):


    d1=take_der(func,grid,der_grid,der_order=1,acc_order=acc_order)

    err=np.mean((d1-np.cos(grid))**2)

    print('Prec order = {} err ={}'.format(acc_order, err))

    plt.plot(grid,d1)
    plt.plot(grid,np.cos(grid))
    plt.show()



