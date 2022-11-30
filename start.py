# метод імітації відпалу Simulated annealing
import random
import time
from math import exp, sin, sqrt, cos, radians

import numpy as np


def func_Rosenbrock(C):
    sum = 0
    
    for i in range(len(C)-1):        
        sum+= 100*((C[i+1]-(C[i])**2)**2) + (C[i]-1)**2
        # sum += C[i]**2
    
    return round(sum)

def func_Beale(C):
    x =  C[0]
    rez = (1.5 - x + x*x)**2 + (2.25 - x + x*x**2 )**2 + (2.652 - x + x*x**3)**2
    return rez 

def func_CrossInTray(C):
    x, y = C
    rez = -0.0001*(np.abs( sin(x) * sin(y) * exp(abs( 100 - ((sqrt(x**2 + y**2)) / np.pi ) ) ) )+1)**0.1
    return rez


class SA:
    #функция, кол-во переменых, размер популяции, границы поиска, T
    def __init__(self, func, var, size, ab, T):
        self.size = size
        self.vars = var #len(ab)
        self.pop = []
        self.func = func
        self.ab = ab #масив меж пошуку для кожної змінної  [[10,12],[12,1000],...[0,1]]
        self.T = T
       
        

    def newPop(self):
        
        for i in range(self.size):

            D = [random.uniform(self.ab[j][0], self.ab[j][1]) for j in range(self.vars)]

            self.pop.append([D, self.func(D)])  #[[1, 4, .. 5], 0]
        

    def getBestFit(self):

        sPop = sorted(self.pop, key=lambda ind: ind[-1])   
        return sPop[0]

    def iter(self):
        return




if __name__ == '__main__':

    VF=0.001 #Критерій зупинки ВФ
    VCH = 0.5 #Критерій зупинки ВЧ
    
    T = 1000

    n = 10
    ab = [[-10, 10]] * n
    
    alg = [SA(func_Beale, 1, 10, [[-4.5,4.5]], T),
            SA(func_CrossInTray, 2, 30, [[-10,10],[-10,10]], T),
            SA(func_Rosenbrock, n, 20, ab, T)]

    XY = [[0.594],
        [1.34941,1.34941],
        [1]*n]

    z = [func_Beale(XY[0]),
            func_CrossInTray(XY[1]),
            func_Rosenbrock(XY[2])]



    f = np.zeros(3)
    for i in range(0,3): alg[i].newPop()

    for i in range(0,3):

        time1 = time.perf_counter()
        while f[i] < 30:

            f[i] += 1

            # sum_p = 0
            # for e in alg[i].pop: sum_p += e.fit
            # VF_fit_prev = sum_p /alg[i].size

            alg[i].iter()

             # VCH    
            # sum = 0.
            # for k in range (alg[i].size-1):
            #     sum+= abs(alg[i].pop[k].Data[0] - alg[i].pop[k+1].Data[0])  + abs(alg[i].pop[k].Data[1] - alg[i].pop[k+1].Data[1])
            # sum = sum / (alg[i].size*2-2)
            # # print(sum)
            # if  sum < VCH:
            #     break            

            # VF
            # sum_n = 0
            # for e in alg[i].pop: sum_n += e.fit
            # VF_fit_new = sum_n /alg[i].size
            # VF_diff = abs(VF_fit_prev - VF_fit_new)
            # # print(VF_fit_new, VF_fit_prev)
            # # print(VF_diff)
            # if  VF_diff < VF:
            #     break


        time2 = time.perf_counter()
        
        print('f(X) = ',z[i], 'X: ', XY[i])
        print('SA: ', alg[i].getBestFit()[-1],   '	X: ', alg[i].getBestFit()[0])
        print('iter:', f[i])
        print(f"Time =  {time2 - time1:0.4f} seconds")
        print()
