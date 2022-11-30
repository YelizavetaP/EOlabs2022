# метод імітації відпалу Simulated annealing
import random
import time
from math import exp, sin, sqrt, cos, radians
import copy

import numpy as np


def func_Rosenbrock(C):
    sum = 0
    
    for i in range(len(C)):        
        # sum+= 100*((C[i+1]-(C[i])**2)**2) + (C[i]-1)**2
        sum += C[i]**2
    
    return round(sum, 6)

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
        self.E = func
        self.ab = ab #масив меж пошуку для кожної змінної  [[10,12],[12,1000],...[0,1]]
        self.T = T
       
        

    def newPop(self):
        
        for i in range(self.size):

            D = [random.uniform(self.ab[j][0], self.ab[j][1]) for j in range(self.vars)]

            self.pop.append([D, self.E(D)])  #[[1, 4, .. 5], 0]
        

    def getBestFit(self):

        sPop = sorted(self.pop, key=lambda ind: ind[-1])   
        return sPop[0]

    def iter(self):
            
        VF=0.001 #Критерій зупинки ВФ
        VCH = 0.5 #Критерій зупинки ВЧ

        t, its = T, 0
        tmin = 10
        while(its < 1000):

            f_sum_1 = 0
            for ind in self.pop: f_sum_1 += ind[-1]
            f_sum_1/=self.size


            for i in range(self.size):

                ind = copy.deepcopy(self.pop[i][0])

                for j in range(self.vars):
                    if (random.random() > 0.5):
                        
                        # print(ind[j])
                        ind[j] = ind[j] +  random.randint(-100, 100)/100 
                        # print(ind[j])
                        # print()


                Ei = self.E(ind)


                # print([ind, Ei])
                # print(self.pop[i][0], self.pop[i][-1])
                # print()
                DEi = Ei - self.pop[i][-1]


                if DEi <= 0:
                    self.pop[i] = [ind, Ei]
                    # print(self.pop[i])
                elif (random.random() < exp(-DEi/t)):
                    # print(exp(-DE/t))
                    self.pop[i] = [ind, Ei]
                    # print(self.pop[i])





            # перевірка критерії виходу з ітерації
            # VCH
            # sum = 0
            # for c in range(self.vars):
            #     sum += abs(self.pop[0][0][c] - self.pop[1][0][c])
            # sum = sum / self.vars
            # if sum <= VCH: break

            # VF 
            # f_sum_2 = 0
            # for ind in self.pop: f_sum_2 += ind[-1]
            # f_sum_2/=self.size
            # diff = abs(f_sum_1-f_sum_2)
            # if diff <= VF: break


            # print(self.getBestFit())

            its += 1
            if t > tmin: t = t/its 
            # print(self.getBestFit()[-1])
        return t, its
        




if __name__ == '__main__':

    T = 50000000.


    n = 10
    ab = [[-50, 50]] * n
    
    alg = [SA(func_Beale, 1, 30, [[-4.5,4.5]], T),
            SA(func_CrossInTray, 2, 30, [[-10,10],[-10,10]], T),
            SA(func_Rosenbrock, n, 40, ab, T)]

    XY = [[0.594],
        [1.34941,1.34941],
        [0]*n]

    z = [func_Beale(XY[0]),
            func_CrossInTray(XY[1]),
            func_Rosenbrock(XY[2])]



    for i in range(0,3): alg[i].newPop()

    for i in range(0, 3):

        time1 = time.perf_counter()
       
        t, its = alg[i].iter()

        time2 = time.perf_counter()
        
        print('f(X) = ',z[i], 'X: ', XY[i])
        print('SA: ', alg[i].getBestFit()[-1],   '	X: ', alg[i].getBestFit()[0])
        print('t, iter:', t, its)
        print(f"Time =  {time2 - time1:0.4f} seconds")
        print()
