# метод фрактальної структуризації
import random
import time
from math import exp, sin, sqrt, fsum
import copy

import numpy as np


def func_Beale(C):
    x =  C[0]
    rez = (1.5 - x + x*x)**2 + (2.25 - x + x*x**2 )**2 + (2.652 - x + x*x**3)**2
    return rez 

def func_CrossInTray(C):
    x, y = C
    rez = -0.0001*(np.abs( sin(x) * sin(y) * exp(abs( 100 - ((sqrt(x**2 + y**2)) / np.pi ) ) ) )+1)**0.1
    return rez


class FS:
    #функция, кол-во переменых, размер популяции, границы поиска, кількість нащадків
    def __init__(self, func, var, size, ab, m):
        self.size = size
        self.vars = var #len(ab)
        self.pop = []
        self.func = func
        self.ab = ab #масив меж пошуку для кожної змінної  [[10,12],[12,1000],...[0,1]]
        self.m = m
       
        

    def newPop(self):
        
        for i in range(self.size):

            D = [random.uniform(self.ab[j][0], self.ab[j][1]) for j in range(self.vars)]

            self.pop.append([D, self.func(D)])  #[[1, 4, .. 5], 0]
        

    def getBestFit(self):
       
        return sorted(self.pop, key=lambda ind: ind[-1])[0]

    def iter(self):

        sigma = 1

        for i in range(self.size):

            S = [[],[]]

            for j in range(self.m):

                x = self.pop[i][0][0] + random.normalvariate(0, sigma)
                if (x) < self.pop[i][0][0]:
                    S[0].append(x)
                else: 
                    S[1].append(x)
            
            l1, l2 = len(S[0]), len(S[1])
            if l1 == 0: l1 = 1
            if l2 == 0: l2 = 1

            xl, xr = fsum(S[0])/l1, fsum(S[1])/l2

            if self.func([xl]) < self.func([xr]) : 
                xnew = xl 
            else: xnew = xr
            self.pop.append([[xnew], self.func([xnew])])

            self.pop = sorted(self.pop, key= lambda ind: ind[-1])[:self.size]

    def iter2(self):

        sigma = 1

        for i in range(self.size):

            S = [[],[]]

            v = random.randint(0, self.vars-1)


            for j in range(self.m):

                x = self.pop[i][0][v] + random.normalvariate(0, sigma)
                if (x) < self.pop[i][0][v]:
                    S[0].append(x)
                else: 
                    S[1].append(x)
                
            l1, l2 = len(S[0]), len(S[1])
            if l1 == 0: l1 = 1
            if l2 == 0: l2 = 1

        xl, xr = fsum(S[0])/l1, fsum(S[1])/l2

        Xl, Xr = copy.deepcopy(self.pop[i][0]), copy.deepcopy(self.pop[i][0])
        Xl[v], Xr[v] = xl, xr

        if self.func(Xl) < self.func(Xr) : 
            xnew = Xl 
        else: xnew = Xr
        self.pop.append([xnew, self.func(xnew)])

        self.pop = sorted(self.pop, key= lambda ind: ind[-1])[:self.size]





if __name__ == '__main__':




    VF=0.00001 #Критерій зупинки ВФ
    VCH = 0.8 #Критерій зупинки ВЧ
    
    m = 7
    
    alg = [FS(func_Beale, 1, 30, [[-4.5,4.5]], m),
            FS(func_CrossInTray, 2, 30, [[-10,10],[-10,10]], m)]

    XY = [[0.594],
        [1.34941,1.34941]]

    z = [func_Beale(XY[0]),
            func_CrossInTray(XY[1])]



    f = np.zeros(2)
    for i in range(0,2): alg[i].newPop()

    for i in range(0,2):

        time1 = time.perf_counter()
        while f[i] < 10:

            f[i] += 1

            # sum_p = 0
            # for e in alg[i].pop: sum_p += e[-1]
            # VF_fit_prev = sum_p /alg[i].size

            # print(f[i])
            if alg[i].vars == 1:
                alg[i].iter()
            else: 
                alg[i].iter2()


            # VCH    
            # sum = 0
            # for c in range(0,alg[i].vars):
            #     sum += abs(abs(alg[i].pop[0][0][c]) - alg[i].pop[1][0][c])
            # sum = sum / alg[i].vars
            # if sum <= VCH: break      

            # VF
            # sum_n = 0
            # for e in alg[i].pop: sum_n += e[-1]
            # VF_fit_new = sum_n /alg[i].size
            # VF_diff = abs(VF_fit_prev - VF_fit_new)
            # # print(VF_fit_new, VF_fit_prev)
            # # print(VF_diff)
            # if  VF_diff < VF:
            #     break


        time2 = time.perf_counter()
        
        print('f(X) = ',z[i], 'X: ', XY[i])
        print('FS: ', alg[i].getBestFit()[-1],   '	X: ', alg[i].getBestFit()[0])
        print('iter:', f[i])
        print(f"Time =  {time2 - time1:0.4f} seconds")
        print()
