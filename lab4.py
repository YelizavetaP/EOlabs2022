# метод деформованих зірок
import random
import time as t
from cmath import exp, sin, sqrt

import numpy as np


def func_Rosenbrock(C):
    sum = 0
    x = C[0]
    for i in range(1):
        sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
    return(round(sum,6))

def func_Beale(C):
    x =  C[0]
    rez = (1.5 - x + x*x)**2 + (2.25 - x + x*x**2 )**2 + (2.652 - x + x*x**3)**2
    return rez 

def func_CrossInTray(C):
    x, y = C
    rez = -0.0001*(np.abs( sin(x) * sin(y) * exp(abs( 100 - ((sqrt(x**2 + y**2)) / np.pi ) ) ) )+1)**0.1
    return rez


# class Individual:
    
#     def __init__(self, data ,func):
#         self.Data = data
#         self.f = func
#         self.fit = self.f(data)


class MDS:
    #функция, кол-во переменых, размер популяции, границы поиска, BF параметри
    def __init__(self, func, var, size, ab, sigma):
        self.size = size
        self.vars = var #len(ab)
        self.pop = []
        self.func = func
        self.ab = ab #масив меж пошуку для кожної змінної  [[10,12],[12,1000],...[0,1]]
        self.sigma = sigma
        

    def newPop(self):
        
        for i in range(self.size):

            D = np.array([random.uniform(self.ab[j][0], self.ab[j][1]) for j in range(self.vars)])
            # self.pop.append(Individual(D, self.func))
            self.pop.append([D, self.func(D)])  #[[1, 4, .. 5], 0]
        

    def getBestFit(self):

        # sPop = sorted(self.pop, key = operator.attrgetter('fit'))
        sPop = sorted(self.pop, key=lambda ind: ind[-1])   # сортируем по возрасту
        return sPop[0]

    def newPopT(self, Pt, Pz, Ps):
        P = Pt + Pz + Ps
        P = sorted(P, key=lambda ind: ind[-1])
        self.pop = P[:self.size]


    def iter(self):

        VF=0.001 #Критерій зупинки ВФ
        VCH = 0.5 #Критерій зупинки ВЧ

        self.newPop()
        print('olg\n' ,self.pop)

        its = 0
        
        time1 = t.perf_counter()
        while its < 10:

            its+=1

            Pt = self.pop
            Pz = []
            Ps = []

            # Pz
            for i in range(self.size):

                Xz = []

                for j in range(self.vars):

                    a, b = self.ab[j]
                    x_new = Pt[i][0][j] + random.normalvariate(0, self.sigma)
                    if x_new < a: x_new = x_new + (b - a)
                    if x_new > b: x_new = x_new + (a - b)
                    Xz.append(x_new)

                Pz.append([Xz, self.func(Xz)])

            # Ps
            for i in range(self.size):

                Xs = []

                for j in range(self.vars):

                    xi = Pt[random.randint(0,self.size-1)]
                    xj = Pt[random.randint(0,self.size-1)]

                    x_new = (xi[0] + xj[0])/2
                    
                    Xs.append(x_new)

                Ps.append([Xs, self.func(Xs)])

            self.newPopT(Pt, Pz, Ps) #елітний відбір нової популяції

            print('new\n' ,self.pop)
            print()
            # перевірка критерії виходу з ітерації
            # VCH
            sum = 0
            for c in range(self.vars):
                sum += abs(self.pop[0][0][c] - self.pop[1][0][c])
            sum = sum / self.vars
            # print(sum)
            # print(self.pop[:2])
            # print()
            if sum <= VCH: break


        time2 = t.perf_counter()
        time = time2 - time1

        return self.getBestFit(), time, its







if __name__ == "__main__":

    alg_test = MDS(func_Beale, 1, 20, [[-4.5,4.5]], (4.5 + 4.5)/10)
    res = alg_test.iter()

    print(res)



    # VF=0.001 #Критерій зупинки ВФ
    # VCH = 0.5 #Критерій зупинки ВЧ
   
    # # def __init__(self,func,  var, s,      ab,        BF1, BF2):
    # alg = [SOS(func_Rosenbrock, 2, 20, [[-50,50],[1,2]], 1, 2),
    #         SOS(func_Beale, 2, 20, [[-4.5, 4.5],[-4.5, 4.5]], 1, 2),
    #         SOS(func_CrossInTray, 2, 20, [[-10,10],[-10,10]], 1, 2)]

    # XY = [[1,1],
    #     [3,0.5],
    #     [1.34941,1.34941]]

    # for i in range(0,3): alg[i].newPop()

    # z = [func_Rosenbrock(XY[0]),
    #         func_Beale(XY[1]),
    #         func_CrossInTray(XY[2])]

    # f = np.zeros(3)

    # for i in range(0,3):

    #     time1 = time.perf_counter()
    #     while f[i] < 30:

    #         f[i] += 1

    #         sum_p = 0
    #         for e in alg[i].pop: sum_p += e.fit
    #         VF_fit_prev = sum_p /alg[i].size

    #         alg[i].sos()

    #          # VCH    
    #         # sum = 0.
    #         # for k in range (alg[i].size-1):
    #         #     sum+= abs(alg[i].pop[k].Data[0] - alg[i].pop[k+1].Data[0])  + abs(alg[i].pop[k].Data[1] - alg[i].pop[k+1].Data[1])
    #         # sum = sum / (alg[i].size*2-2)
    #         # # print(sum)
    #         # if  sum < VCH:
    #         #     break            

    #         # VF
    #         sum_n = 0
    #         for e in alg[i].pop: sum_n += e.fit
    #         VF_fit_new = sum_n /alg[i].size
    #         VF_diff = abs(VF_fit_prev - VF_fit_new)
    #         # print(VF_fit_new, VF_fit_prev)
    #         # print(VF_diff)
    #         if  VF_diff < VF:
    #             break


    #     time2 = time.perf_counter()
        
    #     print('f(x, y) = ',z[i], 'x, y: ', XY[i])
    #     print('SOS: ', alg[i].getBestFit().fit,   '	x,y: ', alg[i].getBestFit().Data)
    #     print('iter:', f[i])
    #     print(f"Time =  {time2 - time1:0.4f} seconds")
    #     print()
    