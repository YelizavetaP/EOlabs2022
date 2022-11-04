# симбіотична оптимізація
import numpy as np
import random
import operator
from cmath import sin, exp, sqrt
import time


def func_Rosenbrock(C):
    sum = 0
    for i in range(1):
        sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
    return(round(sum,6))

def func_Beale(C):
    x, y = C
    rez = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2 )**2 + (2.652 - x + x*y**3)**2
    return rez 

def func_CrossInTray(C):
    x, y = C
    rez = -0.0001*(np.abs( sin(x) * sin(y) * exp(abs( 100 - ((sqrt(x**2 + y**2)) / np.pi ) ) ) )+1)**0.1
    return rez



class Individual:
    
    def __init__(self, data ,func):
        self.Data = data
        self.f = func
        self.fit = self.f(data)


class SOS:
    #функция, кол-во переменых, размер популяции, границы поиска, BF параметри
    def __init__(self, func, var, s, ab, BF1, BF2):
        self.size = s
        self.vars = var #len(ab)
        self.pop = []
        self.func = func
        self.ab = ab #масив меж пошуку для кожної змінної  [[10,12],[12,1000],...[0,1]]
        self.BF1 = BF1
        self.BF2 = BF2
        

    def newPop(self):
        
        for i in range(self.size):

            D = np.array([random.uniform(self.ab[j][0], self.ab[j][1]) for j in range(self.vars)])
            self.pop.append(Individual(D, self.func))
        

    def getBestFit(self):

        sPop = sorted(self.pop, key = operator.attrgetter('fit'))
        return sPop[0]


    def sos(self):

        for i in range(self.size):

            X_best = self.getBestFit()
            Xi = self.pop[i]

            # mutualism
            j = random.randint(0,self.size-1) 
            Xj = self.pop[j]
            
            mut_vector = (Xi.Data + Xj.Data)/2

            new_xi = Xi.Data + random.random() * (X_best.Data - mut_vector * self.BF1)
            new_xj = Xj.Data + random.random() * (X_best.Data - mut_vector * self.BF2)

            newFit_i = self.func(new_xi)
            newFit_j = self.func(new_xj)

            if newFit_i < self.pop[i].fit: self.pop[i] = Individual(np.array(new_xi), self.func)
            if newFit_j < self.pop[j].fit: self.pop[j] = Individual(np.array(new_xj), self.func)

            # Commensalism
            j = random.randint(0,self.size-1) 
            Xj = self.pop[j]

            new_xi = Xi.Data + random.uniform(-1, 1) * (X_best.Data - Xj.Data)

            if (self.func(new_xi) < self.pop[i].fit): self.pop[i] = Individual(np.array(new_xi), self.func)

            # Parasitism
            j = random.randint(0,self.size-1) 
            Xj = self.pop[j]

            parasite_vector = [random.uniform(self.ab[j][0], self.ab[j][1]) for j in range(self.vars)]
            new_fit = self.func(parasite_vector)

            if (new_fit < self.pop[j].fit): self.pop[j] = Individual(np.array(parasite_vector), self.func)
            





if __name__ == "__main__":
    VF=0.001 #Критерій зупинки ВФ
    VCH = 0.5 #Критерій зупинки ВЧ
   
    # def __init__(self,func,  var, s,      ab,        BF1, BF2):
    alg = [SOS(func_Rosenbrock, 2, 20, [[-50,50],[1,2]], 1, 2),
            SOS(func_Beale, 2, 20, [[-4.5, 4.5],[-4.5, 4.5]], 1, 2),
            SOS(func_CrossInTray, 2, 20, [[-10,10],[-10,10]], 1, 2)]

    XY = [[1,1],
        [3,0.5],
        [1.34941,1.34941]]

    for i in range(0,3): alg[i].newPop()

    z = [func_Rosenbrock(XY[0]),
            func_Beale(XY[1]),
            func_CrossInTray(XY[2])]

    f = np.zeros(3)

    for i in range(0,3):

        time1 = time.perf_counter()
        while f[i] < 30:

            f[i] += 1

            sum_p = 0
            for e in alg[i].pop: sum_p += e.fit
            VF_fit_prev = sum_p /alg[i].size

            alg[i].sos()

             # VCH    
            # sum = 0.
            # for k in range (alg[i].size-1):
            #     sum+= abs(alg[i].pop[k].Data[0] - alg[i].pop[k+1].Data[0])  + abs(alg[i].pop[k].Data[1] - alg[i].pop[k+1].Data[1])
            # sum = sum / (alg[i].size*2-2)
            # # print(sum)
            # if  sum < VCH:
            #     break            

            # VF
            sum_n = 0
            for e in alg[i].pop: sum_n += e.fit
            VF_fit_new = sum_n /alg[i].size
            VF_diff = abs(VF_fit_prev - VF_fit_new)
            # print(VF_fit_new, VF_fit_prev)
            # print(VF_diff)
            if  VF_diff < VF:
                break


        time2 = time.perf_counter()
        
        print('f(x, y) = ',z[i], 'x, y: ', XY[i])
        print('SOS: ', alg[i].getBestFit().fit,   '	x,y: ', alg[i].getBestFit().Data)
        print('iter:', f[i])
        print(f"Time =  {time2 - time1:0.4f} seconds")
        print()
    