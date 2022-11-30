import numpy as np
import random
import operator
from cmath import sin, exp, sqrt
import time

                            
def func_Rosenbrock(x, y):
    sum = 0
    C = [x, y]
    for i in range(1):
        sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
    return(round(sum,6))

def func_Beale(x, y):
    rez = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2 )**2 + (2.652 - x + x*y**3)**2
    return rez 

def func_CrossInTray(x, y):
    rez = -0.0001*(np.abs( sin(x) * sin(y) * exp(abs( 100 - ((sqrt(x**2 + y**2)) / np.pi ) ) ) )+1)**0.1
    return rez


class Individual:
    #x,y границы поиска степень 2
    def __init__(self,data,a,b, k, func):
        e1 = (a[1]-a[0])/(2**k-1)
        e2 = (b[1]-b[0])/(2**k-1)
        self.AData = data
        self.CData = [a[0] + self.AData[0]*e1,b[0] + self.AData[1]*e2]
        self.f = func
        self.fit = self.f(self.CData[0],self.CData[1])



class GA:
    #функция, кол-во переменых, размер популяции, границы поиска, вероятности операторов, степень 2
    def __init__(self, func,var,s,a,b, C,M,I, k):
        self.size = s
        self.vars = var
        self.pop = []
        self.func = func
        self.a = a
        self.b = b
        self.C = C
        self.M = M
        self.I = I
        self.k = k

    def newPop(self):
        
        for i in range(self.size):
            D = [random.randint(0,2**self.k) for j in range(self.vars)]
            self.pop.append(Individual(D,self.a,self.b, self.k, self.func))
        

    def getChild(self,a,b):

        p1 = []
        p2 = []
        for i in range(0,self.vars):
            p1.append("0"*(self.k-len(bin(a.AData[i])[2:]))+bin(a.AData[i])[2:]) 
            p2.append("0"*(self.k-len(bin(b.AData[i])[2:]))+bin(b.AData[i])[2:])

        for i in range (self.vars):
            p1[i]=list(p1[i])
            p2[i]=list(p2[i])

        #crossover
        # одноточковий
        #вибір першої точки від 1 до кінця рядка
        k = np.random.randint(1,self.k)
        for j in range(self.vars): 
            for i in range(k,self.k): 
                p1[j][i], p2[j][i] = p2[j][i],p1[j][i] 

        # двоточковий
        #вибір другої точки від першої до кінця рядка
        k2 = np.random.randint(k,self.k)
        for j in range(self.vars): 
            for i in range(k2,self.k): 
                p1[j][i], p2[j][i] = p2[j][i],p1[j][i]

            
        child = p1 if np.random.random()<0.5 else p2

        #mutation
        if np.random.random()<=self.M:
            k = np.random.randint(0,self.k)
            for i in range(self.vars):
                if(child[i][k] == '0'): child[i][k]='1' 
                else: child[i][k] = '0'

        #invesrion
        if np.random.random()<=self.I:
            for i in range(self.vars):
                for j in range(self.k):
                    if(child[i][j]=='0'):
                        child[i][j]='1'
                    else:
                        child[i][j]='0'
            
        return Individual([int(''.join([str(elem) for elem in child[0]]),2),int(''.join([str(elem) for elem in child[1]]),2)],self.a,self.b, self.k, self.func)	

    def sortPop(self):
        self.pop = sorted(self.pop, key = operator.attrgetter('fit'))

    def panmix(self):
        self.sortPop()
        for l in range(0,self.size):
            a = self.pop[random.randint(0,self.size-1)]
            b = self.pop[random.randint(0,self.size-1)]

            flag = 0
            if np.random.random()<self.C:
                ch = self.getChild(a,b)
                for j in range(len(self.pop)):
                    if ch.CData == self.pop[j].CData: 
                        flag = 1
                        break
                if flag == 0: self.pop.append(ch)

        # звичайний відбір (ЗВ)
        # self.pop = random.choices(self.pop, k = self.size)
         
        # відбір із витісненням (ВВ)
        # self.sortPop()
        # newPop = []
        # dif = 0.001
        # temp = 0
        # for n in range(len(self.pop)):
        #     for m in range(len(self.pop)):
        #         if (abs(self.pop[n].fit - self.pop[m].fit) <= dif):
        #             break
        #         else: temp = self.pop[n]
        #     if temp != 0: newPop.append(temp)
        #     if len(newPop) == self.size: break
        # if len(newPop) < self.size: newPop += self.pop[:self.size-len(newPop)]
        # self.pop = newPop
        

        # елітний відбір (ЕВ)
        self.sortPop()
        self.pop = self.pop[:20]


        
    def selection(self):
        
        self.sortPop()
        p = np.array([self.pop[i].fit for i in range(0,self.size-1)])
        m = np.mean(p)
        i = 0
        for l in range(0,self.size-1):
            while i<self.size:
                if p[i]>=m:	break
                i+=1

            a = self.pop[random.randint(0,i)]
            b = self.pop[random.randint(0,self.size-1)]

            flag = 0
            if np.random.random()<self.C:
                ch = self.getChild(a,b)
                for j in range(len(self.pop)):
                    if ch.CData == self.pop[j].CData: 
                        flag = 1
                        break
                if flag == 0: self.pop.append(ch)

        # звичайний відбір (ЗВ)
        self.pop = random.choices(self.pop, k = self.size)
         
        # відбір із витісненням (ВВ)
        # self.sortPop()
        # newPop = []
        # dif = 0.01
        # temp = 0
        # for n in range(len(self.pop)):
        #     for m in range(len(self.pop)):
        #         if (abs(self.pop[n].fit - self.pop[m].fit) <= dif):
        #             break
        #         else: temp = self.pop[n]
        #     if temp != 0: newPop.append(temp)
        #     if len(newPop) == self.size: break
        # if len(newPop) < self.size: newPop += self.pop[:self.size-len(newPop)]
        # self.pop = newPop
        

        # елітний відбір (ЕВ)
        # self.sortPop()
        # self.pop = self.pop[:self.size]


    def inbriding(self):
        self.pop = sorted(self.pop, key = operator.attrgetter('CData'))
        for l in range(0,self.size):
            q = random.randint(1,self.size-2)
            a = self.pop[q]

            if abs(self.pop[q].AData[0] - self.pop[q-1].AData[0]) < abs(self.pop[q+1].AData[0] - self.pop[q].AData[0]):
                b = self.pop[q-1]
            else: b = self.pop[q+1]


            flag = 0
            if np.random.random()<self.C:
                ch = self.getChild(a,b)
                for j in range(len(self.pop)):
                    if ch.CData == self.pop[j].CData: 
                        flag = 1
                        break
                if flag == 0: self.pop.append(ch)

                # звичайний відбір (ЗВ)
        # self.pop = random.choices(self.pop, k = self.size)
         
        # відбір із витісненням (ВВ)
        # self.sortPop()
        # newPop = []
        # dif = 0.001
        # temp = 0
        # for n in range(len(self.pop)):
        #     for m in range(len(self.pop)):
        #         if (abs(self.pop[n].fit - self.pop[m].fit) <= dif):
        #             break
        #         else: temp = self.pop[n]
        #     if temp != 0: newPop.append(temp)
        #     if len(newPop) == self.size: break
        # if len(newPop) < self.size: newPop += self.pop[:self.size-len(newPop)]
        # self.pop = newPop
        

        # елітний відбір (ЕВ)
        self.sortPop()
        self.pop = self.pop[:20]


    def outbriding(self):
        self.pop = sorted(self.pop, key = operator.attrgetter('CData'))
        
        for l in range(0,self.size):
            q = random.randint(1,self.size-2)
            a = self.pop[q]

            if self.size - 1 - q < q:
                b = self.pop[0]
            else: b = self.pop[self.size-1]


            flag = 0
            if np.random.random()<self.C:
                ch = self.getChild(a,b)
                for j in range(len(self.pop)):
                    if ch.CData == self.pop[j].CData: 
                        flag = 1
                        break
                if flag == 0: self.pop.append(ch)

        # звичайний відбір (ЗВ)
        # self.pop = random.choices(self.pop, k = self.size)
         
        # відбір із витісненням (ВВ)
        # self.sortPop()
        # newPop = []
        # dif = 0.001
        # temp = 0
        # for n in range(len(self.pop)):
        #     for m in range(len(self.pop)):
        #         if (abs(self.pop[n].fit - self.pop[m].fit) <= dif):
        #             break
        #         else: temp = self.pop[n]
        #     if temp != 0: newPop.append(temp)
        #     if len(newPop) == self.size: break
        # if len(newPop) < self.size: newPop += self.pop[:self.size-len(newPop)]
        # self.pop = newPop
        

        # елітний відбір (ЕВ)
        self.sortPop()
        self.pop = self.pop[:20]


    
    def getBestFit(self):
        sPop = sorted(self.pop, key = operator.attrgetter('fit'))
        return sPop[0]

    def printPop(self):
        print()
        for i in range(0,self.size):
            print(self.pop[i].CData, self.pop[i].fit)
        # print(np.mean(np.array([self.pop[j].fit for j in range(0,self.size)])))
        # print(self.getBestFit().fit, self.getBestFit().AData, self.getBestFit().CData)
        print()
            

if __name__ == "__main__":

    VF=0.001 #Критерій зупинки ВФ
    VCH = 0.1 #Критерій зупинки ВЧ

    k = 7

    

    #функция, кол-во переменых, размер популяции, границы поиска, вероятности операторов, степень 2
    alg = [GA(func_Rosenbrock,2,20,[-50,50],[1,2], 0.8, 0.05, 0.1, k),
            GA(func_Beale,2,20,[-4.5, 4.5],[-4.5, 4.5], 0.8, 0.05, 0.1, k),
            GA(func_CrossInTray,2,20,[-10,10],[-10,10], 0.8, 0.05, 0.1, k)]

    XY = [[1,1],
        [3,0.5],
        [1.34941,1.34941]]

    for i in range(0,3):
        alg[i].newPop()

    ey = [(alg[i].b[1]-alg[i].b[0])/(2**k-1) for i in range(0,3)]
    ex = [(alg[i].a[1]-alg[i].a[0])/(2**k-1) for i in range(0,3)] 

    z = [func_Rosenbrock(XY[0][0],XY[0][1]),
            func_Beale(XY[1][0],XY[1][1]),
            func_CrossInTray(XY[2][0],XY[2][1])]

    f = np.zeros(3)
    

    for i in range(0,3):
        time1 = time.perf_counter()
        while f[i] < 10000:
            f[i] += 1

            sum_p = 0
            for e in alg[i].pop: sum_p += e.fit
            VF_fit_prev = sum_p /alg[i].size

            # alg[i].panmix()		
            alg[i].selection()
            # alg[i].inbriding()
            # alg[i].outbriding()

             # VCH    
            # sum = 0.
            # for k in range (alg[i].size-1):
            #     sum+= abs(alg[i].pop[k].CData[0] - alg[i].pop[k+1].CData[0])  + abs(alg[i].pop[k].CData[1] - alg[i].pop[k+1].CData[1])
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
        print('GA: ',alg[i].getBestFit().fit,'	x,y: ', alg[i].getBestFit().CData)
        print('iter:', f[i])
        print(f"Time =  {time2 - time1:0.4f} seconds")
        print()
    
                
