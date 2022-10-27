from cmath import exp, sin, sqrt
import numpy as np
import random
import time
from scipy.spatial.distance import hamming


def func_Beale(X):
    rez = (1.5 - X[0] + X[0]*X[1])**2 + (2.25 - X[0] + X[0]*X[1]**2 )**2 + (2.652 - X[0]*X[1]**3)**2
    return rez 
                            
def func_Rosenbrock(C):
    sum = 0
    for i in range(1):
        sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
    return(round(sum,6))

def func_CrossInTray(X):
    rez = -0.0001*(np.abs( sin(X[0]) * sin(X[1]) * exp(abs( 100 - ((sqrt(X[0]**2 + X[1]**2)) / np.pi ) ) ) )+1)**0.1
    return rez



VF=0.0001 
VCH = 0.001 
num_arg = 2

kol_b = 10 #к-ть батьків
kol_d = 7 #к-сть нащадків
kol = kol_d*kol_b+kol_b #к-сть всього
e = 0.001
sigma = 0.1 #середньоквадратичне відхилення 


class Individ:
    def __init__(self,num):
        self.C=[]
        for i in range(num_arg):
            self.C.append(num[i])
        self.fit=0                        
    def fitness(self):
        # self.fit=func_Beale(self.C)
        # self.fit = func_Rosenbrock(self.C)
        self.fit = func_CrossInTray(self.C)
        
    def info(self):  
        self.fitness()
        print("C = ",self.C,"fit = ",self.fit)                       
   







mas2 = []
for j in range (num_arg):
    mas=[]
    for i in range(kol_b):
        mas.append(random.uniform(-10,10))
        random.shuffle(mas)
    mas2.append(mas)
mas2 = np.array(mas2)
mas2 = np.transpose(mas2)

def findDistance(pop):
    sum = 0.
    for i in range (len(pop.B)-2):
        sum+= abs(pop.B[i].C[0]-pop.B[i+1].C[0])+ abs(pop.B[i].C[1]-pop.B[i+1].C[1])
    return sum / (len(pop.B)*2-2)



class population():
    def __init__(self):
        self.B = []
        for i in range(kol_b):
            p = Individ(mas2[i])
            self.B.append(p)
    def fitness(self):
        sum=0.
        for i in self.B:
            i.fitness()
            sum+=i.fit
        self.fit = round(sum/len(self.B),6)

    def info(self):
        for i in range(len(self.B)):
            self.B[i].fitness()
            self.B[i].info()
        self.fitness()
        print("General FIT",self.fit)
    

def sort_fit(pop):
    pop.fitness()
    list=sorted(pop.B,key = lambda A: A.fit)
    for i in range(len(pop.B)):
        pop.B[i]=list[i]
    return pop
def norm(sigma):
    return np.random.normal(0, sigma**2)

def new_population(pop,sigma):
    pop1 = population()
    pop1.B = pop.B 
    sort_fit(pop1)
    pop1.fitness()
    pop2 = population()
    for i in range(kol_b):
        pop2.B[i]=pop1.B[i]
    pop2.fitness()

    

    return pop2, sigma


time1 = time.perf_counter()

pop = population()
pop.info()
N=0
while(N<10000):
    Prev_fit = pop.fit #попередня популяція
    N+=1
    for i in range(kol_b):
        for j in range (kol_d):
            pop.B.append(Individ([round(x+norm(sigma),8) for x in pop.B[i].C]))
        
    pop2, sigma = new_population(pop,sigma)
    pop = population()
    pop = pop2
    # if(findDistance(pop2)<VCH):
    #     break
    if(abs(Prev_fit-pop2.fit)<VF):
    	break


print("X = ", pop2.B[0].C)
print("Y = ", pop2.B[0].fit)
print(N)

time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")
