import numpy as np
from cmath import sin, exp, sqrt
import time

VF=0.0001 #Критерій зупинки ВФ
VCH = 0.001 #Критерій зупинки ВЧ

def de(fobj, bounds, mut=2, crossp=0.7, popsize=20, its=1000): # mut = F crossp = CR
    # Initialization
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    # Evaluation
    pop_denorm = min_b + pop * diff

    fitness = np.asarray([fobj(ind) for ind in pop_denorm]) #Значення fit початкофих індивідів
    best_idx = np.argmin(fitness) #best fit val index
    best = pop_denorm[best_idx] #best fit val index

    for i in range(its):

        # VF_fit_prev = sum(fitness)/len(fitness)
        
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j] #index for every individual
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)] #3 random individuals from pop

            # mutation
            mutant = np.clip(a + mut * (b - c), 0, 1) # clip to 0 - 1 interval
            
            # recombination
            cross_points = np.random.rand(dimensions) < crossp #CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j]) #Замінюємо деякі параметри початкового індивіда pop[j] на параметри мутанта

            trial_denorm = min_b + trial * diff  
            f = fobj(trial_denorm) #fit for new individ

            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        

        pop_denorm = min_b + pop * diff
        # VCH    
        # sum = 0.
        # for k in range (popsize-1):
        #     sum+= abs(pop_denorm[k][0] - pop_denorm[k+1][0])  + abs(pop_denorm[k][1] - pop_denorm[k+1][1])
        # sum = sum / (popsize*2-2)
        # # print(sum)
        # if  sum < VCH:
        #     print(i, "its VCH")
        #     yield best, fitness[best_idx]
        #     break

        # VF
        # VF_fit_new = sum(fitness)/len(fitness)
        # VF_diff = abs(VF_fit_prev - VF_fit_new)
        # # print(VF_fit_new, VF_fit_prev)
        # # print(VF_diff)
        # if  VF_diff < VF:
        #     print(i, "its VF")
        #     yield best, fitness[best_idx]
        #     break


        yield best, fitness[best_idx]


def func_Beale(X):
    rez = (1.5 - X[0] + X[0]*X[1])**2 + (2.25 - X[0] + X[0]*X[1]**2 )**2 + (2.652 - X[0] + X[0]*X[1]**3)**2
    return rez 
                            
def func_Rosenbrock(C):
    sum = 0
    for i in range(1):
        sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
    return(round(sum,6))

def func_CrossInTray(X):
    rez = -0.0001*(np.abs( sin(X[0]) * sin(X[1]) * exp(abs( 100 - ((sqrt(X[0]**2 + X[1]**2)) / np.pi ) ) ) )+1)**0.1
    return rez






# (fobj, bounds, mut=2, crossp=0.7, popsize=20, its=1000):
time1 = time.perf_counter()

# it = list(de(func_Rosenbrock, bounds=[(-50, 50), (-50, 50)], mut = 1, crossp = 0.5, its = 10000))
# it = list(de(func_Beale, bounds=[(-4.5, 4.5), (-4.5, 4.5)], mut = 1, crossp = 0.8, its = 10000))
it = list(de(func_CrossInTray, bounds=[(-10, 10), (-10, 10)], mut = 1, crossp = 0.8, its = 10000))

time2 = time.perf_counter()

print(it[-1])
print(f"Time =  {time2 - time1:0.4f} seconds")