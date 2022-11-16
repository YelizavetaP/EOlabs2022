# метод деформованих зірок
import random
import time as t
from cmath import exp, sin, sqrt, cos

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


# class Individual:
    
#     def __init__(self, data ,func):
#         self.Data = data
#         self.f = func
#         self.fit = self.f(data)


class MODS:
    #функция, кол-во переменых, размер популяции, границы поиска, BF параметри
    def __init__(self, func, var, size, ab):
        self.size = size
        self.vars = var #len(ab)
        self.pop = []
        self.func = func
        self.ab = ab #масив меж пошуку для кожної змінної  [[10,12],[12,1000],...[0,1]]
       
        

    def newPop(self):
        
        for i in range(self.size):

            D = [random.uniform(self.ab[j][0], self.ab[j][1]) for j in range(self.vars)]
            if self.vars > 2:
                 D = [random.uniform(self.ab[j][0], self.ab[j][1]) for j in range(self.vars)]

            self.pop.append([D, self.func(D)])  #[[1, 4, .. 5], 0]
        

    def getBestFit(self):

        sPop = sorted(self.pop, key=lambda ind: ind[-1])   
        return sPop[0]

    def newPopT(self, P_):
        P = []
        for p in P_:
            P += p
        unique_list = []

        P = sorted(P, key=lambda ind: ind[-1])
        self.pop = P[:self.size]
        # print('pop\n',self.pop,'\n')


    def iter(self, sigma):

        VF = 0.001 #Критерій зупинки ВФ
        VCH = 0.01 #Критерій зупинки ВЧ

        self.newPop()

        its = 0
        
        time1 = t.perf_counter()

        while its < 10:
            
            # f_sum_1 = 0
            # for ind in self.pop: f_sum_1 += ind[-1]
            # f_sum_1/=self.size

            Pt = self.pop
            Pz = []
            Ps = []

            # Pz
            for i in range(self.size):

                Xz = []

                for j in range(self.vars):

                    a, b = self.ab[j]
                    x_new = Pt[i][0][j] + random.normalvariate(0, sigma)
                    if x_new < a: x_new = x_new + (b - a)
                    if x_new > b: x_new = x_new + (a - b)
                    Xz.append(x_new)

                Pz.append( [Xz, self.func(Xz)])

            # Ps
            for i in range(self.size):

                Xs = []

                for j in range(self.vars):

                    xi = Pt[random.randint(0,self.size-1)]
                    xj = Pt[random.randint(0,self.size-1)]

                    for k in range(self.vars-1):
                    
                        x_new = xi[i][0][k] + xj[i][0][k]
                        Xs.append(x_new)
                    
                    Xs.append(x_new)

                Ps.append([Xs, self.func(Xs)])

            self.newPopT([Pt, Pz, Ps]) #елітний відбір нової популяції

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

            its+=1

        time2 = t.perf_counter()
        time = time2 - time1
        
        return self.getBestFit(), 'time: ', time, 'iter: ', its


    def iter_n2(self, a):

        VF = 0.001 #Критерій зупинки ВФ
        VCH = 0.01 #Критерій зупинки ВЧ

        self.newPop()

        its = 0
        
        time1 = t.perf_counter()
        
        while its < 10:
            
            # f_sum_1 = 0
            # for ind in self.pop: f_sum_1 += ind[-1]
            # f_sum_1/=self.size

            Pt = self.pop
            Pz, Ps, Pw = [], [], []
            


            # Pz
            while len(Pz) < self.size:

                i, j, k = random.randint(0,self.size-1), random.randint(0,self.size-1), random.randint(0,self.size-1)

                xi = Pt[i]
                xj = Pt[j]
                xk = Pt[k]

                dots = sorted([xi, xj, xk], key=lambda ind: ind[-1]) #знаходимо кразу точку
                
                x0 = [(dots[0][0][0]+dots[1][0][0]+dots[2][0][0])/3, (dots[0][0][1]+dots[1][0][1]+dots[2][0][1])/3] #центр трикутника(центр мас)
                
                # стиснення
                xi_ = [(1+a) * dots[0][0][0] - a * x0[0], (1+a) * dots[0][0][1] - a * x0[1]]
                xj_ = [ (dots[0][0][0] + a * dots[1][0][0]) / (1 + a),  (dots[0][0][1] + a * dots[1][0][1]) / (1 + a)  ]
                xk_ = [ (dots[0][0][0] + a * dots[2][0][0]) / (1 + a),  (dots[0][0][1] + a * dots[2][0][1]) / (1 + a)  ]

                Pz.append([xi_, self.func(xi_)])
                Pz.append([xj_, self.func(xj_)])
                Pz.append([xk_, self.func(xk_)])


                # поворот навколо кращої точки
                beta = random.randint(0,180) # кут повороту
                xj_ = [ dots[1][0][0] + (dots[1][0][0] - dots[0][0][0])*cos(beta) - (dots[1][0][1] - dots[0][0][1])*sin(beta),
                        dots[1][0][1] + (dots[1][0][0] - dots[0][0][0])*sin(beta) - (dots[1][0][1] - dots[0][0][1])*cos(beta)  ]
                
                xk_ = [ dots[2][0][0] + (dots[2][0][0] - dots[0][0][0])*cos(beta) - (dots[2][0][1] - dots[0][0][1])*sin(beta),
                        dots[2][0][1] + (dots[2][0][0] - dots[0][0][0])*sin(beta) - (dots[2][0][1] - dots[0][0][1])*cos(beta)  ]

                Ps.append([xj_, self.func(xj_)])
                Ps.append([xk_, self.func(xk_)])

                # поворот навколо центра
                al = random.randint(0,359) # кут повороту

                xi_ = [(dots[0][0][0] - x0[0]) * cos(al) - (dots[0][0][1] - x0[1]) * sin(al) + x0[0] ,
                       (dots[0][0][0] - x0[0]) * sin(al) + (dots[0][0][1] - x0[1]) * cos(al) + x0[1] ]

                xj_ = [(dots[1][0][0] - x0[0]) * cos(al) - (dots[1][0][1] - x0[1]) * sin(al) + x0[0] ,
                       (dots[1][0][0] - x0[0]) * sin(al) + (dots[1][0][1] - x0[1]) * cos(al) + x0[1] ]

                xk_ = [(dots[2][0][0] - x0[0]) * cos(al) - (dots[2][0][1] - x0[1]) * sin(al) + x0[0] ,
                       (dots[2][0][0] - x0[0]) * sin(al) + (dots[2][0][1] - x0[1]) * cos(al) + x0[1] ]

                Pw.append([xi_, self.func(xi_)])
                Pw.append([xj_, self.func(xj_)])
                Pw.append([xk_, self.func(xk_)])


            self.newPopT([Pt, Pz, Ps, Pw]) #елітний відбір нової популяції



            # перевірка критерії виходу з ітерації
            # VCH
            sum = 0
            for c in range(self.vars):
                sum += abs(self.pop[0][0][c] - self.pop[1][0][c])
            sum = sum / self.vars
            if sum <= VCH: break

            # VF 
            # f_sum_2 = 0
            # for ind in self.pop: f_sum_2 += ind[-1]
            # f_sum_2/=self.size
            # diff = abs(f_sum_1-f_sum_2)
            # if diff <= VF: break

            its+=1

        time2 = t.perf_counter()
        time = time2 - time1
        
        return self.getBestFit(), 'time: ', time, 'iter: ', its


    def iter_n(self, k_):

            VF = 0.0001 #Критерій зупинки ВФ
            VCH = 0.0001 #Критерій зупинки ВЧ
            self.newPop()
            its = 0
            time1 = t.perf_counter()
            
            while its < 1000:
                
                f_sum_1 = 0
                for ind in self.pop: f_sum_1 += ind[-1]
                f_sum_1/=self.size

                Pt = self.pop
                Pz, Pq, Pu = [], [], []

                # Pz
                while len(Pz) < self.size:

                    i_, j, k = random.randint(0,self.size-1), random.randint(0,self.size-1), random.randint(0,self.size-1)


                    dots = sorted([Pt[i_], Pt[j], Pt[k]], key=lambda ind: ind[-1]) #знаходимо кращу точку
                    best = dots[0][0]
                    xj = dots[1][0]
                    xk = dots[2][0]
                    # print(dots)

                    # центроїд
                    c = []
                    for n in range(self.vars):
                        sum = 0
                        for m in range(3):
                            sum += dots[m][0][n]
                        c.append(sum/3)
                    # print('c: ', c)

                    yi = [(k_*best[i] - c[i])/(k_-1) for i in range(self.vars)]
                    yj = [((k_-1)*xj[i] + yi[i])/k_ for i in range(self.vars)]
                    yk = [((k_-1)*xk[i] + yi[i])/k_ for i in range(self.vars)]
                    # print('yi ', yi, '\nyj ', yj, '\nyk ', yk)

                    Pz.append([yi, self.func(yi)])
                    Pz.append([yj, self.func(yj)])
                    Pz.append([yk, self.func(yk)])

                    # поворот
                    al = random.randint(0,180) # кут повороту
                    x_k, x_l = random.sample((0,self.vars-1), 2) #координати що будуть змінені для кожної точки

                    for dot in [best, xj, xk]:
                        
                        xk_val = best[x_k]*cos(al).real - best[x_l]*sin(al).real
                        xl_val = best[x_k]*sin(al).real + best[x_l]*cos(al).real

                        dot[x_k], dot[x_l] = xk_val, xl_val
   
                        Pq.append([dot, self.func(dot)])

                    # стиснення
                    zj = [(k_ * best[i] + xj[i])/(1 + k_) for i in range(self.vars)]
                    zk = [(k_ * best[i] + xk[i])/(1 + k_) for i in range(self.vars)]
                    # print('zj ', zj, '\nzk ', zk)

                    Pu.append([zj, self.func(zj)])
                    Pu.append([zk, self.func(zk)])

                self.newPopT([Pt, Pz, Pq, Pu]) #елітний відбір нової популяції

                # перевірка критерії виходу з ітерації
                # VCH
                # sum = 0
                # for c in range(self.vars):
                #     sum += abs(self.pop[0][0][c] - self.pop[1][0][c])
                # sum = sum / self.vars
                # print(sum)
                # if sum <= VCH: break

                # VF 
                f_sum_2 = 0
                for ind in self.pop: f_sum_2 += ind[-1]
                f_sum_2/=self.size
                diff = abs(f_sum_1-f_sum_2)
                # print(diff)
                if diff <= VF: break

                its+=1

            time2 = t.perf_counter()
            time = time2 - time1
            
            return self.getBestFit(), 'time: ', time, 'iter: ', its




if __name__ == "__main__":

    print(sin(90))
  

    alg_n1 = MODS(func_Beale, 1, 10, [[-4.5,4.5]] )
    res = alg_n1.iter((4.5 + 4.5)/5)

    print('Одновимірна ф-ія: ', res)


    alg_n2 = MODS(func_CrossInTray, 2, 30, [[-10,10],[-10,10]])
    #                    a
    res = alg_n2.iter_n2(2)

    print('Двовимірна ф-ія: ', res)
    n = 3
    ab = [[-10, 10]] * n
    alg_n2 = MODS(func_Rosenbrock, n, 20, ab)
    #                    k
    res = alg_n2.iter_n(5)

    print('n-вимірна ф-ія: ', res)

    