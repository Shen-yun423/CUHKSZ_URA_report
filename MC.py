import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.function_base import average
import matplotlib.pyplot as plt 
# S = linspace(0,0.5,21)
S = np.array([0.03])
s  =0.3
t = 0.5
q = (1-s)/6
tau_s = 3
tau_t = 1
tau_d = 1
Q1 = np.array([[0, t/3, t/3, t/3],
               [t/3, 0, t/3, t/3],
               [2*q, 2*q, 0, 2*q],
               [2*q, 2*q, 2*q, 0]
               ])
p1 = np.array([[1, 2/3, 2/3, 2/3]])

Q2 = np.array([[0, t],
               [3*q, 3*q]
               ])
p2 = np.array([[1, 2]])

Q3 = np.array([[0, t],
               [q, 5*q]
               ])
p3 = np.array([[1, 6]])

Q4 = np.array([[0, t, 0, 0],
               [q, 2*q, 2*q, q],
               [0, 4*q, 0, 2*q],
               [0, 3*q, 3*q, 0]
               ])
p4 = np.array([[1, 6, 3, 2]])

Q5 = np.array([[0, t, 0, 0],
               [q, 2*q, 2*q, q],
               [0, 2*q, q, 3*q],
               [0, q, 3*q, 2*q]
               ])
p5 = np.array([[1, 6, 6, 6]])

Q6 = np.array([[0, t, 0, 0, 0, 0],
               [q, 2*q, 2*q, q, 0, 0],
               [0, 2*q, 0, 2*q, q, q],
               [0, q, 2*q, 0, 2*q, q],
               [0, 0, 2*q, 2*q, q, q],
               [0, 0, 0, 3*q, 3*q, 0]
               ])
p6 = np.array([[1, 6, 6, 6, 6, 2]])

Q7 = np.array([[0, t, 0, 0, 0, 0, 0],
               [q, 2*q, 2*q, q, 0, 0, 0],
               [0, 2*q, 0, 2*q, q, q, 0],
               [0, q, 2*q, 0, q, q, q],
               [0, 0, q, q, q, 2*q, q],
               [0, 0, q, q, 2*q, 0, 2*q],
               [0, 0, 0, q, q, 2*q, 2*q]
               ])
p7 = np.array([[1, 6, 6, 6, 6, 6, 6]])


def func(Q,p,num):
    dim = Q.shape[0]
    Q_m = np.matrix(Q)
    I_m = np.matrix(np.eye(dim))
    A_m = I_m - Q_m
    ksi = np.matrix(np.ones((dim,1))*(s*tau_s + (1-s)*tau_d)) 
    ksi[0] = (1-t)*tau_t + t*tau_d 
    if num == 1:
        ksi[1] = (1-t)*tau_t + t*tau_d 
    
    A_I = A_m.I
    n_vec = A_I * ksi
    coeff = np.matrix(p)
    avg = coeff * n_vec / coeff.sum()
    return np.array(avg).flatten()

list_N = [2 ,3 ,7, 12, 19, 27, 37]
list_Q = [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
list_p = [p1, p2, p3, p4, p5, p6 ,p7]
ratio = []
avg = []
for i in range(7):
    ratio.append(1/list_N[i])
    avg.append(*func(list_Q[i],list_p[i],i+1))
    print(list_N[i], '\t', 1/list_N[i], end='\t')
    print(func(list_Q[i],list_p[i],i+1))

plt.figure(dpi=100)
plt.plot(ratio,avg,'k-o',)
plt.xlabel('# of trap T / # of trap s')
plt.ylabel('Average Reaction Time')
plt.title('Reaction Time vs. # Ratio of Two Traps')
plt.show()

# for s in S:
#     q = (1-s)/6
#     A = np.array([[1, -t, 0, 0, 0, 0],
#                 [-q, 1, -2*q, -q, 0, 0],
#                 [0, -2*q, 1, 0, -2*q, 0],
#                 [0, -q, 0, 1-q, -2*q, 0],
#                 [0, 0, -q, -q, 1-q, -q],
#                 [0, 0, 0, 0, -2*q, 1-2*q]
#                 ])
#     A_m = np.matrix(A)
#     uni_vec = np.matrix(np.ones((6,1)))
#     A_I = A_m.I

#     n_vec = A_I * uni_vec
#     n_avg = 1/24 * (4*n_vec[1,0] + 4*n_vec[2,0] + 4*n_vec[3,0] + 8*n_vec[4,0] + 4*n_vec[5,0])
#     # np.around(n_vec,4)
#     print(np.around(s,2),end='\t')
#     print(*np.around(np.array(n_vec.T),4).flatten(),sep='\t',end='\t')
#     print(np.around(n_avg,4))


# Monte Carlo
# def terminate(start,center):
#     if (start[0] == center) and (start[1] == center):
#         return True
#     else:
#         return False
# def walk(start,s=0):
#     direction = np.random.choice([0,1]) 
#     step = np.random.choice([-1,1],size=(2)) * (1 - np.random.binomial(1,s))
#     step[direction] = 0
#     next_position = (start + step) % 5
#     return next_position

# def average_walk(N):
#     Lattice_unit = np.zeros((N,N)) 
#     center = (N - 1) / 2 
#     Lattice_unit[center,center] = 1
#     average_walk = 0
#     for i in range(10):
#         start = np.random.randint(5,size=(2))
#         steps = 0
#         while not terminate(start,center):
#             start = walk(start)
#             steps += 1
#         average_walk = (average_walk * i + steps) / (i + 1)

#     print(average_walk)