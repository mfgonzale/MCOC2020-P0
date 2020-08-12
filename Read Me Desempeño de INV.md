Tipos de datos:
np.half: 2 bytes
np.single: 4 bytes
np.double: 8 bytes
np.longdouble: 8 bytes

#timing_inv_caso_1_half:
import numpy as np
from numpy import *
from math import *
from time import perf_counter
from numpy import zeros, float32
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

Ns = [3,4,5]

def laplaciano(N, dtype=float32):
    matriz = zeros((N,N), dtype = dtype)
    for i in range(N):
        for j in range(N):
            if i==j:
                matriz[i,j] = 2
            if i+1==j:
                matriz[i,j] = -1
            if i-1==j:
                matriz[i,j] = -1
    return(matriz)



mem = []   #100000
dts = []
for N in Ns:
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = np.linalg.inv(A)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*2
    mem.append(size)
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_1_half.png")
plt.show()

#timing_inv_caso_1_single:

for N in Ns:
    mem = []   #100000
    dts = []
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = np.linalg.inv(A)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*4
    mem.append(size)
    
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_1_single.png")
plt.show()

#timing_inv_caso_1_double:

for N in Ns:
    mem = []   #100000
    dts = []
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = np.linalg.inv(A)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*8
    mem.append(size)
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_1_double.png")
plt.show()

#timing_inv_caso_1_longdouble:

for N in Ns:
    mem = []   #100000
    dts = []
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = np.linalg.inv(A)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*8
    mem.append(size)
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_1_longdouble.png")
plt.show()

#timing_inv_caso_2_half:

mem = []   #100000
dts = []
for N in Ns:
    mem = []   #100000
    dts = []
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = linalg.inv(A, overwrite_a=False)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*2
    mem.append(size)
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_2_half.png")
plt.show()

#timing_inv_caso_2_single:

mem = []   #100000
dts = []
for N in Ns:
    mem = []   #100000
    dts = []
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = linalg.inv(A, overwrite_a=False)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*4
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_2_single.png")
plt.show()

#timing_inv_caso_2_double:

mem = []   #100000
dts = []
for N in Ns:
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = linalg.inv(A, overwrite_a=False)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*8
    
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_2_double.png")
plt.show()

#timing_inv_caso_2_longdouble:

mem = []   #100000
dts = []
for N in Ns:
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = linalg.inv(A, overwrite_a=False)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*8
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_2_longdouble.png")
plt.show()

#timing_inv_caso_3_half:

mem = []   #100000
dts = []
for N in Ns:
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = linalg.inv(A, overwrite_a=True)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*2
    
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_3_half.png")
plt.show()

#timing_inv_caso_3_single:

mem = []   #100000
dts = []
for N in Ns:
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = linalg.inv(A, overwrite_a=True)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*4
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_3_single.png")
plt.show()

#timing_inv_caso_3_double:

mem = []   #100000
dts = []
for N in Ns:
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = linalg.inv(A, overwrite_a=True)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*8
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_3_double.png")
plt.show()

#timing_inv_caso_3_longdouble:

mem = []   #100000
dts = []
for N in Ns:
    A = laplaciano(N, float32)
    t1 = perf_counter()
    A_inv = linalg.inv(A, overwrite_a=True)
    t2 = perf_counter()
    dt = t2 - t1
    dts.append(dt)
    size = (N**2)*8
    
fig, ax = plt.subplots(2,1) 
    
ax[0].loglog(Ns,dts)   
ax[0].set_ylabel("Tiempo transcurrido")
ax[0].set_title("Rendimiento matriz invertida")

ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])
ax[0].set_xlim(10,20000)
ax[0].grid(True)

ax[1].loglog(Ns,mem) 
ax[1].set_xlabel("Tamaño matriz N")
ax[1].set_ylabel("Uso memoria")

ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])
ax[1].set_xlim(10,20000)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("timing_inv_caso_3_longdouble.png")
plt.show()

