# MCOC2020-P0
Mi computador  

Marca/Modelo: Lenovo S41-70  
Tipo: Notebook  
Año de adquisición: 2016    

Procesador:  

Marca/Modelo: Intel(R) Core(TM) i3-5005U  
Velocidad base: 2.00GHz  
Velocidad máxima: 2.00GHz  
Numero de nucleos:2  
Numero de hilos:4  
Arquitectura: 64 bits procesador x64  
Set de instrucciones: Intel® SSE4.1, Intel® SSE4.2, Intel® AVX2  

Tamaños de los cachés del procesador:  
L1d: 2x32 KB  
L1i: 2x32 KB  
L2: 2x256 KB  
L3: 3 MB  

Memoria:  

Total: 8 GB  
Tipo memoria: DDR3  
Velocidad 1600 MHz  
Numero de (SO)DIMM: 1  

Tarjeta Gráfica:  

Marca / Modelo: Gráfica integrada: Inter(R) HD Graphics 5500  

Marca / Modelo: Gráfica dedicada: NVIDIA GeForce GT940M  
                Memoria de video dedicada: 2 GB  
Resolución: 1366 x 768  

Disco 1:  

Marca: Western Digital  
Tipo: HDD  
Tamaño: 1TB  
Particiones: 2  
Sistema de archivos: NTFS  



Dirección MAC de la tarjeta wifi: CC-B0-DA-A6-AC-DD  

Dirección IP (Interna, del router): 192.168.0.3  

Dirección IP (Externa, del ISP): 190.160.0.15  

Proveedor internet: VTR Banda Ancha S.A.  


#Entrega 2:

from scipy import matrix, rand, savetxt  
from time import perf_counter  
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker  
import numpy as np  
from numpy import *  
from math import *  
from mimatmul import mimatmul  
  
  
Ns = [2, 5, 10, 12, 15, 20, 30, 40, 45, 50, 55, 60, 75, 100, 125, 160, 200,250, 350, 500, 600, 800, 1000, 2000, 5000, 10000]  
  
Ncorridas = 10  
  
  
for i in range(Ncorridas):  
    dts = []  
    mem = []  
    name = (f"matmul{i}.txt")  
    fid = open(name, "w")  
    for N in Ns:  
        
        A = np.random.rand(N,N)
        B = np.random.rand(N,N)
        
        t1 = perf_counter()
        C = A@B
        t2 = perf_counter()
        
        dt = t2 - t1
        
        size = 3*(N**2)*8
        
        dts.append(dt)
        mem.append(size)
        
        fid.write(f"{N} {dt} {size}\n")
        
        fid.flush()
        
    fid.close()
     

j = 0  
matriz_N = []  
matriz_dt = []  
matriz_mem = []  
while j < 10:  
    with open(f'matmul{j}.txt','r') as f:  
        lista_N = []  
        lista_dt = []  
        lista_mem = []  
        for linea in f:  
            datos = linea.split(" ")  
            lista_N.append(float(datos[0]))  
            lista_dt.append(float(datos[1]))  
            lista_mem.append(float(datos[2]))  
        matriz_N.append(lista_N)  
        matriz_dt.append(lista_dt)  
        matriz_mem.append(lista_mem)  
          
        f.close()  
    j+=1  
      
fig, ax = plt.subplots(2,1)    
      
a = 0  
while a < 10:    
        ax[0].loglog(matriz_N[a],matriz_dt[a])  
        a+=1  

b = 0  
while b < 10:   
        ax[1].loglog(matriz_N[b],matriz_mem[b])  
        b+=1  
      
      
        
#1)    
ax[0].set_ylabel("Tiempo transcurrido")  
ax[0].set_title("Rendimiento A@B")  
ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])  
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)  
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])  
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])  
ax[0].set_xlim(10,20000)  
ax[0].grid(True)  
  
#2)  
ax[1].set_xlabel("Tamaño matriz N")  
ax[1].set_ylabel("Uso memoria")  
ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])  
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)  
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])  
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])  
ax[1].set_xlim(10,20000)  
ax[1].grid(True)  
  
plt.tight_layout()  
plt.savefig("matmul.png")  
plt.show() 
  
  
#Entrega 3: 
  
def mimatmul(m1,m2):  
    r=[]  
    m=[]  
    for i in range(len(m1)):  
        for j in range(len(m2[0])):  
            sums=0  
            for k in range(len(m2)):  
                sums=sums+(m1[i][k]*m2[k][j])  
            r.append(sums)  
        m.append(r)  
        r=[]  
    return m  
  
from scipy import matrix, rand, savetxt  
from time import perf_counter  
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker  
import numpy as np  
from numpy import *  
from math import *  
from mimatmul import mimatmul  
  
  
Ns = [2, 5, 10, 12, 15, 20, 30, 40, 45, 50, 55, 60, 75, 100, 125, 160, 200,250, 350, 500, 600, 800, 1000, 2000, 5000, 10000]  
  
Ncorridas = 10  
  
  
for i in range(Ncorridas):  
    dts = []  
    mem = []  
    name = (f"matmul{i}.txt")  
    fid = open(name, "w")  
    for N in Ns:  
        
        A = np.random.rand(N,N)  
        B = np.random.rand(N,N)  
          
        t1 = perf_counter()  
        C = mimatmul(A,B)  
        t2 = perf_counter()  
          
        dt = t2 - t1  
          
        size = 3*(N**2)*8  
          
        dts.append(dt)  
        mem.append(size)  
          
        fid.write(f"{N} {dt} {size}\n")  
          
        fid.flush()  
          
    fid.close()  
       
  
j = 0  
matriz_N = []  
matriz_dt = []  
matriz_mem = []  
while j < 10:  
    with open(f'matmul{j}.txt','r') as f:  
        lista_N = []  
        lista_dt = []  
        lista_mem = []  
        for linea in f:  
            datos = linea.split(" ")  
            lista_N.append(float(datos[0]))  
            lista_dt.append(float(datos[1]))  
            lista_mem.append(float(datos[2]))  
        matriz_N.append(lista_N)  
        matriz_dt.append(lista_dt)  
        matriz_mem.append(lista_mem)  
          
        f.close()  
    j+=1  
      
fig, ax = plt.subplots(2,1)    
      
a = 0  
while a < 10:   
        ax[0].loglog(matriz_N[a],matriz_dt[a])  
        a+=1  

b = 0  
while b < 10:   
        ax[1].loglog(matriz_N[b],matriz_mem[b])  
        b+=1  
      
      
        
#1)    
ax[0].set_ylabel("Tiempo transcurrido")  
ax[0].set_title("Rendimiento A@B")  
ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])  
ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)  
ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])  
ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])  
ax[0].set_xlim(10,20000)  
ax[0].grid(True)  
  
#2)  
ax[1].set_xlabel("Tamaño matriz N")  
ax[1].set_ylabel("Uso memoria")  
ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])  
ax[1].set_xticklabels([10,20,50,100,200,500,1000,2000,5000,10000,20000],rotation=45)  
ax[1].set_yticks([100,1000,10000,100000,1000000,10000000,100000000,1000000000])  
ax[1].set_yticklabels(["1 KB","10 KB","100 KB","1 MB","10 MB","100 MB","1 G","10 GB"])  
ax[1].set_xlim(10,20000)  
ax[1].grid(True)  
  
plt.tight_layout()  
plt.savefig("matmul.png")  
plt.show() 
  
  
#Entrega 4:  
  
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
  
mem = []  
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
mem = []  
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
  mem = []  
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
  mem = []  
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
  
mem = []  
dts = []  
for N in Ns:  
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
  
mem = []  
dts = []  
for N in Ns:  
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
  
mem = []  
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
  
mem = []  
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
  
mem = []  
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
  
mem = []  
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
  
mem = []  
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
  
mem = []  
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
  
  
#Entrega 5:  
  
from time import perf_counter  
import scipy as sp  
import scipy.linalg as spLinalg  
import numpy as np  
from numpy import float32, zeros  
import matplotlib.pyplot as plt  
  
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
  
#Tamaños matrices   
Ns = [2,5,10,12,15,20,30,40,45,50,55,60,75,100,125,160,200,250,350,500,600,800,1000]  
  
Ncorridas = 10  
  
names = ["A_invB_inv.txt", "A_invB_npSolve.txt"]  
  
files = [open(name, "w") for name in names]  
  
for N in Ns:  
    dts = np.zeros((Ncorridas, len(files)))  
    print (f"N = {N}")  
      
    for i in range(Ncorridas):  
        print(f"i = {i}")  
          
        #Invirtieno y multiplicando  
        A = laplaciano(N)  
        B = np.ones(N)  
        t1 = perf_counter()  
        A_inv = np.linalg.inv(A)  
        A_invB = A_inv@B  
        t2 = perf_counter()  
        dt = t2 - t1  
        dts[i][0] = dt  
          
          
        #Ocupando np.linalg.solve(A, B)  
        A = laplaciano(N)  
        B = np.ones(N)  
        t1 = perf_counter()  
        A_invB = np.linalg.solve(A, B)  
        t2 = perf_counter()  
        dt = t2 - t1  
        dts[i][1] = dt  
          
    print("dts: ", dts)  
      
    dts_mean = [np.mean(dts[:,j]) for j in range(len(files))]  
      
    print("dts_mean: ", dts_mean)  
      
    #Escribo en el archivo de texto los resultados  
    for j in range(len(files)):  
        files[j].write(f"{N} {dts_mean[j]}\n")  
        files[j].flush()  
[file.close() for file in files]  
  
     
def plotting(names):  
      
    xtks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]  
      
    ytks1 = [0.1e-3, 1e-3, 1e-2, 0.1, 1., 10., 60, 60 * 10]  
    ytklabs1 = ["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"]  
      
    plt.figure()  
      
      
    for name in names:  
        data = np.loadtxt(name)  
        Ns = data[:, 0]  
        dts = data[:, 1]  
          
        print("Ns: ", Ns)  
        print("dts: ", dts)  
          
        plt.loglog(Ns, dts.T, "-o", label=name)  
        plt.ylabel("Tiempo transcurrido (s)")  
        plt.xlabel("Tamaño matriz $N$")  
        plt.grid(True)  
        plt.xticks(xtks,xtks,rotation=45)  
        plt.yticks(ytks1,ytklabs1)  
          
    plt.tight_layout()  
    plt.legend()  
    plt.show()  
    plt.savefig("plot.png")  
      
names = ["A_invB_inv.txt", "A_invB_npSolve.txt"]  
plotting(names)  
  
#Archivos de texto entrega 5:  
  
A_invB_inv:  
  
  
2 0.0001245499997821753  
5 9.270000009564683e-05  
10 0.00012679999963438604  
12 0.00011475000064820051  
15 0.0001687499998297426  
20 8.915000034903642e-05  
30 0.0001830000010158983  
40 0.0002889500001401757  
45 0.00024109999958454864  
50 0.00032929999997577397  
55 0.001593900000443682  
60 0.0005154999998921994  
75 0.0005118000008224044  
100 0.0010457999997015577  
125 0.001239299999724608  
160 0.0019525499992596451  
200 0.01344370000060735  
250 0.003025700000762299  
350 0.0073021499993046746  
500 0.02221654999993916  
600 0.030557449999832897  
800 0.05846744999962539  
1000 0.08278839999911725  
  
  
A_invB_npSolve:  
  
  
2 8.444999912171625e-05  
5 6.865000068501104e-05  
10 0.00011014999927283498  
12 8.615000024292385e-05  
15 0.00040439999884256395  
20 5.580000015470432e-05  
30 0.0009767999999894528  
40 0.0002287500001330045  
45 0.0003261499996369821  
50 0.00023194999994302634  
55 0.00030605000029026996  
60 0.0002678500004549278  
75 0.0018878000000768225  
100 0.0011091500000475207  
125 0.0008554499991078046  
160 0.0009740000004967442  
200 0.005398449999120203  
250 0.0025776999991649063  
350 0.008050350000303297  
500 0.02073290000043926  
600 0.020971549999558192  
800 0.07227399999919726  
1000 0.0813138000003164  



#Entrega 6:  
  
from time import perf_counter  
import scipy as sp  
import scipy.linalg as spLinalg  
import numpy as np  
from numpy import float32, zeros  
import matplotlib.pyplot as plt  
from scipy import linalg  
  
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
  
#Tamaños matrices   
Ns = [2,5,10,12,15,20,30,40,45,50,55,60,75,100,125,160,200,250,350,500,600,800,1000]  
  
Ncorridas = 10  
  
names = ["A_invB_inv.txt", "A_invB_npSolve.txt", "A_invB_spSolve.txt", "A_invB_spSolve_symmetric.txt", "A_invB_spSolve_pos.txt", "A_invB_spSolve_pos_overwrite.txt"]  
  
files = [open(name, "w") for name in names]  
  
for N in Ns:  
    dts = np.zeros((Ncorridas, len(files)))  
    print (f"N = {N}")   
      
    for i in range(Ncorridas):  
        print(f"i = {i}")  
          
        #Invirtieno y multiplicando  
        A = laplaciano(N)  
        B = np.ones(N)  
        t1 = perf_counter()  
        A_inv = np.linalg.inv(A)  
        A_invB = A_inv@B  
        t2 = perf_counter()  
        dt = t2 - t1  
        dts[i][0] = dt  
          
          
        #Ocupando np.linalg.solve(A, B)  
        A = laplaciano(N)  
        B = np.ones(N)  
        t1 = perf_counter()  
        A_invB = np.linalg.solve(A, B)  
        t2 = perf_counter()  
        dt = t2 - t1  
        dts[i][1] = dt  
          
        #Ocupando sp.linalg.solve  
        A = laplaciano(N)  
        B = np.ones(N)  
        t1 = perf_counter()  
        A_invB = linalg.solve(A, B)  
        t2 = perf_counter()  
        dt = t2 - t1  
        dts[i][2] = dt  
          
        #Ocupando symmetric=True  
        A = laplaciano(N)  
        B = np.ones(N)  
        t1 = perf_counter()  
        A_invB = linalg.solve(A, B, sym_pos=True)  
        t2 = perf_counter()  
        dt = t2 - t1  
        dts[i][3] = dt  
          
        #Ocupando assume_a = ‘pos’   
        A = laplaciano(N)  
        B = np.ones(N)  
        t1 = perf_counter()  
        A_invB = linalg.solve(A, B, assume_a="pos")  
        t2 = perf_counter()  
        dt = t2 - t1  
        dts[i][4] = dt  
          
        #Ocupando overwrite=True  
        A = laplaciano(N)  
        B = np.ones(N)  
        t1 = perf_counter()  
        A_invB = linalg.solve(A, B, assume_a="pos", overwrite_a=True)  
        t2 = perf_counter()  
        dt = t2 - t1  
        dts[i][5] = dt  
          
    print("dts: ", dts)  
      
    dts_mean = [np.mean(dts[:,j]) for j in range(len(files))]  
      
    print("dts_mean: ", dts_mean)  
      
    #Escribo en el archivo de texto los resultados  
    for j in range(len(files)):  
        files[j].write(f"{N} {dts_mean[j]}\n")  
        files[j].flush()  
[file.close() for file in files]  
  
     
def plotting(names):  
      
    xtks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]  
      
    ytks1 = [0.1e-3, 1e-3, 1e-2, 0.1, 1., 10., 60, 60 * 10]  
    ytklabs1 = ["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"]  
      
    plt.figure()  
      
      
    for name in names:  
        data = np.loadtxt(name)  
        Ns = data[:, 0]  
        dts = data[:, 1]  
          
        print("Ns: ", Ns)  
        print("dts: ", dts)  
          
        plt.loglog(Ns, dts.T, "-o", label=name)  
        plt.ylabel("Tiempo transcurrido (s)")  
        plt.xlabel("Tamaño matriz $N$")  
        plt.grid(True)  
        plt.xticks(xtks,xtks,rotation=45)  
        plt.yticks(ytks1,ytklabs1)  
          
    plt.tight_layout()  
    plt.legend()  
    plt.show()  
    plt.savefig("plot.png")  
      
names = ["A_invB_inv.txt", "A_invB_npSolve.txt", "A_invB_spSolve.txt", "A_invB_spSolve_symmetric.txt", "A_invB_spSolve_pos.txt", "A_invB_spSolve_pos_overwrite.txt"]  
plotting(names)  
  
  Se puede notar que el desempeño de la primera función es mejor por estar recien empezando el procesador a correr el codigo, y a medida que sigue tomando las otras funciones va bajando en general levemente su desempeño.  


Entrega 7:  
Matrices dispersas y complejidad computacional  
  
from numpy import *  
from time import perf_counter  
from scipy.linalg import solve  
from matplotlib.pyplot import spy  
import matplotlib.pyplot as plt  
from numpy import float32, zeros, ones  
from scipy.sparse import lil_matrix  
from scipy import linalg, zeros  
import numpy as np  
from scipy.sparse import csr_matrix  
from scipy.sparse.linalg import spsolve  
from scipy.sparse.linalg import inv  
import scipy as sp  
  
  
Ns = [100, 200, 400, 800, 1000]  
Ncorridas = 10  
  
def laplaciano_llena(N, dtype=float32):  
    matriz = np.zeros((N,N), dtype = dtype)  
    for i in range(N):  
        for j in range(N):  
            if i==j:  
                matriz[i,j] = 2  
            if i+1==j:  
                matriz[i,j] = -1  
            if i-1==j:  
                matriz[i,j] = -1  
    return(matriz)  
  
  
def laplaciano_dispersa(N, dtype=float32):  
    matriz = lil_matrix((N,N), dtype = dtype) #es como decir zeros   
    for i in range(N):  
        for j in range(N):  
            if i==j:  
                matriz[i,j] = 2  
            if i+1==j:  
                matriz[i,j] = -1  
            if i-1==j:  
                matriz[i,j] = -1  
    return(matriz)  
   
names = ["Caso1_matriz_llena.txt", "Caso1_matriz_dispersa.txt", "Caso2_matriz_llena.txt", "Caso2_matriz_dispersa.txt", "Caso3_matriz_llena.txt", "Caso3_matriz_dispersa.txt"]  
  
files = [open(name, "w") for name in names]  
  
for N in Ns:  
    dts1 = np.zeros((Ncorridas, len(files)))  
    dts2 = np.zeros((Ncorridas, len(files)))  
    print (f"N = {N}")  
      
    for i in range(Ncorridas):  
        print(f"i = {i}")  
        #Caso 1  
        #Matrices llenas  
  
        t1 = perf_counter()  
      
        A = laplaciano_llena(N)  
        B = laplaciano_llena(N)  
          
        t2 = perf_counter()  
          
        C = A@B  
      
        t3 = perf_counter()  
              
        dts1[i][0] = t2 - t1  
        dts2[i][0] = t3 - t2  
          
        #Matriz dispersa  
          
        t1 = perf_counter()  
          
        A_lil = laplaciano_dispersa(N)  
        A_csr = csr_matrix(A_lil)  
        B_lil = laplaciano_dispersa(N)  
        B_csr = csr_matrix(B_lil)  
          
        t2 = perf_counter()  
          
        C = A_csr@B_csr  
      
        t3 = perf_counter()  
          
        dts1[i][1] = t2 - t1  
        dts2[i][1] = t3 - t2  
          
          
        #Caso 2:  
        #Matriz llena  
          
        #print(f"N = {N}")  
        t1 = perf_counter()  
  
        A = laplaciano_llena(N)  
        b = ones(N)  
      
        t2 = perf_counter()  
      
        x = solve(A,b)  
  
        t3 = perf_counter()  
      
        dts1[i][2] = t2 - t1  
        dts2[i][2] = t3 - t2  
          
        #matriz dispersa  
          
        t1 = perf_counter()  
        A_lil = laplaciano_dispersa(N)  
        A_csr = csr_matrix(A_lil)  
        b = ones(N)  
          
        t2 = perf_counter()  
      
        x = spsolve(A_csr,b)  
  
        t3 = perf_counter()  
          
        dts1[i][3] = t2 - t1  
        dts2[i][3] = t3 - t2  
          
        #Caso 3  
          
        #Matriz llena  
  
        t1 = perf_counter()  
        A = laplaciano_llena(N)  
      
        t2 = perf_counter()  
      
        A_inv = sp.linalg.inv(A)  
  
        t3 = perf_counter()  
      
        dts1[i][4] = t2 - t1  
        dts2[i][4] = t3 - t2  
          
        #Matriz dispersa  
          
        t1 = perf_counter()  
      
        A_lil = laplaciano_dispersa(N)  
        A_csr = csr_matrix(A_lil)  
      
        t2 = perf_counter()  
      
        A_inv = sp.sparse.linalg.inv(A_csr)  
  
        t3 = perf_counter()  
      
        dts1[i][5] = t2 - t1  
        dts2[i][5] = t3 - t2  
      
    print("dts1: ", dts1)   
    print("dts2: ", dts2)           
    dts1_mean = [np.mean(dts1[:,j]) for j in range(len(files))] #promedio de cada columna  
    dts2_mean = [np.mean(dts2[:,j]) for j in range(len(files))] #promedio de cada columna  
     
      
    print("dts1_mean: ", dts1_mean)  
    print("dts2_mean: ", dts2_mean)  
    #Escribo en el archivo de texto los resultados  
    for j in range(len(files)):  
        files[j].write(f"{N} {dts1_mean[j]} {dts2_mean[j]}\n")  
        files[j].flush()  
[file.close() for file in files]  
  
  
  
def plotting(names):  
      
    plt.figure()  
      
      
    for name in names:  
        data = np.loadtxt(name)  
        Ns = data[:, 0]  
        dts1 = data[:, 1]  
        dts2 = data[:, 2]  
          
        print("Ns: ", Ns)  
        print("dts1: ", dts1)  
        print("dts2: ", dts2)  
          
        fig, ax = plt.subplots(2,1)    
  
        a = 0  
        while a < 5:     
            ax[0].loglog(Ns[a],dts1[a],"-o")  
            a+=1  
  
        b = 0  
        while b < 5:     
            ax[1].loglog(Ns[b],dts2[b],"-o")  
            b+=1  
          
          
        ax[0].set_ylabel("Tiempo de ensamblado (s)")  
  
        ax[0].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])  
        ax[0].set_xticklabels(["","","","","","","","","","",""],rotation=45)  
        ax[0].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])  
        ax[0].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])  
        ax[0].set_xlim(10,20000)  
        ax[0].grid(True)  
          
  
        ax[1].set_ylabel("Tiempo de solucion (s)")  
        ax[1].set_xlabel("Tamaño matriz $N$")  
        ax[1].set_xticks([10,20,50,100,200,500,1000,2000,5000,10000,20000])  
        ax[1].set_xticklabels(["10","20","50","100","200","500","1000","2000","5000","10000","20000"],rotation=45)  
        ax[1].set_yticks([0.0001,0.001,0.01,0.1,1,10,60,600])  
        ax[1].set_yticklabels(["0.1 ms","1 ms","10 ms","0.1 s","1 s","10 s","1 min","10 min"])  
        ax[1].set_xlim(10,20000)  
        ax[1].grid(True)  
          
    plt.tight_layout()  
    plt.legend()  
    plt.show()  
    plt.savefig("matrices_llenas_y_dispersas.png")  
      
names = ["Caso1_matriz_llena.txt", "Caso1_matriz_dispersa.txt", "Caso2_matriz_llena.txt", "Caso2_matriz_dispersa.txt", "Caso3_matriz_llena.txt", "Caso3_matriz_dispersa.txt"]  
plotting(names)  
  
