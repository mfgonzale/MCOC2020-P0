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
