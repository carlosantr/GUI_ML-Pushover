# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:04:40 2022

@author: User
"""
#Este código tiene la función para generar una curva pushover y sus puntos importantes

#%%Importando librerias necesarias
from openseespy.opensees import * #Opensees
import opsvis as opsv #Opsvis para visualización
import matplotlib.pyplot as plt #Figuras y gráficas
import Analisis as an #Libreria de análisis (gravedad y pushover)
import numpy as np #Funciones math
import pandas as pd #Dataframes y análisis de datos
import time #Toma el tiempo
from joblib import Parallel, delayed #Poner a trabajar el PC en paralelo (más rápido)
from utilidades import BuildRCSection, DimRefuerzo, Puntos, DimViga,comprobacion_norma
import multiprocessing #Contar cpu del PC

#%%Definición de variables fijas
n=100 #Número de modelos a generar
diafragma = 1 #Diafragma de la edificación (colocar 1 si se desea diafragma en cada piso)
pushlimit = 0.05 #Límite del pushover

#%%% Funciones de generación (encargadas de creación de modelo en Opensees)   
def DefModel():#Esta función crea el modelo en Opensees
    wipe()
    model('basic','-ndm',2,'-ndf',3)#Parámetros del modelo 

def DefDimEd(LonY,LonX,ny,nx,yloc,xloc):#Esta función genera los vectores de coordenas de los nodos del edificio
    for i in range(1,ny):
        yloc.append(LonY*i)
    for j in range(1,nx):
        xloc.append(LonX*j)     
    return xloc,yloc

def DefNodos(ny,nx,yloc,xloc):#Esta función crea los nodos y asigna las restricciones del edificio
    #Nodos
    for i in range(nx):
        for j in range(ny):
            nnode = 1000*(i+1)+j
            node(nnode,xloc[i],yloc[j])
    #Apoyos
    empotrado = [1,1,1]
    grado2 = [1,1,0]
    #Empotramiento (nivel 0)
    fixY(0.0,*empotrado)
    
def DefDiafragma(ny,nx):#Esta función asigna los diafragmas del edificio
    if diafragma == 1:
        for j in range(1,ny):
            for i in range(1,nx):
                masternode = 1000 + j
                slavenode = 1000*(i+1) + j
                print(masternode,slavenode)
                equalDOF(masternode,slavenode,1)
                
def DefMat(fc,Fy):#Esta función crea los materiales del edificio y define parámetros del concreto confinado y no confinado
    E = 24000000.0 #Modulo de elasticidad concreto
    ec = 2*fc/E
    fcu = 0.2*fc
    ecu = 0.006
     
    k=1.3
    fcc=fc*k
    ecc= 2*fcc/E
    fucc=0.2*fcc
    eucc=0.02
     
    Es=210000000.0#Modulo de elasticidad acero
    
    uniaxialMaterial('Concrete02', 2, -fc, -ec, -fcu, -ecu) #Concreto no confinado
    uniaxialMaterial('Concrete02', 1, -fcc, -ecc, -fucc, -eucc) #Concreto confinado
    uniaxialMaterial('Steel01', 4, Fy, Es, 0.01)
    uniaxialMaterial('MinMax', 3 , 4, '-min', -0.008, '-max', 0.05) #Acero (MinMax para definir límites)
    
#%%Función de generación de pushover
def PushoverData(Ny,Nx,Ly,Lx,Fc,w,B_col,H_col,B_vig,H_vig,Cuantia_Col,Cuantia_Vig_Sup,Cuantia_Vig_Inf):       
    #Inicio de análisis pushover
    for ESP in [0.001,0.0008,0.0012,0.0006]:#Este ciclo es para definir el paso de la pushover con el que se obtenga un buen resultado (en caso de que no logre converger con algun paso)
        
        #Se llaman las funciones para crear modelo
        nx=Nx+1;ny=Ny+1;xloc=[0.0];yloc=[0.0]
        DefModel()
        xloc,yloc=DefDimEd(Ly,Lx,ny,nx,yloc,xloc)
        DefNodos(ny,nx,yloc,xloc)
        DefDiafragma(ny,nx)
        Fy=420000.00
        DefMat(Fc,Fy)
        
        #%% Creando secciones del modelo
        recub=0.05#Recubrimiento de las secciones (normalmente en edificios es de 5cm)

        #COLUMNA
        As_C=Cuantia_Col*B_col*H_col #Acero necesario
        n_barras_C,A_barras_C,nLineasAcero_C=DimRefuerzo(B_col,As_C,'Columna') #Define el acero comercial y distribución necesaria en columna según cuantía
        Cuantia_Col_Real=(sum(n_barras_C)*A_barras_C[0])/(B_col*H_col)
        Sec_Col=1 #ID de sección de columna
        nInt_C=sum(n_barras_C)-n_barras_C[0]-n_barras_C[-1] #Número de barras intermedias en la columna
        BuildRCSection(Sec_Col,H_col,B_col,recub,recub,1,2,3,n_barras_C[0],A_barras_C[0],n_barras_C[-1],A_barras_C[-1],nInt_C,A_barras_C[1],15,15,20,3) #Creación de columna
        nB_C=n_barras_C
        n_Barras_C=sum(n_barras_C)
        
        #VIGA
        As_V=[Cuantia_Vig_Sup*B_vig*(H_vig-recub),Cuantia_Vig_Inf*B_vig*(H_vig-recub)] #Acero necesario en la parte superior e inferior de la viga
        n_barras_V,A_barras_V,nLineasAcero_V=DimRefuerzo(B_vig,As_V,'Viga')  #Define el acero comercial y distribución necesaria en viga según cuantías
        Cuantia_Vig_Sup_Real=(n_barras_V[0]*A_barras_V[0])/(B_vig*(H_vig-recub))
        Cuantia_Vig_Inf_Real=(n_barras_V[-1]*A_barras_V[-1])/(B_vig*(H_vig-recub))
        n_Barras_V_Sup=n_barras_V[0]
        n_Barras_V_Inf=n_barras_V[-1]  
        Sec_Vig=2 #ID de sección de viga
        nInt_V=0 #Número de barras intermedias (en vigas no cuenta)
        BuildRCSection(Sec_Vig,H_vig,B_vig,recub,recub,1,2,3,n_barras_V[0],A_barras_V[0],n_barras_V[-1],A_barras_V[-1],nInt_V,A_barras_V[1],15,15,20,3) #Creación de viga
        
        #Puntos de integración
        pint = 5 #Puntos de integración en secciones
        beamIntegration('Lobatto', Sec_Col, Sec_Col,pint)
        beamIntegration('Lobatto', Sec_Vig, Sec_Vig,pint)
        
        #Compilado de secciones
        colsecs=[]
        vigsecs=[]
        for p in range(0,Ny):
            colsecs.append(Sec_Col)#Secciones de las columnas en altura
        for q in range(0,Nx):
            vigsecs.append(Sec_Vig)# secciones de las vigas en altura
        print('Secciones OK')
        
        #%% Transformaciones
        lineal = 1
        geomTransf('Linear',lineal)
        pdelta = 2
        geomTransf('PDelta',pdelta)
        cor = 3
        geomTransf('Corotational',cor)
        print('transformaciones ok')
    
        #%%Creando elementos (columnas y vigas)
        #COLUMNAS
        for i in range(nx):
            for j in range(ny-1):
                nodeI = 1000*(i+1) + j
                nodeJ = 1000*(i+1) + (j+1)
                eltag = 1000*(i+1) + j
                element('forceBeamColumn',eltag,nodeI,nodeJ ,pdelta,colsecs[j]) #Creando columnas y asignando ID según nodos     
        print('Columnas generadas')
        
        #VIGAS
        tagvigas=[]
        for j in range(1,ny):
            for i in range(nx-1):
                nodeI = 1000*(i+1) + j
                nodeJ = 1000*(i+2) + j
                eltag = 100000*(i+1) + j
                tagvigas.append(eltag)
                element('forceBeamColumn',eltag,nodeI,nodeJ ,lineal,vigsecs[i-1]) #Creando vigas y asignando ID según nodos    
        print('Vigas generadas')
    
        #%%Asignando carga sobre vigas
        timeSeries('Linear', 1)
        pattern('Plain',1,1)
        eleLoad('-ele',*tagvigas,'-type','beamUniform',-w)
        
        #%%Carga de peso de columnas sobre nodos
        P_col=B_col*H_col*Ly*24 #24 es el peso especifico del concreto [kN/m³]
        timeSeries('Linear', 2)
        pattern('Plain',2,2)
        for i in range(ny-1):
            for j in range(nx):
                load(1000*(j+1)+(i+1),0,-P_col,0)
        
        #%%Análisis de gravedad (verificación)
        an.gravedad()
        loadConst('-time',0.0)
    
        #%%Obtención de fuerzas en elementos por cargas de gravedad
        eletags = getEleTags()
        force = []
        for i in eletags:
            force.append(eleForce(i))
        nodetags = getNodeTags()
    
        #%%Parámetros para pushover y normalizar
        ylocation = np.array(yloc)
        norm = np.sum(ylocation)
        forces = ylocation/norm
        #Para normalizar
        Wt = (w*Nx*Lx*Ny) #Peso del edificio
        ht = Ny*Ly #Altura total del edificio
        #Asignación de patrón de cargas para pushover (triangular)
        timeSeries('Linear',3)
        pattern('Plain',3,3)
        for j in range(ny-1):
            print(1000 + j + 1,forces[j+1])
            load(1000 + j + 1,forces[j+1],0.0,0.0)
        
        #%%Pushover y obtención de puntos
        #Análisis pushover
        dtecho,Vbasal = an.pushover2(pushlimit*yloc[-1], ESP, max(nodetags), 1,[ht,Wt],1e-4)
        #Aproximación de 3 puntos 
        Cortante,Deriva,pre_Vbasalmax,post_Vbasalmax=Puntos(dtecho,Vbasal) #Esta función retorna los 3 puntos de la pushover (plástico, máximo y falla)     
        #Tomando los pushover solamente hasta puntos de interés (Aprox 0.3*Vsmax)
        Index_LimPushover=Vbasal.tolist().index(Cortante[2])+(np.abs(post_Vbasalmax - Cortante[2]*0.1)).argmin()
        D=dtecho[0:Index_LimPushover]
        C=Vbasal[0:Index_LimPushover]
        Der=dtecho
        Cor=Vbasal
        idxfin=Vbasal.tolist().index(Cortante[3])#Indice del punto de falla
        
        wipe() #Cierre del modelo
        
        #%%Comprobación de modelo fallido
        Fails=None
        if (Cortante[3]==Vbasal[-1])|(len(post_Vbasalmax)<=10)|(Cortante[3]==Cortante[2])|(Cortante[3]==0):#Este condicional verifica que el pushover se haya realizado completo y con éxito
            Fails=1
        else:
            break
    
    return D,C,Deriva,Cortante,Cuantia_Col_Real,Cuantia_Vig_Sup_Real,Cuantia_Vig_Inf_Real,nB_C,n_Barras_V_Sup,n_Barras_V_Inf
    