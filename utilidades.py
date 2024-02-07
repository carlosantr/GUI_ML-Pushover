# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:33:00 2022

@author: Carlos Angarita
"""
#Este código contiene funciones de utilidades para la generación de modelos en
#Opensees y comprobaciones de norma.

#%%Importando librerias necesarias
from openseespy.opensees import *
import matplotlib.pyplot as plt
import numpy as np
import opsvis as opsv
import pandas as pd

#%%MomentCurvature y TestMaterial
def MomentCurvature(secTag, axialLoad, maxK, numIncr=100):
    # Script tomado de la librería de OpenSeespy de la web
    # secTag es el tag de la sección
    # axialLoad es la carga axial de la sección
    # maxK es la curvatura
    # numIncr es el número de incrementos
    # Define two nodes at (0,0)
    model('basic','-ndm',2,'-ndf',3)
    node(1, 0.0, 0.0)
    node(2, 0.0, 0.0)

    # Fix all degrees of freedom except axial and bending
    fix(1, 1, 1, 1)
    fix(2, 0, 1, 0)
    
    # Define element
    #                             tag ndI ndJ  secTag
    element('zeroLengthSection',  1,   1,   2,  secTag)

    # Define constant axial load
    timeSeries('Constant', 1)
    pattern('Plain', 1, 1)
    load(2, axialLoad, 0.0, 0.0)

    # Define analysis parameters
    integrator('LoadControl', 0.0)
    system('SparseGeneral', '-piv')
    test('NormUnbalance', 1e-9, 10)
    numberer('Plain')
    constraints('Plain')
    algorithm('Newton')
    analysis('Static')

    # Do one analysis for constant axial load
    analyze(1)
    loadConst('-time',0.0)

    # Define reference moment
    timeSeries('Linear', 2)
    pattern('Plain',2, 2)
    load(2, 0.0, 0.0, 1.0)

    # Compute curvature increment
    dK = maxK / numIncr

    # Use displacement control at node 2 for section analysis
    integrator('DisplacementControl', 2,3,dK,1,dK,dK)
    
    M = [0]
    curv = [0]
    
    # Do the section analysis
    for i in range(numIncr):
        analyze(1)
        curv.append(nodeDisp(2,3))
        M.append(getTime())
    plt.figure()
    plt.plot(curv,M)
    plt.xlabel('Curvatura')
    plt.ylabel('Momento (kN-m)')
    
    return M,curv

def testMaterial(matTag,displ):
    # wipe()
    
    model('basic','-ndm',2,'-ndf',3)
    # h = getNodeTags()

    node(1,0.0,0.0)
    node(2,0.0,0.0)
    
    fix(1,1,1,1)
    fix(2,1,1,0)
    
    controlnode = 2
    element('zeroLength',1,1,2,'-mat',matTag,'-dir',6)
    
    recorder('Node','-file','MPhi.out','-time','-node',2,'-dof',3,'disp')
    recorder('Element','-file','Moment.out','-time','-ele',1,'force')
    
    ratio = 1/1000
    
    timeSeries('Linear',1)
    pattern('Plain',1,1)
    load(2,0.0,0.0,1.0)
    
    constraints('Plain')
    numberer('Plain')
    system('BandGeneral')
    test('EnergyIncr',1e-6,1000)
    algorithm('Newton')
    
    currentDisp = 0.0
    Disp = [0]
    F = [0]
    nSteps = 1000
    
    for i in displ:
        Dincr = ratio*i/nSteps
        integrator('DisplacementControl',controlnode,3,Dincr)
        analysis('Static')
        
        if Dincr > 0:
            Dmax = Dincr*nSteps
            ok = 0
            while ok == 0 and currentDisp < Dmax:
                ok = analyze(1)
                currentDisp = nodeDisp(controlnode,3)
                F.append(getTime())
                Disp.append(currentDisp)
        elif Dincr < 0:
            Dmax = Dincr*nSteps
            ok = 0
            while ok == 0 and currentDisp > Dmax:
                ok = analyze(1)
                currentDisp = nodeDisp(controlnode,3)
                F.append(getTime())
                Disp.append(currentDisp)
    Fcurr = getTime()
    if ok != 0:
        print('Fallo la convergencia en ',Fcurr)
    else:
        print('Analisis completo')
    
    plt.figure()
    plt.plot(Disp,F)
    plt.xlabel('deformación unitaria (m/m)')
    plt.ylabel('esfuerzo (kPa)')
    return Disp,F

#%%Diagrama de interacción
def interaction_diagram_kN(b,h,recub,nLineasAcero,nBarras,ABarras,fc,fy):#Este código genera el diagrama de interacción de una columna

    #Encontrando posicion de y área de lineas de acero
    As=[]
    pos=[]#Posicion de lineas de barras de acero definidas en As
    nBarras=np.float_(nBarras)
    for i in range(0,nLineasAcero):
        pos.append(recub+i*(h-2*recub)/(nLineasAcero-1))
        As.append(nBarras[i]*ABarras[i])    

    Es=210000000; #modulo de elasticidad del acero
    ec=0.003; #del concreto
    es=fy/Es; #del acero
    
    pos=np.array(pos)
    As=np.array(As)
    
    if (len(pos)!=len(As)):
        print('ERROR: numero de barras no coincide con su ubicacion')    
    else:
        
        c=[] #puntos al eje neutro
        for i in range(1,101):
            c.append(h*i/100)
        points = len(c)
        yg = h/2; #posicion del centroide
        arm=yg-np.array(pos) #brazo de las barras de acero
        Ts = -sum(As)*fy #maxima fuerza de tension sin momento
        Pc = 0.8*(0.85*fc*(b*h-sum(As))+sum(As)*fy) #maxima fuerza de compresion sin momento
        
        ei=[];fi=[];
        fs=np.zeros((points,len(pos)));
        Fi=[];Fs=[];Mi=[];Ms=[]
        Fc=[];Mc=[];Pt=[];Pn=[];Mn=[];phi=[]
        
        for i in range(points):
            ci=c[i]
            ei.append(ec*(1-(pos/ci)))
            fi.append(ei[i]*Es)
            
            for j in range(len(pos)):
                if (fi[i][j]<-fy):
                    fs[i][j]=-fy
                elif (fi[i][j]>fy):
                    fs[i][j]=fy
                else:
                    fs[i][j]=fi[i][j]
            
            Fi.append(np.array(fs[i][:])*As)
            Fs.append(sum(Fi[i][:]))
            Mi.append(Fi[i][:]*arm)
            Ms.append(sum(Mi[i][:]))
            Fc.append(0.85*fc*0.85*ci*b)
            Mc.append(Fc[i]*(yg-0.85*ci/2))
            Pt.append(Fc[i]+Fs[i])
            Pn.append(min(Pt[i],Pc))
            Mn.append(Mc[i]+Ms[i])
            phi.append(0.65+(-ei[i][-1]-es)*250/3)
            
            if (phi[i]<0.65):
                phi[i]=0.65
            elif(phi[i]>0.9):
                phi[i]=0.9
            else:
                phi[i]=phi[i]
            
        phi.insert(0,0.9);phi.append(0.65)
        Pn.insert(0,Ts);Pn.append(Pc)
        M1=sum(As*arm*-fy)
        Mn.insert(0,M1);Mn.append(0)
        
        Pn=np.array(Pn)
        Mn=np.array(Mn)
        
        # plt.plot(Mn,Pn)
        # plt.xlabel('Moment (kN-m)')
        # plt.ylabel('Axial load (kN)')
        # plt.title('Interaction Diagram')
        
        return Pn[Mn>=0],Mn[Mn>=0]

#%%Creación de secciones
def BuildRCSection(ID,HSec,BSec,coverH,coverB,coreID,coverID,steelID,numBarsTop,barAreaTop,numBarsBot,barAreaBot,numBarsIntTot,barAreaInt,nfCoreY,nfCoreZ,nfCoverY,nfCoverZ):#Esta función crea las secciones de columnas y vigas de un modelo (con asignación de acero, concreto confinado y no confinado)
    # Define a procedure which generates a rectangular reinforced concrete section
	# with one layer of steel at the top & bottom, skin reinforcement and a 
	# confined core.
	#		by: Silvia Mazzoni, 2006
	#			adapted from Michael H. Scott, 2003
	# 
	# Formal arguments
	#    id - tag for the section that is generated by this procedure
	#    HSec - depth of section, along local-y axis
	#    BSec - width of section, along local-z axis
	#    cH - distance from section boundary to neutral axis of reinforcement
	#    cB - distance from section boundary to side of reinforcement
	#    coreID - material tag for the core patch
	#    coverID - material tag for the cover patches
	#    steelID - material tag for the reinforcing steel
	#    numBarsTop - number of reinforcing bars in the top layer
	#    numBarsBot - number of reinforcing bars in the bottom layer
	#    numBarsIntTot - TOTAL number of reinforcing bars on the intermediate layers, symmetric about z axis and 2 bars per layer-- needs to be an even integer
	#    barAreaTop - cross-sectional area of each reinforcing bar in top layer
	#    barAreaBot - cross-sectional area of each reinforcing bar in bottom layer
	#    barAreaInt - cross-sectional area of each reinforcing bar in intermediate layer 
	#    nfCoreY - number of fibers in the core patch in the y direction
	#    nfCoreZ - number of fibers in the core patch in the z direction
	#    nfCoverY - number of fibers in the cover patches with long sides in the y direction
	#    nfCoverZ - number of fibers in the cover patches with long sides in the z direction
    
    #Variables necesarias
    coverY = HSec/2.0
    coverZ = BSec/2.0
    coreY = coverY - coverH
    coreZ = coverZ - coverB
    numBarsInt = int(numBarsIntTot/2)
    nespacios=numBarsInt+1
    a=HSec-2*coverH
    b=a/nespacios
    GJ = 1e6
    
    #Creación de sección
    section('Fiber',ID,'-GJ',GJ)
    patch('quad',coreID,nfCoreZ,nfCoreY,-coreY,coreZ,-coreY,-coreZ,coreY,-coreZ,coreY,coreZ) #Concreto confinado
    patch('quad',coverID,2,nfCoverY,-coverY,coverZ,-coreY,coreZ,coreY,coreZ,coverY,coverZ) #Concreto no confinado 1
    patch('quad',coverID,2,nfCoverY,-coreY,-coreZ,-coverY,-coverZ,coverY,-coverZ,coreY,-coreZ) #Concreto no confinado 2
    patch('quad',coverID,nfCoverZ,2,-coverY,coverZ,-coverY,-coverZ,-coreY,-coreZ,-coreY,coreZ) #Concreto no confinado 3
    patch('quad',coverID,nfCoverZ,2,coreY,coreZ,coreY,-coreZ,coverY,-coverZ,coverY,coverZ) #Concreto no confinado 4
    #Cuando el número de barras intermedia es 0, se trata de una viga, sino se asignan las de columna
    if(numBarsInt>0):
        layer('straight',steelID,numBarsInt,barAreaInt,-coreY+b,coreZ,coreY-b,coreZ) #Barras de acero intermedias 1
        layer('straight',steelID,numBarsInt,barAreaInt,-coreY+b,-coreZ,coreY-b,-coreZ) #Barras de acero intermedias 2
    layer('straight',steelID,numBarsTop,barAreaTop,coreY,coreZ,coreY,-coreZ) #Barras de acero superiores
    layer('straight',steelID,numBarsBot,barAreaBot,-coreY,coreZ,-coreY,-coreZ) #Barras de acero inferiores
    return 

#%%Dimensiones de vigas
def DimViga(Lx):#Esta función define las dimensiones de viga de un edificio según su longitud de viga
    
    #Selección de base, de manera aleatoria, teniendo en cuenta incertidumbre de que cumpla o no Columna Fuerte - Viga Debil. Además de cumplir que: Lx/16<=H_viga<=Lx/12
    if (Lx<=5.2):
        bv=np.random.choice([0.25,0.3,0.35,0.4,0.45,0.25,0.3])
    else:
        bv=np.random.choice([0.35,0.4,0.45,0.5,0.5])
    
    #Definición de altura de viga: 1.5*B_viga<=H_viga<=1.2*B_viga
    hv=0.05*np.round(np.random.uniform(bv*1.2,bv*1.5)/0.05) 
    return bv,hv

#%%Asignación de refuerzo        
def DimRefuerzo(Base,Area,Elemento):#Esta función define una distribución de barras común en la sección (barras comerciales) que se asemejen al acero requerido según la cuantía
    
    n_barras=[];A_barras=[];nT=[];nB=[];nLineasAcero=0;
    n_posibilidad=[]
    A_posibilidad=[]
    nLineas_posibilidad=[]
    #Tabla de barras de refuerzo
    table=pd.DataFrame([
        [2,3,4,5,6,7,8],#Número de barra
        [0.32,0.71,1.29,1.99,2.84,3.87,5.1],#Área en cm²
        ])
    table=table.transpose()
    table.set_axis(['#Barra','Area'],axis=1,inplace=True)
    table['Area']=table['Area']/10000
    table['Lim4']=table['Area']*4
    table['Lim6']=table['Area']*6
    table['Lim8']=table['Area']*8
    table['Lim10']=table['Area']*10
    table['Lim12']=table['Area']*12
    table['Lim14']=table['Area']*14
   
    #Encontrando la mejor combinación de barras para COLUMNAS
    if (Elemento=='Columna'):#En cada ciclo se encuentra una combinación diferente, y de manera aleatoria al final se elige una combinación que se ajuste a la cantidad de acero requerido
    
        if (Base<=0.3):
            for i in range(2,len(table)):#A
                if(Area<=table.loc[i,'Lim4']):#Asigna 4 barras que mejor se ajusten
                    n_posibilidad.append([2,2])
                    A_posibilidad.append([table.loc[i,'Area'],table.loc[i,'Area']])
                    nLineas_posibilidad.append(2)
                    break

        for p in range(2,len(table)):#C
            if(Area<=table.loc[p,'Lim8']):#Asigna 8 barras que mejor se ajusten
                n_posibilidad.append([3,2,3])
                A_posibilidad.append([table.loc[p,'Area'],table.loc[p,'Area'],table.loc[p,'Area']])
                nLineas_posibilidad.append(3)
                break

                                    
        if (Base>=0.35):
            for u in range(2,len(table)):#E
                if(Area<=table.loc[u,'Lim12']):#Asigna 12 barras que mejor se ajusten
                    n_posibilidad.append([4,2,2,4])
                    A_posibilidad.append([table.loc[u,'Area'],table.loc[u,'Area'],table.loc[u,'Area'],table.loc[u,'Area']])
                    nLineas_posibilidad.append(4)
                    break

        #Selección aleatoria de combinación de barras de acero
        idx=np.round(np.random.uniform(0.5,len(n_posibilidad)+0.5)).astype(int)
        n_barras.append(n_posibilidad[idx-1])
        A_barras.append(A_posibilidad[idx-1])
        nLineasAcero=nLineas_posibilidad[idx-1]
      
    #Encontrando la mejor combinación de barras para VIGAS                                                  
    if(Elemento=='Viga'):
        #Recorre números de barra y cantidades de barras para encontrar una combinación para el acero superior
        for j in range(2,10):#Asignar máximo 9 barras arriba
            for i in range(1,len(table)):
                if(Area[0]<=j*table.loc[i,'Area']):
                    nT=[j,table.loc[i,'Area']]#Número de barras y área
                    break
            if (len(nT)>0):
                break
            
        #Recorre números de barra y cantidades de barras para encontrar una combinación para el acero inferior
        for j in range(2,9):#Asignar máximo 8 barras abajo
            for i in range(1,len(table)):    
                if(Area[1]<=j*table.loc[i,'Area']):
                    nB=[j,table.loc[i,'Area']]#Número de barras y área
                    break
            if (len(nB)>0):
                break
        #Almacenando resultados
        n_barras.append([nT[0],0,nB[0]])
        A_barras.append([nT[1],0,nB[1]])
        nLineasAcero=2
    
    #Regresando el número de barras (superior, intermedia e inferior), sus áreas y el número de líneas de acero que tendrá la sección
    return n_barras[0], A_barras[0], nLineasAcero

#%%Puntos de pushover
def Puntos(dtecho,Vbasal):
    
    #Encontrando el PUNTO MÁXIMO
    Vbasalmax=max(Vbasal) #Cortante máximo   
    pre_Vbasalmax=Vbasal[0:Vbasal.tolist().index(Vbasalmax)]#Vector previo al punto máximo
    Vbasal_Pos=Vbasal[Vbasal>0]#Vector de cortante solamente positivo
    post_Vbasalmax=Vbasal[Vbasal.tolist().index(Vbasalmax):len(Vbasal_Pos)]#Vector posterior al punto máximo
    #Condicion de máximo
    if (len(post_Vbasalmax)>=5): #Condicional para asegurar que después del máximo la pushover continua (no ha convergido)
        if (max(post_Vbasalmax)>Vbasalmax):#Condicional para asegurarse que es el máximo real de la pushover
            Vbasalmax=max(post_Vbasalmax)
            pre_Vbasalmax=Vbasal[0:Vbasal.tolist().index(Vbasalmax)]#Vector previo al punto máximo
            Vbasal_Pos=Vbasal[Vbasal>0]#Vector de Vbasal solo positivo
            post_Vbasalmax=Vbasal[Vbasal.tolist().index(Vbasalmax)+5:len(Vbasal_Pos)]#Vector posterior al punto máximo
    
    #Encontrando el PUNTO DE FALLA (se define como el punto donde la curva cae drásticamente, es decir, cuando el gradiente de cortante respecto al desplazamiento es muy grande)
    index_Fin=0
    grad_anterior=1000
    #Encontrando el índice
    for T in range (Vbasal.tolist().index(Vbasalmax)-1,len(Vbasal)):#Ciclo para recorrer la curva después del punto máximo
        if (T!=len(Vbasal)-1):#Condicional para asegurarse de estar en el rango de la curva (número de puntos)
            grad_actual=(Vbasal[T+1]-Vbasal[T])#Cálculo del gradiente entre el siguiente punto de la curva y el actual  
        if (abs(grad_actual) >= 50*abs(grad_anterior))&(grad_actual<-1):#Condicional para comprobar que la diferencia de gradientes entre los puntod anteriores y siguientes es significativa (punto de falla), o no.
            index_Fin=T #Índice del punto de falla
            break
        grad_anterior=grad_actual #Almacenando el gradiente actual como "anterior", para el siguiente punto
    #Guardando cortante del punto final
    if(Vbasal[index_Fin]<=Vbasalmax*0.8)&(len(post_Vbasalmax)>=3):#Condicional para que si el punto final es inferior al 80% del cortante máximo, tome como final ese 80%, pues después no es significativo.
        Vb_fin=post_Vbasalmax[(np.abs(post_Vbasalmax - Vbasalmax*0.8)).argmin()]
    elif (index_Fin!=0)&(len(post_Vbasalmax)>=4):#Tomando el cortante del índice encontrado
        Vb_fin=Vbasal[index_Fin] 
    else:#Cuando el pushover falla y no genera mas datos después del punto máximo
        Vb_fin=Vbasalmax
        post_Vbasalmax=[1]            
    
    #Resultados
    #CORTANTE (el primero punto se define como el 70% del máximo)
    Points_Cortante=[0,pre_Vbasalmax[(np.abs(pre_Vbasalmax - Vbasalmax*0.7)).argmin()],  
                     Vbasalmax,
                     Vb_fin]
    #DESPLAZAMIENTOS (se encuentran como el desplazamiento obtenido para los cortantes definidos)
    Points_Deriva=[0,dtecho[Vbasal.tolist().index(Points_Cortante[1])],
                   dtecho[Vbasal.tolist().index(Points_Cortante[2])],
                   dtecho[Vbasal.tolist().index(Points_Cortante[3])]]   
    
    return Points_Cortante,Points_Deriva,pre_Vbasalmax,post_Vbasalmax

#%%Comprobación de norma
def comprobacion_norma(B_vig,H_vig,recub,Fy,Fc,w,Lx,Cuantia_Vig,Cuantia_Col):#Esta función se encarga de definir si la combinación de parámetros del edificio creado, cumple o no con el criterio de columna fuerte - viga débil
    
    #Variables de cumplimiento
    OK=0 #General
    OKV=0 #Viga
    OKC=0 #Columna

    #CRITERIO 1
    As_V=Cuantia_Vig*B_vig*(H_vig-recub)#Área de acero arriba y abajo
    Mn_V = As_V*Fy*((H_vig-recub)-(As_V*Fy/(Fc*B_vig)))#Momento de la viga
    if(Mn_V>=(w*(Lx**2)/12)):#Comprobacion del momento nominal de la viga con WL²/11
        OKV=1
 
    #CRITERIO 2 
    B_col=B_vig;H_col=B_col  
    As_C=Cuantia_Col*B_col*H_col#Área de acero de columna
    n_barras_C,A_barras_C,nLineasAcero_C=DimRefuerzo(B_col,As_C,'Columna')#Número de barras y áreas de columnas
    Point_Diagrama=0.15*Fc*(B_col*H_col)#Aproximación para el momento del diagrama de interacción para comprobar SCWB
    Pd,Md=interaction_diagram_kN(B_col, H_col, recub, nLineasAcero_C, n_barras_C, A_barras_C, Fc, Fy)
    index_near=(np.abs(Point_Diagrama-Pd)).argmin()
    #Interpolando entre puntos del diagrama de interacción, para encontrar el momento nominal exacto
    if (Pd[index_near]<Point_Diagrama):
        m=(Md[index_near+1]-Md[index_near])/(Pd[index_near+1]-Pd[index_near])
        Mn_C = Md[index_near]+((Point_Diagrama-Pd[index_near])*m)#Momento nominal de columna
    else:
        m=(Md[index_near]-Md[index_near-1])/(Pd[index_near]-Pd[index_near-1])
        Mn_C = Md[index_near]+((Pd[index_near]-Point_Diagrama)*m)#Momento nominal de columna
    #Comprobacion de SCWB
    if(Mn_C>1.2*Mn_V):
        OKC=1
    
    #¿CUMPLE?
    if(OKV==OKC==1):
        OK=1

    return OK

