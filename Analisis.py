# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 15:27:44 2022

@author: Carlos Angarita
"""
from openseespy.opensees import *
import matplotlib.pyplot as plt
import numpy as np

#%% Función para realizar un análisis de gravedad
def gravedad():
    
# Create the system of equation, a sparse solver with partial pivoting
    system('BandGeneral')

# Create the constraint handler, the transformation method
    constraints('Transformation')

# Create the DOF numberer, the reverse Cuthill-McKee algorithm
    numberer('RCM')

# Create the convergence test, the norm of the residual with a tolerance of
# 1e-12 and a max number of iterations of 10
    test('NormDispIncr', 1.0e-12, 10, 3)

# Create the solution algorithm, a Newton-Raphson algorithm
    algorithm('Newton')

# Create the integration scheme, the LoadControl scheme using steps of 0.1
    integrator('LoadControl', 0.1)

# Create the analysis object
    analysis('Static')

    ok = analyze(10)
    
    if ok != 0:
        print('Análisis de gravedad fallido')
        sys.exit()
    else:
        print('Análisis de gravedad completado')

#%%Funcion para realizar análisis pushover
def pushover2(Dmax,Dincr,IDctrlNode,IDctrlDOF,norm=[-1,1],Tol=1e-8):
    
    # creación del recorder de techo y definición de la tolerancia
    #recorder('Node','-file','techo.out','-time','-node',IDctrlNode,'-dof',IDctrlDOF,'disp')
    maxNumIter = 10
      
    # configuración básica del análisis
    wipeAnalysis()
    constraints('Plain')
    numberer('RCM')
    system('BandGeneral')
    test('EnergyIncr', Tol, maxNumIter)
    algorithm('Newton')    
    integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
    analysis('Static')
    
    # Otras opciones de análisis    
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    # rutina del análisis
    
    Nsteps =  int(Dmax/ Dincr) 
    dtecho = [nodeDisp(IDctrlNode,IDctrlDOF)]
    Vbasal = [getTime()]
    
    for k in range(Nsteps):
        ok = analyze(1)
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            print('configuración por defecto no converge en desplazamiento: ',nodeDisp(IDctrlNode,IDctrlDOF))
            for j in algoritmo:
                if j < 4:
                    algorithm(algoritmo[j], '-initial')
    
                else:
                    algorithm(algoritmo[j])
                
                # el test se hace 50 veces más
                test('EnergyIncr', Tol, maxNumIter*50)
                ok = analyze(1)
                if ok == 0:
                    # si converge vuelve a las opciones iniciales de análisi
                    test('EnergyIncr', Tol, maxNumIter)
                    algorithm('Newton')
                    break
                    
        if ok != 0:
            print('Pushover analisis fallido')
            print('Desplazamiento alcanzado: ',nodeDisp(IDctrlNode,IDctrlDOF),'m')
            break
    
        
        dtecho.append(nodeDisp(IDctrlNode,IDctrlDOF))
        Vbasal.append(getTime())
        
    #plt.figure()
    #plt.plot(dtecho,Vbasal)
    #plt.xlabel('desplazamiento de techo (m)')
    #plt.ylabel('corte basal (kN)')
    
    techo = np.array(dtecho)
    V = np.array(Vbasal)
    
    
    if norm[0] != -1:
        deriva = techo/norm[0]*100
        VW = V/norm[1]
        #plt.figure()
        #plt.plot(deriva,VW)
        #plt.xlabel('Deriva de techo (%)')
        #plt.ylabel('V/W')
    
    return techo, V
