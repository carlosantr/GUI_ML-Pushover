# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:49:38 2022

@author: Carlos Angarita
"""
#This code generate a Graphical User Interface (GUI) for pushover curves prediction
#The GUI presents: 1. Input parameters, 2. Structure profile, 3. Beams and columns cross-section, and 4.Pushover curve prediction
#Este código genera una interfaz para predicción de curvas pushover, dividiendo

#%% Libraries
from tkinter.ttk import *
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from utilidades import comprobacion_norma, DimRefuerzo
from keras.models import load_model
import joblib

#%%Importing the ML models
#Artificial Neural Networks (ANN)
red_Plas_D=load_model('ML_Models/ANN/ANN_Yielding_D.h5')
red_Max_D=load_model('ML_Models/ANN/ANN_Maximum_D.h5')
red_Fin_D=load_model('ML_Models/ANN/ANN_Failure_D.h5')
red_Max_Vs=load_model('ML_Models/ANN/ANN_Maximum_Vs.h5')
red_Fin_Vs=load_model('ML_Models/ANN/ANN_Failure_Vs.h5')
#Random FoREST (RF)
regressor_Plas_D = joblib.load('ML_Models/RF/RF_Yielding_D.pkl')
regressor_Max_D = joblib.load('ML_Models/RF/RF_Maximum_D.pkl')
regressor_Fin_D = joblib.load('ML_Models/RF/RF_Failure_D.pkl')
regressor_Max_Vs = joblib.load('ML_Models/RF/RF_Maximum_Vs.pkl')
regressor_Fin_Vs = joblib.load('ML_Models/RF/RF_Failure_Vs.pkl')

#%% Activation functions
def Press(): #This function is called when the button is pressed
    A=[Nx,Ny,Lx,Ly,Fc,w,Bcol,Hcol,Bbeam,Hbeam,AMcol,AMbeam_TOP,AMbeam_BOTTOM]#Vector with the user values
    At=["Nx","Ny","Lx","Ly","Fc","w","Bcol","Hcol","Bbeam","Hbeam","ρc","ρb-top","ρb-bot"]
    Check=[]
    Range=[[2,5],[2,5],[4,8],[2.5,4],[17000,35000],[10,30],[0.25,0.5],[0.25,0.5],[0.25,0.5],[0.3,0.65],[0.0112198,0.0278204],[0.0051,0.0124898],[0.0034,0.009088]]#Allowed range
    NY=int(Ny.get());NX=int(Nx.get());LY=float(Ly.get());LX=float(Lx.get());FC=float(Fc.get());W=float(w.get());B_COL=float(Bcol.get());H_COL=float(Hcol.get());B_VIG=float(Bbeam.get());H_VIG=float(Hbeam.get());CUANTIA_VIG_SUP=float(AMbeam_TOP.get());CUANTIA_VIG_INF=float(AMbeam_BOTTOM.get());CUANTIA_COL=float(AMcol.get());
   
    #Checking inputs
    for i in range(len(A)):#This loop is for selecting each user value
        if A[i]:#Verify if there is some value entered
            try:
                #Checking the value format (integer or float)
                if(i<=1):
                    K=int(str(A[i].get()))
                else:
                    K=float(str(A[i].get()))
                
                if(K>=Range[i][0])&(K<=Range[i][1]):#Checking the allowed range for each user value
                    #Checking normative requirements
                    LimH1=np.round(float(0.05*np.round((LX/16)/0.05)),2)#Beam height limit 1
                    LimH2=np.round(float(0.05*np.round((LX/12)/0.05)),2)#Beam height limit 2
                    LimB1=np.round(float(0.05*np.round((H_VIG/1.4)/0.05)),2)#Beam base limit 1
                    LimB2=np.round(float(0.05*np.round((H_VIG/1.2)/0.05)),2)#Beam base limit 2
                    OK_P=comprobacion_norma(B_VIG, H_VIG, 0.05, 420000, FC, W, LX, CUANTIA_VIG_SUP, CUANTIA_COL)#Checking SCWB requirement with user values
                    if(H_VIG>=LimH1)&(H_VIG<=LimH2)&(B_VIG>=LimB1)&(B_VIG<=LimB2)&(OK_P==1):#Checking if normative requirements are satisfied
                        Check.append(True)
                    else:#If normative requirements aren't satisfied, the print a warning of the problem
                        Check.append(False)
                        if(H_VIG<LimH1)|(H_VIG>LimH2):
                            adv=Label(in_frame,width=30,font=("Arial",7),bg="#ECAFAF",text="Hbeam don't satisfy norm requirements (Lx)").place(x=75,y=318)
                        elif(B_VIG<LimB1)|(B_VIG>LimB2):
                            adv=Label(in_frame,width=30,font=("Arial",7),bg="#ECAFAF",text="Bbeam don't satisfy norm requirements (Hbeam)").place(x=75,y=318)
                        elif(OK_P==0):
                            adv=Label(in_frame,width=30,font=("Arial",7),bg="#ECAFAF",text="Don't satisfy norm requirements (SCWB)").place(x=75,y=318)                          
                        break
                else:#If user values aren't in the allowed range, the print a warning of the problem
                    Check.append(False)
                    adv=Label(in_frame,width=30,font=("Arial",7),bg="#ECAFAF",text="Value out of range for "+At[i]).place(x=75,y=318)
                    break
                
            except ValueError:#If the entered value is incorrect, then print a warning
                Check.append(False)
                adv=Label(in_frame,width=30,font=("Arial",7),bg="#ECAFAF",text="Incorrect value for "+At[i]).place(x=75,y=318)
                break
    
    #Calculating if everthing is OK
    if (sum(Check)==13):
        adv=Label(in_frame,width=30,font=("Arial",7),bg="#88F36E",text="OK").place(x=75,y=318)#Text to indicate that user values are good to calculate prediction
        nB_C,nB_V_Top,nB_V_Bot=Modeling(Range)#Function to plot the pushover prediction
        Drawing(nB_C,nB_V_Top,nB_V_Bot)#Function to draw structure profile, beams and columns
        
def Drawing(nB_C,nB_V_Top,nB_V_Bot):#This function draw the structure profile, beams and columns
    #Structure profile
    Str=Canvas(Win,width="180",height="180",bg="#EEEEEE")
    Str.pack()
    Str.place(x=29,y=395)
    spacex=170/int(Nx.get())
    spacey=170/int(Ny.get())
    Str.create_line(5,5,5,175);Str.create_line(0,175,10,175)
    Str.create_line(5,5,175,5) 
    for i in range(1,int(Nx.get())+1):
        Str.create_line(i*spacex+5,5,i*spacex+5,175)
        Str.create_line(i*spacex,175,i*spacex+10,175) 
    for i in range(1,int(Ny.get())):
        Str.create_line(5,i*spacey+5,175,i*spacey+5) 
    #Columns
    Col=Canvas(Win,width="180",height="195",bg="#EEEEEE")
    Col.pack()
    Col.place(x=260,y=390)
    xC=1000*float(Bcol.get())/5;yC=1000*float(Hcol.get())/5
    x1C=(180-xC)/2;y1C=(195-yC)/2
    x2C=x1C+xC;y2C=y1C+yC
    Col.create_rectangle(x1C,y1C,x2C,y2C,width=3.5)
    Col.create_text(x2C+8.5,y1C+(yC/2), text = 'H = '+str(float(Hcol.get())), angle = 270,font=("Arial",8))
    Col.create_text(x1C+(xC/2),y1C-8.5, text = 'B = '+str(float(Bcol.get())), font=("Arial",8))
    #Reinforcement
    recub=5
    dim=7
    sepy=(abs(y2C-y1C)-recub*2-dim)/(len(nB_C)-1)
    for i in range(nB_C[0]):#Top
        sep=(abs(x2C-x1C)-recub*2-dim)/(nB_C[0]-1)
        Col.create_oval(x1C+recub+(sep*i),y1C+recub,x1C+recub+dim+(sep*i),y1C+recub+dim,fill='black')
    for i in range(nB_C[-1]):#Bottom
        sep=(abs(x2C-x1C)-recub*2-dim)/(nB_C[-1]-1)
        Col.create_oval(x1C+recub+(sep*i),y2C-recub-dim,x1C+recub+dim+(sep*i),y2C-recub,fill='black')
    if (len(nB_C)>=3):
        for i in range(nB_C[1]):#Fila 3
            sep=(abs(x2C-x1C)-recub*2-dim)/(nB_C[1]-1)
            Col.create_oval(x1C+recub+(sep*i),y1C+recub+sepy,x1C+recub+dim+(sep*i),y1C+recub+sepy+dim,fill='black')
        if (len(nB_C)==4):
            for i in range(nB_C[2]):#Fila 3
                sep=(abs(x2C-x1C)-recub*2-dim)/(nB_C[2]-1)
                Col.create_oval(x1C+recub+(sep*i),y1C+recub+2*sepy,x1C+recub+dim+(sep*i),y1C+recub+2*sepy+dim,fill='black')  
    #Beams
    Bea=Canvas(Win,width="180",height="195",bg="#EEEEEE")
    Bea.pack()
    Bea.place(x=491,y=390)
    xB=1000*float(Bbeam.get())/5;yB=1000*float(Hbeam.get())/5
    x1B=(180-xB)/2;y1B=(195-yB)/2
    x2B=x1B+xB;y2B=y1B+yB
    Bea.create_rectangle(x1B,y1B,x2B,y2B,width=3.5)
    Bea.create_text(x2B+8.5,y1B+(yB/2), text = 'H = '+str(float(Hbeam.get())), angle = 270,font=("Arial",8))
    Bea.create_text(x1B+(xB/2),y1B-8.5, text = 'B = '+str(float(Bbeam.get())), font=("Arial",8))
    #Reinforcement
    for i in range(nB_V_Top):#Top
        sep=(abs(x2B-x1B)-recub*2-dim)/(nB_V_Top-1)
        Bea.create_oval(x1B+recub+(sep*i),y1B+recub,x1B+recub+dim+(sep*i),y1B+recub+dim,fill='black')
    for i in range(nB_V_Bot):#Bottom
        sep=(abs(x2B-x1B)-recub*2-dim)/(nB_V_Bot-1)
        Bea.create_oval(x1B+recub+(sep*i),y2B-recub-dim,x1B+recub+dim+(sep*i),y2B-recub,fill='black')
    
def Modeling(Range):#This funtion plot the pushover prediction
    NY=int(Ny.get());NX=int(Nx.get());LY=float(Ly.get());LX=float(Lx.get());FC=float(Fc.get());W=float(w.get());B_COL=float(Bcol.get());H_COL=float(Hcol.get());B_VIG=float(Bbeam.get());H_VIG=float(Hbeam.get());CUANTIA_VIG_SUP=float(AMbeam_TOP.get());CUANTIA_VIG_INF=float(AMbeam_BOTTOM.get());CUANTIA_COL=float(AMcol.get());
    Wt = (W*NX*LX*NY)
    ht = NY*LY
    
    #Opensees Pushover
    # D,C,Deriva,Cortante,C_ColR,Ct_VigR,Cb_VigR,nB_C,nB_V_Top,nB_V_Bot=PushoverData(NY,NX,LY,LY,FC,W,B_COL,H_COL,B_VIG,H_VIG,CUANTIA_COL,CUANTIA_VIG_SUP,CUANTIA_VIG_INF) 
    # D=(D/ht)*100
    # Deriva=np.array(Deriva)*100/ht
    # C=C/Wt
    # Cortante=np.array(Cortante)/Wt
    recub=0.05#Recubrimiento
    #Columna
    As_C=CUANTIA_COL*B_VIG*H_VIG #Acero necesario
    nB_C,A_barras_C,nLineasAcero_C=DimRefuerzo(B_VIG,As_C,'Columna') #Define el acero comercial y distribución necesaria en columna según cuantía
    C_ColR=(sum(nB_C)*A_barras_C[0])/(B_VIG*B_VIG)
    #Vigas
    As_V=[CUANTIA_VIG_SUP*B_VIG*(H_VIG-recub),CUANTIA_VIG_INF*B_VIG*(H_VIG-recub)]
    n_barras_V,A_barras_V,nLineasAcero_V=DimRefuerzo(B_VIG,As_V,'Viga')  #Define el acero comercial y distribución necesaria en viga según cuantías
    Ct_VigR=(n_barras_V[0]*A_barras_V[0])/(B_VIG*(H_VIG-recub))
    Cb_VigR=(n_barras_V[-1]*A_barras_V[-1])/(B_VIG*(H_VIG-recub))
    nB_V_Top=n_barras_V[0]
    nB_V_Bot=n_barras_V[-1]  
       
    #Model results and Opensees
    #ANN 
    PRED_RN_Plas_D=red_Plas_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0]
    PRED_RN_Max_D=red_Max_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0]
    PRED_RN_Fin_D=red_Fin_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0] 
    PRED_RN_Max_Vs=red_Max_Vs.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0] 
    PRED_RN_Fin_Vs=red_Fin_Vs.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0] 
    #RF
    PRED_RF_Plas_D=regressor_Plas_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0]
    PRED_RF_Max_D=regressor_Max_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0]
    PRED_RF_Fin_D=regressor_Fin_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0] 
    PRED_RF_Max_Vs=regressor_Max_Vs.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0] 
    PRED_RF_Fin_Vs=regressor_Fin_Vs.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0] 
    
    Vs_RF=np.array([0,PRED_RF_Max_Vs*0.7,PRED_RF_Max_Vs,PRED_RF_Fin_Vs])#*Wt
    D_RF=np.array([0,PRED_RF_Plas_D,PRED_RF_Max_D,PRED_RF_Fin_D])#*ht/100
    Vs_RN=np.array([0,PRED_RN_Max_Vs*0.7,PRED_RN_Max_Vs,PRED_RN_Fin_Vs])#*Wt
    D_RN=np.array([0,PRED_RN_Plas_D,PRED_RN_Max_D,PRED_RN_Fin_D])#*ht/100
    
    #Graphic
    fig,ax=plt.subplots(dpi=65,figsize=(4.85,4.85),facecolor="#EEEEEE")
    plt.yticks(rotation=90)
    plt.grid()
    ax.set_xlabel("Drift [m]")
    ax.set_ylabel("Base Shear [kN]")
    ax.set_facecolor("#EEEEEE")
    ax.plot(D_RF,Vs_RF, 'r')
    ax.plot(D_RN,Vs_RN, 'b')
    # ax.plot(D,C, 'black')
    # ax.scatter(Deriva,Cortante)
    plt.legend(['RF','ANN',"Opensees","Points"],loc='lower right')
    
    #Inserting as Canvas
    Model=FigureCanvasTkAgg(fig,master=Win)
    Model.get_tk_widget().place(x=372.5,y=20)
    Model.draw()
    Label(Win,text="Pushover Prediction",fg="#0C0268",font=("Arial Black",15)).place(x=412.5,y=5.5)

    return nB_C,nB_V_Top,nB_V_Bot

#%% Creating window
Win=Tk()

#%% Adjusting window
Win.title('GUI - Pushover prediction for 2D RC building frames')#Title of the window
Win.resizable(False,False)#Dimensionalidad de la ventana (Width,Height)
Win.geometry("700x600")#Interface size
Win.iconbitmap('Pushover.ico')#Interface icon
Win.config(bg="#919191")#Color

#%% Input Frame
in_frame=Frame(Win,width="342.5",height="342.5")#Creating frame
in_frame.place(x=5,y=5)#Position
in_frame.config(bg="#EEEEEE")#Color
in_frame.config(bd=1)#Border
in_frame.config(relief="solid")

Label(in_frame,text="Inputs",fg="#0C0268",font=("Arial Black",15)).place(x=130.5,y=0) #Title of the frame (text)

#Input definition (entry)
Nx=StringVar();Ny=StringVar();Lx=StringVar();Ly=StringVar();Fc=StringVar();w=StringVar()
Bcol=StringVar();Hcol=StringVar();Bbeam=StringVar();Hbeam=StringVar();AMcol=StringVar();AMbeam_TOP=StringVar();AMbeam_BOTTOM=StringVar()

#In this section is defined the Label of the text (input variables) and the Entry box for the user to write a value. And the place into the frame
#Left zone
Nx_txt=Label(in_frame,text="Nx").place(x=80,y=37.5) #TEXT 
Nx_in=Entry(in_frame,textvariable=Nx,justify="center").place(x=28,y=57.5) #ENTRY BOX
Ny_txt=Label(in_frame,text="Ny").place(x=80,y=77.5)
Ny_in=Entry(in_frame,textvariable=Ny,justify="center").place(x=28,y=97.5)
Lx_txt=Label(in_frame,text="Lx").place(x=80,y=117.5)
Lx_in=Entry(in_frame,textvariable=Lx,justify="center").place(x=28,y=137.5)
Ly_txt=Label(in_frame,text="Ly").place(x=80,y=157.5)
Ly_in=Entry(in_frame,textvariable=Ly,justify="center").place(x=28,y=177.5)
Fc_txt=Label(in_frame,text="Fc").place(x=80,y=197.5)
Fc_in=Entry(in_frame,textvariable=Fc,justify="center").place(x=28,y=217.5)
w_txt=Label(in_frame,text="W").place(x=80,y=237.5)
w_in=Entry(in_frame,textvariable=w,justify="center").place(x=28,y=257.5)
#Right zone
Bcol_txt=Label(in_frame,text="Bcol").place(x=233,y=37.5)
Bcol_in=Entry(in_frame,textvariable=Bcol,justify="center").place(x=186,y=57.5)
Hcol_txt=Label(in_frame,text="Hcol").place(x=233,y=77.5)
Hcol_in=Entry(in_frame,textvariable=Hcol,justify="center").place(x=186,y=97.5)
Bbeam_txt=Label(in_frame,text="Bbeam").place(x=227,y=117.5)
Bbeam_in=Entry(in_frame,textvariable=Bbeam,justify="center").place(x=186,y=137.5)
Hbeam_txt=Label(in_frame,text="Hbeam").place(x=225,y=157.5)
Hbeam_in=Entry(in_frame,textvariable=Hbeam,justify="center").place(x=186,y=177.5)
AMcol_txt=Label(in_frame,text="ρc").place(x=238,y=197.5)
AMcol_in=Entry(in_frame,textvariable=AMcol,justify="center").place(x=186,y=217.5)
AMbeamUP_txt=Label(in_frame,text="ρb-top").place(x=195,y=237.5)
AMbeamUP_in=Entry(in_frame,textvariable=AMbeam_TOP,justify="center",width=8).place(x=192,y=257.5)
AMbeamDOWN_txt=Label(in_frame,text="ρb-bot").place(x=255,y=237.5)
AMbeamDOWN_in=Entry(in_frame,textvariable=AMbeam_BOTTOM,justify="center",width=8).place(x=252,y=257.5)

#Button (when the user press on the button, the function "press" is called)
button=Button(in_frame,command=Press,text="Predict",width=10,bg="#C0C0C0",fg="#0C0268",font=("Arial Black",9)).place(x=123,y=285.5)

#%% Pushover curve Frame
pc_frame=Frame(Win,width="342.5",height="342.5")#Creating frame
pc_frame.place(x=352.5,y=5)#Position
pc_frame.config(bg="#EEEEEE")#Color
pc_frame.config(bd=1)#Border
pc_frame.config(relief="solid")

Label(Win,text="Pushover Prediction",fg="#0C0268",font=("Arial Black",15)).place(x=412.5,y=5.5)#Title of the frame (text)

#%% Structure Frame
secS_frame=Frame(Win,width="228",height="242.5")#Creating frame
secS_frame.place(x=5,y=352.5)#Position
secS_frame.config(bg="#EEEEEE")#Color
secS_frame.config(bd=1)#Border
secS_frame.config(relief="solid")

Label(secS_frame,text="Structure",fg="#0C0268",font=("Arial Black",15)).place(x=60,y=0)#Title of the frame (text)

#%% Section Beam Frame
#Creando frame
secC_frame=Frame(Win,width="228",height="242.5")#Creating frame
secC_frame.place(x=236,y=352.5)#Position
secC_frame.config(bg="#EEEEEE")#Color
secC_frame.config(bd=1)#Border
secC_frame.config(relief="solid")

Label(secC_frame,text="Columns",fg="#0C0268",font=("Arial Black",15)).place(x=63,y=0)#Title of the frame (text)

#%% Section Column Frame
secB_frame=Frame(Win,width="228",height="242.5")#Creating frame
secB_frame.place(x=467,y=352.5)#Position
secB_frame.config(bg="#EEEEEE")#Color
secB_frame.config(bd=1)#Border
secB_frame.config(relief="solid")

Label(secB_frame,text="Beams",fg="#0C0268",font=("Arial Black",15)).place(x=71,y=0)#Title of the frame (text)

#%%Loop for frame interface
Win.mainloop()