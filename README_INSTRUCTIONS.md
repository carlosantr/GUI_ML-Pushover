# GUI_ML-Pushover
Graphical User Interface (GUI) expected to be used by practitioners to quickly carry out an approximated pushover curve that provides relevant information on the seismic capacity of low-rise RC frame buildings, using RF and ANN Machine Learning models. The next steps are required to run the GUI:

0. Download the repository as a zip file and extract the elements on your own PC.
1. Open the code called "GUI.py" in a Python IDE (Spyder 5.4.3 is recommended)
2. Installation of libraries: You have to install from your console in python the next libraries:
   *Scikit-Learn (write in console: pip install --user scikit-learn==1.2.1)
   *Joblib (write in console: pip install --user joblib==1.3.2) and
   *Keras (wite in console: pip install --user keras==2.12.0)
3. Restart the kernel (or close and open again the IDE).
4. Run the entire code.
5. Enter your input data in the boxes, taking into account the model ranges presented in "Table 5. Input parameters" (from the paper "Machine Learning – based approach for predicting pushover curves of low-rise reinforced concrete frame buildings"). The ranges supporter by the GUI are: Nx (2-5), Ny (2-5), Lx (4-8), Ly (2.5-4), Fc (17000-35000), W (10-30), Bcol/Hcol/Bbeam (0.25-0.5), Hbeam (0.3-0.65), ρc (0.01122-0.02782), ρb-top (0.0051-0.0125) and ρb-bot (0.0034-0.0091).
6. Press the Predict button.
   6.1. If an error occurs or the seismic code consideration are not accomplished, a message is presented and the values must be changed 
        until the error or warning are cprrected.
7. When the "OK" green message is presented, the prediction is done.
