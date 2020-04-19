""" utilityfunctions.py
    Archivo que contiene los metodos para multivariate-linear-regression.py

    Author: Gabriel Aldahir Lopez Soto
    Email: gabriel.lopez@gmail.com
    Institution: Universidad de Monterrey
    First created: Thu 30 March, 2020
"""
# Importa las librerias estandard y la libreria utilityfunctions
import numpy as np
import pandas as pd
from random import randint

# Se inicializa el promedio y la desviacion 
mean = []
std = []

def eval_hypothesis_function(w, x):
    """
    Evaluar la funcion de hipotesis con la w y x

    INPUTS
    :parametro 1: matriz w con los paramametros
    :parametro 2: matriz x con las caracteristicas

    OUTPUTS
    :return: matriz con la funcion evaluada

    """
    # print(w.T)
    # print(len(x))
    # print(len(w.T))
    # print(len(x.T))
    # exit(1)
    # print('---------')
    # for x,y in zip(x.T,x.T):
    #     print(x)
    # exit(1)
    return 1/(1+np.exp(np.matmul(-w.T, x)))

def remap(x):
    """
    Crea un arreglo con 1's para despues agregarlos a la matriz x
    y despues aplicarle trasnpuesta

    INPUTS
    :parametro 1: matriz x con las caracteristicas ya escaladas

    OUTPUTS
    :return: matriz con 1's agregados y transpuesta

    """
    Nr = x.shape[0]
    x = np.hstack((np.ones((Nr,1)),x)).T
    return x

def compute_gradient_of_cost_function(x, y, w):
    """
    Calcular la funcion de costo de la gradiente descendiente
    se evaluea la funcion, despues de resta la funcion hipotesis 
    con las y, se multiplican por las x.T, se suman y se dividen 
    entre el numero de muestras, despues de hace un reshape

    INPUTS
    :parametro 1: matriz w con los parametros
    :parametro 2: matriz x con las caracteristicas
    :parametro 2: matriz y con los valores de costo ultimo milla

    OUTPUTS
    :return: matriz con el gradiente de la funcion de costo

    """
    features, Nr = x.shape
    
    hypothesis_function = eval_hypothesis_function(w, x)

    residual =  np.subtract(hypothesis_function.T, y)

    multi = (residual.T*x)

    suma = np.sum(multi,axis=1)

    gradient_of_cost_function = (suma/Nr)

    gradient_of_cost_function = np.reshape(gradient_of_cost_function,(features,1))


    return gradient_of_cost_function


def compute_L2_norm(gradient_of_cost_function):
    """
    Calcular la L2 norm

    INPUTS
    :parametro 1: matriz de gradiente de la funcion de costo

    OUTPUTS
    :return: calcular L2 norm

    """
    return np.linalg.norm(gradient_of_cost_function)

def scale_features(dataX,label,*arg):
    """
    Aplicar el escalamiento de caracteristicas dependiendo 
    cada caracteristica

    INPUTS
    :parametro 1: matriz dataX con las caracteristicas
    :parametro 2: string con la etiqueta de que proceso se hara
    :parametro 2: argumento opcional para la media y desviacion estandar

    OUTPUTS
    :return: matriz con las caracteristicas escaladas

    """
    if label == "training":
        meanT = dataX.mean()
        stdT = dataX.std()
        dataScaled = ((dataX - meanT ) / stdT)
        return dataScaled, meanT, stdT
    if label == "testing":
        dataScaled = ((dataX - arg[0] ) / arg[1])
        return dataScaled

# def predict_last_mile(x_testing,w):
#     """
#     Calcular la el valor de la ultima milla con base a los datos
#     de prueba y los parametros w optimos

#     INPUTS
#     :parametro 1: matriz x_testing con los datos de entrenamientos
#     :parametro 2: matriz w con los parametros optimos

#     OUTPUTS
#     :return: matriz con las y predecidas

#     """
#     return np.matmul(w.T, x_testing.T)


def predict_log(x_testing,w):
    """
    Calcular la el valor de la ultima milla con base a los datos
    de prueba y los parametros w optimos

    INPUTS
    :parametro 1: matriz x_testing con los datos de entrenamientos
    :parametro 2: matriz w con los parametros optimos

    OUTPUTS
    :return: matriz con las y predecidas

    """
    # for x in zip(x_testing):
        # np.matmul(w.T, x)
    return np.matmul(w.T, x_testing)

def confusionMatrix(predicted_class,y_testing):
    """
    Calcular la el valor de la ultima milla con base a los datos
    de prueba y los parametros w optimos

    INPUTS
    :parametro 1: matriz x_testing con los datos de entrenamientos
    :parametro 2: matriz w con los parametros optimos

    OUTPUTS
    :return: matriz con las y predecidas

    # """
    tp = tn = fn = fp = accuracy = precision = recall = specifity = f1 = 0
    for x,y in zip(predicted_class,y_testing):
        if(x > 0 and y == 1):
            tp += 1
        if(x < 0 and y == 0):
            tn += 1
        if(x < 0 and y == 1):
            fn += 1
        if(x > 0 and y == 0):
            fp += 1
    print("TP: "+str(tp))
    print("TN: "+str(tn))
    print("FP: "+str(fp))
    print("FN: "+str(fn))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp)/(tp+fp)
    recall = (tp/(tp+fn))
    specifity = (tn/(tn+fp))
    f1 = (2.0*((float(precision)*float(recall))/(float(precision)+float(recall))))
    print("Accuracy:"+str(accuracy))
    print("Precision:"+str(precision))
    print("Recall:"+str(recall))
    print("Specifity:"+str(specifity))
    print("F1: " + str(f1))
    return None

# def show_last(price):
#     """
#     Desplegar los valores de la ultima milla predecidos

#     INPUTS
#     :parametro 1: matriz con los valores de la ultima milla

#     OUTPUTS

#     """
#     print("-"*28)
#     print("Last Mile Cost")
#     print("-"*28)
#     for x,y in zip(price,price):
#         print(x)

# def load_predict(path_and_filename,mean,std):
#     """
#     Cargar los archivos CSV de datos de prueba, 
#     desplegar los valores de prueba escalados y hacerles 
#     el remap

#     INPUTS
#     :parametro 1: direccion y nombre del archivo
#     :parametro 2: promedio de las caracteristicas
#     :parametro 2: desviacion estandar de las caracteristicas

#     OUTPUTS
#     :return: matriz con los valores de x escalados

#     """
#     try:
#         testing_data = pd.read_csv(path_and_filename)

#     except IOError:
#       print ("Error: El archivo no existe")
#       exit(0)
    

#     filas = len(testing_data)

#     columnas = len(list(testing_data))

#     dataX = pd.DataFrame.to_numpy(testing_data.iloc[:,0:columnas])

#     print("-"*28)
#     print("Training Data")
#     print("-"*28)
#     for x,y in zip(dataX,dataX):
#         print(x)

#     dataXscaled=[]

#     for featureX,meanX,stdX in zip (dataX.T,mean,std):
#         dataScaled= scale_features(featureX,"testing",meanX,stdX)
#         dataXscaled.append(dataScaled)

#     dataXscaled = np.array(dataXscaled).T

#     print("-"*28)
#     print("Training Data Scaled")
#     print("-"*28)
#     for x,y in zip(dataXscaled,dataXscaled):
#         print(x)

#     dataXscaled = remap(dataXscaled)
#     return dataXscaled.T


def load_data(path_and_filename):
    """
    Cargar los archivos CSV de datos de entrenamiento, 
    desplegar los valores de entrenamiento escalados y hacerles 
    el remap

    INPUTS
    :parametro 1: direccion y nombre del archivo

    OUTPUTS
    :return: matriz con los valores de x escalados, datosY
    promedio, desviacion, columnas

    """
    try:
        training_data = pd.read_csv(path_and_filename)

    except IOError:
      print ("Error: El archivo no existe")
      exit(0)

    filas = len(training_data)
    columnas = len(list(training_data))

    dataX = pd.DataFrame.to_numpy(training_data.iloc[:,0:columnas-1])

    dataY = pd.DataFrame.to_numpy(training_data.iloc[:,-1]).reshape(filas,1)

    testingDataX = []
    testingDataY = []

    testingPercent = round(len(dataX)*.20)

    for x in range(0,testingPercent):

        delNumber = randint(0,len(dataX)-1)

        testingDataX.append(dataX[delNumber])
        dataX = np.delete(dataX, delNumber, 0)

        testingDataY.append(dataY[delNumber])
        dataY = np.delete(dataY, delNumber, 0)

    testingDataX = np.array(testingDataX)
    testingDataY = np.array(testingDataY)

    dataXscaled=[]

    for featureX in dataX.T:
        dataScaled, meanX, stdX = scale_features(featureX,"training")
        dataXscaled.append(dataScaled)
        mean.append(meanX)
        std.append(stdX)

    dataXscaled = np.array(dataXscaled).T


    dataXscaledTesting=[]

    for featureX,meanX,stdX in zip (testingDataX.T,mean,std):
        dataScaled= scale_features(featureX,"testing",meanX,stdX)
        dataXscaledTesting.append(dataScaled)

    dataXscaledTesting = np.array(dataXscaledTesting).T

    dataXscaledTesting = remap(dataXscaledTesting) 
    dataXscaled = remap(dataXscaled)

    dataXscaled = dataXscaled.T #nuevo
    dataXscaledTesting = dataXscaledTesting.T #nuevo


    testingDataX = remap(testingDataX) 
    dataX = remap(dataX)

    dataX = dataX.T #nuevo
    testingDataX = testingDataX.T #nuevo

    # print('----')
    # print(len(dataX.T))
    # print('----')
    # print(len(testingDataX.T))
    # print('----')
    # print(len(dataXscaled.T))
    # print('----')
    # print(len(dataXscaledTesting.T))
    # exit(1)
    return dataXscaled.T, dataY, dataXscaledTesting.T, testingDataY, columnas
    # return dataX.T, dataY, testingDataX.T, testingDataY, columnas

def show_w(w):
    """
    Desplegar los valores optimos de la w

    INPUTS
    :parametro 1: matriz con los valores optimos de la w

    OUTPUTS

    """
    print("-"*28)
    print("W parameter")
    print("-"*28)
    for x in range(len(w)):
        print("w",x,":",float(w[x]))


def gradient_descent(x_training, y_training, w, stopping_criteria, learning_rate):
    """
    Calcula la gradiente descendiente y comprueba si es la optima

    INPUTS
    :parametro 1: matriz x con datos de entrenamiento
    :parametro 2: matriz y con datos de entrenamiento
    :parametro 3: matriz w con parametros optimos
    :parametro 4: criterio de paro
    :parametro 5: learning rate

    OUTPUTS
    :return: matriz w con los parametros optimos

    """
    L2_norm = 100.0
    i = 0
    while L2_norm > stopping_criteria:

        gradient_of_cost_function = compute_gradient_of_cost_function(x_training,y_training,w)

        w = w - learning_rate*gradient_of_cost_function
        # if(i==10):
            # exit(1)
        # print('------')
        # print(L2_norm)
        # print('------')

        L2_norm = compute_L2_norm(gradient_of_cost_function)

        i += 1
    # print(i)
    return w