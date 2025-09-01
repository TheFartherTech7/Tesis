# -------------------------------------------------------
# Autor: Pablo Emilio
# Proyecto: Analisis de titulaciones
# Descripción: Ajuste de curvas con R^2" Fecha 02/07/2025
# --------------------------------------------------------



#Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#Funciones
def ajustar_modelo(x,y,grado=3):
    modelo = np.poly1d(np.polyfit(x,y,grado))
    r2 = r2_score(y, modelo(x))
    return modelo, r2

#Código principal
def main():
    # Leer datos
    df = pd.read_csv("C:/Users/Usuario/OneDrive - Instituto Tecnológico de Morelia/Documentos/Tesis/analisis_datos_4_0.csv.csv")
    
    # Limpieza de columnas por si tienen espacios
    df.columns = df.columns.str.strip()
    
    # Filtrar si se desea (ej: por reactivo)
    #df = df[df["Reactivo"]== "Melaza"]
    
    x = df["Concentracion de titulante"].values
    y = df["pH"].values
    df = df[df["Reactivo"]== "Melaza"]
    #Ajuste
    modelo, r2 = ajustar_modelo(x, y, grado =3)
    
    #Mostrar resultados
    print(f"Ecuacion: {modelo}")
    print(f"R^2: {r2:.4f}")
    
    #Graficar
    plt.scatter(x, y, label="Datos")
    x_fit = np.linspace(min(x), max(x), 200)
    plt.plot(x_fit, modelo(x_fit), color= 'red', label=f"Ajuste (R^2={r2:-4f})")
    plt.xlabel("Concentracion de titulante")
    plt.ylabel("pH")
    plt.title("Curva de titulación")
    plt.legend()
    plt.grid(True)
    plt.show




if __name__ == "__main__":
    main()