import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# Leer datos
archivo = "datos_titulacion.csv"
df = pd.read_csv(archivo)
df.columns = df.columns.str.strip()

# Obtener lista de reactivos únicos
reactivos = sorted(df["Reactivo"].unique())

# Diccionario para asignar números
menu_reactivos = {i + 1: reactivo for i, reactivo in enumerate(reactivos)}

# Función para ajustar modelos polinómicos de grado 1 a 4
def ajustar_modelo(X, y):
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_grado = 0
    mejor_ecuacion = ""
    mejor_poly = None

    for grado in range(1, 5):
        poly = PolynomialFeatures(degree=grado)
        X_poly = poly.fit_transform(X)
        modelo = LinearRegression()
        modelo.fit(X_poly, y)
        y_pred = modelo.predict(X_poly)
        r2 = r2_score(y, y_pred)

        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_modelo = modelo
            mejor_grado = grado
            mejor_poly = poly
            mejor_ecuacion = mostrar_ecuacion(modelo, poly)

    return mejor_modelo, mejor_grado, mejor_r2, mejor_ecuacion, mejor_poly

# Formatear ecuación polinómica
def mostrar_ecuacion(modelo, poly):
    coefs = modelo.coef_
    inter = modelo.intercept_
    terms = poly.get_feature_names_out(["X1", "X2"])
    ecuacion = f"pH = {inter:.3f}"
    for coef, term in zip(coefs[1:], terms[1:]):
        if abs(coef) > 1e-3:
            ecuacion += f" + ({coef:.3f})*{term}"
    return ecuacion

# Gráfica 3D para múltiples concentraciones
def graficar_superficie_multiple(sub_df, reactivo, modelo, poly, grado):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colores = ['r', 'g', 'b', 'm', 'c', 'y']
    etiquetas = []

    for i, conc in enumerate(sorted(sub_df["Concentracion de reactivo"].unique())):
        datos = sub_df[sub_df["Concentracion de reactivo"] == conc]
        X = datos[["Concentracion de reactivo", "Concentracion de titulante"]].values
        y = datos["pH"].values
        ax.scatter(X[:, 0], X[:, 1], y, c=colores[i % len(colores)], label=f"{conc} {datos['Unidad'].iloc[0]}")

    x1_lin = np.linspace(sub_df["Concentracion de reactivo"].min(), sub_df["Concentracion de reactivo"].max(), 30)
    x2_lin = np.linspace(sub_df["Concentracion de titulante"].min(), sub_df["Concentracion de titulante"].max(), 30)
    x1_grid, x2_grid = np.meshgrid(x1_lin, x2_lin)
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    X_poly = poly.transform(X_grid)
    y_pred = modelo.predict(X_poly).reshape(x1_grid.shape)

    ax.plot_surface(x1_grid, x2_grid, y_pred, cmap='viridis', alpha=0.6)
    ax.set_title(f"{reactivo} | Modelo grado {grado}")
    ax.set_xlabel("Conc. Reactivo")
    ax.set_ylabel("Conc. Titulante")
    ax.set_zlabel("pH")
    ax.legend()
    plt.tight_layout()
    plt.show()

    guardar = input("\n¿Deseas guardar esta gráfica como imagen? (Sí/No): ").strip().lower()
    if guardar in ["sí", "si"]:
        nombre = f"{reactivo}_concentraciones.png".replace(" ", "_")
        fig.savefig(nombre)
        print(f"Imagen guardada como: {nombre}")

# Análisis individual
def analizar_reactivo():
    print("\nSelecciona un reactivo:")
    for num, nombre in menu_reactivos.items():
        print(f"{num}. {nombre}")

    entrada = input("\nEscribe el número o nombre del reactivo: ").strip().lower()
    reactivo = None
    if entrada.isdigit():
        reactivo = menu_reactivos.get(int(entrada))
    else:
        for nombre in reactivos:
            if nombre.lower() == entrada:
                reactivo = nombre
                break

    if not reactivo:
        print("\nOpción inválida. Intenta de nuevo.")
        return

    sub_df = df[df["Reactivo"] == reactivo]
    X = sub_df[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = sub_df["pH"].values
    modelo, grado, r2, eq, poly = ajustar_modelo(X, y)

    print("\n--- Resultado ---")
    print(f"Reactivo: {reactivo}")
    print(f"Mejor modelo: Grado {grado}")
    print(f"R^2 global del reactivo: {r2:.4f}")
    print(f"Ecuación: {eq}")

    graficar_superficie_multiple(sub_df, reactivo, modelo, poly, grado)

# Análisis global
def analisis_global():
    X = df[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = df["pH"].values
    modelo, grado, r2, eq, poly = ajustar_modelo(X, y)

    print("\n--- Modelo Global ---")
    print(f"Grado: {grado}")
    print(f"R^2 global: {r2:.4f}")
    print(f"Ecuación: {eq}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', label="Datos reales")

    x1_lin = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    x2_lin = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    x1_grid, x2_grid = np.meshgrid(x1_lin, x2_lin)
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    X_poly = poly.transform(X_grid)
    y_pred = modelo.predict(X_poly).reshape(x1_grid.shape)

    ax.plot_surface(x1_grid, x2_grid, y_pred, cmap='viridis', alpha=0.7)
    ax.set_title(f"Modelo Global | Grado {grado}")
    ax.set_xlabel("Conc. Reactivo")
    ax.set_ylabel("Conc. Titulante")
    ax.set_zlabel("pH")
    ax.legend()
    plt.tight_layout()
    plt.show()

    guardar = input("\n¿Deseas guardar esta gráfica como imagen? (Sí/No): ").strip().lower()
    if guardar in ["sí", "si"]:
        nombre = f"modelo_global.png"
        fig.savefig(nombre)
        print(f"Imagen guardada como: {nombre}")

# Menú principal
while True:
    print("\n=== Análisis de Titulaciones ===")
    print("1. Analizar un reactivo individual")
    print("2. Realizar análisis global")
    print("3. Salir")
    opcion = input("Selecciona una opción: ").strip()

    if opcion == "1":
        analizar_reactivo()
    elif opcion == "2":
        analisis_global()
    elif opcion == "3":
        print("Saliendo...")
        break
    else:
        print("\nOpción no válida. Intenta de nuevo.")
