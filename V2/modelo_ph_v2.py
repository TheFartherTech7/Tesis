
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from itertools import combinations
import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# === Funciones auxiliares ===

def cargar_datos(archivo="datos_expandido.csv"):
    df = pd.read_csv(archivo)
    df.columns = df.columns.str.strip()
    return df

def mostrar_menu(lista):
    for i, item in enumerate(lista, 1):
        print(f"{i}. {item}")
    print()

def obtener_indices_seleccionados(lista):
    seleccion = input("Selecciona por número(s) separados por coma (ej. 1,3,5): ")
    try:
        indices = [int(i)-1 for i in seleccion.split(",") if int(i)-1 < len(lista)]
        return [lista[i] for i in indices]
    except:
        return []

def mostrar_ecuacion_legible(modelo, poly, nombres_vars, umbral=1e-3):
    coefs = modelo.coef_
    inter = modelo.intercept_
    nombres = poly.get_feature_names_out(nombres_vars)
    ecuacion = f"{inter:.3f}"
    for coef, nombre in zip(coefs, nombres[1:]):
        if abs(coef) >= umbral:
            nombre = nombre.replace("^", "**")
            ecuacion += f" + {coef:.3f}*{nombre}"
    return ecuacion

def ajustar_modelo(df, features, target="pH", max_grado=4):
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_grado = 1
    mejor_poly = None

    X = df[features].values
    y = df[target].values

    for grado in range(1, max_grado + 1):
        poly = PolynomialFeatures(degree=grado, include_bias=False)
        X_poly = poly.fit_transform(X)
        modelo = LinearRegression().fit(X_poly, y)
        r2 = r2_score(y, modelo.predict(X_poly))
        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_modelo = modelo
            mejor_grado = grado
            mejor_poly = poly

    return mejor_modelo, mejor_poly, mejor_grado, mejor_r2

def graficar_3d(df, features, modelo, poly, titulo):
    if len(features) != 2:
        print("Solo se puede graficar en 3D si seleccionas exactamente 2 variables independientes.")
        return

    x = df[features[0]]
    y = df[features[1]]
    z = df["pH"]
    X = df[features].values
    X_poly = poly.transform(X)
    z_pred = modelo.predict(X_poly)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="blue", label="Datos")

    xi, yi = np.meshgrid(
        np.linspace(x.min(), x.max(), 30),
        np.linspace(y.min(), y.max(), 30)
    )
    zi = modelo.predict(poly.transform(np.c_[xi.ravel(), yi.ravel()]))
    ax.plot_surface(xi, yi, zi.reshape(xi.shape), alpha=0.4, color="orange")

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel("pH")
    ax.set_title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if input("¿Deseas guardar la imagen? (s/n): ").lower() == "s":
        fig.savefig(f"grafica_{titulo.replace(' ', '_')}.png")
        print("Imagen guardada.")

def exportar_resultados(nombre_archivo, resultados):
    df = pd.DataFrame(resultados)
    df.to_csv(nombre_archivo, index=False)
    print(f"Resultados guardados en {nombre_archivo}")

# === Funciones principales ===

def analizar_individual(df, reactivos):
    mostrar_menu(reactivos)
    seleccion = obtener_indices_seleccionados(reactivos)
    if not seleccion:
        print("Selección inválida.")
        return
    reactivo = seleccion[0]
    sub_df = df[df[reactivo] > 0]
    if len(sub_df) < 5:
        print("No hay suficientes datos para ese reactivo.")
        return
    features = [reactivo, "Titulante"]
    modelo, poly, grado, r2 = ajustar_modelo(sub_df, features)
    eq = mostrar_ecuacion_legible(modelo, poly, features)
    print(f"Grado: {grado}, R²: {r2:.3f}")
    print("Ecuación:")
    print(eq)
    graficar_3d(sub_df, features, modelo, poly, f"Modelo {reactivo}")

def comparar_reactivos(df, reactivos):
    mostrar_menu(reactivos)
    seleccion = obtener_indices_seleccionados(reactivos)
    if not seleccion:
        print("Selección inválida.")
        return
    sub_df = df[(df[seleccion] > 0).any(axis=1)]
    features = seleccion + ["Titulante"]
    modelo, poly, grado, r2 = ajustar_modelo(sub_df, features)
    eq = mostrar_ecuacion_legible(modelo, poly, features)
    print(f"Grado: {grado}, R²: {r2:.3f}")
    print("Ecuación combinada:")
    print(eq)

def analisis_global(df, reactivos):
    features = reactivos + ["Titulante"]
    modelo, poly, grado, r2 = ajustar_modelo(df, features)
    eq = mostrar_ecuacion_legible(modelo, poly, features)
    print(f"Modelo global: Grado: {grado}, R²: {r2:.3f}")
    print("Ecuación:")
    print(eq)
    exportar_resultados("modelo_global.csv", [{
        "Tipo": "Global",
        "Grado": grado,
        "R2": r2,
        "Ecuacion": eq
    }])

def busqueda_greedy(df, reactivos, umbral=0.80):
    mejores = []
    mejor_r2 = 0
    for r in reactivos:
        sub_df = df[df[r] > 0]
        if len(sub_df) < 5:
            continue
        modelo, poly, grado, r2 = ajustar_modelo(sub_df, [r, "Titulante"])
        if r2 > mejor_r2:
            mejores = [r]
            mejor_r2 = r2

    candidatos = set(reactivos) - set(mejores)

    while candidatos:
        mejoras = []
        for c in candidatos:
            comb = mejores + [c]
            sub_df = df[(df[comb] > 0).any(axis=1)]
            modelo, poly, grado, r2 = ajustar_modelo(sub_df, comb + ["Titulante"])
            if r2 >= umbral:
                mejoras.append((r2, c, modelo, poly, grado))

        if not mejoras:
            break

        mejoras.sort(reverse=True)
        r2, c, modelo, poly, grado = mejoras[0]
        mejores.append(c)
        candidatos.remove(c)

    features = mejores + ["Titulante"]
    sub_df = df[(df[mejores] > 0).any(axis=1)]
    modelo, poly, grado, r2 = ajustar_modelo(sub_df, features)
    eq = mostrar_ecuacion_legible(modelo, poly, features)
    print(f"Reactivos seleccionados: {mejores}")
    print(f"Grado: {grado}, R²: {r2:.3f}")
    print("Ecuación final:")
    print(eq)

def menu():
    df = cargar_datos()
    reactivos = [col for col in df.columns if col not in ["Titulante", "pH"]]

    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Analizar reactivo individual")
        print("2. Comparar múltiples reactivos")
        print("3. Análisis global")
        print("4. Búsqueda greedy de R² ≥ 0.80")
        print("5. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            analizar_individual(df, reactivos)
        elif opcion == "2":
            comparar_reactivos(df, reactivos)
        elif opcion == "3":
            analisis_global(df, reactivos)
        elif opcion == "4":
            busqueda_greedy(df, reactivos)
        elif opcion == "5":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    menu()