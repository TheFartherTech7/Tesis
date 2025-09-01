import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import re
from scipy.stats import f, chi2

# Leer datos
df = pd.read_csv("datos_titulacion.csv")
df.columns = df.columns.str.strip()

reactivos = sorted(df["Reactivo"].unique())
menu_reactivos = {i + 1: reactivo for i, reactivo in enumerate(reactivos)}

# ======================== FUNCIONES ESTADÍSTICAS ========================

def calcular_f_stat(y_true, y_pred, num_params):
    """
    Calcula el estadístico F para el modelo ajustado.
    y_true: valores observados
    y_pred: valores predichos por el modelo
    num_params: número de parámetros (coeficientes) del modelo (incluye intercepto)
    """
    n = len(y_true)
    p = num_params - 1  # número de predictores (excluyendo intercepto)
    rss = np.sum((y_true - y_pred)**2)  # Residual Sum of Squares
    tss = np.sum((y_true - np.mean(y_true))**2)  # Total Sum of Squares

    if rss == 0:
        return np.inf  # perfecto ajuste

    msr = (tss - rss) / p  # Mean Square Regression
    mse = rss / (n - p - 1)  # Mean Square Error

    if mse == 0:
        return np.inf

    F = msr / mse
    return F

def calcular_chi_cuadrada(y_true, y_pred):
    """
    Calcula el estadístico chi cuadrada para el modelo.
    Aquí usamos residuos al cuadrado normalizados por varianza estimada (MSE).
    """
    residuos = y_true - y_pred
    mse = np.mean(residuos**2)
    if mse == 0:
        return 0  # ajuste perfecto, chi cuadrada = 0
    chi_sq = np.sum((residuos**2) / mse)
    return chi_sq

def interpretar_modelo(r2, F, chi_sq, n, p, alfa=0.05):
    """
    Interpreta si el modelo es estadísticamente significativo con base en:
    - R^2 (explicación de varianza)
    - Estadístico F y su p-valor
    - Chi-cuadrada y su p-valor
    Parámetros:
    - n: número de muestras
    - p: número de predictores (excluyendo intercepto)
    - alfa: nivel de significancia
    Devuelve una cadena con resumen.
    """
    resumen = []
    resumen.append(f"R^2: {r2:.5f}")

    # p-valor de F
    if np.isinf(F):
        p_valor_f = 0.0
    else:
        p_valor_f = 1 - f.cdf(F, p, n - p - 1)

    resumen.append(f"Estadístico F: {F:.3f}")
    resumen.append(f"p-valor (F): {p_valor_f:.5f}")

    # p-valor chi cuadrada
    dof = n - p - 1  # grados de libertad aproximados
    p_valor_chi = 1 - chi2.cdf(chi_sq, dof)
    resumen.append(f"Chi-cuadrada: {chi_sq:.3f}")
    resumen.append(f"p-valor (Chi-cuadrada): {p_valor_chi:.5f}")

    if p_valor_f < alfa:
        resumen.append("El modelo es estadísticamente significativo (según F).")
    else:
        resumen.append("El modelo NO es estadísticamente significativo (según F).")

    if p_valor_chi < alfa:
        resumen.append("El ajuste es bueno según el test Chi-cuadrada.")
    else:
        resumen.append("El ajuste NO es bueno según el test Chi-cuadrada.")

    return "\n".join(resumen)

# ======================== FUNCIONES PRINCIPALES ========================

def mostrar_ecuacion_legible(coefs, terms):
    from collections import defaultdict
    grupos_x1 = defaultdict(list)
    grupos_x2_solo = defaultdict(list)

    for coef, term in zip(coefs[1:], terms[1:]):
        if abs(coef) < 1e-5:
            continue
        term_clean = term.replace(" ", "").replace("^", "**")
        pot_x1 = 0
        if "X1**3" in term_clean: pot_x1 = 3
        elif "X1**2" in term_clean: pot_x1 = 2
        elif "X1*" in term_clean or term_clean == "X1": pot_x1 = 1

        if pot_x1 > 0:
            grupos_x1[pot_x1].append((coef, term_clean))
        else:
            pot_x2 = 0
            if "X2**4" in term_clean: pot_x2 = 4
            elif "X2**3" in term_clean: pot_x2 = 3
            elif "X2**2" in term_clean: pot_x2 = 2
            elif "X2*" in term_clean or term_clean == "X2": pot_x2 = 1
            grupos_x2_solo[pot_x2].append((coef, term_clean))

    constante = coefs[0]
    ecuacion = "pH = \n"
    bloques = []

    import re

    for pot in sorted(grupos_x1.keys(), reverse=True):
        term_strs = []
        for coef, t in grupos_x1[pot]:
            part = re.split(r'X1(\*\*[\d]+)?\*?', t)[-1]
            part = part if part else "1"
            term_strs.append(f"{coef:.5f}{'*' + part if part != '1' else ''}")
        bloque = f"    X1^{pot}*(" + " + ".join(term_strs) + ")"
        bloques.append(bloque)

    for pot in sorted(grupos_x2_solo.keys(), reverse=True):
        term_strs = []
        for coef, t in grupos_x2_solo[pot]:
            part = re.split(r'X2(\*\*[\d]+)?\*?', t)[-1]
            part = part if part else "1"
            term_strs.append(f"{coef:.5f}{'*' + part if part != '1' else ''}")
        bloque = f"    X2^{pot}*(" + " + ".join(term_strs) + ")"
        bloques.append(bloque)

    bloques.append(f"    {constante:.5f}")
    ecuacion += " +\n".join(bloques)
    return ecuacion

def mostrar_ecuacion(modelo, poly):
    coefs = modelo.coef_
    terms = poly.get_feature_names_out(["X1", "X2"])
    ecuacion = mostrar_ecuacion_legible(coefs, terms)
    return ecuacion.replace("**", "^")

def ajustar_modelo(X, y):
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_grado = 0
    mejor_poly = None
    mejor_ecuacion = ""

    for grado in range(1, 4):
        poly = PolynomialFeatures(degree=grado)
        X_poly = poly.fit_transform(X)
        modelo = LinearRegression().fit(X_poly, y)
        r2 = r2_score(y, modelo.predict(X_poly))
        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_modelo = modelo
            mejor_grado = grado
            mejor_poly = poly
            mejor_ecuacion = mostrar_ecuacion(modelo, poly)

    return mejor_modelo, mejor_grado, mejor_r2, mejor_ecuacion, mejor_poly

def limpiar_nombre_archivo(nombre):
    import re
    return re.sub(r'[^A-Za-z0-9_.-]', '_', nombre)

def guardar_resultados_csv(titulo, grado, r2, eq):
    nombre_archivo = f"resultados_{limpiar_nombre_archivo(titulo.lower())}.csv"
    resultados_df = pd.DataFrame([{"Titulo": titulo, "Grado": grado, "R2": r2, "Ecuacion": eq}])
    resultados_df.to_csv(nombre_archivo, index=False)
    print(f"Resultados guardados como: {nombre_archivo}")

def graficar_superficie(df_subset, modelo, poly, grado, titulo="Modelo", cmap='viridis'):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    X = df_subset[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = df_subset["pH"].values

    ax.scatter(X[:, 0], X[:, 1], y, c='r', label="Datos reales", s=30, alpha=0.7)

    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    y_pred = modelo.predict(poly.transform(X_grid)).reshape(x1_grid.shape)

    ax.plot_surface(x1_grid, x2_grid, y_pred, cmap=cmap, alpha=0.5, edgecolor='none')

    ax.set_title(f"{titulo} | Grado {grado}")
    ax.set_xlabel("Conc. Reactivo")
    ax.set_ylabel("Conc. Titulante")
    ax.set_zlabel("pH")
    ax.set_zlim(0, 14)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

    while True:
        decision = input("\n¿Deseas guardar esta gráfica y/o el CSV? (Sí/No): ").strip().lower()
        if decision in ["sí", "si"]:
            while True:
                opcion = input("¿Qué deseas guardar? (1: Gráfica, 2: CSV, 3: Ambos): ").strip()
                if opcion in ["1", "2", "3"]:
                    if opcion in ["1", "3"]:
                        nombre = f"{limpiar_nombre_archivo(titulo)}.png"
                        fig.savefig(nombre, dpi=300)
                        print(f"Imagen guardada como: {nombre}")
                    return opcion in ["2", "3"]
                else:
                    print("Opción inválida. Ingresa 1, 2 o 3.")
        elif decision in ["no", "n"]:
            return False
        else:
            print("Respuesta no válida. Por favor, responde Sí o No.")

def analizar_reactivo():
    print("\nSelecciona un reactivo:")
    for num, nombre in menu_reactivos.items():
        print(f"{num}. {nombre}")
    entrada = input("\nEscribe el número o nombre del reactivo: ").strip().lower()

    reactivo = None
    if entrada.isdigit():
        reactivo = menu_reactivos.get(int(entrada))
    else:
        for r in reactivos:
            if r.lower() == entrada:
                reactivo = r
                break

    if not reactivo:
        print("\nOpción inválida. Intenta de nuevo.")
        return

    sub_df = df[df["Reactivo"] == reactivo]
    X = sub_df[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = sub_df["pH"].values
    modelo, grado, r2, eq, poly = ajustar_modelo(X, y)

    y_pred = modelo.predict(poly.transform(X))

    # Estadísticos
    num_params = len(modelo.coef_)
    n = len(y)
    p = num_params - 1
    F = calcular_f_stat(y, y_pred, num_params)
    chi_sq = calcular_chi_cuadrada(y, y_pred)
    interpretacion = interpretar_modelo(r2, F, chi_sq, n, p)

    print("\n--- Resultado ---")
    print(f"Reactivo: {reactivo}")
    print(f"Mejor modelo: Grado {grado}")
    print(f"R^2: {r2:.5f}")
    print(f"Ecuación:\n{eq}\n")
    print("--- Análisis estadístico ---")
    print(interpretacion)

    guardar_csv = graficar_superficie(sub_df, modelo, poly, grado, titulo=reactivo)
    if guardar_csv:
        guardar_resultados_csv(reactivo, grado, r2, eq)


# ======================== MENÚ PRINCIPAL ========================

def main():
    while True:
        print("\n=== Análisis de Titulaciones ===")
        print("1. Analizar un reactivo individual")
        print("2. Salir")

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            analizar_reactivo()
        elif opcion == "2":
            print("Saliendo...")
            break
        else:
            print("\nOpción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
