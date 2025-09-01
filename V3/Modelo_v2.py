import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import f, chi2
import os

# Evitar warnings por símbolos Unicode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Cargar datos
archivo = 'datos_expandido.csv'
df = pd.read_csv(archivo)

# Directorio de resultados
os.makedirs("resultados", exist_ok=True)

# Función para ajustar modelo polinomial y calcular estadísticas
def ajustar_modelo(df, features, target="pH", max_grado=4):
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_grado = 1
    mejor_poly = None
    mejor_y_pred = None

    X = df[features].values
    y = df[target].values

    for grado in range(1, max_grado + 1):
        poly = PolynomialFeatures(degree=grado)
        X_poly = poly.fit_transform(X)
        modelo = LinearRegression().fit(X_poly, y)
        y_pred = modelo.predict(X_poly)
        r2 = r2_score(y, y_pred)
        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_modelo = modelo
            mejor_grado = grado
            mejor_poly = poly
            mejor_y_pred = y_pred

    # Calcular estadísticas
    X_poly_final = mejor_poly.fit_transform(X)
    y_pred_final = mejor_modelo.predict(X_poly_final)
    residuals = y - y_pred_final
    mse = np.mean(residuals ** 2)
    chi2_val = np.sum((residuals ** 2) / y_pred_final)
    chi2_p = 1 - chi2.cdf(chi2_val, df=len(y) - X_poly_final.shape[1])

    n = len(y)
    k = X_poly_final.shape[1] - 1
    f_val = (mejor_r2 / k) / ((1 - mejor_r2) / (n - k - 1))
    f_p = 1 - f.cdf(f_val, k, n - k - 1)

    return mejor_modelo, mejor_poly, mejor_r2, mejor_grado, mejor_y_pred, f_val, f_p, chi2_val, chi2_p

# Función para graficar en 3D
def graficar_3d(df, poly, modelo, features, y_pred, nombre):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = df[features].values
    Y = df["pH"].values
    ax.scatter(X[:, 0], X[:, 1], Y, color='blue', label="Datos reales")
    ax.scatter(X[:, 0], X[:, 1], y_pred, color='red', label="Modelo")
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel("pH")
    plt.title(f"Modelo 3D - {nombre}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/modelo_3d_{nombre}.png")
    plt.close()

# Función para guardar resultados en CSV
def guardar_resultados_csv(lista_resultados):
    df_resultados = pd.DataFrame(lista_resultados)
    df_resultados.to_csv("resultados/resultados_modelo.csv", index=False)

# Función para graficar comparaciones
def graficar_comparaciones(df_resultados):
    for columna in ["R2", "F", "Chi2"]:
        plt.figure()
        plt.bar(df_resultados["Reactivo"], df_resultados[columna], color='teal')
        plt.ylabel(columna)
        plt.xticks(rotation=45)
        plt.title(f"Comparación de {columna} entre reactivos")
        plt.tight_layout()
        plt.savefig(f"resultados/comparacion_{columna.lower()}.png")
        plt.close()

# Función principal para analizar reactivo
def analizar_reactivo(nombre_reactivo):
    df_reactivo = df[[f"Concentracion_{nombre_reactivo}", "Concentracion_titulante", f"pH_{nombre_reactivo}"].dropna()]
    df_reactivo.columns = ["Reactivo", "Titulante", "pH"]
    modelo, poly, r2, grado, y_pred, f_val, f_p, chi2_val, chi2_p = ajustar_modelo(
        df_reactivo, ["Reactivo", "Titulante"], "pH"
    )
    graficar_3d(df_reactivo, poly, modelo, ["Reactivo", "Titulante"], y_pred, nombre_reactivo)
    return {
        "Reactivo": nombre_reactivo,
        "R2": r2,
        "Grado": grado,
        "F": f_val,
        "p-valor F": f_p,
        "Chi2": chi2_val,
        "p-valor Chi2": chi2_p,
        "Significativo": f_p < 0.05
    }

# Menú interactivo
reactivos = [col.replace("pH_", "") for col in df.columns if col.startswith("pH_")]
resultados = []
while True:
    print("\nReactivos disponibles:")
    for i, r in enumerate(reactivos):
        print(f"{i+1}. {r}")
    opcion = input("Seleccione un reactivo (o escriba 'salir'): ")
    if opcion.lower() == "salir":
        break
    try:
        idx = int(opcion) - 1
        if 0 <= idx < len(reactivos):
            nombre = reactivos[idx]
            print(f"Analizando {nombre}...")
            resultado = analizar_reactivo(nombre)
            resultados.append(resultado)
        else:
            print("Índice fuera de rango.")
    except ValueError:
        print("Entrada inválida.")

if resultados:
    guardar_resultados_csv(resultados)
    graficar_comparaciones(pd.DataFrame(resultados))
    print("\n✅ Análisis finalizado. Resultados guardados en carpeta 'resultados'.")
else:
    print("\nNo se analizaron reactivos.")
