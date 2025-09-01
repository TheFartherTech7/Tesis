import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import f, chi2
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import griddata
from scipy.interpolate import SmoothBivariateSpline
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict




warnings.filterwarnings("ignore")

# Leer datos
df = pd.read_csv("datos_titulacion.csv")
df.columns = df.columns.str.strip()

reactivos = sorted(df["Reactivo"].unique())
menu_reactivos = {i + 1: reactivo for i, reactivo in enumerate(reactivos)}

PH_MIN = 3
PH_MAX = 9

# Funciones estadísticas
def calcular_estadisticas(y_real, y_pred, num_parametros):
    n = len(y_real)
    ss_total = np.sum((y_real - np.mean(y_real))**2)
    ss_res = np.sum((y_real - y_pred)**2)
    r2 = 1 - (ss_res / ss_total)
    mse = ss_res / n

    if n > num_parametros:
        f_stat = ((ss_total - ss_res) / (num_parametros - 1)) / (ss_res / (n - num_parametros))
        p_f = 1 - f.cdf(f_stat, num_parametros - 1, n - num_parametros)
    else:
        f_stat = np.nan
        p_f = np.nan

    # Chi-cuadrada con varianza estimada (MSE) para normalizar
    if mse > 0:
        chi_sq = np.sum((y_real - y_pred) ** 2 / mse)
        p_chi = 1 - chi2.cdf(chi_sq, df=n - num_parametros)
    else:
        chi_sq = 0
        p_chi = 1

    return r2, mse, f_stat, p_f, chi_sq, p_chi


def imprimir_ecuacion_polinomial(modelo, poly):
    terminos = poly.get_feature_names_out(["C_reactivo", "C_titulante"])
    coeficientes = modelo.coef_
    intercepto = modelo.intercept_

    ecuacion = f"pH = {intercepto:.4f}"
    for coef, termino in zip(coeficientes[1:], terminos[1:]):  # Saltamos el término constante
        # Reemplazamos nombres para la salida
        termino_simple = termino.replace("C_reactivo", "Cr").replace("C_titulante", "Ct")
        signo = " + " if coef >= 0 else " - "
        ecuacion += f"{signo}{abs(coef):.4f}*{termino_simple}"

    print(ecuacion)

        

    
# Función principal
def analizar_reactivo(nombre_reactivo, grado=3):
    datos = df[df["Reactivo"] == nombre_reactivo]
    X = datos[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = datos["pH"].values

    poly = PolynomialFeatures(degree=grado)
    X_poly = poly.fit_transform(X)
    modelo = LinearRegression().fit(X_poly, y)
    y_pred = np.clip(modelo.predict(X_poly), 3.0, 9.0)
    
    imprimir_ecuacion_polinomial(modelo, poly)
    
        # Calcular errores individuales y agregarlos al DataFrame
    errores_abs = y - y_pred
    error_pct = np.abs(errores_abs / y) * 100
    mape = np.mean(error_pct)
    
    datos = datos.copy()  # Para evitar modificar el original
    datos["pH_predicho"] = np.round(y_pred,2)
    datos["Error_abs"] = errores_abs
    datos["Error_pct"] = error_pct

    print("\nErrores por punto de datos:")
    print(datos[["Concentracion de reactivo", "Concentracion de titulante", "pH", "pH_predicho", "Error_pct"]].round(3))
    print(f"\n📊 MAPE (Error porcentual medio): {mape:.2f}%")

    # Opción para exportar
    exportar = input("\n¿Deseas exportar los resultados a un CSV? (s/n): ").strip().lower()
    if exportar == 's':
        nombre_archivo = f"{nombre_reactivo.replace(' ', '_')}.csv"
        datos.to_csv(nombre_archivo, index=False)
        print(f"  ✅ Resultados exportados como '{nombre_archivo}'")

    r2, mse, f_stat, p_f, chi_sq, p_chi = calcular_estadisticas(y, y_pred, X_poly.shape[1])

    print(f"\nResultados para {nombre_reactivo} (grado {grado}):")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  F-estadístico: {f_stat:.2f}, p-valor: {p_f:.3e} -> {'Significativo' if p_f < 0.05 else 'No significativo'}")
    print(f"  Chi²: {chi_sq:.2f}, p-valor: {p_chi:.3e} -> {'El modelo se ajusta bien a los datos' if p_chi > 0.05 else 'El modelo NO se ajusta bien a los datos'}")


    # Gráfica 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c='blue', label='Datos reales')
    ax.scatter(X[:, 0], X[:, 1], y_pred, c='red', marker='x', label='Modelo')
    ax.set_xlabel("Conc. Reactivo")
    ax.set_ylabel("Conc. Titulante")
    ax.set_zlabel("pH")
    ax.set_title(f"{nombre_reactivo} - Ajuste grado {grado}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Interpolación en malla 2D
    opcion_malla = input("\n¿Deseas generar una malla de interpolación? (s/n): ").strip().lower()
    if opcion_malla == 's':
        try:
            pasos = int(input("  ¿Cuántos pasos deseas por dimensión? (ej. 20): "))
        except:
            pasos = 20  # valor por defecto

        opcion_metodo = input("¿Qué tipo de interpolación quieres? (1) Normal (2) Suave spline: ").strip()
    
        if opcion_metodo == '1':
            df_interpolado = interpolar_malla(modelo, poly, datos, pasos, nombre_reactivo)
        elif opcion_metodo == '2':
            df_interpolado = interpolar_malla_suave(datos, pasos, nombre_reactivo)
        else:
            print("Opción inválida, se usará interpolación normal por defecto.")
            df_interpolado = interpolar_malla(modelo, poly, datos, pasos, nombre_reactivo)
    else:
        menu()


    # ... (continúa tu código que maneja la exportación y muestra de la malla)

        archivo_malla = f"{nombre_reactivo.replace(' ', '_')}_malla_interpolada.csv"

        # Revisar si ya existe la malla
        if os.path.exists(archivo_malla):
            print(f"📁 Ya existe el archivo '{archivo_malla}'. Leyendo archivo existente...")
            df_interpolado = pd.read_csv(archivo_malla)
        else:
            df_interpolado = interpolar_malla(modelo, poly, datos, pasos, nombre_reactivo)


        exportar_malla = input("  ¿Exportar malla a CSV? (s/n): ").strip().lower()
        if exportar_malla == 's':
            df_interpolado.to_csv(archivo_malla, index=False)
            print(f"✅ Malla exportada como '{archivo_malla}'")
        else:
            print("ℹ️ Malla generada pero no exportada.")


        # Mostrar la gráfica 3D en cualquier caso
        

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df_interpolado["Conc. Reactivo"], 
                        df_interpolado["Conc. Titulante"], 
                        df_interpolado["pH_estimado"], 
                        cmap='viridis', edgecolor='none', alpha=0.9)

        ax.set_xlabel('Concentración Reactivo')
        ax.set_ylabel('Concentración Titulante')
        ax.set_zlabel('pH Estimado')
        ax.set_title(f'Superficie 3D - {nombre_reactivo}')
        plt.show()

        # Interpolación personalizada
        while True:
            opcion = input("\n¿Deseas predecir pH con nuevas concentraciones? (s/n). Escriba 'salir' si desea cerrar el programa: ").strip().lower()
            if opcion == 's':
                try:
                    print(f"ℹ️ Rango válido de Concentración Reactivo: {df_interpolado['Conc. Reactivo'].min():.3f} a {df_interpolado['Conc. Reactivo'].max():.2f}")
                    print(f"ℹ️ Rango válido de Concentración Titulante: {df_interpolado['Conc. Titulante'].min():.4f} a {df_interpolado['Conc. Titulante'].max():.2f}")
                    c1 = float(input("  Ingresa concentración de reactivo: "))
                    c2 = float(input("  Ingresa concentración de titulante: "))

                    ph_interp = predecir_con_interpolacion(df_interpolado, c1, c2)
                    if ph_interp is not None:
                        print(f"  ✅ pH estimado por interpolación: {ph_interp:.2f}")
                    else:
                        # Si el punto está fuera del rango de la malla, usa el modelo directo
                        print("  ⚠️ Punto fuera del rango de la malla, se usará el modelo ajustado.")

                        punto = poly.transform([[c1, c2]])
                        predicho = np.clip(modelo.predict(punto)[0], PH_MIN, PH_MAX)

                        # Cálculo del intervalo de confianza
                        X_poly_full = poly.transform(datos[["Concentracion de reactivo", "Concentracion de titulante"]].values)
                        residuos = y - modelo.predict(X_poly_full)
                        mse = np.mean(residuos ** 2)

                        XTX_inv = np.linalg.inv(X_poly_full.T @ X_poly_full)
                        var_pred = mse * (punto @ XTX_inv @ punto.T)[0][0]

                        from scipy.stats import t
                        t_val = t.ppf(0.975, df=len(y) - X_poly_full.shape[1])
                        margen = t_val * np.sqrt(var_pred)
                        lim_inf = max(predicho - margen, PH_MIN)
                        lim_sup = min(predicho + margen, PH_MAX)

                        print(f"  ✅ pH estimado: {predicho:.2f}")
                        print(f"  🔍 Intervalo de confianza (95%): [{lim_inf:.2f}, {lim_sup:.2f}]")

                        # Comparar con datos reales si existe
                        match = datos[
                            (datos["Concentracion de reactivo"] == c1) &
                            (datos["Concentracion de titulante"] == c2)
                        ]
                        if not match.empty:
                            ph_real = match["pH"].values[0]
                            err_pct = np.abs((ph_real - predicho) / ph_real * 100)
                            print(f"  📌 pH real (desde CSV): {ph_real:.2f}")
                            print(f"  Error porcentual: {err_pct:.2f}%")
                        else:
                            print("  ⚠️ No se encontró ese punto exacto en los datos originales.")

                except Exception as e:
                    print(f"  ❌ Error al predecir: {e}")
            elif opcion == 'n':
                continue
            elif opcion == 'salir':
                break
            

        

def interpolar_malla(modelo, poly, datos, pasos=10, nombre_reactivo =""):
    min_c1, max_c1 = datos["Concentracion de reactivo"].min(), datos["Concentracion de reactivo"].max()
    min_c2, max_c2 = datos["Concentracion de titulante"].min(), datos["Concentracion de titulante"].max()
    
    c1_vals = np.linspace(min_c1,max_c1, pasos)
    c2_vals = np.linspace(min_c2, max_c2, pasos)
    
    # Obtener rango experimental de pH
    ph_min_exp = datos["pH"].min()
    ph_max_exp = datos["pH"].max()
    
    interpolaciones = []
    
    for c1 in c1_vals:
        for c2 in c2_vals:
            
            punto = [[c1, c2]]
            # Evaluar vecindad local
            nbrs = NearestNeighbors(n_neighbors=4).fit(datos[["Concentracion de reactivo", "Concentracion de titulante"]])
            _, indices = nbrs.kneighbors([[c1, c2]])
            vecinos = datos.iloc[indices[0]]
            vecinos_ph = vecinos["pH"].values

            # Interpolar solo si hay vecinos razonablemente cerca
            if vecinos.shape[0] >= 3:
                punto_poly = poly.transform(punto)
                pred = modelo.predict(punto_poly)[0]  # Aquí faltaba poly.transform
                pred = np.clip(pred, max(vecinos_ph.min(), ph_min_exp - 0.5),
                                    min(vecinos_ph.max(), ph_max_exp + 0.5))
            else:
                pred = np.nan  # Deja un hueco si no hay datos confiables

            interpolaciones.append((c1, c2, pred))
    
    df_interpolado = pd.DataFrame(interpolaciones, columns=["Conc. Reactivo", "Conc. Titulante", "pH_estimado"])
    
    print("\n📌 Interpolación dentro del rango de datos:")
    print(df_interpolado.round(3))
    
    # --------- Punto 3: Visualización de la malla interpolada ---------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    X = df_interpolado["Conc. Reactivo"]
    Y = df_interpolado["Conc. Titulante"]
    Z = df_interpolado["pH_estimado"]

    # Elimina puntos NaN para graficar correctamente
    mask = ~np.isnan(Z)
    ax.plot_trisurf(X[mask], Y[mask], Z[mask], cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_title(f"🌐 Superficie Interpolada - {nombre_reactivo}")
    ax.set_xlabel("Concentración Reactivo")
    ax.set_ylabel("Concentración Titulante")
    ax.set_zlabel("pH estimado")
    
    plt.tight_layout()
    plt.show()
    
    return df_interpolado  # <-- CORRECTAMENTE afuera de los for loops
  # <- Asegúrate de mantener esto al final añade esto al final

def interpolar_malla_suave(datos, pasos=20, nombre_reactivo=""):
    """
    Interpolación suave 2D con SmoothBivariateSpline para evitar picos bruscos.
    """
    x = datos["Concentracion de reactivo"].values
    y = datos["Concentracion de titulante"].values
    z = datos["pH"].values

    # Ajuste spline con suavizado (s > 0)
    spline = SmoothBivariateSpline(x, y, z, s=0.1)  # Ajusta 's' para más o menos suavizado

    c1_vals = np.linspace(x.min(), x.max(), pasos)
    c2_vals = np.linspace(y.min(), y.max(), pasos)

    interpolaciones = []

    for c1 in c1_vals:
        for c2 in c2_vals:
            pred = spline.ev(c1, c2)
            pred = np.clip(pred, PH_MIN, PH_MAX)
            interpolaciones.append((c1, c2, pred))

    df_interpolado = pd.DataFrame(interpolaciones, columns=["Conc. Reactivo", "Conc. Titulante", "pH_estimado"])

    print(f"\n📌 Interpolación suave con spline para {nombre_reactivo}:")
    print(df_interpolado.round(3))

    return df_interpolado

def predecir_con_interpolacion(df_malla, c1, c2):
    puntos = df_malla[["Conc. Reactivo", "Conc. Titulante"]].values
    valores = df_malla["pH_estimado"].values
    punto_consulta = np.array([[c1, c2]])
    
    min_c1, max_c1 = df_malla["Conc. Reactivo"].min(), df_malla["Conc. Reactivo"].max()
    min_c2, max_c2 = df_malla["Conc. Titulante"].min(), df_malla["Conc. Titulante"].max()
    
    rbf = RBFInterpolator(puntos, valores, smoothing=1.0, kernel = 'thin_plate_spline')
    try:
        ph_est = rbf(punto_consulta)[0]
    except Exception:
        ph_est = np.nan

    if not np.isnan(ph_est):
        # Filtrado local con vecinos
        nbrs = NearestNeighbors(n_neighbors=4).fit(puntos)
        distances, indices = nbrs.kneighbors(punto_consulta)
        vecinos_pH = valores[indices[0]]
        pH_min = max(vecinos_pH.min(), df_malla["pH_estimado"].min())
        pH_max = min(vecinos_pH.max(), df_malla["pH_estimado"].max())
        ph_est = np.clip(ph_est, pH_min, pH_max)
        
    return ph_est
  
# Menú de selección
def menu():
    print("\nReactivos disponibles:")
    for i, r in menu_reactivos.items():
        print(f"  {i}. {r}")
    try:
        opcion = int(input("Selecciona el número de reactivo: "))
        if opcion in menu_reactivos:
            grado = int(input("Grado del modelo polinomial (ej. 2, 3, 4): "))
            analizar_reactivo(menu_reactivos[opcion], grado)
        else:
            print("❌ Opción inválida.")
    except ValueError:
        print("❌ Entrada no válida.")

if __name__ == "__main__":
    menu()

