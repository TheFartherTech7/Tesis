import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from scipy.stats import stats
import seaborn as sns
import os

def unir_csv_suavizados(directorio):
    archivos = [f for f in os.listdir(directorio) if f.endswith("suavizado.csv")]
    
    if not archivos:
        print("❌ No se encontraron archivos que terminen en 'suavizado.csv'.")
        return
    
    dfs = []
    for archivo in archivos:
        ruta_completa = os.path.join(directorio, archivo)
        try:
            df = pd.read_csv(ruta_completa)
            # Extraer el nombre del reactivo del archivo
            reactivo = archivo.replace("suavizado.csv", "").replace("superficie_", "").replace("_", "").strip()
            df["Reactivo"] = reactivo
            dfs.append(df)
            print(f"✅ Archivo añadido: {archivo}")
        except Exception as e:
            print(f"⚠️ Error al leer {archivo}: {e}")
    
    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        columnas_ordenadas = ["Reactivo", "Concenrtacion de reactivo", "Concentracion de titulante", "pH"]
        df_final = df_final[columnas_ordenadas]
        salida = os.path.join(directorio, "datos_suavizados_combinados.csv")
        df_final.to_csv(salida, index=False)
        print(f"\n📦 Archivo combinado exportado como: {salida}")
    else:
        print("❌ No se pudieron procesar archivos válidos.")


def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    return df

def filtrar_reactivo(df, reactivo):
    return df[df["Reactivo"] == reactivo].copy()

def entrenar_modelo(df, modelo='polinomial', grado=3):
    X = df[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = df["pH"].values

    if modelo == 'polinomial':
        poly = PolynomialFeatures(degree=grado)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        return lambda X_new: model.predict(poly.transform(X_new))
    
    elif modelo == 'svr':
        scaler = StandardScaler()
        svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, epsilon=0.1))
        svr_model.fit(X, y)
        return lambda X_new: svr_model.predict(X_new)

def generar_grid(df, resolucion=100):
    x = df["Concentracion de reactivo"]
    y = df["Concentracion de titulante"]
    x_lin = np.linspace(x.min(), x.max(), resolucion)
    y_lin = np.linspace(y.min(), y.max(), resolucion)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    puntos = np.c_[X_grid.ravel(), Y_grid.ravel()]
    return X_grid, Y_grid, puntos

def graficar_superficie(X_grid, Y_grid, Z_grid, titulo="Superficie suavizada"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel("Concentración de Reactivo (M)")
    ax.set_ylabel("Concentración de Titulante (M)")
    ax.set_zlabel("pH")
    ax.set_title(titulo)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

def exportar_grid(X_grid, Y_grid, Z_grid, nombre_archivo):
    df_out = pd.DataFrame({
        "Concentracion de reactivo": X_grid.ravel(),
        "Concentracion de titulante": Y_grid.ravel(),
        "pH": Z_grid.ravel()
    })
    df_out.to_csv(nombre_archivo, index=False)
    print(f"Archivo exportado como: {nombre_archivo}")
    
    
def menu_estimador_ph():
    df = pd.read_csv("datos_suavizados_combinados.csv")

    # 💡 Limpiar directamente la columna "Reactivo"
    df["Reactivo"] = df["Reactivo"].str.replace("svr", "", case=False, regex=True).str.strip()

    # ✅ Usamos directamente la columna limpia
    reactivos = sorted(df["Reactivo"].unique())

    print("🔬 Reactivos disponibles:")
    for i, r in enumerate(reactivos, 1):
        print(f"{i}. {r}")

    seleccion = input("\n👉 Selecciona un número de reactivo: ").strip()

    try:
        idx = int(seleccion) - 1
        if idx < 0 or idx >= len(reactivos):
            raise ValueError
        reactivo = reactivos[idx]
    except ValueError:
        print("❌ Selección inválida.")
        return

    df_r = df[df["Reactivo"] == reactivo].copy()
    
        # Preparar variables independientes (X) y dependiente (y)
    X = df_r[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = df_r["pH"].values

    # Entrenar el modelo SVR
    
    modelo_svr = SVR(kernel="rbf")  # Puedes ajustar el kernel si ya sabes cuál va mejor
    modelo_svr.fit(X, y)
    # Modelo de regresión lineal simple para evaluación estadística
    model = LinearRegression()
    model.fit(X, y)
    # Predicción
    y_pred = modelo_svr.predict(X)

    # Métricas
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)

    # Mostrar
    print(f"🔹 R²     = {r2:.4f}")
    print(f"🔹 MSE    = {mse:.4f}")
    print(f"🔹 RMSE   = {rmse:.4f}")
    print(f"🔹 MAE    = {mae:.4f}")
   
   


    x_min, x_max = df_r["Concentracion de reactivo"].min(), df_r["Concentracion de reactivo"].max()
    y_min, y_max = df_r["Concentracion de titulante"].min(), df_r["Concentracion de titulante"].max()

    while True:
        try:
            print(f"\nℹ️ Rango válido de Conc. Reactivo: {x_min:.4f} a {x_max:.4f}")
            c1 = float(input("💧 Ingresa concentración de reactivo: "))
            print(f"ℹ️ Rango válido de Conc. Titulante: {y_min:.4f} a {y_max:.4f}")
            c2 = float(input("🧪 Ingresa concentración de titulante: "))

            fuera_rango = False
            if not (x_min <= c1 <= x_max):
                print("⚠️ ¡Cuidado! La concentración del reactivo está fuera del rango de la malla.")
                fuera_rango = True
            if not (y_min <= c2 <= y_max):
                print("⚠️ ¡Cuidado! La concentración del titulante está fuera del rango de la malla.")
                fuera_rango = True

            if fuera_rango:
                print("⚠️ El valor de pH puede no ser confiable porque estás extrapolando fuera de los datos disponibles.")

        except ValueError:
            print("❌ Valor inválido.")
            continue  # volver a pedir

        df_r["distancia"] = np.sqrt((df_r["Concentracion de reactivo"] - c1)**2 + (df_r["Concentracion de titulante"] - c2)**2)
        punto_cercano = df_r.loc[df_r["distancia"].idxmin()]
        ph_estimado = punto_cercano["pH"]

        ph_estimado = max(3, min(9, ph_estimado))

        ph_min = df_r.groupby("Concentracion de titulante")["pH"].min().min()
        ph_max = df_r.groupby("Concentracion de titulante")["pH"].max().max()

        if ph_estimado < ph_min:
            ph_estimado = ph_min
            print("⚠️ Se alcanzó el límite inferior de la malla, tenga precaución.")
        elif ph_estimado > ph_max:
            ph_estimado = ph_max
            print("⚠️ Se alcanzó el límite superior de la malla, tenga precaución.")

        print(f"\n📌 pH estimado para {reactivo} con C_reactivo = {c1:.4f} M y C_titulante = {c2:.4f} M: pH ≈ {ph_estimado:.2f}")

        X_grid, Y_grid = np.meshgrid(
            np.unique(df_r["Concentracion de reactivo"]),
            np.unique(df_r["Concentracion de titulante"])
        )
        Z_grid = df_r.pivot(index="Concentracion de titulante", columns="Concentracion de reactivo", values="pH").values

        # 🔍 Filtrar valores luego de alcanzar el límite de pH (3.00 o 9.00)
        limite_alcanzado = False
        Z_filtrado = []
        for fila in Z_grid:
            fila_filtrada = []
            for z in fila:
                if not limite_alcanzado:
                    fila_filtrada.append(z)
                    if z <= 3.00 or z >= 9.00:
                        limite_alcanzado = True
                else:
                    fila_filtrada.append(np.nan)
            Z_filtrado.append(fila_filtrada)
        Z_filtrado = np.array(Z_filtrado)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8, edgecolor='none')
        ax.scatter(c1, c2, ph_estimado, color='red', s=100, label='pH estimado')
        ax.set_xlabel("Concentración de Reactivo (M)")
        ax.set_ylabel("Concentración de Titulante (M)")
        ax.set_zlabel("pH")
        ax.set_zlim(3, 9)
        ax.set_title(f"{reactivo} - pH estimado")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Preguntar si seguir con el mismo reactivo
        while True:
            continuar = input("\n¿Quieres probar otra concentración con este reactivo (s), cambiar de reactivo (r), o salir (x)? ").lower()
            if continuar == 's':
                break  # rompe este while interno, pero mantiene el de las concentraciones
            elif continuar == 'r':
                menu_estimador_ph()
                return
            elif continuar == 'x':
                print("👋 Hasta luego.")
                return
            else:
                print("❌ Opción inválida.")


   

# ... [todas tus importaciones y funciones iguales hasta aquí] ...

def main():
    print("=== Selecciona una opción ===")
    print("1. 🔄 Unir archivos suavizados")
    print("2. 🧪 Entrenar y graficar modelo para un reactivo")
    print("3. 🔍 Estimar pH a partir de concentraciones")

    opcion = input("👉 Ingresa 1, 2 o 3: ").strip()

    if opcion == "1":
        ruta_directorio = input("📁 Ingresa la ruta de la carpeta con los archivos suavizados: ")
        unir_csv_suavizados(ruta_directorio)

    elif opcion == "2":
        ruta = input("📂 Ingresa la ruta del archivo CSV: ")
        df = cargar_datos(ruta)
        
        reactivo = input(f"🧪 Ingresa el nombre del reactivo a analizar (ej. CaCl2): ")
        df_r = filtrar_reactivo(df, reactivo)

        modelo = input("⚙️ Modelo a usar (polinomial / svr): ").strip().lower()
        grado = 3
        if modelo == 'polinomial':
            grado = int(input("🔢 Grado del polinomio (ej. 2 o 3): "))
        
        predictor = entrenar_modelo(df_r, modelo=modelo, grado=grado)
        X_grid, Y_grid, puntos = generar_grid(df_r)

        Z_pred = predictor(puntos).reshape(X_grid.shape)
        graficar_superficie(X_grid, Y_grid, Z_pred, f"{reactivo} - Superficie Suavizada ({modelo})")

        exportar = input("💾 ¿Deseas exportar la malla como CSV? (s/n): ").lower()
        if exportar == 's':
            exportar_grid(X_grid, Y_grid, Z_pred, f"superficie_{reactivo}_{modelo}_suavizado.csv")

    elif opcion == "3":
        menu_estimador_ph()

    else:
        print("❌ Opción inválida. Intenta con 1, 2 o 3.")
        
if __name__ == "__main__":
    main()
