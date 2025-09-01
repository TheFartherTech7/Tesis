import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# Leer datos
df = pd.read_csv("datos_titulacion.csv")
df.columns = df.columns.str.strip()

reactivos = sorted(df["Reactivo"].unique())
menu_reactivos = {i + 1: reactivo for i, reactivo in enumerate(reactivos)}

# ======================== FUNCIONES PRINCIPALES ========================

def mostrar_ecuacion_legible(coefs, terms):
    from collections import defaultdict
    grupos_x1 = defaultdict(list)
    grupos_x2_solo = defaultdict(list)

    for coef, term in zip(coefs[1:], terms[1:]):
        if abs(coef) < 1e-3:
            continue
        term_clean = term.replace(" ", "*").replace("^", "**")
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

    for pot in sorted(grupos_x1.keys(), reverse=True):
        bloque = f"    X1^{pot}*(" + " + ".join(
            [f"{coef:.3f}{'*' + t.split('*',1)[1] if '*' in t else ''}" for coef, t in grupos_x1[pot]]
        ) + ")"
        bloques.append(bloque)

    for pot in sorted(grupos_x2_solo.keys(), reverse=True):
        bloque = f"    X2^{pot}*(" + " + ".join(
            [f"{coef:.3f}{'*' + t.split('*',1)[1] if '*' in t else ''}" for coef, t in grupos_x2_solo[pot]]
        ) + ")"
        bloques.append(bloque)

    bloques.append(f"    {constante:.3f}")
    ecuacion += " +\n".join(bloques)
    return ecuacion

def mostrar_ecuacion(modelo, poly):
    coefs = modelo.coef_
    terms = poly.get_feature_names_out(["X1", "X2"])
    ecuacion = mostrar_ecuacion_legible(coefs, terms)
    return ecuacion.replace("**", "^")

def ajustar_modelo(X, y):
    mejor_r2, mejor_modelo, mejor_grado, mejor_poly, mejor_ecuacion = -np.inf, None, 0, None, ""
    for grado in range(1, 5):
        poly = PolynomialFeatures(degree=grado)
        X_poly = poly.fit_transform(X)
        modelo = LinearRegression().fit(X_poly, y)
        r2 = r2_score(y, modelo.predict(X_poly))
        if r2 > mejor_r2:
            mejor_r2, mejor_modelo, mejor_grado, mejor_poly = r2, modelo, grado, poly
            mejor_ecuacion = mostrar_ecuacion(modelo, poly)
    return mejor_modelo, mejor_grado, mejor_r2, mejor_ecuacion, mejor_poly

def guardar_resultados_csv(titulo, grado, r2, eq):
    nombre_archivo = f"resultados_{titulo.replace(' ', '_').lower()}.csv"
    resultados_df = pd.DataFrame([{"Titulo": titulo, "Grado": grado, "R2": r2, "Ecuacion": eq}])
    resultados_df.to_csv(nombre_archivo, index=False)
    print(f"Resultados guardados como: {nombre_archivo}")

def graficar_superficie(df_subset, modelo, poly, grado, titulo="Modelo", cmap='viridis'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = df_subset[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = df_subset["pH"].values
    ax.scatter(X[:, 0], X[:, 1], y, c='r', label="Datos reales")

    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    y_pred = modelo.predict(poly.transform(X_grid)).reshape(x1_grid.shape)

    ax.plot_surface(x1_grid, x2_grid, y_pred, cmap=cmap, alpha=0.6)
    ax.set_title(f"{titulo} | Grado {grado}")
    ax.set_xlabel("Conc. Reactivo")
    ax.set_ylabel("Conc. Titulante")
    ax.set_zlabel("pH")
    ax.set_zlim(0, 14)
    ax.legend()
    plt.tight_layout()
    plt.show()

    decision = input("\n¿Deseas guardar esta gráfica y/o el CSV? (Sí/No): ").strip().lower()
    if decision in ["sí", "si"]:
        opcion = input("¿Qué deseas guardar? (1: Gráfica, 2: CSV, 3: Ambos): ").strip()
        if opcion in ["1", "3"]:
            nombre = f"{titulo.replace(' ', '_')}.png"
            fig.savefig(nombre)
            print(f"Imagen guardada como: {nombre}")
        return opcion in ["2", "3"]
    return False

def entrenar_y_calcular_r2_para_conjunto(conjunto_reactivos):
    if not conjunto_reactivos:
        return -np.inf, None, None, None, None
    sub_df = df[df["Reactivo"].isin(conjunto_reactivos)]
    X = sub_df[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = sub_df["pH"].values
    modelo, grado, r2, eq, poly = ajustar_modelo(X, y)
    return r2, modelo, grado, eq, poly

def busqueda_greedy_max_r2_reactivos():
    UMBRAL_R2 = 0.80
    print(f"\nIniciando búsqueda greedy con umbral R² ≥ {UMBRAL_R2}...\n")
    mejor_r2_global = -np.inf
    mejor_conjunto_global = set()
    mejor_modelo_global = None
    mejor_grado_global = 0
    mejor_ecuacion_global = ""
    mejor_poly_global = None

    candidatos_iniciales = []
    for reactivo in reactivos:
        sub_df = df[df["Reactivo"] == reactivo]
        X = sub_df[["Concentracion de reactivo", "Concentracion de titulante"]].values
        y = sub_df["pH"].values
        modelo, grado, r2, eq, poly = ajustar_modelo(X, y)
        if r2 >= UMBRAL_R2:
            candidatos_iniciales.append((r2, reactivo, modelo, grado, eq, poly))

    if not candidatos_iniciales:
        print("Ningún reactivo individual supera el umbral de R².")
        return

    candidatos_iniciales.sort(reverse=True)
    _, mejor_reactivo, modelo, grado, eq, poly = candidatos_iniciales[0]
    conjunto_actual = {mejor_reactivo}
    reactivos_restantes = set(reactivos) - conjunto_actual
    mejor_r2_global = _
    mejor_conjunto_global = conjunto_actual.copy()
    mejor_modelo_global = modelo
    mejor_grado_global = grado
    mejor_ecuacion_global = eq
    mejor_poly_global = poly
    print(f"Reactivo inicial: {mejor_reactivo} | R²: {mejor_r2_global:.4f}")

    while reactivos_restantes:
        mejora = False
        for r in reactivos_restantes:
            nuevo_conjunto = conjunto_actual.union({r})
            r2, modelo, grado, eq, poly = entrenar_y_calcular_r2_para_conjunto(nuevo_conjunto)
            if r2 >= UMBRAL_R2 and len(nuevo_conjunto) > len(mejor_conjunto_global):
                mejora = True
                mejor_candidato = r
                mejor_r2_global = r2
                conjunto_actual.add(r)
                reactivos_restantes.remove(r)
                mejor_conjunto_global = conjunto_actual.copy()
                mejor_modelo_global = modelo
                mejor_grado_global = grado
                mejor_ecuacion_global = eq
                mejor_poly_global = poly
                print(f"Agregado: {r} | R²: {r2:.4f} | Total reactivos: {len(mejor_conjunto_global)}")
                break
        if not mejora:
            break

    print("\n=== Resultado final búsqueda greedy ===")
    print(f"Reactivos seleccionados ({len(mejor_conjunto_global)}): {sorted(mejor_conjunto_global)}")
    print(f"R² global final: {mejor_r2_global:.4f}")
    print(f"Ecuación final:\n{mejor_ecuacion_global}")
    if mejor_modelo_global:
        guardar_csv = graficar_superficie(df[df["Reactivo"].isin(mejor_conjunto_global)],
                                          mejor_modelo_global, mejor_poly_global,
                                          mejor_grado_global, titulo="Modelo Greedy con Umbral R2")
        if guardar_csv:
            guardar_resultados_csv("Modelo Greedy con Umbral R2", mejor_grado_global, mejor_r2_global, mejor_ecuacion_global)

def analizar_reactivo():
    print("\nSelecciona un reactivo:")
    for num, nombre in menu_reactivos.items():
        print(f"{num}. {nombre}")
    entrada = input("\nEscribe el número o nombre del reactivo: ").strip().lower()
    reactivo = menu_reactivos.get(int(entrada)) if entrada.isdigit() else next((r for r in reactivos if r.lower() == entrada), None)
    if not reactivo:
        print("\nOpción inválida. Intenta de nuevo.")
        return
    sub_df = df[df["Reactivo"] == reactivo]
    X = sub_df[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = sub_df["pH"].values
    modelo, grado, r2, eq, poly = ajustar_modelo(X, y)
    print("\n--- Resultado ---")
    print(f"Reactivo: {reactivo}\nMejor modelo: Grado {grado}\nR^2: {r2:.4f}\nEcuación:\n{eq}")
    guardar_csv = graficar_superficie(sub_df, modelo, poly, grado, titulo=reactivo)
    if guardar_csv:
        guardar_resultados_csv(reactivo, grado, r2, eq)

def analisis_global(df_entrada, titulo="Modelo Global"):
    X = df_entrada[["Concentracion de reactivo", "Concentracion de titulante"]].values
    y = df_entrada["pH"].values
    modelo, grado, r2, eq, poly = ajustar_modelo(X, y)
    print(f"\n--- {titulo} ---\nGrado: {grado}\nR^2: {r2:.4f}\nEcuación:\n{eq}")
    guardar_csv = graficar_superficie(df_entrada, modelo, poly, grado, titulo=titulo)
    if guardar_csv:
        guardar_resultados_csv(titulo, grado, r2, eq)

# ======================== MENÚ PRINCIPAL ========================

while True:
    print("\n=== Análisis de Titulaciones ===")
    print("1. Analizar un reactivo individual")
    print("2. Comparar múltiples reactivos con un solo modelo")
    print("3. Realizar análisis global")
    print("4. Búsqueda automática greedy para maximizar R² y reactivos")
    print("5. Salir")

    opcion = input("Selecciona una opción: ").strip()

    if opcion == "1":
        analizar_reactivo()
    elif opcion == "2":
        print("\nSelecciona los reactivos a comparar (por números o nombres separados por coma):")
        for num, nombre in menu_reactivos.items():
            print(f"{num}. {nombre}")
        entrada = input("\nEscribe tus opciones: ").strip().lower()
        entradas = [e.strip() for e in entrada.split(",")]
        seleccionados = set()
        for entrada in entradas:
            if entrada.isdigit():
                reactivo = menu_reactivos.get(int(entrada))
                if reactivo:
                    seleccionados.add(reactivo)
            else:
                reactivo = next((r for r in reactivos if r.lower() == entrada), None)
                if reactivo:
                    seleccionados.add(reactivo)
                else:
                    print(f"Opción inválida: {entrada}")
        if not seleccionados:
            print("No se seleccionaron reactivos válidos.")
            continue
        sub_df = df[df["Reactivo"].isin(seleccionados)]
        analisis_global(sub_df, titulo="Modelo Múltiples Reactivos")
    elif opcion == "3":
        analisis_global(df)
    elif opcion == "4":
        busqueda_greedy_max_r2_reactivos()
    elif opcion == "5":
        print("Saliendo...")
        break
    else:
        print("\nOpción no válida. Intenta de nuevo.")
