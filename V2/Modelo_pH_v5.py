import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import f
from scipy.stats import chi2
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

def calcular_f_stat(y_true, y_pred, n_parametros):
    n = len(y_true)
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    explained = tss - rss
    df_model = n_parametros - 1
    df_error = n - n_parametros
    ms_model = explained / df_model
    ms_error = rss / df_error
    
    f_stat = ms_model / ms_error
    p_value = 1 - f.cdf(f_stat, df_model, df_error)
    
    return f_stat, p_value, df_model, df_error

  
def calcular_chi_cuadrada(y_real, y_pred):
    epsilon = 1e-8  # para evitar división entre cero
    return np.sum(((y_real - y_pred) ** 2) / (y_pred + epsilon))

def calcular_grados_de_libertad(n_observaciones, n_parametros):
    """Calcula los grados de libertad para el modelo."""
    return n_observaciones - n_parametros

def interpretar_chi_cuadrada(chi_sq, df, alpha=0.05):
    """Compara el valor Chi² con el valor crítico y reporta si el ajuste es estadísticamente adecuado."""
    p_valor = 1 - chi2.cdf(chi_sq, df)
    chi_crit = chi2.ppf(1 - alpha, df)
    print(f"\n📊 χ²: {chi_sq:.4f}")
    print(f"🎯 Grados de libertad: {df}")
    print(f"🔍 Valor crítico (α={alpha}): {chi_crit:.4f}")
    print(f"🧪 p-valor: {p_valor:.6f}")
    if chi_sq < chi_crit:
        print("✅ El modelo se considera adecuado estadísticamente (no se rechaza H0).")
    else:
        print("❌ El modelo no ajusta adecuadamente los datos (se rechaza H0).")


def obtener_indices_seleccionados(lista):
    seleccion = input("Selecciona por número(s) separados por coma (ej. 1,3,5): ")
    try:
        indices = [int(i)-1 for i in seleccion.split(",") if int(i)-1 < len(lista)]
        return [lista[i] for i in indices]
    except:
        return []

def mostrar_ecuacion_legible(coefs, terms):
    from collections import defaultdict
    grupos_x = defaultdict(list)
    grupos_z = defaultdict(list)

    for coef, term in zip(coefs[1:], terms[1:]):
        if abs(coef) < 1e-3:
            continue

        term_clean = term.replace(" ", "*").replace("^", "**")

        if "Z" in term_clean:
            # Agrupar por potencias de Z
            pot_z = 0
            if "Z**4" in term_clean: pot_z = 4
            elif "Z**3" in term_clean: pot_z = 3
            elif "Z**2" in term_clean: pot_z = 2
            elif "*Z" in term_clean or term_clean.endswith("Z"): pot_z = 1
            grupos_z[pot_z].append((coef, term_clean))
        else:
            # Agrupar por cantidad de Xs
            pot_x = term_clean.count("X")
            grupos_x[pot_x].append((coef, term_clean))

    constante = coefs[0]
    ecuacion = "pH = \n"
    bloques = []

    for pot in sorted(grupos_x.keys(), reverse=True):
        bloque = "    " + " + ".join(
            [f"{coef:.3f}*{t}" for coef, t in grupos_x[pot]]
        )
        bloques.append(bloque)

    for pot in sorted(grupos_z.keys(), reverse=True):
        bloque = "    " + " + ".join(
            [f"{coef:.3f}*{t}" for coef, t in grupos_z[pot]]
        )
        bloques.append(bloque)

    bloques.append(f"    {constante:.3f}")
    ecuacion += " +\n".join(bloques)
    ecuacion = ecuacion.replace("**", "^")
    return ecuacion

def simplificar_ecuacion_simbolica(equation_str):
    from sympy import symbols, sympify, simplify, collect
    import re

    # Definir los símbolos hasta X10 + Z
    Xs = symbols('X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12')
    Z = symbols('Z')
    all_vars = list(Xs) + [Z]

    # Quitar encabezado y limpiar formato
    cuerpo = re.sub(r"pH\s*=\s*", "", equation_str)
    cuerpo = cuerpo.replace("\n", "").replace("^", "**")

    try:
        expr = sympify(cuerpo, locals={str(v): v for v in all_vars})
        expr_simpl = simplify(expr)
        expr_grouped = collect(expr_simpl, all_vars)
        return str(expr_grouped).replace("**", "^")
    except Exception as e:
        return f"⚠️ Error al simplificar: {e}"

def evaluar_ecuacion_personalizada():
    from sympy import symbols, sympify
    import re

    print("\n🔎 Introduce tu ecuación en formato simbólico (usa ^ en lugar de **). Pega y presiona Enter dos veces cuando termines:")
    
    print("Ejemplo: 5 + 2*X1^2 - 3*Z")
    print("-" * 50)
    
    # Leer varias líneas hasta que se dé Enter 2 veces seguidas
    lineas = []
    while True:
        linea = input()
        if linea.strip() == "" and lineas and lineas[-1] == "":
            break
        lineas.append(linea)
    
    # Unir líneas y limpiar caracteres invisibles, espacios y errores comunes
    eq_str = " ".join(lineas)
    eq_str = eq_str.replace("^", "**")
    eq_str = eq_str.replace("–", "-").replace("−", "-")  # guiones raros
    eq_str = re.sub(r"[^\x00-\x7F]+", "", eq_str)  # eliminar caracteres no ASCII
    eq_str = eq_str.strip()

    
    if not eq_str:
        print("❌ No se ingresó una ecuación.")
        return

    # Definir variables posibles
    Xs = symbols('X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12')
    Z = symbols('Z')
    all_vars = list(Xs) + [Z]

    try:
        print(f"\n📋 Ecuación limpia interpretada:\n{eq_str}\n")
        expr = sympify(eq_str, locals={str(v): v for v in all_vars})
    except Exception as e:
        print(f"⚠️ Error al interpretar la ecuación: {e}")
        return

    # Detectar variables presentes
    usadas = sorted(expr.free_symbols, key=lambda x: str(x))
    variables = {}

    print("\nIntroduce los valores correspondientes a cada variable usada:")
    for var in usadas:
        while True:
            val = input(f"{var} = ")
            try:
                variables[var] = float(val) if val else 0.0
                break
            except ValueError:
                print("❌ Valor inválido. Introduce un número.")

    try:
        resultado = expr.evalf(subs=variables)
        print(f"\n✅ Resultado de la evaluación: pH = {resultado:.3f}")

        if resultado < 0 or resultado > 14:
            print("⚠️ Advertencia: El resultado está fuera del rango típico de pH (0-14). Revisa tu ecuación o los valores introducidos.")

    except Exception as e:
        print(f"⚠️ Error al evaluar la expresión: {e}")



def ajustar_modelo(df, features, target="pH", max_grado=4):
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_grado = 1
    mejor_poly = None
    mejor_chi2 = None

    X = df[features].values
    y = df[target].values

    for grado in range(1, max_grado + 1):
        poly = PolynomialFeatures(degree=grado, include_bias=False)
        X_poly = poly.fit_transform(X)
        modelo = LinearRegression().fit(X_poly, y)
        y_pred = modelo.predict(X_poly)
        r2 = r2_score(y, y_pred)
        chi2 = calcular_chi_cuadrada(y, y_pred)
        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_chi2 = chi2
            mejor_modelo = modelo
            mejor_grado = grado
            mejor_poly = poly

    return mejor_modelo, mejor_poly, mejor_grado, mejor_r2, mejor_chi2


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
    modelo, poly, grado, r2, chi2, coefs = ajustar_modelo(sub_df, features)
    terms = poly.get_feature_names_out(["X1", "Z"])[:len(modelo.coef_)+1]
    eq = mostrar_ecuacion_legible(np.append(modelo.intercept_, modelo.coef_), terms)

    print(f"Grado: {grado}, R²: {r2:.3f}, χ²: {chi2:.3f}")
        # Calcular grados de libertad e interpretar Chi²
    y_pred = modelo.predict(poly.transform(sub_df[features].values))
    n_obs = len(sub_df)
    n_parametros = len(modelo.coef_) + 1  # +1 por el intercepto
    df_libertad = calcular_grados_de_libertad(n_obs, n_parametros)
    interpretar_chi_cuadrada(chi2, df_libertad)
    f_val, p_val, df1, df2 = calcular_f_stat(sub_df["pH"].values, y_pred, n_parametros)
    print(f"\n📈 Estadístico F: {f_val:.3f}")
    print(f"🎯 p-valor F: {p_val:.6f}")
    if p_val < 0.05:
        print("✅ El modelo es estadísticamente significativo (se rechaza H0).")
    else:
        print("❌ El modelo no es estadísticamente significativo (no se rechaza H0).")

    print("Ecuación:")
    print("\n🧠 Ecuación simbólica simplificada:")
    print(simplificar_ecuacion_simbolica(eq))

    graficar_3d(sub_df, features, modelo, poly, f"Modelo {reactivo}")

def comparar_reactivos(df, reactivos):
    mostrar_menu(reactivos)
    seleccion = obtener_indices_seleccionados(reactivos)
    if not seleccion:
        print("Selección inválida.")
        return
    sub_df = df[(df[seleccion] > 0).any(axis=1)]
    features = seleccion + ["Titulante"]
    modelo, poly, grado, r2, chi2, coefs = ajustar_modelo(sub_df, features)
    # Nombres tipo X1, X2, ..., para los reactivos + Z al final
    var_names = [f"X{i+1}" for i in range(len(features) - 1)] + ["Z"]
    terms = poly.get_feature_names_out(var_names)[:len(modelo.coef_)+1]

    eq = mostrar_ecuacion_legible(np.append(modelo.intercept_, modelo.coef_), terms)


    print(f"Grado: {grado}, R²: {r2:.3f}, χ²: {chi2:.3f}")
    
        # Calcular grados de libertad e interpretar Chi²
    y_pred = modelo.predict(poly.transform(sub_df[features].values))
    n_obs = len(sub_df)
    n_parametros = len(modelo.coef_) + 1  # +1 por el intercepto
    df_libertad = calcular_grados_de_libertad(n_obs, n_parametros)
    interpretar_chi_cuadrada(chi2, df_libertad)
    f_val, p_val, df1, df2 = calcular_f_stat(sub_df["pH"].values, y_pred, n_parametros)
    print(f"\n📈 Estadístico F: {f_val:.3f}")
    print(f"🎯 p-valor F: {p_val:.6f}")
    if p_val < 0.05:
        print("✅ El modelo es estadísticamente significativo (se rechaza H0).")
    else:
        print("❌ El modelo no es estadísticamente significativo (no se rechaza H0).")

    print("Ecuación combinada:")
    print("\n🧠 Ecuación simbólica simplificada:")
    print(simplificar_ecuacion_simbolica(eq))


def analisis_global(df, reactivos):
    features = reactivos + ["Titulante"]
    modelo, poly, grado, r2 = ajustar_modelo(df, features)
    # Nombres tipo X1, X2, ..., para los reactivos + Z al final
    var_names = [f"X{i+1}" for i in range(len(features) - 1)] + ["Z"]
    terms = poly.get_feature_names_out(var_names)[:len(modelo.coef_)+1]

    eq = mostrar_ecuacion_legible(np.append(modelo.intercept_, modelo.coef_), terms)


    print(f"Modelo global: Grado: {grado}, R²: {r2:.3f}")
    print("Ecuación:")
    print("\n🧠 Ecuación simbólica simplificada:")
    print(simplificar_ecuacion_simbolica(eq))

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
    modelo, poly, grado, r2, chi2, coefs = ajustar_modelo(sub_df, features)
    # Nombres tipo X1, X2, ..., para los reactivos + Z al final
    var_names = [f"X{i+1}" for i in range(len(features) - 1)] + ["Z"]
    terms = poly.get_feature_names_out(var_names)[:len(modelo.coef_)+1]

    eq = mostrar_ecuacion_legible(np.append(modelo.intercept_, modelo.coef_), terms)


    print(f"Reactivos seleccionados: {mejores}")
    print(f"Grado: {grado}, R²: {r2:.3f}, χ²: {chi2:.3f}")
    print("Ecuación final:")
    print("\n🧠 Ecuación simbólica simplificada:")
    print(simplificar_ecuacion_simbolica(eq))

def evaluar_ecuacion_desde_csv():
    from sympy import symbols, sympify
    import re

    archivo = "modelo_global.csv"
    try:
        df = pd.read_csv(archivo)
        eq_raw = df.loc[0, "Ecuacion"]
    except Exception as e:
        print(f"❌ No se pudo leer el archivo: {e}")
        return

    # Limpiar y formatear
    eq_str = re.sub(r"pH\s*=\s*", "", eq_raw)
    eq_str = eq_str.replace("^", "**").replace("–", "-").replace("−", "-")
    eq_str = re.sub(r"[^\x00-\x7F]+", "", eq_str).strip()

    # Definir variables
    Xs = symbols('X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12')
    Z = symbols('Z')
    all_vars = list(Xs) + [Z]

    try:
        expr = sympify(eq_str, locals={str(v): v for v in all_vars})
    except Exception as e:
        print(f"⚠️ Error al interpretar la ecuación: {e}")
        return

    usadas = sorted(expr.free_symbols, key=lambda x: str(x))
    variables = {}

    print("\n📥 Introduce los valores para evaluar la ecuación:")
    for var in usadas:
        while True:
            val = input(f"{var} = ")
            try:
                variables[var] = float(val) if val else 0.0
                break
            except ValueError:
                print("❌ Valor inválido. Intenta de nuevo.")

    try:
        resultado = expr.evalf(subs=variables)
        print(f"\n✅ Resultado de la evaluación: pH = {resultado:.3f}")

        if resultado < 0 or resultado > 14:
            print("⚠️ Advertencia: El resultado está fuera del rango típico de pH (0-14). Revisa tu ecuación o los valores introducidos.")

    except Exception as e:
        print(f"⚠️ Error al evaluar la expresión: {e}")

def menu():
    df = cargar_datos()
    reactivos = [col for col in df.columns if col not in ["Titulante", "pH"]]

    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Analizar reactivo individual")
        print("2. Comparar múltiples reactivos")
        print("3. Análisis global")
        print("4. Búsqueda greedy de R² ≥ 0.80")
        print("5. Evaluar ecuación manualmente")
        print("6. Evaluar ecuación desde archivo CSV")
        print("7. Salir")

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
            evaluar_ecuacion_personalizada()
        elif opcion == "6":
            evaluar_ecuacion_desde_csv()
        elif opcion == "7":
            print("Saliendo del programa.")
            break


        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    menu()