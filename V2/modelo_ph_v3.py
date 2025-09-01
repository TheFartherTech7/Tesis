
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

























