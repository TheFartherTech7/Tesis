import pandas as pd

# Cargar archivo original
df = pd.read_csv("datos_titulacion.csv")
df.columns = df.columns.str.strip()  # Limpiar espacios en los nombres

# Lista de reactivos únicos
reactivos = df["Reactivo"].unique()

# Crear filas con todos los reactivos en 0, excepto el correspondiente
filas_transformadas = []

for _, fila in df.iterrows():
    nueva_fila = {reactivo: 0 for reactivo in reactivos}
    reactivo_actual = fila["Reactivo"]
    nueva_fila[reactivo_actual] = fila["Concentracion de reactivo"]
    nueva_fila["Titulante"] = fila["Concentracion de titulante"]
    nueva_fila["pH"] = fila["pH"]
    filas_transformadas.append(nueva_fila)

# Crear nuevo DataFrame
df_expandido = pd.DataFrame(filas_transformadas)

# Guardar
df_expandido.to_csv("datos_expandido.csv", index=False)
print("✅ Listo. Archivo expandido guardado como: datos_expandido.csv")
