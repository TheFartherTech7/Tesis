import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def corrected_henderson_model(X_eq, A, B, k, alpha, Cr):
    """Ecuación de Henderson corregida con asíntotas ajustables"""
    exponent = -k * (X_eq - alpha * Cr)
    return A + (B - A) / (1 + np.exp(exponent))


def molar_to_equivalent(Ct, valence=1):
    """Convierte concentración molar a equivalentes/L"""
    return Ct * valence

def fit_titration_data(df_subset, pH0_dict):
    """Ajusta la ecuación usando pH0 medido experimentalmente"""
    try:
        print(f"\n=== DEBUGGING INICIO ===")
        print(f"DataFrame subset shape: {df_subset.shape}")
        print(f"Columnas: {df_subset.columns.tolist()}")
        print(f"Primeras filas:\n{df_subset.head()}")
        
        # Extraer datos
        Ct = df_subset['Concentracion de titulante'].values
        X_eq = molar_to_equivalent(Ct)
        Y = df_subset['pH'].values
        Cr = df_subset['Concentracion de reactivo'].iloc[0]
        reactivo = df_subset['Reactivo'].iloc[0]
        
        print(f"Reactivo: {reactivo}, Concentración: {Cr}g/L")
        print(f"Ct original: {Ct}")
        print(f"X_eq: {X_eq}")
        print(f"Y: {Y}")
        
        # Obtener pH0 experimental
        pH0 = pH0_dict.get((reactivo, Cr))
        if pH0 is None:
            print(f"Advertencia: No se encontró pH0 para {reactivo} {Cr}g/L")
            return None
        
        # Filtrado de datos inválidos
        mask = (~np.isnan(Ct)) & (~np.isnan(Y)) & (Y >= 2.5) & (Y <= 9.5)
        print(f"Mask: {mask}")
        print(f"Valores NaN en Ct: {np.isnan(Ct)}")
        print(f"Valores NaN en Y: {np.isnan(Y)}")
        print(f"Y fuera de rango: {(Y < 2.5) | (Y > 9.5)}")
        
        X_eq = X_eq[mask]
        Y = Y[mask]
        
        print(f"X_eq filtrado: {X_eq}")
        print(f"Y filtrado: {Y}")
        print(f"Número de puntos después de filtrado: {len(X_eq)}")
        
        if len(X_eq) < 4:
            print(f"🚨 Muy pocos datos para {reactivo} {Cr}g/L: {len(X_eq)} puntos")
            print("=== DEBUGGING FIN ===")
            return None
        
        # ... resto del código de ajuste ...
        
    except Exception as e:
        print(f"Error crítico en ajuste: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_initial_pH_values(df, reactivo):
    """Solicita al usuario los valores de pH inicial para las concentraciones del reactivo seleccionado"""
    pH0_dict = {}
    
    print(f"\n=== INGRESO DE pH INICIAL PARA {reactivo} ===")
    print("Por favor ingrese los valores de pH inicial medidos experimentalmente:")
    
    # Obtenemos las concentraciones únicas para este reactivo
    concentrations = df[df['Reactivo'] == reactivo]['Concentracion de reactivo'].unique()
    
    for conc in sorted(concentrations):
        while True:
            try:
                pH0 = float(input(f"pH inicial para {reactivo} {conc}g/L: "))
                if 2.5 <= pH0 <= 9.5:
                    # Almacenamos como tupla (reactivo, concentración) como clave
                    pH0_dict[(reactivo, conc)] = pH0
                    break
                else:
                    print("El pH debe estar entre 2.5 y 9.5")
            except ValueError:
                print("Ingrese un valor numérico válido")
    
    return pH0_dict

def plot_titration_curves(df, reactivo, pH0_dict):
    """Grafica usando pH0 experimental"""
    subsets = []
    concentrations = sorted(df[df['Reactivo'] == reactivo]['Concentracion de reactivo'].unique())
    colors = ['blue', 'green', 'purple']
    
    if len(concentrations) > 3:
        concentrations = concentrations[:3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, conc in enumerate(concentrations):
        subset = df[(df['Reactivo'] == reactivo) & 
                   (df['Concentracion de reactivo'] == conc)]
        model = fit_titration_data(subset, pH0_dict)
        
        if model is None:
            continue
            
        # Marcar el pH inicial en el gráfico
        ax1.scatter(0, model['pH0'], color=colors[i], marker='x', s=100)
        
        # Escala pH normal
        ax1.scatter(model['X_eq'], model['Y'], color=colors[i], 
                   label=f'{conc}g/L (datos)', alpha=0.7)
        ax1.plot(np.sort(model['X_eq']), model['Y_pred'][np.argsort(model['X_eq'])], 
                color=colors[i], label=f'{conc}g/L (modelo)')
        
        # Escala [H+]
        H_exp = 10**(-model['Y'])
        H_pred = 10**(-model['Y_pred'])
        H0 = 10**(-model['pH0'])
        ax2.scatter(0, H0, color=colors[i], marker='x', s=100)
        ax2.scatter(model['X_eq'], H_exp, color=colors[i], alpha=0.7)
        ax2.plot(np.sort(model['X_eq']), H_pred[np.argsort(model['X_eq'])], 
                color=colors[i])
    
    # Configurar gráficos
    ax1.set_title(f'Titulación {reactivo} - Escala pH')
    ax1.set_xlabel('Equivalentes de titulante (eq/L)')
    ax1.set_ylabel('pH')
    ax1.set_ylim(2, 10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title(f'Titulación {reactivo} - Escala [H+]')
    ax2.set_xlabel('Equivalentes de titulante (eq/L)')
    ax2.set_ylabel('[H+] (mol/L)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()

def show_statistics(model):
    """Muestra resultados con pH0 experimental"""
    if not model:
        print("Modelo no disponible")
        return
    
    A, B, k, alpha = model['params']
    print("\n=== PARÁMETROS AJUSTADOS ===")
    print(f"pH inicial (medido): {model['pH0']:.2f}")
    print(f"pH inicial (modelo): {model['pH0_pred']:.2f}")
    print(f"Error en pH0: {model['pH0_error']:.4f}")
    print(f"A (pH mínimo): {A:.2f}")
    print(f"B (pH máximo): {B:.2f}")
    print(f"k: {k:.1f}")
    print(f"Alpha (eq/g): {alpha:.4f}")
    
    residuals = model['Y'] - model['Y_pred']
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((model['Y'] - np.mean(model['Y']))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    print(f"\nR²: {r2:.4f}")

def interactive_menu(df, reactivo, pH0_dict):
    """Menú interactivo actualizado"""
    while True:
        print("\nOpciones:")
        print("1. Predecir pH para concentración dada")
        print("2. Ver gráficas de titulación")
        print("3. Volver al menú principal")
        
        choice = input("Seleccione opción (1-3): ").strip()
        
        if choice == '1':
            try:
                Cr = float(input(f"Concentración de {reactivo} (g/L): "))
                Ct = float(input("Concentración de titulante (mol/L): "))
                valence = int(input("Valencia del titulante (1 para HCl/NaOH): ") or 1)
                
                # Obtener pH0 para esta concentración
                pH0 = pH0_dict.get((reactivo, Cr))
                if pH0 is None:
                    print(f"No hay pH0 registrado para {Cr}g/L")
                    continue
                
                X_eq = molar_to_equivalent(Ct, valence)
                
                # ❌ PROBLEMA: Esto solo toma 5 filas
                # temp_subset = df[(df['Reactivo'] == reactivo) & 
                #                (df['Concentracion de reactivo'] == Cr)].iloc[:5]
                
                # ✅ SOLUCIÓN: Tomar TODAS las filas para esa concentración
                temp_subset = df[(df['Reactivo'] == reactivo) & 
                               (df['Concentracion de reactivo'] == Cr)].copy()
                
                model = fit_titration_data(temp_subset, pH0_dict)
                
                if model:
                    A, B, k, alpha = model['params']
                    # Función de predicción
                    exponent = -k * (X_eq - alpha * Cr)
                    pH_pred = A + (B - A) / (1 + np.exp(exponent))
                    print(f"\npH estimado: {pH_pred:.2f}")
                    print(f"Concentración de [H+]: {10**-pH_pred:.2e} mol/L")
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif choice == '2':
            plot_titration_curves(df, reactivo, pH0_dict)
                
        elif choice == '3':
            break

def main():
    """Función principal actualizada"""
    file_path = 'datos_titulacion.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo {file_path}")
        return
    
    try:
        df = pd.read_csv(file_path)
        required_cols = ['Reactivo', 'Concentracion de reactivo', 
                        'Concentracion de titulante', 'pH']
        if not all(col in df.columns for col in required_cols):
            print("Error: El archivo no tiene las columnas requeridas")
            return
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return
    
    # Creamos un diccionario global para almacenar todos los pH0
    global_pH0_dict = {}
    
    while True:
        reactivos = df['Reactivo'].unique()
        print("\n" + "="*50)
        print("Reactivos disponibles:")
        for i, r in enumerate(reactivos, 1):
            print(f"{i}. {r}")
        print("0. Salir")
        
        seleccion = input("\nIngrese número de reactivo a analizar: ").strip()
        
        if seleccion == '0':
            break
            
        try:
            idx = int(seleccion) - 1
            reactivo = reactivos[idx]
        except:
            print("Entrada inválida")
            continue
        
        # Verificar si ya tenemos datos de pH para este reactivo
        existing_pH0 = {conc: global_pH0_dict.get((reactivo, conc)) 
                       for conc in df[df['Reactivo'] == reactivo]['Concentracion de reactivo'].unique()}
        
        if all(pH is not None for pH in existing_pH0.values()):
            print("\nValores de pH inicial ya ingresados:")
            for conc, pH in existing_pH0.items():
                print(f"{reactivo} {conc}g/L: pH = {pH}")
            usar_existentes = input("\n¿Usar estos valores? (s/n): ").strip().lower()
            
            if usar_existentes != 's':
                # Obtener nuevos valores de pH inicial para este reactivo
                new_pH0_dict = get_initial_pH_values(df, reactivo)
                # Actualizar el diccionario global
                global_pH0_dict.update(new_pH0_dict)
        else:
            # Obtener valores de pH inicial para este reactivo
            new_pH0_dict = get_initial_pH_values(df, reactivo)
            global_pH0_dict.update(new_pH0_dict)
        
        # Usar el primer conjunto de datos para mostrar parámetros
        subset = df[df['Reactivo'] == reactivo].copy()
        model = fit_titration_data(subset, global_pH0_dict)
        
        if model:
            show_statistics(model)
            interactive_menu(df, reactivo, global_pH0_dict)
        else:
            print("No se pudo ajustar el modelo para este reactivo")

if __name__ == "__main__":
    main()