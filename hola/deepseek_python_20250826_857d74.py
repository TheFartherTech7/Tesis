import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# CATALOGACIÓN DE REACTIVOS COMPLEJOS
REACTIVOS_COMPLEJOS = ['Ex. Levadura', 'Melaza', 'Suero leche']

def es_reactivo_complejo(reactivo):
    """Determina si un reactivo es complejo basado en la lista predefinida"""
    return any(compuesto.lower() in reactivo.lower() for compuesto in REACTIVOS_COMPLEJOS)

def cargar_datos(archivo_csv):
    """Carga y valida los datos del archivo CSV"""
    try:
        df = pd.read_csv(archivo_csv)
        print(f"Datos cargados correctamente. Filas: {len(df)}")
        print(f"Columnas disponibles: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None

def calcular_equivalentes(concentracion_titulante, es_acido=True):
    """
    Convierte concentración de titulante a equivalentes/L
    Negative para ácido, positivo para base
    """
    if es_acido:
        return -concentracion_titulante  # Equivalentes de H+
    else:
        return concentracion_titulante   # Equivalentes de OH-

def determinar_tipo_titulante(df):
    """Determina si el titulante es ácido o base basado en los datos"""
    ph_inicial = df['pH'].iloc[0] if len(df) > 0 else 7.0
    ph_final = df['pH'].iloc[-1] if len(df) > 0 else 7.0
    
    if ph_final < ph_inicial:
        return True
    else:
        return False

# FUNCIÓN UNIFICADA CON TRANSICIÓN SUAVE
def modelo_unificado(Xeq, Cr, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max):
    """
    Ecuación unificada con transición suave entre regiones ácida y alcalina
    """
    # Término de transición (centrado en el punto de equivalencia)
    transicion = Xeq - alpha * Cr
    
    # Funciones de peso suaves (sigmoideas)
    peso_acido = 1 / (1 + np.exp(10 * transicion))  # Peso para región ácida
    peso_alcalino = 1 / (1 + np.exp(-10 * transicion))  # Peso para región alcalina
    
    # Contribuciones de cada región
    parte_acida = peso_acido * (pH0 - (pH0 - pH_min) / (1 + np.exp(k_acida * transicion)))
    parte_alcalina = peso_alcalino * (pH0 + (pH_max - pH0) / (1 + np.exp(-k_alcalina * transicion)))
    
    return parte_acida + parte_alcalina

def modelo_unificado_ajuste(Xeq, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max, Cr):
    """Wrapper para curve_fit con la función unificada"""
    return modelo_unificado(Xeq, Cr, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max)

# FUNCIONES ORIGINALES (mantenidas para comparación)
def pH_acido(Xeq, Cr, alpha, k_acida, pH0, pH_min):
    return pH0 - (pH0 - pH_min) / (1 + np.exp(k_acida * (Xeq - alpha * Cr)))

def pH_alcalino(Xeq, Cr, alpha, k_alcalina, pH0, pH_max):
    return pH0 + (pH_max - pH0) / (1 + np.exp(-k_alcalina * (Xeq - alpha * Cr)))

def modelo_simple_ajuste(Xeq, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max, Cr, pH_medidos):
    resultados = []
    for i, x in enumerate(Xeq):
        if pH_medidos[i] < pH0:
            resultados.append(pH_acido(x, Cr, alpha, k_acida, pH0, pH_min))
        else:
            resultados.append(pH_alcalino(x, Cr, alpha, k_alcalina, pH0, pH_max))
    return np.array(resultados)

def calcular_alpha_inicial_acido(Xeq_measured, Cr, k, pH0, pH_min, pH_measured):
    try:
        termino = (pH0 - pH_min) / max(0.001, (pH0 - pH_measured)) - 1
        if termino <= 0:
            return 0.1
        termino_log = np.log(max(1e-10, termino))
        alpha = (Xeq_measured - (1/k) * termino_log) / Cr
        return max(0.001, min(10.0, alpha))
    except Exception as e:
        print(f"    Error calculando alpha ácido: {e}")
        return 0.1

def calcular_alpha_inicial_alcalino(Xeq_measured, Cr, k, pH0, pH_max, pH_measured):
    try:
        termino = (pH_max - pH0) / max(0.001, (pH_measured - pH0)) - 1
        if termino <= 0:
            return 0.1
        termino_log = np.log(max(1e-10, termino))
        alpha = (Xeq_measured + (1/k) * termino_log) / Cr
        return max(0.001, min(10.0, alpha))
    except Exception as e:
        print(f"    Error calculando alpha alcalino: {e}")
        return 0.1

def optimizar_parametros_simple(Xeq_datos, pH_datos, Cr, pH0, pH_min, pH_max):
    """Optimiza parámetros para compuestos simples usando modelo unificado"""
    
    print("  Usando modelo unificado con transición suave")
    
    # Valores iniciales
    k_acida_inicial = 1000
    k_alcalina_inicial = 1000
    
    # Calcular alpha inicial promedio
    alphas_iniciales = []
    for i in range(len(Xeq_datos)):
        try:
            if pH_datos[i] < pH0:
                alpha_i = calcular_alpha_inicial_acido(Xeq_datos[i], Cr, k_acida_inicial, pH0, pH_min, pH_datos[i])
            else:
                alpha_i = calcular_alpha_inicial_alcalino(Xeq_datos[i], Cr, k_alcalina_inicial, pH0, pH_max, pH_datos[i])
            
            if not np.isnan(alpha_i) and np.isfinite(alpha_i) and 0 < alpha_i < 10:
                alphas_iniciales.append(alpha_i)
        except:
            continue
    
    alpha_inicial = np.mean(alphas_iniciales) if alphas_iniciales else 0.1
    
    print(f"  Valores iniciales: alpha={alpha_inicial:.6f}, k_acida={k_acida_inicial}, k_alcalina={k_alcalina_inicial}")
    
    # Optimización con modelo unificado
    try:
        parametros_opt, covarianza = curve_fit(
            lambda x, a, k_a, k_al: modelo_unificado_ajuste(x, a, k_a, k_al, pH0, pH_min, pH_max, Cr),
            Xeq_datos, 
            pH_datos,
            p0=[alpha_inicial, k_acida_inicial, k_alcalina_inicial],
            bounds=([0.0001, 1, 1], [10, 1000000, 1000000]),
            maxfev=10000
        )
        
        alpha_opt, k_acida_opt, k_alcalina_opt = parametros_opt
        print(f"  Parámetros optimizados: alpha={alpha_opt:.6f}, k_acida={k_acida_opt:.2f}, k_alcalina={k_alcalina_opt:.2f}")
        
        # Predecir con modelo unificado
        pH_pred = modelo_unificado_ajuste(Xeq_datos, alpha_opt, k_acida_opt, k_alcalina_opt, pH0, pH_min, pH_max, Cr)
        
        return alpha_opt, k_acida_opt, k_alcalina_opt, covarianza, pH_pred
        
    except Exception as e:
        print(f"  Error en optimización unificada: {e}")
        return None, None, None, None, None

def optimizar_parametros_complejo(Xeq_datos, pH_datos, Cr, pH0, pH_min, pH_max):
    """Optimiza parámetros para compuestos complejos usando modelo unificado"""
    
    print("  Usando modelo unificado para compuesto complejo")
    
    # Valores iniciales más conservadores para medios complejos
    alpha_inicial = 0.1
    k_acida_inicial = 100
    k_alcalina_inicial = 1000
    
    try:
        parametros_opt, covarianza = curve_fit(
            lambda x, a, k_a, k_al: modelo_unificado_ajuste(x, a, k_a, k_al, pH0, pH_min, pH_max, Cr),
            Xeq_datos, 
            pH_datos,
            p0=[alpha_inicial, k_acida_inicial, k_alcalina_inicial],
            bounds=([0.0001, 1, 10], [5, 1000, 50000]),
            maxfev=20000
        )
        
        alpha_opt, k_acida_opt, k_alcalina_opt = parametros_opt
        
        # Predecir con modelo unificado
        pH_pred = modelo_unificado_ajuste(Xeq_datos, alpha_opt, k_acida_opt, k_alcalina_opt, pH0, pH_min, pH_max, Cr)
        r2 = r2_score(pH_datos, pH_pred)
        
        print(f"  R² = {r2:.4f}, alpha={alpha_opt:.6f}, k_acida={k_acida_opt:.2f}, k_alcalina={k_alcalina_opt:.2f}")
        
        return alpha_opt, k_acida_opt, k_alcalina_opt, covarianza, pH_pred
        
    except Exception as e:
        print(f"  Error en optimización unificada para complejos: {e}")
        return None, None, None, None, None

# ... (las funciones de métricas, estadísticas y gráficos se mantienen igual)

def calcular_metricas_error(y_real, y_pred, n_params, cov_matrix=None):
    """Calcula métricas de error del ajuste"""
    if len(y_real) != len(y_pred) or len(y_real) == 0:
        return {}
    
    r2 = r2_score(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    
    try:
        mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    except:
        mape = np.nan
    
    metricas = {
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'N_puntos': len(y_real),
        'N_parametros': n_params
    }
    
    if cov_matrix is not None:
        param_errors = np.sqrt(np.diag(cov_matrix))
        metricas['Param_errors'] = param_errors
    
    return metricas

def analizar_reactivo(df, reactivo_seleccionado):
    """Analiza todas las concentraciones de un reactivo específico"""
    
    print(f"\n=== ANALIZANDO REACTIVO: {reactivo_seleccionado} ===")
    
    df_reactivo = df[df['Reactivo'] == reactivo_seleccionado]
    
    if df_reactivo.empty:
        print(f"No hay datos para el reactivo: {reactivo_seleccionado}")
        return
    
    concentraciones = df_reactivo['Concentracion de reactivo'].unique()
    print(f"Concentraciones encontradas: {concentraciones}")
    
    es_acido = determinar_tipo_titulante(df_reactivo)
    resultados = {}
    
    for conc in concentraciones:
        print(f"\n--- Analizando concentración: {conc}g/L ---")
        
        df_conc = df_reactivo[df_reactivo['Concentracion de reactivo'] == conc]
        
        Xeq_datos = calcular_equivalentes(df_conc['Concentracion de titulante'].values, es_acido)
        pH_datos = df_conc['pH'].values
        
        idx_cero = np.argmin(np.abs(df_conc['Concentracion de titulante'].values))
        pH0 = pH_datos[idx_cero]
        
        pH_min = np.min(pH_datos)
        pH_max = np.max(pH_datos)
        
        print(f"  pH0 (X_eq=0): {pH0:.2f}, pH_min: {pH_min:.2f}, pH_max: {pH_max:.2f}")
        print(f"  Rango de equivalentes: [{np.min(Xeq_datos):.4f}, {np.max(Xeq_datos):.4f}]")
        
        # Siempre usar modelo unificado (más robusto y continuo)
        if es_reactivo_complejo(reactivo_seleccionado):
            resultado_opt = optimizar_parametros_complejo(Xeq_datos, pH_datos, conc, pH0, pH_min, pH_max)
        else:
            resultado_opt = optimizar_parametros_simple(Xeq_datos, pH_datos, conc, pH0, pH_min, pH_max)
        
        if resultado_opt[0] is not None:
            alpha_opt, k_acida_opt, k_alcalina_opt, covarianza, pH_pred = resultado_opt
            
            n_params = 3
            metricas = calcular_metricas_error(pH_datos, pH_pred, n_params, covarianza)
            
            resultados[conc] = {
                'alpha': alpha_opt,
                'k_acida': k_acida_opt,
                'k_alcalina': k_alcalina_opt,
                'pH0': pH0,
                'pH_min': pH_min,
                'pH_max': pH_max,
                'metricas': metricas,
                'Xeq_datos': Xeq_datos,
                'pH_datos': pH_datos,
                'pH_pred': pH_pred
            }
            
            print(f"  R²: {metricas['R²']:.4f}, RMSE: {metricas['RMSE']:.4f}")
        else:
            print(f"  No se pudo optimizar para concentración {conc}g/L")
    
    return resultados

# ... (el resto de las funciones se mantienen igual)

def main():
    """Función principal del programa"""
    print("=== PROGRAMA DE ANÁLISIS DE TITULACIONES ===")
    print("Modelo: Ecuación unificada con transición suave")
    
    archivo_csv = 'datos_titulacion.csv'
    df = cargar_datos(archivo_csv)
    
    if df is None or len(df) == 0:
        print("No se pudieron cargar los datos. Verifica el archivo CSV.")
        return
    
    reactivos_disponibles = list(df['Reactivo'].unique())
    print(f"\nReactivos disponibles: {reactivos_disponibles}")
    
    while True:
        print(f"\nReactivos disponibles:")
        for i, reactivo in enumerate(reactivos_disponibles, 1):
            print(f"{i}. {reactivo}")
        
        seleccion = input("\nIngresa el número del reactivo a analizar o 'salir' para terminar: ").strip()
        
        if seleccion.lower() == 'salir':
            break
        
        if seleccion.isdigit():
            idx = int(seleccion) - 1
            if 0 <= idx < len(reactivos_disponibles):
                reactivo_seleccionado = reactivos_disponibles[idx]
                
                resultados = analizar_reactivo(df, reactivo_seleccionado)
                
                if resultados:
                    estadisticas_globales = calcular_estadisticas_globales(resultados)
                    mostrar_estadisticas_globales(estadisticas_globales, reactivo_seleccionado)
                    
                    # Opciones para ver detalles adicionales
                    opcion = input("\n¿Deseas ver gráficos de resultados? (sí/no): ").strip().lower()
                    if opcion in ['si', 'sí', 's', 'yes', 'y']:
                        es_acido = determinar_tipo_titulante(df[df['Reactivo'] == reactivo_seleccionado])
                        graficar_resultados_combinados(resultados, reactivo_seleccionado, es_acido)
            else:
                print(f"Error: '{seleccion}' no es una selección válida.")
        else:
            print("Por favor, ingresa un número válido.")
    
    print("\n¡Análisis completado!")

if __name__ == "__main__":
    main()