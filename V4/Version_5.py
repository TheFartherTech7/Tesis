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
    # Verificar si hay valores de pH que disminuyen (ácido) o aumentan (base)
    ph_inicial = df['pH'].iloc[0] if len(df) > 0 else 7.0
    ph_final = df['pH'].iloc[-1] if len(df) > 0 else 7.0
    
    if ph_final < ph_inicial:
        print("Titulante detectado: Ácido (HCl)")
        return True
    else:
        print("Titulante detectado: Base (NaOH)")
        return False

# FUNCIONES ORIGINALES PARA COMPUESTOS SIMPLES
def pH_acido(Xeq, Cr, alpha, k_acida, pH0, pH_min):
    """Ecuación para pH ácido (pH < pH0) - CORREGIDA"""
    return pH0 - (pH0 - pH_min) / (1 + np.exp(k_acida * (Xeq - alpha * Cr)))

def pH_alcalino(Xeq, Cr, alpha, k_alcalina, pH0, pH_max):
    """Ecuación para pH alcalino (pH > pH0)"""
    return pH0 + (pH_max - pH0) / (1 + np.exp(-k_alcalina * (Xeq - alpha * Cr)))

def modelo_simple_ajuste(Xeq, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max, Cr, pH_medidos):
    """Función de ajuste para compuestos simples"""
    resultados = []
    for i, x in enumerate(Xeq):
        if pH_medidos[i] < pH0:
            resultados.append(pH_acido(x, Cr, alpha, k_acida, pH0, pH_min))
        else:
            resultados.append(pH_alcalino(x, Cr, alpha, k_alcalina, pH0, pH_max))
    return np.array(resultados)

# NUEVAS FUNCIONES PARA COMPUESTOS COMPLEJOS CON TRANSICIÓN SUAVIZADA
def pH_acido_complejo_v1(Xeq, Cr, alpha, k_acida, pH0, pH_min):
    """Versión 1: Sigmoidal modificada con transición suave"""
    termino = (Xeq - alpha * Cr)
    transicion_suave = 1 / (1 + np.exp(-k_acida * termino / 10))
    return pH0 - (pH0 - pH_min) * transicion_suave

def pH_acido_complejo_v2(Xeq, Cr, alpha, k_acida, pH0, pH_min, beta=0.3):
    """Versión 2: Mezcla lineal-sigmoidal para transición suave"""
    termino = (Xeq - alpha * Cr)
    sigmoidal = 1 / (1 + np.exp(-k_acida * termino))
    lineal = beta * (termino / (np.abs(termino) + 1e-10)) * (1 - np.exp(-np.abs(termino)/10))
    return pH0 - (pH0 - pH_min) * (sigmoidal + lineal)

def pH_acido_complejo_v3(Xeq, Cr, alpha, k_acida, pH0, pH_min):
    """Versión 3: Sigmoidal con constante k reducida"""
    k_efectiva = k_acida * 0.1
    return pH0 - (pH0 - pH_min) / (1 + np.exp(k_efectiva * (Xeq - alpha * Cr)))

def pH_alcalino_complejo(Xeq, Cr, alpha, k_alcalina, pH0, pH_max):
    """Ecuación para pH alcalino (pH > pH0) - para compuestos complejos"""
    return pH0 + (pH_max - pH0) / (1 + np.exp(-k_alcalina * (Xeq - alpha * Cr)))

def modelo_complejo_ajuste(Xeq, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max, Cr, pH_medidos, version=1):
    """Función de ajuste para compuestos complejos con diferentes versiones"""
    resultados = []
    for i, x in enumerate(Xeq):
        if pH_medidos[i] < pH0:
            if version == 1:
                resultados.append(pH_acido_complejo_v1(x, Cr, alpha, k_acida, pH0, pH_min))
            elif version == 2:
                resultados.append(pH_acido_complejo_v2(x, Cr, alpha, k_acida, pH0, pH_min))
            else:
                resultados.append(pH_acido_complejo_v3(x, Cr, alpha, k_acida, pH0, pH_min))
        else:
            resultados.append(pH_alcalino_complejo(x, Cr, alpha, k_alcalina, pH0, pH_max))
    return np.array(resultados)

def calcular_alpha_inicial_acido(Xeq_measured, Cr, k, pH0, pH_min, pH_measured):
    """Calcula alpha inicial para región ácida - FÓRMULA CORREGIDA"""
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
    """Calcula alpha inicial para región alcalina"""
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
    """Optimiza parámetros para compuestos simples (modelo original)"""
    
    # Separar datos en regiones ácida y alcalina
    mascara_acida = pH_datos < pH0
    mascara_alcalina = pH_datos >= pH0
    
    Xeq_acidos = Xeq_datos[mascara_acida]
    pH_acidos = pH_datos[mascara_acida]
    
    Xeq_alcalinos = Xeq_datos[mascara_alcalina]
    pH_alcalinos = pH_datos[mascara_alcalina]
    
    # Valores iniciales
    k_acida_inicial = 1000
    k_alcalina_inicial = 1000
    
    # Calcular alpha inicial para cada región
    alphas_iniciales_acidos = []
    for i in range(len(Xeq_acidos)):
        try:
            alpha_i = calcular_alpha_inicial_acido(Xeq_acidos[i], Cr, k_acida_inicial, pH0, pH_min, pH_acidos[i])
            if not np.isnan(alpha_i) and np.isfinite(alpha_i) and 0 < alpha_i < 10:
                alphas_iniciales_acidos.append(alpha_i)
        except:
            continue
    
    alphas_iniciales_alcalinos = []
    for i in range(len(Xeq_alcalinos)):
        try:
            alpha_i = calcular_alpha_inicial_alcalino(Xeq_alcalinos[i], Cr, k_alcalina_inicial, pH0, pH_max, pH_alcalinos[i])
            if not np.isnan(alpha_i) and np.isfinite(alpha_i) and 0 < alpha_i < 10:
                alphas_iniciales_alcalinos.append(alpha_i)
        except:
            continue
    
    # Promediar los alpha de ambas regiones
    alpha_acida = np.mean(alphas_iniciales_acidos) if alphas_iniciales_acidos else 0.1
    alpha_alcalina = np.mean(alphas_iniciales_alcalinos) if alphas_iniciales_alcalinos else 0.1
    alpha_inicial = (alpha_acida + alpha_alcalina) / 2
    
    print(f"  Valores iniciales: alpha={alpha_inicial:.6f}, k_acida={k_acida_inicial}, k_alcalina={k_alcalina_inicial}")
    
    # Función wrapper para curve_fit
    def funcion_ajuste(Xeq, alpha, k_acida, k_alcalina):
        return modelo_simple_ajuste(Xeq, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max, Cr, pH_datos)
    
    # Optimización
    try:
        parametros_opt, covarianza = curve_fit(
            funcion_ajuste, 
            Xeq_datos, 
            pH_datos,
            p0=[alpha_inicial, k_acida_inicial, k_alcalina_inicial],
            bounds=([0.0001, 1, 1], [10, 1000000, 1000000]),
            maxfev=10000
        )
        
        alpha_opt, k_acida_opt, k_alcalina_opt = parametros_opt
        print(f"  Parámetros optimizados: alpha={alpha_opt:.6f}, k_acida={k_acida_opt:.2f}, k_alcalina={k_alcalina_opt:.2f}")
        
        return alpha_opt, k_acida_opt, k_alcalina_opt, covarianza
        
    except Exception as e:
        print(f"  Error en optimización simple: {e}")
        return None, None, None, None

def optimizar_parametros_complejo(Xeq_datos, pH_datos, Cr, pH0, pH_min, pH_max):
    """Optimiza parámetros para compuestos complejos con transición suavizada"""
    
    print("  Usando modelo para compuesto complejo con transición suavizada")
    
    # Probar diferentes versiones de la función ácida
    mejores_resultados = None
    mejor_r2 = -np.inf
    mejor_version = 1
    
    for version in [1, 2, 3]:
        print(f"  Probando versión {version} de la función ácida...")
        
        try:
            # Valores iniciales más conservadores para medios complejos
            alpha_inicial = 0.1
            k_acida_inicial = 100  # Valor más bajo para transición suave
            k_alcalina_inicial = 1000  # Mantenemos alto para región alcalina
            
            def funcion_ajuste(Xeq, alpha, k_acida, k_alcalina):
                return modelo_complejo_ajuste(Xeq, alpha, k_acida, k_alcalina, 
                                            pH0, pH_min, pH_max, Cr, pH_datos, version)
            
            # Optimización con límites más adecuados
            parametros_opt, covarianza = curve_fit(
                funcion_ajuste, 
                Xeq_datos, 
                pH_datos,
                p0=[alpha_inicial, k_acida_inicial, k_alcalina_inicial],
                bounds=([0.0001, 1, 10], [5, 1000, 50000]),
                maxfev=20000
            )
            
            alpha_opt, k_acida_opt, k_alcalina_opt = parametros_opt
            
            # Calcular R² para esta versión
            pH_pred = modelo_complejo_ajuste(Xeq_datos, alpha_opt, k_acida_opt, k_alcalina_opt,
                                           pH0, pH_min, pH_max, Cr, pH_datos, version)
            r2 = r2_score(pH_datos, pH_pred)
            
            print(f"  Versión {version}: R² = {r2:.4f}, alpha={alpha_opt:.6f}, "
                  f"k_acida={k_acida_opt:.2f}, k_alcalina={k_alcalina_opt:.2f}")
            
            if r2 > mejor_r2:
                mejor_r2 = r2
                mejores_resultados = (alpha_opt, k_acida_opt, k_alcalina_opt, covarianza)
                mejor_version = version
                
        except Exception as e:
            print(f"  Error en versión {version}: {e}")
            continue
    
    if mejores_resultados:
        alpha_opt, k_acida_opt, k_alcalina_opt, covarianza = mejores_resultados
        print(f"  Mejor versión: {mejor_version}, R² = {mejor_r2:.4f}")
        print(f"  Parámetros optimizados: alpha={alpha_opt:.6f}, "
              f"k_acida={k_acida_opt:.2f}, k_alcalina={k_alcalina_opt:.2f}")
        
        return alpha_opt, k_acida_opt, k_alcalina_opt, covarianza, mejor_version
        
    print("  No se pudo optimizar con ninguna versión")
    return None, None, None, None, None

def calcular_f_statistic(y_real, y_pred, n_params):
    """
    Calcula F-statistic y p-value para modelos no lineales
    
    F = (SSR / p) / (SSE / (n - p - 1))
    donde:
    SSR = Sum of Squares Regression
    SSE = Sum of Squares Error
    p = número de parámetros
    n = número de observaciones
    """
    n = len(y_real)
    
    # Sum of Squares Error
    sse = np.sum((y_real - y_pred)**2)
    
    # Sum of Squares Regression
    ssr = np.sum((y_pred - np.mean(y_real))**2)
    
    # Sum of Squares Total
    sst = np.sum((y_real - np.mean(y_real))**2)
    
    # Grados de libertad
    df_regression = n_params
    df_error = n - n_params - 1
    df_total = n - 1
    
    # Mean Squares
    msr = ssr / df_regression
    mse = sse / df_error
    
    # F-statistic
    f_statistic = msr / mse
    
    # p-value
    from scipy.stats import f
    p_value = 1 - f.cdf(f_statistic, df_regression, df_error)
    
    return {
        'F-statistic': f_statistic,
        'p-value': p_value,
        'SSR': ssr,
        'SSE': sse,
        'SST': sst,
        'MSR': msr,
        'MSE': mse,
        'df_regression': df_regression,
        'df_error': df_error
    }

def calcular_aic_bic(y_real, y_pred, n_params):
    """Calcula AIC y BIC para modelos no lineales"""
    n = len(y_real)
    sse = np.sum((y_real - y_pred)**2)
    
    # AIC = n * ln(SSE/n) + 2 * k
    aic = n * np.log(sse/n) + 2 * n_params
    
    # BIC = n * ln(SSE/n) + k * ln(n)
    bic = n * np.log(sse/n) + n_params * np.log(n)
    
    return {'AIC': aic, 'BIC': bic}

def calcular_error_estandar_parametros(cov_matrix):
    """Calcula error estándar de los parámetros a partir de la matriz de covarianza"""
    if cov_matrix is not None:
        param_errors = np.sqrt(np.diag(cov_matrix))
        return param_errors
    return None

def prueba_normalidad_residuos(residuos):
    """Prueba de normalidad de residuos usando Shapiro-Wilk"""
    from scipy.stats import shapiro
    stat, p_value = shapiro(residuos)
    return {'Shapiro-Wilk_stat': stat, 'Shapiro-Wilk_p': p_value}

def calcular_metricas_error(y_real, y_pred, n_params, cov_matrix=None):
    """Calcula métricas de error del ajuste incluyendo estadísticas avanzadas"""
    if len(y_real) != len(y_pred) or len(y_real) == 0:
        return {}
    
    # Métricas básicas
    r2 = r2_score(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    
    try:
        mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    except:
        mape = np.nan
    
    # Estadísticas avanzadas
    f_stats = calcular_f_statistic(y_real, y_pred, n_params)
    aic_bic = calcular_aic_bic(y_real, y_pred, n_params)
    
    metricas = {
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'N_puntos': len(y_real),
        'N_parametros': n_params,
        'F-statistic': f_stats['F-statistic'],
        'p-value': f_stats['p-value'],
        'AIC': aic_bic['AIC'],
        'BIC': aic_bic['BIC'],
        'SSR': f_stats['SSR'],
        'SSE': f_stats['SSE'],
        'SST': f_stats['SST']
    }
    
    # Error estándar de parámetros si hay matriz de covarianza
    if cov_matrix is not None:
        param_errors = calcular_error_estandar_parametros(cov_matrix)
        metricas['Param_errors'] = param_errors
    
    # Prueba de normalidad de residuos
    residuos = y_real - y_pred
    normalidad = prueba_normalidad_residuos(residuos)
    metricas.update(normalidad)
    
    return metricas

def graficar_resultados_combinados(resultados, reactivo_seleccionado, es_acido):
    """Crea gráficos combinados de pH vs equivalentes para todas las concentraciones"""
    
    if not resultados:
        return None
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Gráfico 1: pH vs Equivalentes/L
    ax = axs[0]
    for conc, result in resultados.items():
        Xeq_datos = result['Xeq_datos']
        pH_datos = result['pH_datos']
        pH_pred = result['pH_pred']
        
        ax.scatter(Xeq_datos, pH_datos, alpha=0.7, s=50, label=f'{conc}g/L (datos)')
        ax.plot(np.sort(Xeq_datos), pH_pred[np.argsort(Xeq_datos)], linewidth=2, label=f'{conc}g/L (ajuste)')
    
    ax.set_xlabel('Equivalentes/L (eq/L)')
    ax.set_ylabel('pH')
    ax.set_title(f'{reactivo_seleccionado} - pH vs Equivalentes de titulante')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Gráfico 2: [H+] vs Equivalentes/L
    ax = axs[1]
    for conc, result in resultados.items():
        Xeq_datos = result['Xeq_datos']
        pH_datos = result['pH_datos']
        pH_pred = result['pH_pred']
        
        H_plus_experimental = 10**(-pH_datos)
        H_plus_predicho = 10**(-pH_pred)
        
        ax.scatter(Xeq_datos, H_plus_experimental, alpha=0.7, s=50, label=f'{conc}g/L (datos)')
        ax.plot(np.sort(Xeq_datos), H_plus_predicho[np.argsort(Xeq_datos)], linewidth=2, label=f'{conc}g/L (ajuste)')
    
    ax.set_xlabel('Equivalentes/L (eq/L)')
    ax.set_ylabel('[H⁺] (mol/L)')
    ax.set_yscale('log')
    ax.set_title(f'{reactivo_seleccionado} - [H⁺] vs Equivalentes de titulante')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Gráfico 3: Parámetros alpha por concentración
    ax = axs[2]
    concentraciones = list(resultados.keys())
    alpha_values = [resultados[conc]['alpha'] for conc in concentraciones]
    
    ax.bar(range(len(concentraciones)), alpha_values, alpha=0.7)
    ax.set_xlabel('Concentración (g/L)')
    ax.set_ylabel('Alpha (eq/g)')
    ax.set_title(f'{reactivo_seleccionado} - Parámetro Alpha por concentración')
    ax.set_xticks(range(len(concentraciones)))
    ax.set_xticklabels(concentraciones)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Constantes k por concentración
    ax = axs[3]
    k_acida_values = [resultados[conc]['k_acida'] for conc in concentraciones]
    k_alcalina_values = [resultados[conc]['k_alcalina'] for conc in concentraciones]
    
    x_pos = np.arange(len(concentraciones))
    width = 0.35
    
    ax.bar(x_pos - width/2, k_acida_values, width, alpha=0.7, label='k_ácida')
    ax.bar(x_pos + width/2, k_alcalina_values, width, alpha=0.7, label='k_alcalina')
    
    ax.set_xlabel('Concentración (g/L)')
    ax.set_ylabel('Valor de k')
    ax.set_title(f'{reactivo_seleccionado} - Constantes k por concentración')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(concentraciones)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analizar_reactivo(df, reactivo_seleccionado):
    """Analiza todas las concentraciones de un reactivo específico"""
    
    print(f"\n=== ANALIZANDO REACTIVO: {reactivo_seleccionado} ===")
    
    # Filtrar datos del reactivo
    df_reactivo = df[df['Reactivo'] == reactivo_seleccionado]
    
    if df_reactivo.empty:
        print(f"No hay datos para el reactivo: {reactivo_seleccionado}")
        return
    
    # Obtener concentraciones únicas
    concentraciones = df_reactivo['Concentracion de reactivo'].unique()
    print(f"Concentraciones encontradas: {concentraciones}")
    
    # Determinar tipo de titulante
    es_acido = determinar_tipo_titulante(df_reactivo)
    
    resultados = {}
    
    for conc in concentraciones:
        print(f"\n--- Analizando concentración: {conc}g/L ---")
        
        # Filtrar datos para esta concentración
        df_conc = df_reactivo[df_reactivo['Concentracion de reactivo'] == conc]
        
        # Convertir concentración de titulante a equivalentes
        Xeq_datos = calcular_equivalentes(df_conc['Concentracion de titulante'].values, es_acido)
        pH_datos = df_conc['pH'].values
        
        # ENCONTRAR EL pH0 CUANDO X_eq = 0 (sin titulante)
        idx_cero = np.argmin(np.abs(df_conc['Concentracion de titulante'].values))
        pH0 = pH_datos[idx_cero]
        
        # Determinar pH_min, pH_max
        pH_min = np.min(pH_datos)
        pH_max = np.max(pH_datos)
        
        print(f"  pH0 (X_eq=0): {pH0:.2f}, pH_min: {pH_min:.2f}, pH_max: {pH_max:.2f}")
        print(f"  Rango de equivalentes: [{np.min(Xeq_datos):.4f}, {np.max(Xeq_datos):.4f}]")
        
        # Seleccionar modelo según tipo de reactivo
        if es_reactivo_complejo(reactivo_seleccionado):
            resultado_opt = optimizar_parametros_complejo(Xeq_datos, pH_datos, conc, pH0, pH_min, pH_max)
            if resultado_opt[0] is not None:
                alpha_opt, k_acida_opt, k_alcalina_opt, covarianza, version = resultado_opt
                # Predecir pH con la mejor versión
                pH_pred = modelo_complejo_ajuste(Xeq_datos, alpha_opt, k_acida_opt, k_alcalina_opt, 
                                               pH0, pH_min, pH_max, conc, pH_datos, version)
            else:
                print(f"  No se pudo optimizar para concentración {conc}g/L")
                continue
        else:
            resultado_opt = optimizar_parametros_simple(Xeq_datos, pH_datos, conc, pH0, pH_min, pH_max)
            if resultado_opt[0] is not None:
                alpha_opt, k_acida_opt, k_alcalina_opt, covarianza = resultado_opt
                pH_pred = modelo_simple_ajuste(Xeq_datos, alpha_opt, k_acida_opt, k_alcalina_opt, 
                                              pH0, pH_min, pH_max, conc, pH_datos)
            else:
                print(f"  No se pudo optimizar para concentración {conc}g/L")
                continue
        
        # Calcular métricas de error
        # Calcular métricas de error
        n_params = 3  # alpha, k_acida, k_alcalina
        metricas = calcular_metricas_error(pH_datos, pH_pred, n_params)
        
        # Guardar resultados
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
        
        # Mostrar métricas
        print(f"  Métricas de ajuste:")
        for metrica, valor in metricas.items():
            if metrica != 'N_puntos':
                print(f"    {metrica}: {valor:.4f}")
    
    # Graficar todos los resultados combinados
    if resultados:
        graficar_resultados_combinados(resultados, reactivo_seleccionado, es_acido)
    
    return resultados

def main():
    """Función principal del programa"""
    print("=== PROGRAMA DE ANÁLISIS DE TITULACIONES ===")
    print("Modelos: Simple (sales/azúcares) y Complejo (proteínas/polisacáridos)")
    
    # Cargar datos
    archivo_csv = 'datos_titulacion.csv'
    df = cargar_datos(archivo_csv)
    
    if df is None:
        print("No se pudieron cargar los datos. Verifica el archivo CSV.")
        return
    
    # Mostrar estadísticas básicas
    print("\n=== ESTADÍSTICAS BÁSICAS ===")
    reactivos_disponibles = list(df['Reactivo'].unique())
    print(f"Reactivos disponibles: {reactivos_disponibles}")
    print(f"Número de experimentos: {len(df)}")
    
    # Mostrar clasificación
    print("\n=== CLASIFICACIÓN DE REACTIVOS ===")
    for reactivo in reactivos_disponibles:
        tipo = "Complejo" if es_reactivo_complejo(reactivo) else "Simple"
        print(f"{reactivo}: {tipo}")
    
    # Seleccionar reactivo para analizar
    while True:
        print(f"\nReactivos disponibles:")
        for i, reactivo in enumerate(reactivos_disponibles, 1):
            tipo = "(Complejo)" if es_reactivo_complejo(reactivo) else "(Simple)"
            print(f"{i}. {reactivo} {tipo}")
        
        seleccion = input("\nIngresa el número del reactivo a analizar, escribe el nombre completo, o 'salir' para terminar: ").strip()
        
        if seleccion.lower() == 'salir':
            break
        
        # Determinar si la selección es un número o texto
        reactivo_seleccionado = None
        if seleccion.isdigit():
            idx = int(seleccion) - 1
            if 0 <= idx < len(reactivos_disponibles):
                reactivo_seleccionado = reactivos_disponibles[idx]
        else:
            if seleccion in reactivos_disponibles:
                reactivo_seleccionado = seleccion
        
        if reactivo_seleccionado is None:
            print(f"Error: '{seleccion}' no es una selección válida.")
            continue
        
        # Analizar el reactivo seleccionado
        resultados = analizar_reactivo(df, reactivo_seleccionado)
        
        if resultados:
            print(f"\n=== RESUMEN PARA {reactivo_seleccionado} ===")
            for conc, result in resultados.items():
                print(f"\nConcentración: {conc}g/L")
                print(f"Alpha: {result['alpha']:.6f} eq/g")
                print(f"K_ácida: {result['k_acida']:.2f}")
                print(f"K_alcalina: {result['k_alcalina']:.2f}")
                print(f"R²: {result['metricas']['R²']:.4f}")
    
    print("\n¡Análisis completado!")

if __name__ == "__main__":
    main()