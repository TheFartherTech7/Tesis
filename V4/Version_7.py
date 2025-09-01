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

# MODELO UNIFICADO CON TRANSICIÓN SUAVE (REEMPLAZA TODAS LAS FUNCIONES ANTERIORES)
def modelo_unificado(Xeq, Cr, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max):
    """
    Ecuación unificada con transición suave entre regiones ácida y alcalina
    """
    # Punto de equivalencia
    Xeq_equiv = alpha * Cr
    
    # Función de transición suave (c = 10 fijo)
    s = 1 / (1 + np.exp(-10 * Xeq))
    
    # Región ácida (Xeq < 0)
    pH_acida = pH0 - (pH0 - pH_min) / (1 + np.exp(k_acida * (Xeq + Xeq_equiv)))
    
    # Región alcalina (Xeq > 0)  
    pH_alcalina = pH0 + (pH_max - pH0) / (1 + np.exp(-k_alcalina * (Xeq - Xeq_equiv)))
    
    # Mezcla suave
    return (1 - s) * pH_acida + s * pH_alcalina

def modelo_unificado_ajuste(Xeq, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max, Cr):
    """Wrapper para curve_fit"""
    return modelo_unificado(Xeq, Cr, alpha, k_acida, k_alcalina, pH0, pH_min, pH_max)

def calcular_alpha_inicial(Xeq_measured, Cr, k, pH0, pH_limite, pH_measured, es_acido=True):
    """Calcula alpha inicial para cualquier región"""
    try:
        if es_acido:
            termino = (pH0 - pH_limite) / max(0.001, (pH0 - pH_measured)) - 1
        else:
            termino = (pH_limite - pH0) / max(0.001, (pH_measured - pH0)) - 1
        
        if termino <= 0:
            return 0.1
        
        termino_log = np.log(max(1e-10, termino))
        
        if es_acido:
            alpha = (Xeq_measured - (1/k) * termino_log) / Cr
        else:
            alpha = (Xeq_measured + (1/k) * termino_log) / Cr
            
        return max(0.001, min(10.0, alpha))
    except Exception as e:
        print(f"    Error calculando alpha: {e}")
        return 0.1

def optimizar_parametros_unificado(Xeq_datos, pH_datos, Cr, pH0, pH_min, pH_max, es_complejo=False):
    """Optimiza parámetros usando el modelo unificado"""
    
    print("  Usando modelo unificado con transición suave")
    
    # Valores iniciales según tipo de reactivo
    if es_complejo:
        k_acida_inicial = 100
        k_alcalina_inicial = 1000
        bounds = ([0.0001, 1, 10], [5, 1000, 50000])
    else:
        k_acida_inicial = 1000
        k_alcalina_inicial = 1000
        bounds = ([0.0001, 1, 1], [10, 1000000, 1000000])
    
    # Calcular alpha inicial promedio
    alphas_iniciales = []
    for i in range(len(Xeq_datos)):
        try:
            if pH_datos[i] < pH0:
                alpha_i = calcular_alpha_inicial(Xeq_datos[i], Cr, k_acida_inicial, pH0, pH_min, pH_datos[i], es_acido=True)
            else:
                alpha_i = calcular_alpha_inicial(Xeq_datos[i], Cr, k_alcalina_inicial, pH0, pH_max, pH_datos[i], es_acido=False)
            
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
            bounds=bounds,
            maxfev=20000 if es_complejo else 10000
        )
        
        alpha_opt, k_acida_opt, k_alcalina_opt = parametros_opt
        
        # Predecir con modelo unificado
        pH_pred = modelo_unificado_ajuste(Xeq_datos, alpha_opt, k_acida_opt, k_alcalina_opt, pH0, pH_min, pH_max, Cr)
        
        print(f"  Parámetros optimizados: alpha={alpha_opt:.6f}, k_acida={k_acida_opt:.2f}, k_alcalina={k_alcalina_opt:.2f}")
        
        return alpha_opt, k_acida_opt, k_alcalina_opt, covarianza, pH_pred
        
    except Exception as e:
        print(f"  Error en optimización unificada: {e}")
        return None, None, None, None, None

# ... (LAS FUNCIONES DE MÉTRICAS, ESTADÍSTICAS Y GRÁFICOS SE MANTIENEN IGUAL)
# calcular_f_statistic, calcular_aic_bic, calcular_error_estandar_parametros,
# prueba_normalidad_residuos, calcular_metricas_error, calcular_estadisticas_globales,
# mostrar_estadisticas_individuales, mostrar_estadisticas_globales, graficar_resultados_combinados
# SE MANTIENEN EXACTAMENTE IGUAL

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
    """Calcula AIC и BIC para modelos no lineales"""
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

def calcular_estadisticas_globales(resultados):
    """Calcula estadísticas globales para todas las concentraciones del reactivo"""
    if not resultados:
        return None
    
    # Estadísticas básicas de parámetros
    estadisticas_globales = {
        'alpha_promedio': np.mean([r['alpha'] for r in resultados.values()]),
        'alpha_desviacion': np.std([r['alpha'] for r in resultados.values()]),
        'alpha_cv': (np.std([r['alpha'] for r in resultados.values()]) / 
                     np.mean([r['alpha'] for r in resultados.values()])) * 100,
        
        'k_acida_promedio': np.mean([r['k_acida'] for r in resultados.values()]),
        'k_acida_desviacion': np.std([r['k_acida'] for r in resultados.values()]),
        
        'k_alcalina_promedio': np.mean([r['k_alcalina'] for r in resultados.values()]),
        'k_alcalina_desviacion': np.std([r['k_alcalina'] for r in resultados.values()]),
        
        'r2_promedio': np.mean([r['metricas']['R²'] for r in resultados.values()]),
        'r2_min': np.min([r['metricas']['R²'] for r in resultados.values()]),
        'r2_max': np.max([r['metricas']['R²'] for r in resultados.values()]),
        
        'rmse_promedio': np.mean([r['metricas']['RMSE'] for r in resultados.values()]),
        'mae_promedio': np.mean([r['metricas']['MAE'] for r in resultados.values()]),
        'mape_promedio': np.mean([r['metricas']['MAPE'] for r in resultados.values()]),
        
        'fstat_promedio': np.mean([r['metricas']['F-statistic'] for r in resultados.values()]),
        'pvalue_promedio': np.mean([r['metricas']['p-value'] for r in resultados.values()]),
        'aic_promedio': np.mean([r['metricas']['AIC'] for r in resultados.values()]),
        'bic_promedio': np.mean([r['metricas']['BIC'] for r in resultados.values()]),
        
        'shapiro_stat_promedio': np.mean([r['metricas']['Shapiro-Wilk_stat'] for r in resultados.values()]),
        'shapiro_p_promedio': np.mean([r['metricas']['Shapiro-Wilk_p'] for r in resultados.values()]),
        
        'n_concentraciones': len(resultados),
        'total_puntos': sum([r['metricas']['N_puntos'] for r in resultados.values()])
    }
    
    # Calcular estadísticas globales de bondad de ajuste (sumando todas las concentraciones)
    ssr_total = sum([r['metricas']['SSR'] for r in resultados.values()])
    sse_total = sum([r['metricas']['SSE'] for r in resultados.values()])
    sst_total = sum([r['metricas']['SST'] for r in resultados.values()])
    
    total_puntos = estadisticas_globales['total_puntos']
    total_parametros = 3 * len(resultados)  # 3 parámetros por concentración
    
    # Grados de libertad globales
    df_regression_total = total_parametros
    df_error_total = total_puntos - total_parametros - 1
    
    # Mean Squares globales
    msr_total = ssr_total / df_regression_total if df_regression_total > 0 else 0
    mse_total = sse_total / df_error_total if df_error_total > 0 else 0
    
    # F-statistic global
    f_statistic_total = msr_total / mse_total if mse_total > 0 else 0
    
    # p-value global
    from scipy.stats import f
    p_value_total = 1 - f.cdf(f_statistic_total, df_regression_total, df_error_total) if mse_total > 0 else 1.0
    
    # AIC y BIC globales (sumando todos los puntos)
    aic_total = sum([r['metricas']['AIC'] for r in resultados.values()])
    bic_total = sum([r['metricas']['BIC'] for r in resultados.values()])
    
    # Calcular MAPE global ponderado
    mape_total = np.average([r['metricas']['MAPE'] for r in resultados.values()], 
                           weights=[r['metricas']['N_puntos'] for r in resultados.values()])
    
    # Agregar estadísticas de bondad de ajuste global
    estadisticas_globales.update({
        'F-statistic_global': f_statistic_total,
        'p-value_global': p_value_total,
        'SSR_global': ssr_total,
        'SSE_global': sse_total,
        'SST_global': sst_total,
        'MSR_global': msr_total,
        'MSE_global': mse_total,
        'df_regression_global': df_regression_total,
        'df_error_global': df_error_total,
        'AIC_global': aic_total,
        'BIC_global': bic_total,
        'MAPE_global': mape_total,
        'R²_global': 1 - (sse_total / sst_total) if sst_total > 0 else 0
    })
    
    return estadisticas_globales

def mostrar_estadisticas_individuales(resultados, reactivo_seleccionado):
    """Muestra estadísticas individuales por concentración"""
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS INDIVIDUALES - {reactivo_seleccionado}")
    print(f"{'='*60}")
    
    for conc, result in resultados.items():
        print(f"\nConcentración: {conc}g/L")
        print(f"Alpha: {result['alpha']:.6f} ± {result['metricas'].get('Param_errors', [0,0,0])[0]:.6f} eq/g")
        print(f"K_ácida: {result['k_acida']:.2f} ± {result['metricas'].get('Param_errors', [0,0,0])[1]:.2f}")
        print(f"K_alcalina: {result['k_alcalina']:.2f} ± {result['metricas'].get('Param_errors', [0,0,0])[2]:.2f}")
        
        print(f"\nMétricas de ajuste:")
        print(f"  R²: {result['metricas']['R²']:.4f}")
        print(f"  MAE: {result['metricas']['MAE']:.4f}")
        print(f"  RMSE: {result['metricas']['RMSE']:.4f}")
        print(f"  MAPE: {result['metricas']['MAPE']:.4f}%")
        print(f"  F-statistic: {result['metricas']['F-statistic']:.4f}")
        print(f"  p-value: {result['metricas']['p-value']:.6f}")
        print(f"  AIC: {result['metricas']['AIC']:.4f}")
        print(f"  BIC: {result['metricas']['BIC']:.4f}")
        
        print(f"\nSumas de cuadrados:")
        print(f"  SSR: {result['metricas']['SSR']:.4f}")
        print(f"  SSE: {result['metricas']['SSE']:.4f}")
        print(f"  SST: {result['metricas']['SST']:.4f}")
        
        
        
        print(f"\nPrueba de normalidad (Shapiro-Wilk):")
        print(f"  Estadístico: {result['metricas']['Shapiro-Wilk_stat']:.4f}")
        print(f"  p-value: {result['metricas']['Shapiro-Wilk_p']:.4f}")
        
        print(f"\nNúmero de puntos: {result['metricas']['N_puntos']}")
        print(f"Número de parámetros: {result['metricas']['N_parametros']}")

def mostrar_estadisticas_globales(estadisticas_globales, reactivo_seleccionado):
    """Muestra estadísticas globales del reactivo incluyendo bondad de ajuste"""
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS GLOBALES - {reactivo_seleccionado}")
    print(f"{'='*60}")
    
    print(f"\nParámetro Alpha:")
    print(f"  Promedio: {estadisticas_globales['alpha_promedio']:.6f} eq/g")
    print(f"  Desviación estándar: {estadisticas_globales['alpha_desviacion']:.6f}")
    print(f"  Coeficiente de variación: {estadisticas_globales['alpha_cv']:.2f}%")
    
    print(f"\nConstante K_ácida:")
    print(f"  Promedio: {estadisticas_globales['k_acida_promedio']:.2f}")
    print(f"  Desviación estándar: {estadisticas_globales['k_acida_desviacion']:.2f}")
    
    print(f"\nConstante K_alcalina:")
    print(f"  Promedio: {estadisticas_globales['k_alcalina_promedio']:.2f}")
    print(f"  Desviación estándar: {estadisticas_globales['k_alcalina_desviacion']:.2f}")
    
    print(f"\nMÉTRICAS DE AJUSTE PROMEDIO:")
    print(f"  R² promedio: {estadisticas_globales['r2_promedio']:.4f}")
    print(f"  R² mínimo: {estadisticas_globales['r2_min']:.4f}")
    print(f"  R² máximo: {estadisticas_globales['r2_max']:.4f}")
    print(f"  RMSE promedio: {estadisticas_globales['rmse_promedio']:.4f}")
    print(f"  MAE promedio: {estadisticas_globales['mae_promedio']:.4f}")
    print(f"  MAPE promedio: {estadisticas_globales['mape_promedio']:.4f}%")
    print(f"  F-statistic promedio: {estadisticas_globales['fstat_promedio']:.2f}")
    print(f"  p-value promedio: {estadisticas_globales['pvalue_promedio']:.6f}")
    print(f"  AIC promedio: {estadisticas_globales['aic_promedio']:.2f}")
    print(f"  BIC promedio: {estadisticas_globales['bic_promedio']:.2f}")
    
    print(f"\nPRUEBA DE NORMALIDAD PROMEDIO:")
    print(f"  Shapiro-Wilk stat promedio: {estadisticas_globales['shapiro_stat_promedio']:.4f}")
    print(f"  Shapiro-Wilk p-value promedio: {estadisticas_globales['shapiro_p_promedio']:.4f}")
    
    print(f"\nSUMAS DE CUADRADOS GLOBALES:")
    print(f"  SSR global: {estadisticas_globales['SSR_global']:.4f}")
    print(f"  SSE global: {estadisticas_globales['SSE_global']:.4f}")
    print(f"  SST global: {estadisticas_globales['SST_global']:.4f}")
    print(f"  MSR global: {estadisticas_globales['MSR_global']:.4f}")
    print(f"  MSE global: {estadisticas_globales['MSE_global']:.4f}")
    
    print(f"\nGRADOS DE LIBERTAD GLOBALES:")
    print(f"  Regresión: {estadisticas_globales['df_regression_global']}")
    print(f"  Error: {estadisticas_globales['df_error_global']}")
    
    print(f"\nRESUMEN:")
    print(f"  Número de concentraciones: {estadisticas_globales['n_concentraciones']}")
    print(f"  Total de puntos experimentales: {estadisticas_globales['total_puntos']}")

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
    alpha_errors = [resultados[conc]['metricas'].get('Param_errors', [0,0,0])[0] for conc in concentraciones]
    
    ax.bar(range(len(concentraciones)), alpha_values, alpha=0.7, yerr=alpha_errors, capsize=5)
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
    k_acida_errors = [resultados[conc]['metricas'].get('Param_errors', [0,0,0])[1] for conc in concentraciones]
    k_alcalina_errors = [resultados[conc]['metricas'].get('Param_errors', [0,0,0])[2] for conc in concentraciones]
    
    x_pos = np.arange(len(concentraciones))
    width = 0.35
    
    ax.bar(x_pos - width/2, k_acida_values, width, alpha=0.7, label='k_ácida', yerr=k_acida_errors, capsize=5)
    ax.bar(x_pos + width/2, k_alcalina_values, width, alpha=0.7, label='k_alcalina', yerr=k_alcalina_errors, capsize=5)
    
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
    
    df_reactivo = df[df['Reactivo'] == reactivo_seleccionado]
    
    if df_reactivo.empty:
        print(f"No hay datos para el reactivo: {reactivo_seleccionado}")
        return
    
    concentraciones = df_reactivo['Concentracion de reactivo'].unique()
    print(f"Concentraciones encontradas: {concentraciones}")
    
    es_acido = determinar_tipo_titulante(df_reactivo)
    es_complejo = es_reactivo_complejo(reactivo_seleccionado)
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
        
        # SIEMPRE usar modelo unificado (eliminamos la distinción simple/complejo en el modelo)
        resultado_opt = optimizar_parametros_unificado(Xeq_datos, pH_datos, conc, pH0, pH_min, pH_max, es_complejo)
        
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
            tipo = "Complejo" if es_reactivo_complejo(reactivo) else "Simple"
            print(f"{i}. {reactivo} ({tipo})")
        
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