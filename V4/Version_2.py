import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
import sys
import os

# Configuración de warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def safe_gradient(y, x):
    """Calcula gradiente numérico de forma segura"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(x) < 2:
            return np.zeros_like(y)
        return np.gradient(y, x)

def calculate_equivalents(concentration):
    """Convierte concentración molar a equivalentes"""
    return np.where(np.isnan(concentration), 0, concentration * 1)

def modified_henderson_eq(X_eq, pH0, beta, k, n, C_eq):
    """Ecuación modificada con protección numérica"""
    exponent = np.clip(-k * (X_eq - C_eq), -700, 700)
    denominator = (1 + np.exp(exponent))
    return pH0 + beta / np.clip(denominator, 1e-10, None)

def get_initial_guess(X, Y):
    """Calcula valores iniciales que respeten los bounds"""
    if len(X) == 0 or len(Y) == 0:
        return [3.0, 1.0, 1000, 1.0, 0.001]
    
    pH0 = np.clip(np.nanpercentile(Y, 5), 2.8, 3.2)
    max_Y = np.nanpercentile(Y, 95)
    beta = np.clip((max_Y - pH0) / np.clip(np.nanmax(X), 1e-6, None), 0.1, 10.0)
    
    try:
        dYdX = safe_gradient(Y, X)
        if len(dYdX) > 0:
            C_eq = X[np.nanargmax(np.abs(dYdX))]
        else:
            C_eq = np.nanmedian(X)
    except:
        C_eq = np.nanmedian(X)
    
    C_eq = np.clip(C_eq, np.nanmin(X), np.nanmax(X))
    k = 1000  # Valor inicial estándar
    
    return [pH0, beta, k, 1.0, C_eq]

def calculate_statistics(Y_obs, Y_pred, params, n_params):
    """
    Calcula métricas estadísticas con rigor estadístico, incluyendo:
    - R² y R² ajustado
    - Chi-cuadrado normalizado y su p-value
    - Estadístico F y p-value
    - MSE y MAPE
    - Validación de supuestos estadísticos
    """
    # 1. Validación básica de longitudes
    if len(Y_obs) != len(Y_pred):
        raise ValueError("Y_obs y Y_pred deben tener la misma longitud")
    
    n = len(Y_obs)
    
    # 2. Validación única de condiciones inválidas
    if n <= n_params or n < 3 or (n - n_params) <= 0:
        return {k: np.nan for k in ['r2', 'r2_adj', 'chi_sqr', 'chi_sqr_p_value',
                                  'f_stat', 'f_stat_p_value', 'mse', 'mape',
                                  'degrees_of_freedom', 'sigma', 'normality_p',
                                  'ss_total', 'ss_reg', 'ss_res']}
    residuals = Y_obs - Y_pred
    degrees_of_freedom = n - n_params
    
    # 1. Cálculo de varianza residual
    sigma_sq = np.sum(residuals**2) / degrees_of_freedom
    
    # 2. Coeficientes de determinación
    ss_total = np.sum((Y_obs - np.mean(Y_obs))**2)
    ss_res = np.sum(residuals**2)
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / degrees_of_freedom
    
    # 3. Estadístico Chi-cuadrado
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_sqr = np.sum(residuals**2 / sigma_sq) if sigma_sq != 0 else np.nan
    chi_sqr_p_value = 1 - stats.chi2.cdf(chi_sqr, degrees_of_freedom) if not np.isnan(chi_sqr) else np.nan
    
    # 4. ANOVA
    ss_total = np.sum((Y_obs - np.mean(Y_obs))**2)
    ss_reg = np.sum((Y_pred - np.mean(Y_obs))**2)
    ss_res = np.sum(residuals**2)
    
    df_reg = n_params - 1
    df_resid = degrees_of_freedom
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f_stat = (ss_reg / df_reg) / (ss_res / df_resid) if (ss_res != 0 and df_resid != 0) else np.nan
    f_stat_p_value = 1 - stats.f.cdf(f_stat, df_reg, df_resid) if not np.isnan(f_stat) else np.nan
    
    # Cálculo CORRECTO de chi-cuadrado para regresión no lineal:
    sigma_sq = np.sum(residuals**2) / degrees_of_freedom  # Varianza residual
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_sqr = np.sum((residuals**2) / sigma_sq) if sigma_sq > 0 else np.nan
        chi_sqr_p_value = stats.chi2.sf(chi_sqr, degrees_of_freedom) if not np.isnan(chi_sqr) else np.nan
    
    # 5. Métricas de error
    mse = mean_squared_error(Y_obs, Y_pred)
    # Por esta versión más robusta:
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.abs(residuals / Y_obs)
        mape = np.nanmean(relative_errors[~np.isinf(relative_errors)]) * 100
    
    # 6. Test de normalidad
    normality_p = np.nan
    if 3 <= n <= 5000:
        try:
            _, normality_p = stats.shapiro(residuals)
        except:
            pass
    
    return {
        'r2': r2,
        'r2_adj': r2_adj,
        'chi_sqr': chi_sqr,
        'chi_sqr_p_value': chi_sqr_p_value,
        'f_stat': f_stat,
        'f_stat_p_value': f_stat_p_value,
        'mse': mse,
        'mape': mape,
        'degrees_of_freedom': degrees_of_freedom,
        'sigma': np.sqrt(sigma_sq),
        'normality_p': normality_p,
        'ss_total': ss_total,
        'ss_reg': ss_reg,
        'ss_res': ss_res
    }


def bootstrap_ci(model, n_iterations=1000):
    """Versión mejorada con manejo de errores"""
    try:
        X = model['X']
        Y = model['Y']
        params = model['params']
        n = len(X)
        
        bootstrap_params = []
        np.random.seed(42)  # Reproducibilidad
        
        for _ in range(n_iterations):
            try:
                # Muestreo con reemplazo
                indices = np.random.choice(n, n, replace=True)
                X_sample = X[indices]
                Y_sample = Y[indices]
                
                # Ajuste con más tolerancia a errores
                params_i, _ = curve_fit(
                    modified_henderson_eq,
                    X_sample, Y_sample,
                    p0=params,
                    bounds=([2.8, 0.5, 10, 0.1, np.nanmin(X)],
                           [10.0, 8.0, 5000, 2.0, np.nanmax(X)]),
                    maxfev=20000,  # Aumentar iteraciones máximas
                    ftol=1e-4,     # Tolerancia más flexible
                    xtol=1e-4
                )
                bootstrap_params.append(params_i)
            except:
                continue  # Ignora muestras problemáticas
        
        if len(bootstrap_params) < 100:  # Mínimo 100 muestras válidas
            raise ValueError("Insuficientes muestras bootstrap válidas")
        
        bootstrap_params = np.array(bootstrap_params)
        ci_lower = np.percentile(bootstrap_params, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_params, 97.5, axis=0)
        
        return {
            'original': params,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_valid_samples': len(bootstrap_params)  # Para diagnóstico
        }
        
    except Exception as e:
        print(f"Error en bootstrap: {str(e)}")
        return None

def heteroscedasticity_test(residuals, X):
    try:
        from statsmodels.stats.diagnostic import het_white
        from statsmodels.tools.tools import add_constant
        
        # Añadir término constante y términos cuadráticos
        X_design = add_constant(np.column_stack([X, X**2]))  # ¡Corrección clave!
        white_test = het_white(residuals, X_design)
        return {
            'test_stat': white_test[0],
            'p_value': white_test[1],
            'f_stat': white_test[2],
            'f_p_value': white_test[3]
        }
    except Exception as e:
        print(f"Error en test de heterocedasticidad: {str(e)}")
        return None

def apply_boxcox_transform(Y):
    """Aplica transformación Box-Cox con manejo robusto de datos"""
    Y = np.asarray(Y).copy()
    
    # 1. Manejo de NaN e infinitos
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Los datos contienen valores NaN o infinitos")
    
    # 2. Manejo de valores no positivos
    min_Y = np.min(Y)
    if min_Y <= 0:
        offset = -min_Y + 0.001  # Asegura valores estrictamente positivos
        Y = Y + offset
    else:
        offset = 0
    
    # 3. Aplicar Box-Cox
    try:
        Y_transformed, lambda_ = stats.boxcox(Y)
        return Y_transformed, lambda_, offset
    except Exception as e:
        raise ValueError(f"Error en Box-Cox: {str(e)}")

def inv_boxcox(y, lmbda, offset=0):
    if lmbda == 0:
        return np.exp(y) - offset
    else:
        return (y * lmbda + 1)**(1/lmbda) - offset  # Fórmula correcta

def fit_titration_data(df_subset, transform_y=False):
    """Ajuste robusto con manejo de errores mejorado"""
    
    # Diccionario base que siempre se devuelve
    base_result = {
        'params': None,
        'X': None,
        'Y': None,
        'Y_pred': None,
        'covariance': None,
        'stats': None,
        'transform_y': transform_y,
        'lambda_boxcox': None
    }
    
    if not isinstance(df_subset, pd.DataFrame):
        print("Error: Se esperaba un DataFrame de pandas")
        return base_result
    
    if df_subset.empty:
        print("Error: El DataFrame está vacío")
        return base_result
        
    try:
        X = calculate_equivalents(df_subset['Concentracion de titulante'].values)
        Y = df_subset['pH'].values
        
        # Filtrado robusto
        valid_mask = (~np.isnan(X)) & (~np.isnan(Y)) & (Y >= 2.5) & (Y <= 9.5)
        X = X[valid_mask]
        Y = Y[valid_mask]
        
        if len(X) < 3:
            return None
        
        # Transformación Box-Cox si se solicita
        lambda_ = None
        if transform_y:
            try:
                Y_transformed, lambda_, offset = apply_boxcox_transform(Y)
                Y_original = Y.copy()
                Y = Y_transformed
            except Exception as e:
                print(f"Error en transformación Box-Cox: {str(e)}")
                print("Valores de Y:", Y)
                print("NaN en Y:", np.isnan(Y).sum())
                print("Valores <=0 en Y:", (Y <= 0).sum())
                return None
        
        if "K2HPO4" in df_subset['Reactivo'].unique():
            bounds = (
                # Límites inferiores       [  pH0,  beta,   k,    n,      C_eq ]
                [2.8,                    0.5,    10,    0.1,    np.nanmin(X)],
                
                # Límites superiores
                [10.0,                   8.0,    5000,  2.0,    np.nanmax(X)]  # Permite pH0 hasta 10
            )
        else:
            bounds = (
                # Límites inferiores       [  pH0,  beta,   k,    n,      C_eq ]
                [2.8,                    0.5,    10,    0.1,    np.nanmin(X)],
                
                # Límites superiores
                [9.0,                    8.0,    5000,  2.0,    np.nanmax(X)]
            )
             
        p0 = get_initial_guess(X, Y)
        p0 = [np.clip(p0[i], bounds[0][i], bounds[1][i]) for i in range(len(p0))]
        
        # Ajuste de curva
        params, covariance = curve_fit(
            modified_henderson_eq,
            X, Y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
            method='trf',
            nan_policy='omit'
        )
        
        # Validación post-ajuste
        if np.any(np.isnan(params)) or not all(bounds[0][i] <= params[i] <= bounds[1][i] for i in range(len(params))):
            raise ValueError("Parámetros inválidos después del ajuste")
        
        # Calcular predicciones
        Y_pred = modified_henderson_eq(X, *params)
        
        # Revertir transformación si se aplicó
        if transform_y and lambda_ is not None:
            Y_pred = inv_boxcox(Y_pred, lambda_)
            Y = Y_original
        
        # Calcular métricas estadísticas
        stats = calculate_statistics(Y, Y_pred, params, len(params))
        
        return {
            'params': params,
            'X': X,
            'Y': Y,
            'Y_pred': Y_pred,
            'covariance': covariance,
            'stats': stats,
            'transform_y': transform_y,
            'lambda_boxcox': lambda_
        }
        
    except Exception as e:
        print(f"Error en ajuste: {str(e)}")
        return None

def plot_comparison_dual(df, reactivo):
    """Muestra gráficos en escala pH y [H⁺] sin normalización"""
    if reactivo not in df['Reactivo'].unique():
        print(f"No hay datos para {reactivo}")
        return
    
    concentraciones = sorted(df[df['Reactivo'] == reactivo]['Concentracion de reactivo'].unique())
    colors = ['#006400', '#800080', "#0A16BE"]  # Verde oscuro, Púrpura, Azul
    colors = colors * (len(concentraciones) // len(colors) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    results = []
    
    # Bandera para mostrar el label del IC solo una vez
    ic_label_added = False
    
    for i, conc in enumerate(concentraciones):
        subset = df[(df['Reactivo'] == reactivo) & 
                   (df['Concentracion de reactivo'] == conc)]
        
        result = fit_titration_data(subset)
        if result is None:
            print(f"No se pudo ajustar {reactivo} {conc}g/L")
            continue
            
        results.append((conc, result))
        X = result['X']
        Y_pred = result['Y_pred']
        
        # --- Gráfico 1: Escala pH (original) ---
        # Bandas de confianza
        if bootstrap_result := bootstrap_ci(result):
            Y_samples = []
            for params_sample in bootstrap_result['bootstrap_params']:
                if np.any(np.isnan(params_sample)):
                    continue
                Y_samples.append(modified_henderson_eq(X, *params_sample))
            
            if Y_samples:
                Y_samples = np.array(Y_samples)
                Y_lower = np.nanpercentile(Y_samples, 2.5, axis=0)
                Y_upper = np.nanpercentile(Y_samples, 97.5, axis=0)
                
                label = 'IC 95%' if not ic_label_added else None
                ax1.fill_between(X, Y_lower, Y_upper, color=colors[i], alpha=0.1, label=label)
                ic_label_added = True
        
        # Datos y ajuste
        ax1.scatter(X, result['Y'], color=colors[i],
                   label=f'{conc}g/L (datos)', alpha=0.7, s=60)
        ax1.plot(np.sort(X), Y_pred[np.argsort(X)],
                color=colors[i], linewidth=2.5,
                label=f'{conc}g/L (ajuste)')
        
        # Punto de equivalencia
        if len(result['params']) >= 5:
            ax1.axvline(x=result['params'][4], color=colors[i], linestyle=':', alpha=0.5)
        
        # --- Gráfico 2: Escala [H⁺] ---
        Y_H = 10**(-result['Y'])  # Conversión directa a [H⁺]
        Y_H_pred = 10**(-Y_pred)  # Convertir predicciones a [H⁺]
        
        ax2.scatter(X, Y_H, color=colors[i], alpha=0.7, s=60)
        ax2.plot(np.sort(X), Y_H_pred[np.argsort(X)], color=colors[i], linewidth=2.5)
        ax2.set_yscale('log')
    
    if not results:
        plt.close(fig)
        return
    
    # Configuración de gráficos
    ax1.set_title(f"Titulación de {reactivo} (escala pH)\nIC bootstrap (95%)", fontsize=14)
    ax1.set_xlabel('Equivalentes de titulante (H+/OH-)/L', fontsize=12)
    ax1.set_ylabel('pH', fontsize=12)
    ax1.grid(True, alpha=0.2)
    
    ax2.set_title(f"Titulación de {reactivo} (escala [H⁺])", fontsize=14)
    ax2.set_xlabel('Equivalentes de titulante (H+/OH-)/L', fontsize=12)
    ax2.set_ylabel('[H⁺] (mol/L)', fontsize=12)
    ax2.grid(True, alpha=0.2)
    
    # Leyenda solo para ax1 (pH)
    handles, labels = ax1.get_legend_handles_labels()
    if ic_label_added and 'IC 95%' not in labels:
        handles.append(plt.Rectangle((0,0), 1, 1, fc=colors[0], alpha=0.2))
        labels.append('IC 95%')
    ax1.legend(handles, labels, fontsize=10)
    
    plt.tight_layout()
    plt.show()
    return results

def plot_residual_analysis(result):
    """Diagnóstico de residuos del modelo mejorado"""
    if result is None:
        print("No hay datos para analizar residuos")
        return
    
    residuals = result['Y'] - result['Y_pred']
    X = result['X']
    
    # Crear figura más grande para acomodar más subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Gráfico 1: Histograma de residuos
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.axvline(0, color='r', linestyle='--')
    ax1.set_title("Distribución de Residuos")
    
    # Gráfico 2: QQ-plot
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(residuals, plot=ax2)
    ax2.set_title("QQ-plot de Residuos")
    
    # Gráfico 3: Residuos vs Predicciones
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(result['Y_pred'], residuals, alpha=0.7)
    ax3.axhline(0, color='r', linestyle='--')
    ax3.set_xlabel("Valores Predichos")
    ax3.set_ylabel("Residuos")
    ax3.set_title("Residuos vs Predicciones")
    
    # Gráfico 4: Residuos vs X
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(X, residuals, alpha=0.7)
    ax5.axhline(0, color='r', linestyle='--')
    ax5.set_xlabel("Variable Independiente (X)")
    ax5.set_ylabel("Residuos")
    ax5.set_title("Residuos vs Variable Independiente")
    
    # Mostrar tests estadísticos en consola
    print("\n=== DIAGNÓSTICO DE RESIDUOS ===")
    
    # Test de normalidad
    if len(residuals) >= 3:
        _, normality_p = stats.shapiro(residuals)
        print(f"\nTest de normalidad (Shapiro-Wilk):")
        print(f"p-value = {normality_p:.4f}")
        print("Interpretación:", "Normal (p ≥ 0.05)" if normality_p >= 0.05 else "No normal (p < 0.05)")
    
    # Test de heterocedasticidad
    het_test = heteroscedasticity_test(residuals, X)
    if het_test:
        print(f"\nTest de White para heterocedasticidad:")
        print(f"Estadístico = {het_test['test_stat']:.4f}, p-value = {het_test['p_value']:.4f}")
        print(f"Estadístico F = {het_test['f_stat']:.4f}, p-value = {het_test['f_p_value']:.4f}")
        print("Interpretación:", "Homocedástico (p ≥ 0.05)" if het_test['p_value'] >= 0.05 else "Heterocedástico (p < 0.05)")
    
    # Intervalo de confianza bootstrap (versión específica para residuos)
    try:
        ci = bootstrap_ci_residuos(result)
        if ci:
            print(f"\nIntervalo de confianza bootstrap (95%) para la media de residuos:")
            print(f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            print("\nNo se pudo calcular el intervalo bootstrap para residuos.")
    except Exception as e:
        print(f"\nError al calcular IC para residuos: {str(e)}")
    
    plt.tight_layout()
    plt.show()

# Nueva función específica para residuos
def bootstrap_ci_residuos(model, n_iterations=1000):
    """Versión especial para IC de la media de residuos"""
    try:
        residuals = model['Y'] - model['Y_pred']
        if len(residuals) == 0:
            return None
            
        # Método percentil básico
        bootstrap_means = []
        n = len(residuals)
        np.random.seed(42)  # Reproducibilidad
        
        for _ in range(n_iterations):
            sample = np.random.choice(residuals, n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        return (ci_lower, ci_upper)  # Devuelve una tupla simple
        
    except Exception as e:
        print(f"Error en bootstrap de residuos: {str(e)}")
        return None

def predict_pH(model, Cr, Ct):
    """Predice el pH para concentraciones dadas"""
    if model is None or not isinstance(model, dict) or 'params' not in model or len(model['params']) != 5:
        print("Error: Modelo inválido o incompleto")
        return None
    
    try:
        X_eq = calculate_equivalents(np.array([float(Ct)]))[0]
        return modified_henderson_eq(X_eq, *model['params'])
    except (ValueError, TypeError) as e:
        print(f"Error en cálculo de pH: {str(e)}")
        return None

def show_statistics(model):
    """Muestra métricas estadísticas con formato profesional"""
    if model is None or model.get('stats') is None:
        print("\nNo se pudieron calcular estadísticas del modelo")
        print("Posibles causas:")
        print("- Insuficientes datos válidos")
        print("- Problemas en el ajuste de curva")
        print("- Valores de pH fuera de rango")
        return
    
    stats = model['stats']
    print("\n=== ANÁLISIS ESTADÍSTICO DEL MODELO ===")
    print("\n--- Bondad de Ajuste ---")
    print(f"R²: {stats['r2']:.4f} (explica el {stats['r2']*100:.1f}% de la varianza)")
    print(f"R² ajustado: {stats['r2_adj']:.4f}")
    print(f"MSE: {stats['mse']:.4f}")
    print(f"MAPE: {stats['mape']:.2f}%")
    print(f"Desviación estándar residual (σ): {stats['sigma']:.4f}")
    
    # ANOVA
    print("\n--- Análisis de Varianza (ANOVA) ---")
    print(f"Suma de cuadrados total (SST): {stats['ss_total']:.4f}")
    print(f"Suma de cuadrados de regresión (SSR): {stats['ss_reg']:.4f}")
    print(f"Suma de cuadrados residual (SSE): {stats['ss_res']:.4f}")
    print(f"Estadístico F: {stats['f_stat']:.4f} (p-value: {stats['f_stat_p_value']:.4g})")
    print(f"χ²(DF={stats['degrees_of_freedom']}) = {stats['chi_sqr']:.1f}, p = {stats['chi_sqr_p_value']:.3f}")
    
    # Intervalos de confianza
    print("\n--- Intervalos de Confianza Bootstrap (95%) ---")
    param_names = ['pH0', 'beta', 'k', 'C_eq']
    bootstrap_result = bootstrap_ci(model)

    if bootstrap_result:
        for i, name in enumerate(param_names):
            # Formato especial para C_eq
            if name == 'C_eq':
                original = bootstrap_result['original'][i]
                lower = bootstrap_result['ci_lower'][i]
                upper = bootstrap_result['ci_upper'][i]
                
                # Determinar si son protones o hidroxilos
                ion_type = "H⁺" if original < 0 else "OH⁻"
                
                print(f"{name}: {abs(original):.4f} eq de {ion_type}/L "
                    f"[{abs(lower):.4f}, {abs(upper):.4f}]")
            else:
                print(f"{name}: {bootstrap_result['original'][i]:.4f} "
                    f"[{bootstrap_result['ci_lower'][i]:.4f}, "
                    f"{bootstrap_result['ci_upper'][i]:.4f}]")
    else:
        print("No se pudieron calcular los intervalos de confianza")
    
    # Validación de supuestos
    print("\n--- Validación de Supuestos ---")
    print(f"Test de normalidad (Shapiro-Wilk p-value): {stats['normality_p']:.4g}")
    print("\nNota: Los intervalos bootstrap son válidos incluso con residuos no normales")
    print("===================================\n")
    

def interactive_menu(df, reactivo, model):
    """Menú interactivo para análisis de titulación"""
    while True:
        print("\nOpciones:")
        print("1. Calcular pH para concentraciones específicas")
        print("2. Ver gráfico comparativo")
        print("3. Ver diagnóstico de residuos")
        print("4. Cambiar de reactivo")
        print("5. Salir del programa")
        
        choice = input("Seleccione una opción (1-5): ").strip()
        
        if choice == '1':
            # Mostrar rangos disponibles ANTES de pedir inputs
            print(f"\nRangos experimentales para {reactivo}:")
            reactivo_data = df[df['Reactivo'] == reactivo]
            
            min_Cr = reactivo_data['Concentracion de reactivo'].min()
            max_Cr = reactivo_data['Concentracion de reactivo'].max()
            mean_Cr = (max_Cr + min_Cr)/2
            print(f"- Concentración de reactivo: {min_Cr:.2f}, {mean_Cr:.2f} y {max_Cr:.2f} g/L")
            
            min_Ct = reactivo_data['Concentracion de titulante'].min()
            max_Ct = reactivo_data['Concentracion de titulante'].max()
            print(f"- Concentración de titulante: {min_Ct:.4f}, {max_Ct:.4f} mol/L")
            
            try:
                Cr = float(input("\nIngrese la concentración de reactivo (g/L): "))
                Ct = float(input("Ingrese la concentración de titulante mol/L: "))
                
                subset = df[(df['Reactivo'] == reactivo) & 
                          (df['Concentracion de reactivo'] == Cr)]
                
                if subset.empty:
                    print(f"\nAdvertencia: No hay datos experimentales para {Cr}g/L de {reactivo}")
                    print("El resultado será una extrapolación del modelo.")
                
                pH_pred = predict_pH(model, Cr, Ct)
                
                if pH_pred is not None:
                    print(f"\nResultado: pH estimado = {pH_pred:.2f}")
                    
                    if not subset.empty:
                        result = fit_titration_data(subset)
                        if result is not None:
                            show_statistics(result['stats'])
                else:
                    print("No se pudo calcular el pH. Verifique los valores ingresados.")
                    
            except ValueError:
                print("Error: Ingrese valores numéricos válidos.")
            
        elif choice == '2':
            plot_comparison_dual(df, reactivo)
            
        elif choice == '3':
            if model is not None:
                plot_residual_analysis(model)
            else:
                print("No hay modelo disponible para análisis")
                
        elif choice == '4':
            return 'change'
            
        elif choice == '5':
            return 'exit'

def main():
    """Función principal del programa"""
    file_path = 'datos_titulacion.csv'
    
    # 1. Verificación del archivo
    if not os.path.exists(file_path):
        print(f"Error: El archivo '{file_path}' no existe en:")
        print(f"-> {os.getcwd()}")
        print("\nSugerencias:")
        print("- Verifique el nombre del archivo")
        print("- Coloque el archivo en la misma carpeta que este script")
        return
    
    # 2. Carga y validación de datos
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            print("Error: El archivo está vacío")
            return
        
        required_columns = ['Reactivo', 'Concentracion de reactivo', 
                          'Concentracion de titulante', 'pH']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print("Error: Faltan columnas requeridas:")
            for col in missing_cols:
                print(f"- {col}")
            print("\nColumnas encontradas:")
            print(df.columns.tolist())
            return
        
        # Verificar valores nulos
        for col in ['Reactivo', 'pH']:
            if df[col].isnull().any():
                print(f"Advertencia: La columna '{col}' contiene valores nulos")
        
    except pd.errors.EmptyDataError:
        print("Error: El archivo CSV está vacío o corrupto")
        return
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return
    
    # 3. Bucle principal
    while True:
        reactivos = df['Reactivo'].unique()
        
        print("\n" + "="*50)
        print("Reactivos disponibles:")
        for i, r in enumerate(reactivos, 1):
            print(f"{i}. {r}")
        
        reactivo = input("\nIngrese el reactivo a analizar (número o nombre, 'salir' para terminar): ").strip()
        
        if reactivo.lower() == 'salir':
            break
            
        # Validar selección
        selected = None
        try:
            num = int(reactivo)
            if 1 <= num <= len(reactivos):
                selected = reactivos[num-1]
        except ValueError:
            matches = [r for r in reactivos if r.lower() == reactivo.lower()]
            if matches:
                selected = matches[0]
        
        if not selected:
            print(f"\nError: '{reactivo}' no es válido")
            print("Ingrese un número o nombre de la lista")
            continue
        
        # Ajustar modelo
        subset = df[df['Reactivo'] == selected]
        model = fit_titration_data(subset)
        
        if model is None:
            print(f"\nNo se pudo ajustar modelo para {selected}. Posibles causas:")
            print("- Insuficientes datos")
            print("- Valores de pH fuera de rango (2.5-9.5)")
            print("- Problemas en el ajuste de curva")
            continue
        
        # Mostrar resultados
        print("\n" + "="*50)
        print(f"RESULTADOS PARA: {selected}")
        show_statistics(model)
        
        # Menú interactivo
        result = interactive_menu(df, selected, model)
        
        if result == 'exit':
            break

    print("\nPrograma terminado. ¡Hasta pronto!")

if __name__ == "__main__":
    main()