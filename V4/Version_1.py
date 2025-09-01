import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
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
    denominator = (1 + np.exp(exponent))**n
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
    if len(Y_obs) != len(Y_pred):
        raise ValueError("Y_obs y Y_pred deben tener la misma longitud")
    if len(Y_obs) <= n_params:
        return {k: np.nan for k in ['r2', 'r2_adj', 'chi_sqr', 'chi_sqr_p_value',
                                  'f_stat', 'f_stat_p_value', 'mse', 'mape',
                                  'degrees_of_freedom', 'sigma', 'normality_p']}
    
    n = len(Y_obs)
    residuals = Y_obs - Y_pred
    degrees_of_freedom = n - n_params
    
    # 1. Cálculo de varianza residual
    sigma_sq = np.sum(residuals**2) / degrees_of_freedom
    
    # 2. Coeficientes de determinación
    r2 = r2_score(Y_obs, Y_pred)
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
    
    # 5. Métricas de error
    mse = mean_squared_error(Y_obs, Y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs(np.where(Y_obs != 0, residuals/Y_obs, 0))) * 100
    
    # 6. Test de normalidad
    if 3 <= n <= 5000:
        _, normality_p = stats.shapiro(residuals)
    else:
        normality_p = np.nan
    
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
    if not isinstance(df_subset, pd.DataFrame):
        raise TypeError("Se esperaba un DataFrame de pandas")
    if df_subset.empty:
        return None
        
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
        
        # Configurar límites y valores iniciales
        bounds = (
            [2.8, 0.1, 1, 0.3, np.nanmin(X)],   # Límites inferiores
            [3.2, 10.0, 1e6, 5.0, np.nanmax(X)]  # Límites superiores
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

def plot_comparison(df, reactivo):
    """Visualización mejorada con gráficos comparativos"""
    if reactivo not in df['Reactivo'].unique():
        print(f"No hay datos para {reactivo}")
        return
    
    concentraciones = sorted(df[df['Reactivo'] == reactivo]['Concentracion de reactivo'].unique())
    colors = ['#006400', '#800080', "#0A16BE"]  # Verde oscuro, Púrpura, Azul
    colors = colors * (len(concentraciones) // len(colors) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    results = []
    
    for i, conc in enumerate(concentraciones):
        subset = df[(df['Reactivo'] == reactivo) & 
                   (df['Concentracion de reactivo'] == conc)]
        
        result = fit_titration_data(subset)
        if result is None:
            print(f"No se pudo ajustar {reactivo} {conc}g/L")
            continue
            
        results.append((conc, result))
        
        # Gráfico de dispersión y línea de ajuste
        sc = ax.scatter(result['X'], result['Y'], color=colors[i],
                       label=f'{conc}g/L (datos)', alpha=0.7, s=60)
        line, = ax.plot(np.sort(result['X']), 
                       result['Y_pred'][np.argsort(result['X'])],
                       color=colors[i], linewidth=2.5,
                       label=f'{conc}g/L (ajuste)')
        
        # Línea vertical en el punto de equivalencia
        if len(result['params']) >= 5:
            ax.axvline(x=result['params'][4], color=colors[i], linestyle=':', alpha=0.5)
    
    if not results:
        plt.close(fig)
        print(f"No se pudo generar gráfico para {reactivo}")
        return
    
    ax.set_title(f"Titulación de {reactivo}\n", fontsize=16)
    ax.set_xlabel('Equivalentes de titulante (eq de H+ u OH-/L)', fontsize=12)
    ax.set_ylabel('pH', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_residual_analysis(result):
    """Diagnóstico de residuos del modelo"""
    if result is None:
        print("No hay datos para analizar residuos")
        return
    
    residuals = result['Y'] - result['Y_pred']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histograma de residuos
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.axvline(0, color='r', linestyle='--')
    ax1.set_title("Distribución de Residuos")
    
    # QQ-plot
    stats.probplot(residuals, plot=ax2)
    ax2.set_title("QQ-plot de Residuos")
    
    # Residuos vs Predicciones
    ax3.scatter(result['Y_pred'], residuals, alpha=0.7)
    ax3.axhline(0, color='r', linestyle='--')
    ax3.set_xlabel("Valores Predichos")
    ax3.set_ylabel("Residuos")
    ax3.set_title("Residuos vs Predicciones")
    
    plt.tight_layout()
    plt.show()

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

def show_statistics(stats):
    """Muestra métricas estadísticas con formato profesional"""
    print("\n=== ANÁLISIS ESTADÍSTICO DEL MODELO ===")
    print("\n--- Bondad de Ajuste ---")
    print(f"R²: {stats['r2']:.4f} (explica el {stats['r2']*100:.1f}% de la varianza)")
    print(f"R² ajustado: {stats['r2_adj']:.4f}")
    print(f"MSE: {stats['mse']:.4f}")
    print(f"MAPE: {stats['mape']:.2f}%")
    print(f"Desviación estándar residual (σ): {stats['sigma']:.4f}")
    
    print("\n--- Análisis de Varianza (ANOVA) ---")
    print(f"Suma de cuadrados total (SST): {stats['ss_total']:.4f}")
    print(f"Suma de cuadrados de regresión (SSR): {stats['ss_reg']:.4f}")
    print(f"Suma de cuadrados residual (SSE): {stats['ss_res']:.4f}")
    print(f"Estadístico F: {stats['f_stat']:.4f} (p-value: {stats['f_stat_p_value']:.4g})")
    
    print("\n--- Test Chi-cuadrado de Bondad de Ajuste ---")
    print(f"Estadístico χ²: {stats['chi_sqr']:.4f} (p-value: {stats['chi_sqr_p_value']:.4g})")
    print(f"Grados de libertad: {stats['degrees_of_freedom']}")
    
    print("\n--- Validación de Supuestos ---")
    print(f"Test de normalidad (Shapiro-Wilk p-value): {stats['normality_p']:.4g}")
    
    print("\n--- Interpretación ---")
    print("p-value < 0.05: El modelo muestra problemas significativos")
    print("p-value ≥ 0.05: El ajuste es estadísticamente adecuado")
    print("===================================\n")

def interactive_menu(df, reactivo, model):
    """Menú interactivo para análisis de titulación"""
    while True:
        print("\nOpciones:")
        print("1. Calcular pH para concentraciones específicas")
        print("2. Ver gráfico comparativo")
        print("3. Ver diagnóstico de residuos")
        print("4. Reajustar modelo con transformación Box-Cox")
        print("5. Cambiar de reactivo")
        print("6. Salir del programa")
        
        choice = input("Seleccione una opción (1-6): ").strip()
        
        if choice == '1':
            try:
                Cr = float(input("Ingrese la concentración de reactivo (g/L): "))
                Ct = float(input("Ingrese la concentración de titulante (mol/L): "))
                
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
            plot_comparison(df, reactivo)
            
        elif choice == '3':
            if model is not None:
                plot_residual_analysis(model)
            else:
                print("No hay modelo disponible para análisis")
                
        elif choice == '4':
            subset = df[df['Reactivo'] == reactivo]
            new_model = fit_titration_data(subset, transform_y=True)
            if new_model is not None:
                model = new_model
                print("\nModelo reajustado con transformación Box-Cox (λ = {:.3f})".format(
                    new_model['lambda_boxcox']))
                show_statistics(model['stats'])
            else:
                print("No se pudo reajustar el modelo")
                
        elif choice == '5':
            return 'change'
            
        elif choice == '6':
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
        show_statistics(model['stats'])
        
        # Menú interactivo
        result = interactive_menu(df, selected, model)
        
        if result == 'exit':
            break

    print("\nPrograma terminado. ¡Hasta pronto!")

if __name__ == "__main__":
    main()