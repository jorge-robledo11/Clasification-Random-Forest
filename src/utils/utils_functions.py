# • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ Pyfunctions ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ •
# Librerías y/o depedencias
from scipy import stats
from matplotlib import gridspec
from IPython.display import display, Latex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import empiricaldist
sns.set_theme(context='notebook', style=plt.style.use('dark_background'))


# Capturar variables
# Función para capturar los tipos de variables
def capture_variables(data:pd.DataFrame) -> tuple:
    
    """
    Function to capture the types of Dataframe variables

    Args:
        dataframe: DataFrame
    
    Return:
        variables: A tuple of lists
    
    The order to unpack variables:
    1. numericals
    2. continous
    3. categoricals
    4. discretes
    5. temporaries
    """

    numericals = list(data.select_dtypes(include = [np.int32, np.int64, np.float32, np.float64]).columns)
    categoricals = list(data.select_dtypes(include = ['category', 'object', 'bool']).columns)
    temporaries = list(data.select_dtypes(include = ['datetime', 'timedelta']).columns)
    discretes = [col for col in data[numericals] if len(data[numericals][col].unique()) <= 10]
    continuous = [col for col in data[numericals] if col not in discretes]

    # Variables
    print('\t\tTipos de variables')
    print(f'Hay {len(continuous)} variables continuas')
    print(f'Hay {len(discretes)} variables discretas')
    print(f'Hay {len(temporaries)} variables temporales')
    print(f'Hay {len(categoricals)} variables categóricas')
    
    variables = tuple((continuous, categoricals, discretes, temporaries))

    # Retornamos una tupla de listas
    return variables
            

# Función para graficar los datos con valores nulos
def plotting_nan_values(data:pd.DataFrame) -> any:

    """
    Function to plot nan values

    Args:
        data: DataFrame
    
    Return:
        Dataviz
    """

    vars_with_nan = [var for var in data.columns if data[var].isnull().sum() > 0]
    
    if len(vars_with_nan) == 0:
        print('No se encontraron variables con nulos')
    
    else:
        # Plotting
        plt.figure(figsize=(14, 6))
        data[vars_with_nan].isnull().mean().sort_values(ascending=False).plot.bar(color='royalblue', edgecolor='skyblue', lw=0.75)
        plt.ylabel('Percentage of missing data')
        plt.xticks(fontsize=10, rotation=25)
        plt.yticks(fontsize=10)
        plt.grid(True)
        plt.tight_layout()


# Función para obtener la matriz de correlaciones entre los predictores
def correlation_matrix(data:pd.DataFrame, continuous:list) -> any:
    
    """
    Function to plot correlation_matrix

    Args:
        data: DataFrame
        continuous: list
    
    Return:
        Dataviz
    """
    
    correlations = data[continuous].corr(method='pearson', numeric_only=True)
    plt.figure(figsize=(17, 10))
    sns.heatmap(correlations, vmax=1, annot=True, cmap='gist_yarg', linewidths=1, square=True)
    plt.title('Matriz de Correlaciones\n', fontsize=14)
    plt.xticks(fontsize=10, rotation=25)
    plt.yticks(fontsize=10, rotation=25)
    plt.tight_layout()


# Covarianza entre los predictores
# Función para obtener una matriz de covarianza con los predictores
def covariance_matrix(data:pd.DataFrame):
    
    """
    Function to get mapped covariance matrix

    Args:
        data: DataFrame
    
    Return:
        DataFrame
    """
    
    cov_matrix = data.cov()
    
    # Crear una matriz de ceros con el mismo tamaño que la matriz de covarianza
    zeros_matrix = np.zeros(cov_matrix.shape)
    
    # Crear una matriz diagonal de ceros reemplazando los valores de la diagonal de la matriz con ceros
    diagonal_zeros_matrix = np.diag(zeros_matrix)
    
    # Reemplazar la diagonal de la matriz de covarianza con la matriz diagonal de ceros
    np.fill_diagonal(cov_matrix.to_numpy(), diagonal_zeros_matrix)
    
    # Mapear los valores con etiquetas para saber cómo covarian los predictores
    cov_matrix = cov_matrix.applymap(lambda x: 'Positivo' if x > 0 else 'Negativo' if x < 0 else '')
    
    return cov_matrix


# Función para graficar la covarianza entre los predictores
def plotting_covariance(X:pd.DataFrame, continuous:list, n_iter:int) -> any:
    
    """
    Function to plot covariance matrix choosing some random predictors

    Args:
        X: DataFrame
        continuous: list
        n_iter: int

    Return:
        DataViz
    """

    # Semilla para efectos de reproducibilidad
    np.random.seed(n_iter)

    for _ in range(n_iter):
        # Creamos una figura con tres subfiguras
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        plt.suptitle('Covariance Plots\n', fontsize=15)

        # Seleccionamos dos variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la primera subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax1, color='red', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax1.grid(color='white', linestyle='-', linewidth=0.25)

        # Seleccionamos dos nuevas variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la segunda subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax2, color='green', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax2.grid(color='white', linestyle='-', linewidth=0.25)

        # Seleccionamos otras dos variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la tercera subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax3, color='blue', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax3.grid(color='white', linestyle='-', linewidth=0.25)

        # Mostramos la figura
        fig.tight_layout()
        plt.show()


# Diagnóstico de variables
# Función para observar el comportamiento de variables continuas
def diagnostic_plots(data:pd.DataFrame, variables:list) -> any:

    """
    Function to get diagnostic graphics into 
    numerical (continous and discretes) predictors

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
        
    for var in data[variables]:
        fig, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle('Diagnostic Plots', fontsize=16)
        plt.rcParams.update({'figure.max_open_warning': 0}) # Evitar un warning

        # Histogram Plot
        plt.subplot(1, 4, 1)
        plt.title('Histogram Plot')
        sns.histplot(data[var], bins=25, color='midnightblue', edgecolor='white', lw=0.5)
        plt.axvline(data[var].mean(), color='#E51A4C', ls='dashed', lw=1.5, label='Mean')
        plt.axvline(data[var].median(), color='gold', ls='dashed', lw=1.5, label='Median')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=25)
        plt.xlabel(var)
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.legend(fontsize=10)
        
        # CDF Plot
        plt.subplot(1, 4, 2)
        plt.title('CDF Plot')
        xs = np.linspace(data[var].min(), data[var].max())
        ys = stats.norm(data[var].mean(), data[var].std()).cdf(xs) # Distribución normal a partir de unos datos
        plt.plot(xs, ys, color='cornflowerblue', ls='dashed')
        empiricaldist.Cdf.from_seq(data[var], normalize=True).plot(color='chartreuse')
        plt.xlabel(var)
        plt.xticks(rotation=25)
        plt.ylabel('Probabilidad')
        plt.legend(['Distribución normal', var], fontsize=8, loc='upper left')
        plt.grid(color='white', linestyle='-', linewidth=0.25)

        # PDF Plot
        plt.subplot(1, 4, 3)
        plt.title('PDF Plot')
        kurtosis = stats.kurtosis(data[var], nan_policy='omit') # Kurtosis
        skew = stats.skew(data[var], nan_policy='omit') # Sesgo
        xs = np.linspace(data[var].min(), data[var].max())
        ys = stats.norm(data[var].mean(), data[var].std()).pdf(xs) # Distribución normal a partir de unos datos
        plt.plot(xs, ys, color='cornflowerblue', ls='dashed')
        sns.kdeplot(data=data, x=data[var], fill=True, lw=0.75, color='crimson', alpha=0.5, edgecolor='white')
        plt.text(s=f'Skew: {skew:0.2f}\nKurtosis: {kurtosis:0.2f}',
                 x=0.25, y=0.65, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='center', horizontalalignment='center')
        plt.ylabel('Densidad')
        plt.xticks(rotation=25)
        plt.xlabel(var)
        plt.xlim()
        plt.legend(['Distribución normal', var], fontsize=8, loc='upper right')
        plt.grid(color='white', linestyle='-', linewidth=0.25)

        # Boxplot & Stripplot
        plt.subplot(1, 4, 4)
        plt.title('Boxplot')
        sns.boxplot(data=data[var], width=0.4, color='silver', sym='*',
                    boxprops=dict(lw=1, edgecolor='white'),
                    whiskerprops=dict(color='white', lw=1),
                    capprops=dict(color='white', lw=1),
                    medianprops=dict(),
                    flierprops=dict(color='red', lw=1, marker='o', markerfacecolor='red'))
        plt.axhline(data[var].quantile(0.75), color='magenta', ls='dotted', lw=1.5, label='IQR 75%')
        plt.axhline(data[var].median(), color='gold', ls='dashed', lw=1.5, label='Median')
        plt.axhline(data[var].quantile(0.25), color='cyan', ls='dotted', lw=1.5, label='IQR 25%')
        plt.xlabel(var)
        plt.tick_params(labelbottom=False)
        plt.ylabel('Unidades')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        
        fig.tight_layout()


# Test de Normalidad de D’Agostino y Pearson
# Función para observar el comportamiento de las variables continuas en una prueba de normalidad
# Y realizar un contraste de hipótesis para saber si se asemeja a una distribución normal
def normality_test(data, variables):

    display(Latex('Si el $pvalue$ < 0.05; se rechaza la $H_0$ sugiere que los datos no se ajustan de manera significativa a una distribución normal'))
    
    # Configurar figura
    fig = plt.figure(figsize=(24, 20))
    plt.suptitle('Prueba de Normalidad', fontsize=18)
    gs = gridspec.GridSpec(nrows=len(variables) // 3+1, ncols=3, figure=fig)
    
    for i, var in enumerate(variables):

        ax = fig.add_subplot(gs[i//3, i % 3])

        # Gráfico Q-Q
        stats.probplot(data[var], dist='norm', plot=ax)
        ax.set_xlabel(var)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels())
        ax.grid(color='white', linestyle='-', linewidth=0.25)

        # P-value
        p_value = stats.normaltest(data[var])[1]
        ax.text(0.8, 0.9, f"p-value={p_value:0.3f}", transform=ax.transAxes, fontsize=13) 

    plt.tight_layout(pad=3)
    plt.show()


# Definir la transformación de Yeo-Johnson
def yeo_johnson_transform(x):
    y = np.where(x >= 0, x + 1, 1 / (1 - x))
    y = np.sign(y) * (np.abs(y) ** 0.5)
    return y

def gaussian_transformation(data:pd.DataFrame, variables:list) -> dict:
    
    """
    Function to get Gaussian transformations of the variables

    Args:
        data: DataFrame
        variables: list
    
    Return:
        results: dict
    """
    
    # Definir las transformaciones gaussianas a utilizar
    transformaciones_gaussianas = {
        'Log': np.log,
        'Sqrt': np.sqrt, 
        'Reciprocal': lambda x: 1/x, 
        'Exp': lambda x: x**2, 
        'Yeo-Johnson': yeo_johnson_transform
        }
    
    # Crear un diccionario para almacenar los resultados de las pruebas de normalidad
    results = dict()

    # Iterar a través de las variables y las transformaciones
    for var in data[variables].columns:
        mejores_p_value = 0
        mejor_transformacion = None
        
        for nombre_transformacion, transformacion in transformaciones_gaussianas.items():
            # Aplicar la transformación a la columna
            variable_transformada = transformacion(data[var])
            
            # Calcular el p-value de la prueba de normalidad
            p_value = stats.normaltest(variable_transformada)[1]
            
            # Actualizar el mejor p-value y transformación si es necesario
            if p_value > mejores_p_value:
                mejores_p_value = p_value
                mejor_transformacion = nombre_transformacion
        
        # Almacenar el resultado en el diccionario
        results[var] = mejor_transformacion
        
    return results


# Graficar la comparativa entre las variables originales y su respectiva transformación
def graficar_transformaciones(data:pd.DataFrame, continuous:list, transformacion:dict) -> any:
    
    """
    Function to plot compare Gaussian transformations of the variables and their original state

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
    
    # Definir las transformaciones gaussianas a utilizar
    transformaciones_gaussianas = {
        'Log': np.log,
        'Sqrt': np.sqrt, 
        'Reciprocal': lambda x: 1/x, 
        'Exp': lambda x: x**2, 
        'Yeo-Johnson': yeo_johnson_transform
        }
    
    data = data.copy()
    data = data[continuous]
    
    for variable, transformacion_name in transformacion.items():
        # Obtener datos originales
        data_original = data[variable]
        
        # Obtener la transformación correspondiente
        transformacion_func = transformaciones_gaussianas.get(transformacion_name)
        
        # Aplicar transformación
        data_transformada = transformacion_func(data_original)

        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Graficar histograma datos originales 
        hist_kws = {'color': 'royalblue', 'lw': 0.5}
        sns.histplot(data_original, ax=ax1, kde=True, bins=50, **hist_kws)
        ax1.set_title('Original')
        ax1.grid(color='white', linestyle='-', linewidth=0.25)

        # Graficar histograma datos transformados
        sns.histplot(data_transformada, ax=ax2, kde=True, bins=50, **hist_kws)
        ax2.set_title(f'{transformacion_name}')
        ax2.grid(color='white', linestyle='-', linewidth=0.25)
        
        # Cambiar color del KDE en ambos gráficos
        for ax in [ax1, ax2]:
            for line in ax.lines:
                line.set_color('crimson')

        # Mostrar figura
        plt.tight_layout()
        plt.show()
