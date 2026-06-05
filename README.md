# Regresion Polinomial: Teoria y Practica

## Que es la regresion polinomial

Cuando tienes datos del mundo real y quieres encontrar la funcion matematica que mejor los describe, usas regresion. Si esa funcion es una linea recta, se llama regresion lineal. Si es una curva, se llama regresion polinomial.

La regresion polinomial busca los coeficientes de un polinomio de la forma:

```
y = a0 + a1*x + a2*x^2 + a3*x^3 + ... + an*x^n
```

Donde cada `a` es un coeficiente que el modelo debe encontrar, y `n` es el grado del polinomio.

---

## Por que usamos exponentes

Sin exponentes, una funcion solo puede producir lineas rectas. Los exponentes introducen curvatura. Cada exponente aporta un tipo distinto de forma:

- `x^1` produce una linea recta
- `x^2` produce una parabola (forma de U o arco)
- `x^3` produce una curva en S
- `x^4` produce una forma de W suavizada

Al combinar varios de estos terminos con distintos coeficientes, puedes moldear la curva para que se ajuste a casi cualquier patron de datos.

---

## Como se encuentran los coeficientes

### Cuando tienes el mismo numero de puntos que coeficientes

Si tienes exactamente tantos puntos como coeficientes desconocidos, puedes plantear un sistema de ecuaciones y resolverlo exactamente. Por ejemplo, con 3 puntos y un polinomio de grado 2 (que tiene 3 coeficientes: a0, a1, a2), sustituyes cada punto en la formula y obtienes 3 ecuaciones con 3 incognitas. Eso se resuelve con eliminacion gaussiana: restas ecuaciones entre si para eliminar variables una por una hasta quedarte con una sola incognita, la resuelves, y sustituyes hacia atras para encontrar las demas.

### Cuando tienes mas puntos que coeficientes (el caso real)

En la practica, casi siempre tienes muchos mas datos que coeficientes. En ese caso, ninguna curva puede pasar exactamente por todos los puntos. El objetivo cambia: buscas la curva que se acerque lo mas posible a todos los puntos en conjunto.

Para medir que tan bien se ajusta la curva, calculas el error en cada punto:

```
error_i = valor_real_i - valor_predicho_i
```

Luego sumas todos los errores al cuadrado (para que los errores positivos y negativos no se cancelen entre si):

```
S = error_1^2 + error_2^2 + ... + error_n^2
```

El objetivo es encontrar los coeficientes que hagan S lo mas pequeño posible. Para eso usas derivadas: la derivada de S respecto a cada coeficiente te dice como cambia el error cuando cambias ese coeficiente. En el punto minimo de cualquier funcion, la pendiente es cero, por lo que igualas cada derivada a cero. Eso genera un sistema de ecuaciones que se resuelve con eliminacion gaussiana. A este metodo se le llama minimos cuadrados.

---

## Cuantos coeficientes tiene un polinomio

La formula es simple:

```
numero de coeficientes = grado del polinomio + 1
```

Por ejemplo:
- Grado 1: 2 coeficientes (a0 y a1)
- Grado 2: 3 coeficientes (a0, a1, a2)
- Grado 3: 4 coeficientes (a0, a1, a2, a3)

Esto importa porque necesitas al menos tantos puntos como coeficientes para que el modelo tenga sentido.

---

## El dilema del grado

Elegir el grado del polinomio es la decision mas importante del proceso. Hay tres escenarios posibles:

### Subajuste

El modelo es demasiado simple para describir los datos. Ocurre cuando el grado es muy bajo. El error es alto tanto en los datos de entrenamiento como en datos nuevos.

### Buen ajuste

El modelo captura la tendencia real de los datos sin volverse innecesariamente complejo. El error es bajo y similar tanto en datos de entrenamiento como en datos nuevos.

### Sobreajuste

El modelo es tan complejo que memorizo los datos en lugar de aprender la tendencia. Ocurre cuando el grado es muy alto. El error en entrenamiento es muy bajo, pero el error en datos nuevos es alto. La curva zigzaguea entre los puntos de forma absurda.

---

## Como elegir el grado correcto

No existe una formula para saberlo de antemano. Se descubre experimentando con los siguientes pasos:

1. Grafica los datos y observa su forma general para tener una idea inicial del grado a probar.
2. Divide el dataset en dos partes: 80% para entrenamiento y 20% para prueba. Los datos de prueba se esconden hasta el final.
3. Entrena el modelo con distintos grados (1, 2, 3, 4...) y mide el error en ambas partes.
4. El grado correcto es el que produce el menor error en los datos de prueba.

La senal de sobreajuste es clara: el error de entrenamiento sigue bajando pero el error de prueba empieza a subir.

---

## Metricas de evaluacion

### R cuadrado (R2)

Mide que proporcion de la variacion en los datos explica el modelo. Va de 0 a 1. Un R2 de 0.90 significa que el modelo explica el 90% de la variacion. Cuanto mas cerca de 1, mejor.

### RMSE (Raiz del Error Cuadratico Medio)

Es el promedio de los errores de prediccion, en las mismas unidades que los datos originales. Si predices temperaturas y el RMSE es 2, el modelo se equivoca en promedio 2 grados. Cuanto mas pequeño, mejor.

---

## El dataset: Auto MPG

Para este ejercicio usaremos el dataset **Auto MPG**, disponible publicamente en el repositorio de UCI Machine Learning:

```
https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
```

Este dataset contiene informacion sobre el consumo de combustible de distintos automoviles de los anos 70 y 80. Tiene las siguientes columnas:

- `mpg`: millas por galon (consumo de combustible, variable objetivo)
- `cylinders`: numero de cilindros del motor
- `displacement`: desplazamiento del motor en pulgadas cubicas
- `horsepower`: caballos de fuerza
- `weight`: peso del vehiculo en libras
- `acceleration`: tiempo en segundos de 0 a 60 mph
- `model_year`: ano del modelo
- `origin`: origen del auto (1=americano, 2=europeo, 3=japones)
- `car_name`: nombre del auto

Para este ejercicio usaremos `horsepower` como variable de entrada (x) y `mpg` como variable de salida (y). La relacion entre ambas no es lineal: a medida que aumentan los caballos de fuerza, el consumo baja, pero no de forma constante. Una curva polinomial describe esta relacion mejor que una linea recta.

---

# Pasos de la Práctica

---

### Paso 1: Importar las bibliotecas necesarias

Antes de hacer cualquier cosa necesitamos cargar las herramientas que vamos a usar. Cada biblioteca tiene un proposito especifico:

- `pandas`: carga y manipula tablas de datos
- `numpy`: operaciones matematicas con arrays
- `matplotlib.pyplot`: hace graficas
- `PolynomialFeatures`: transforma x en [x, x^2, x^3...] segun el grado que elijas
- `LinearRegression`: encuentra los coeficientes optimos usando minimos cuadrados
- `train_test_split`: divide los datos en entrenamiento y prueba
- `r2_score` y `mean_squared_error`: miden que tan bueno es el modelo

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
```

---

### Paso 2: Cargar el dataset

El archivo no tiene encabezados, asi que debes asignarlos manualmente. Tambien tiene valores faltantes marcados con `?` en la columna horsepower, por lo que le indicamos a pandas que los trate como nulos desde el principio con `na_values='?'`.

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

columnas = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'model_year', 'origin', 'car_name']

df = pd.read_csv(url, sep='\s+', names=columnas, na_values='?')

df.head()
```

El resultado esperado es una tabla con 9 columnas donde cada fila es un auto. Si ves numeros en todas las columnas numericas, el archivo se cargo correctamente.

---

### Paso 3: Explorar y limpiar los datos

Antes de entrenar cualquier modelo, siempre debes entender bien los datos que tienes: cuantas filas hay, que tipos de datos tiene cada columna, y si hay valores faltantes que puedan causar errores.

```python
# tamaño del dataset: (filas, columnas)
print(df.shape)

# Tipo de dato de cada columna
print(df.dtypes)

# Cuantos valores nulos hay en cada columna
print(df.isnull().sum())
```

Cuando ejecutes esto veras que `horsepower` tiene 6 valores nulos. Los eliminamos:

```python
df = df.dropna()

# Verificamos que horsepower es numerica
df['horsepower'] = pd.to_numeric(df['horsepower'])

# Comprobamos el tamaño despues de limpiar
print(df.shape)
```

Antes de limpiar habia 398 filas. Despues quedan 392. Esas 6 filas eran los autos con horsepower desconocido.

---

### Paso 4: Visualizar la relacion entre horsepower y mpg

Antes de entrenar cualquier modelo, grafica los datos crudos. Esta es una de las practicas mas importantes en ciencia de datos: ver los datos antes de modelarlos. La forma visual te da informacion que los numeros solos no te dan.

```python
plt.figure(figsize=(8, 5))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.5, color='steelblue')
plt.xlabel('Caballos de fuerza (horsepower)')
plt.ylabel('Millas por galon (mpg)')
plt.title('Relacion entre potencia y consumo de combustible')
plt.grid(True)
plt.show()
```

Observa la forma de la nube de puntos. Baja rapidamente al principio y luego se aplana. Eso es una pista visual clara de que necesitas al menos un termino x^2: una linea recta no puede capturar esa curvatura.

---

### Paso 5: Preparar los datos para el modelo

Separamos la variable de entrada de la variable objetivo, y dividimos en entrenamiento y prueba. Los datos de prueba los guardamos a un lado y no los tocamos hasta la evaluacion final. Simulamos que esos datos son del futuro y todavia no los conocemos.

```python
# Variable de entrada (doble corchete para que quede como array 2D)
X = df[['horsepower']]

# Variable objetivo
y = df['mpg']

# Division 80% entrenamiento, 20% prueba
# random_state=42 garantiza que siempre obtengas la misma division
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Entrenamiento: {X_train.shape[0]} filas')
print(f'Prueba:        {X_test.shape[0]} filas')
```

Resultado esperado:
```
Entrenamiento: 313 filas
Prueba:        79 filas
```

---

### Paso 6: Entrenar modelos con distintos grados y comparar

Este es el paso central. Probamos grados del 1 al 5 y medimos el error de cada uno en entrenamiento y en prueba.

El truco de sklearn es que no existe una funcion unica llamada "regresion polinomial". En cambio, se usan dos pasos encadenados:

1. `PolynomialFeatures` transforma tu columna horsepower en multiples columnas: [hp, hp^2, hp^3...]. Esto convierte el problema en una regresion lineal con varias variables.
2. `LinearRegression` encuentra los coeficientes de esas columnas usando minimos cuadrados.

El resultado es exactamente una regresion polinomial.

```python
resultados = []

for grado in range(1, 6):

    # Paso A: crear el transformador para este grado
    poly = PolynomialFeatures(degree=grado, include_bias=False)

    # Paso B: transformar los datos
    # fit_transform sobre entrenamiento: aprende la escala y transforma
    # solo transform sobre prueba: usa la escala aprendida, no aprende nueva
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)

    # Paso C: entrenar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train_poly, y_train)

    # Paso D: predecir
    pred_train = modelo.predict(X_train_poly)
    pred_test  = modelo.predict(X_test_poly)

    # Paso E: calcular metricas
    r2_train   = r2_score(y_train, pred_train)
    r2_test    = r2_score(y_test, pred_test)
    rmse_train = mean_squared_error(y_train, pred_train) ** 0.5
    rmse_test  = mean_squared_error(y_test, pred_test) ** 0.5

    resultados.append({
        'grado'     : grado,
        'r2_train'  : round(r2_train, 4),
        'r2_test'   : round(r2_test, 4),
        'rmse_train': round(rmse_train, 4),
        'rmse_test' : round(rmse_test, 4)
    })

pd.DataFrame(resultados)
```

Resultado esperado aproximado:

```
   grado  r2_train  r2_test  rmse_train  rmse_test
0      1    0.6059   0.6080      4.7144     4.7362
1      2    0.6876   0.6923      4.1883     4.1848
2      3    0.7013   0.6989      4.1009     4.1281
3      4    0.7018   0.6901      4.0975     4.1920
4      5    0.7031   0.6730      4.0880     4.3012
```

Observa que el rmse_test es minimo en grado 2 o 3, y a partir de ahi empieza a subir aunque el rmse_train siga bajando. Esa es exactamente la señal de sobreajuste.

---

### Paso 7: Visualizar el modelo ganador

Con el grado que obtuvo el menor rmse_test, entrenamos el modelo final y dibujamos la curva sobre los puntos originales. Creamos un array de 300 valores de x espaciados uniformemente para que la curva se vea suave en lugar de angulosa.

```python
grado_ganador = 2  # ajusta segun tus resultados del paso anterior

poly_final   = PolynomialFeatures(degree=grado_ganador, include_bias=False)
X_train_poly = poly_final.fit_transform(X_train)
modelo_final = LinearRegression()
modelo_final.fit(X_train_poly, y_train)

# Array de x para dibujar la curva suavemente entre el minimo y maximo del dataset
x_curva      = np.linspace(X['horsepower'].min(), X['horsepower'].max(), 300).reshape(-1, 1)
x_curva_poly = poly_final.transform(x_curva)
y_curva      = modelo_final.predict(x_curva_poly)

# Grafica
plt.figure(figsize=(8, 5))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.4, color='gray', label='Datos reales')
plt.plot(x_curva, y_curva, color='steelblue', linewidth=2, label=f'Polinomio grado {grado_ganador}')
plt.xlabel('Caballos de fuerza (horsepower)')
plt.ylabel('Millas por galon (mpg)')
plt.title('Regresion polinomial ajustada')
plt.legend()
plt.grid(True)
plt.show()
```

La curva azul debe seguir la forma general de los puntos grises sin zigzaguear entre ellos.

---

### Paso 8: Interpretar los coeficientes

Los coeficientes del modelo te dicen exactamente como esta construida la curva matematicamente.

```python
print(f'Intercepto (a0): {modelo_final.intercept_:.4f}')
print(f'Coeficientes:    {modelo_final.coef_}')
```

Resultado esperado para grado 2:
```
Intercepto (a0): 56.9001
Coeficientes:    [-0.4662  0.0012]
```

Esto significa que la curva encontrada es:

```
mpg = 56.90 - 0.4662 * hp + 0.0012 * hp^2
```

El coeficiente de hp es negativo: a mas caballos de fuerza, el mpg baja. El coeficiente de hp^2 es positivo pero muy pequeno: amortigua la caida, haciendo que la curva se aplane en lugar de seguir bajando indefinidamente. Eso es exactamente lo que se ve en la grafica.

---

### Paso 9: Hacer predicciones con el modelo

Una vez entrenado el modelo, puedes predecir el mpg de cualquier auto con tal de que le des su horsepower. Recuerda que antes de predecir debes transformar el valor con PolynomialFeatures, exactamente igual que hiciste con los datos de entrenamiento.

```python
autos_nuevos = np.array([[70], [100], [150], [200]])
autos_poly   = poly_final.transform(autos_nuevos)
predicciones = modelo_final.predict(autos_poly)

for hp, mpg_pred in zip(autos_nuevos.flatten(), predicciones):
    print(f'Auto con {hp} hp  ->  {mpg_pred:.1f} mpg predichos')
```

Resultado esperado aproximado:
```
Auto con 70 hp   ->  30.2 mpg predichos
Auto con 100 hp  ->  24.5 mpg predichos
Auto con 150 hp  ->  17.8 mpg predichos
Auto con 200 hp  ->  14.3 mpg predichos
```

Los valores bajan a medida que sube la potencia, y la caida es cada vez mas lenta. Eso es consistente con lo que muestra la grafica y con la logica fisica: los autos muy potentes son ineficientes, pero pasar de 150 a 200 hp no penaliza tanto el consumo como pasar de 70 a 100 hp.

---

## Nota sobre el uso de sklearn

sklearn no tiene una funcion llamada "regresion polinomial" directamente. Lo que hace internamente es:

1. `PolynomialFeatures` transforma tu columna x en multiples columnas [x, x^2, x^3...]
2. `LinearRegression` encuentra los coeficientes de esas columnas usando minimos cuadrados

El resultado es exactamente una regresion polinomial. La separacion en dos pasos existe porque la misma logica de minimos cuadrados que sirve para una linea recta sirve tambien para una curva, siempre que primero transformes las entradas.