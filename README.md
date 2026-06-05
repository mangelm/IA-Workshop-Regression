# Proyecto 1: Regresión Lineal Simple
### Dataset: Salary Dataset — Predicción de Salario por Años de Experiencia

---

## Objetivo del Proyecto

En este proyecto aprenderás a construir un modelo de **Regresión Lineal Simple** desde cero usando Python. Utilizaremos un dataset real que relaciona los **años de experiencia** de un empleado con su **salario anual**. Al finalizar, serás capaz de:

- Comprender qué es y cómo funciona la regresión lineal simple.
- Preprocesar datos para un modelo de Machine Learning.
- Entrenar, evaluar y ajustar un modelo de regresión.
- Interpretar métricas de evaluación.

---

## ¿Qué es la Regresión Lineal Simple?

La **Regresión Lineal Simple** es un método estadístico que modela la relación entre **una variable de entrada (X)** y **una variable de salida (y)** mediante una línea recta. La fórmula es:

```
ŷ = β₀ + β₁ · X
```

Donde:
- `ŷ` → Valor predicho (salario estimado)
- `β₀` → Intercepto (valor de y cuando X = 0)
- `β₁` → Pendiente (cuánto cambia y por cada unidad que aumenta X)
- `X` → Variable independiente (años de experiencia)

> **Intuición:** Imagina dibujar la línea que "mejor se ajusta" a una nube de puntos en un gráfico de dispersión. Eso es exactamente lo que hace este algoritmo.

---

## Dataset

**Fuente:** [Salary Dataset — Simple Linear Regression (Kaggle)](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression)

| Columna | Descripción |
|---|---|
| `YearsExperience` | Años de experiencia laboral del empleado |
| `Salary` | Salario anual en USD |

---

## Paso 0 — Preparar el Entorno

Antes de comenzar, asegúrate de tener instaladas las librerías necesarias.

```python
# Ejecuta esta celda si no tienes las librerías instaladas
# !pip install pandas numpy matplotlib seaborn scikit-learn
```

```python
# ─── Importaciones ───────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
print("Librerías cargadas correctamente")
```

---

## Paso 1 — Cargar y Explorar los Datos

Carga el archivo CSV descargado de Kaggle. El archivo se llama `Salary_dataset.csv`.

```python
# Carga el dataset
df = pd.read_csv("Salary_dataset.csv")

# Vistazo inicial
print("Primeras 5 filas del dataset:")
df.head()
```

```python
# Información general del dataset
df.shape
```

```python
df.info()
```

```python
df.isnull().sum()
```

```python
# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
df.describe().round(2)
```

> **¿Qué observar?**
> - `count`: número de registros — verifica que sean 30.
> - `mean`: promedio — ¿cuánto gana en promedio alguien con experiencia promedio?
> - `min` / `max`: rangos de los datos.

---

## Paso 2 — Análisis Exploratorio de Datos (EDA)

Antes de entrenar cualquier modelo, debemos entender visualmente nuestros datos.

```python
# ─── 2.1 Verificar valores nulos ─────────────────────────────────────────────
print("Valores nulos por columna:")
print(df.isnull().sum())
```

> **Valor esperado:** 0 valores nulos. Si hay nulos, habrá que tratarlos antes de continuar.

```python
# ─── 2.2 Subplot con 2 histogramas ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['YearsExperience'], bins=10, color='steelblue', edgecolor='white')
axes[0].set_title('Distribución: Años de Experiencia')
axes[0].set_xlabel('Años')
axes[0].set_ylabel('Frecuencia')

axes[1].hist(df['Salary'], bins=10, color='coral', edgecolor='white')
axes[1].set_title('Distribución: Salario')
axes[1].set_xlabel('Salario (USD)')
axes[1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()
```

```python
# ─── 2.3 scatter plot de YearsExperience (eje X) vs Salary (eje Y) ───────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(df['YearsExperience'], df['Salary'], color='steelblue', s=80, alpha=0.8, edgecolors='white')
plt.title('Salario vs. Años de Experiencia', fontsize=14)
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario (USD)')
plt.tight_layout()
plt.show()
```

> **¿Qué observar en el scatter plot?**
> - Si los puntos forman una **línea ascendente**, hay una relación lineal positiva → ¡ideal para regresión lineal!
> - Si los puntos están dispersos sin patrón, la regresión lineal podría no ser el mejor modelo.

```python
# ─── 2.4 Correlación de Pearson ──────────────────────────────────────────────
correlacion = df['YearsExperience'].corr(df['Salary'])
print(f" Correlación de Pearson (X, y): {correlacion:.4f}")
```

>  **Interpretación de la correlación:**
> | Valor | Interpretación |
> |---|---|
> | 0.9 – 1.0 | Correlación muy fuerte positiva  |
> | 0.7 – 0.9 | Correlación fuerte positiva |
> | 0.4 – 0.7 | Correlación moderada |
> | 0.0 – 0.4 | Correlación débil |
>
> Para este dataset, esperamos un valor **> 0.95** — excelente para regresión lineal.

---

##  Paso 3 — Preparación de los Datos

```python
# ─── 3.1 Separar variables ───────────────────────────────────────────────────
# X: variable independiente (entrada)
# y: variable dependiente (salida a predecir)

X = df[['YearsExperience']]   # DataFrame 2D — sklearn lo requiere así
y = df['Salary']               # Series 1D

print(f" Forma de X: {X.shape}")
print(f" Forma de y: {y.shape}")
```

>  **Nota importante:** Usamos `df[['YearsExperience']]` (doble corchete) para que X sea un DataFrame 2D. Scikit-learn espera matrices 2D como entrada, no vectores 1D.

```python
# ─── 3.2 División en conjuntos de entrenamiento y prueba ──────────────────────
# 80% para entrenar, 20% para evaluar
# random_state=42 garantiza reproducibilidad (siempre el mismo split)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f" Datos de entrenamiento: {X_train.shape[0]} muestras")
print(f" Datos de prueba:        {X_test.shape[0]} muestras")
```

>  **¿Por qué dividir los datos?**
> - **Entrenamiento (80%):** El modelo aprende de estos datos.
> - **Prueba (20%):** Evaluamos el modelo con datos que NUNCA ha visto. Así sabemos si realmente aprendió o solo memorizó.
>
> Si evaluáramos con los mismos datos de entrenamiento, podríamos creer que el modelo es excelente cuando en realidad solo memorizó.

---

##  Paso 4 — Entrenar el Modelo

```python
# ─── 4.1 Crear e instanciar el modelo ────────────────────────────────────────
modelo = LinearRegression()

# El método .fit() encuentra los valores óptimos de β₀ y β₁
modelo.fit(X_train, y_train)

print(" Modelo entrenado exitosamente")
print("─" * 40)
print(f" Intercepto  (β₀): {modelo.intercept_:,.2f}")
print(f" Pendiente   (β₁): {modelo.coef_[0]:,.2f}")
```

>  **Interpretación de los coeficientes:**
> - **β₀ (intercepto):** Es el salario base teórico cuando los años de experiencia son 0. Puede ser un valor negativo o no significativo en la práctica, pero es necesario matemáticamente.
> - **β₁ (pendiente):** Por cada año adicional de experiencia, el salario aumenta en β₁ dólares. Por ejemplo, si β₁ = 9,450, entonces cada año extra de experiencia equivale a ~$9,450 más de salario.

```python
# ─── 4.3 Visualizar la línea de regresión con un scatter plot ────────────────────────────────────
plt.figure(figsize=(8, 5))

# Datos de entrenamiento
plt.scatter(X_train, y_train, color='steelblue', s=80, alpha=0.8,
            edgecolors='white', label='Datos entrenamiento')
# Datos de prueba
plt.scatter(X_test, y_test, color='coral', s=80, alpha=0.8,
            edgecolors='white', label='Datos prueba')

# Línea de regresión
X_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(X_linea, modelo.predict(X_linea), color='black', linewidth=2,
         label='Línea de regresión')

plt.title('Regresión Lineal Simple: Salario vs. Experiencia', fontsize=13)
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario (USD)')
plt.legend()
plt.tight_layout()
plt.show()
```

---

# 5 — Evaluación del Modelo

Una vez entrenado el modelo, necesitamos medir qué tan bien hace predicciones.

```python
# ─── 5.1 Generar predicciones sobre datos de prueba ──────────────────────────
y_pred = modelo.predict(X_test)

# ─── 5.2 Tabla comparativa: valores reales vs predichos ──────────────────────
comparacion = pd.DataFrame({
    'Años_Exp': X_test['YearsExperience'].values,
    'Salario_Real': y_test.values,
    'Salario_Predicho': y_pred.round(2),
    'Error': (y_test.values - y_pred).round(2)
})
print("📋 Comparación Real vs Predicho:")
print(comparacion.to_string(index=False))
```

```python
# ─── 5.3 Calcular métricas de evaluación ─────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("=" * 45)
pr        MÉTRICAS DE EVALUACIÓN")
print("=" * 45)
print(f"  MAE  (Error Absoluto Medio):    ${mae:>12,.2f}")
print(f"  MSE  (Error Cuadrático Medio):  ${mse:>12,.2f}")
print(f"  RMSE (Raíz Error Cuadrático):   ${rmse:>12,.2f}")
print(f"  R²   (Coeficiente Determinación): {r2:>10.4f}")
print("=" * 45)
```

> ###  Guía de Interpretación de Métricas
>
> | Métrica | Fórmula | Qué mide | Valor ideal |
> |---|---|---|---|
> | **MAE** | `mean(|y - ŷ|)` | Error promedio en las mismas unidades de y | Lo más bajo posible |
> | **MSE** | `mean((y - ŷ)²)` | Penaliza errores grandes más que MSE | Lo más bajo posible |
> | **RMSE** | `√MSE` | Error en las mismas unidades (más interpretable que MSE) | Lo más bajo posible |
> | **R²** | `1 - SS_res/SS_tot` | % de varianza explicada por el modelo | Cercano a 1.0 |
>
> ####  Valores esperados para este dataset:
> - **R² ≥ 0.90** → El modelo explica al menos el 90% de la variación en salarios. Para este dataset con alta correlación, esperamos **R² ≈ 0.95 o mayor**.
> - **MAE < $5,000** → En promedio, el modelo se equivoca menos de $5,000 por predicción. Aceptable para salarios en el rango de $37k–$122k.
> - **RMSE** → Debe estar en un rango similar al MAE. Si RMSE >> MAE, hay algunos outliers que el modelo no predice bien.

---

## 📉 Paso 6 — Análisis de Residuos

Los **residuos** son la diferencia entre el valor real y el predicho: `e = y - ŷ`. Analizar los residuos nos permite verificar si el modelo es adecuado.

```python
residuos = y_test.values - y_pred

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ─── 6.1 Residuos vs Valores Predichos ───────────────────────────────────────
axes[0].scatter(y_pred, residuos, color='steelblue', s=70, alpha=0.8, edgecolors='white')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title('Residuos vs. Valores Predichos')
axes[0].set_xlabel('Valores Predichos (ŷ)')
axes[0].set_ylabel('Residuos (y - ŷ)')

# ─── 6.2 Distribución de residuos ────────────────────────────────────────────
axes[1].hist(residuos, bins=8, color='coral', edgecolor='white')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Distribución de Residuos')
axes[1].set_xlabel('Residuo')
axes[1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()
```

>  **¿Qué esperar en los gráficos de residuos?**
>
> **Gráfico izquierdo (Residuos vs Predichos):**
> -  **Bueno:** Los puntos distribuidos aleatoriamente alrededor de la línea y = 0, sin patrón visible.
> -  **Problema:** Si ves una curva o patrón en forma de U, los datos tienen relación no lineal y deberías usar regresión polinómica.
>
> **Gráfico derecho (Distribución de residuos):**
> -  **Bueno:** Distribución aproximadamente simétrica alrededor de 0 (campana de Gauss).
> -  **Problema:** Si está muy sesgada, el modelo tiene un error sistemático.

---

## 🔧 Paso 7 — ¿Qué hacer si el modelo no predice bien?

Si los resultados no son satisfactorios (por ejemplo, R² < 0.80), considera estas estrategias:

```python
# ─── Estrategia 1: Revisar si hay outliers que afectan el modelo ─────────────
# Identificar predicciones con error > 2 desviaciones estándar
umbral = 2 * residuos.std()
outliers_idx = np.where(np.abs(residuos) > umbral)[0]

print(f"  Predicciones con error grande (>{umbral:.0f} USD):")
if len(outliers_idx) > 0:
    print(comparacion.iloc[outliers_idx])
else:
    print("   No se detectaron outliers significativos")
```

```python
# ─── Estrategia 2: Evaluar también sobre datos de entrenamiento ──────────────
# Si R²_train >> R²_test → hay overfitting (pero en regresión lineal simple es poco probable)
y_pred_train = modelo.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred)

print(f"R² entrenamiento: {r2_train:.4f}")
print(f"R² prueba:        {r2_test:.4f}")
print(f"Diferencia:       {abs(r2_train - r2_test):.4f}")

if abs(r2_train - r2_test) > 0.10:
    print("  Posible overfitting — gran diferencia entre entrenamiento y prueba")
else:
    print(" Modelo generaliza bien — diferencia pequeña entre conjuntos")
```

```python
# ─── Estrategia 3: Validación cruzada (K-Fold) ───────────────────────────────
# Más robusto que un único train/test split, especialmente con datasets pequeños
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2')

print(" Validación Cruzada (5-Fold):")
print(f"  R² por fold: {cv_scores.round(4)}")
print(f"  R² promedio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

>  **Validación Cruzada (K-Fold):** Divide los datos en K partes iguales. En cada iteración, usa K-1 partes para entrenar y 1 para evaluar. Esto da una estimación más confiable del rendimiento real del modelo, especialmente cuando el dataset es pequeño (como nuestros 30 registros).

---

##  Paso 8 — Hacer Predicciones Nuevas

```python
# ─── Predecir el salario para distintos años de experiencia ──────────────────
nuevos_datos = pd.DataFrame({'YearsExperience': [1.5, 5.0, 8.0, 10.0, 12.0]})
predicciones = modelo.predict(nuevos_datos)

print(" Predicciones de salario:")
print("-" * 40)
for exp, sal in zip(nuevos_datos['YearsExperience'], predicciones):
    print(f"  {exp:>5.1f} años de experiencia → ${sal:>10,.2f} USD")
```

---

##  Paso 9 — Resumen y Conclusiones

```python
# ─── Resumen completo del modelo ─────────────────────────────────────────────
print("=" * 50)
print("        RESUMEN DEL MODELO")
print("=" * 50)
print(f"\n  Ecuación: Salario = {modelo.intercept_:,.2f} + {modelo.coef_[0]:,.2f} × Años")
print(f"\n  Métricas sobre conjunto de prueba:")
print(f"    • R²   = {r2:.4f}  {' Excelente' if r2 >= 0.90 else ' Mejorable'}")
print(f"    • MAE  = ${mae:,.2f}")
print(f"    • RMSE = ${rmse:,.2f}")
print(f"\n  Validación cruzada (5-Fold):")
print(f"    • R² promedio = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 50)
```

> ###  Checklist de un buen modelo de Regresión Lineal Simple
>
> - [ ] R² ≥ 0.85 en datos de prueba
> - [ ] MAE razonable respecto al rango de y
> - [ ] Diferencia R²_train − R²_test < 0.10
> - [ ] Residuos distribuidos aleatoriamente (sin patrón)
> - [ ] Residuos con distribución aproximadamente normal
> - [ ] R² de validación cruzada consistente con R² de prueba

---

##  Conceptos Clave Aprendidos

| Concepto | Descripción |
|---|---|
| **Regresión Lineal** | Modelo que ajusta una línea recta entre X e y |
| **β₀ y β₁** | Intercepto y pendiente de la recta |
| **Train/Test Split** | División de datos para evaluación honesta |
| **R²** | Porcentaje de varianza explicada (0 a 1) |
| **MAE / RMSE** | Magnitud del error en unidades originales |
| **Análisis de residuos** | Diagnóstico de supuestos del modelo |
| **Validación cruzada** | Evaluación robusta con múltiples splits |

---

*Proyecto 1 de 5 — Serie: Modelos de Regresión con Python*