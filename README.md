# 📊 Regresión - Notebook de Machine Learning

Este notebook proporciona una guía completa sobre diferentes técnicas de regresión en Machine Learning, desde conceptos básicos hasta implementaciones avanzadas.

## 📋 Contenido

### 1. Regresión Lineal Simple
- Predicción de salarios basada en años de experiencia
- Dataset: [Salary Dataset](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression)
- Implementación con Scikit-learn
- Visualización de datos y línea de regresión

### 2. Métricas de Evaluación
Análisis detallado de las principales métricas para evaluar modelos de regresión:

- **MSE (Mean Squared Error)**: Error cuadrático medio
- **RMSE (Root Mean Squared Error)**: Raíz del error cuadrático medio
- **MAE (Mean Absolute Error)**: Error absoluto medio
- **Max Abs Error**: Error máximo absoluto
- **R² (Coefficient of Determination)**: Coeficiente de determinación
- **R² Ajustado**: R² ajustado para evitar sobreajuste
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual medio absoluto

### 3. Regresión Polinómica Simple
- Ajuste de modelos polinómicos a datos no lineales
- Comparación con regresión lineal simple

### 4. Regresión Lineal Múltiple
- Predicción con múltiples características
- Dataset: California Housing
- Análisis de relaciones multivariables

### 5. Regresión Polinómica Múltiple
- Modelos polinómicos con múltiples características
- Técnicas avanzadas de feature engineering

### 6. Train-Test Split
Dos enfoques comparados:

#### 🦾 Sin Train-Test Split
- Exploración inicial y análisis descriptivo
- Cuando el dataset es muy pequeño
- Casos de uso apropiados

#### 🦾 Con Train-Test Split
- Evaluación robusta del modelo
- Detección de overfitting
- Validación adecuada del rendimiento

## 🛠️ Requisitos

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from mpl_toolkits.mplot3d import Axes3D
```

## 📦 Instalación

```bash
pip install pandas numpy matplotlib scikit-learn
```

## 🚀 Uso

1. Clona o descarga el notebook
2. Asegúrate de tener todas las dependencias instaladas
3. Ejecuta las celdas secuencialmente para seguir el flujo completo
4. Los datasets se cargan automáticamente desde Kaggle y Scikit-learn

## 📊 Datasets Utilizados

1. **Salary Dataset**: Relación entre años de experiencia y salario
2. **California Housing**: Predicción de precios de viviendas con múltiples características

## 🎯 Objetivos de Aprendizaje

- Comprender los fundamentos de la regresión lineal y polinómica
- Aprender a evaluar modelos usando métricas apropiadas
- Diferenciar entre regresión simple y múltiple
- Implementar buenas prácticas con train-test split
- Visualizar resultados y realizar predicciones

## 📈 Visualizaciones

El notebook incluye múltiples visualizaciones:
- Gráficos de dispersión con líneas de regresión
- Comparaciones entre valores reales y predichos
- Gráficos 3D para regresión múltiple
- Análisis de errores y residuos

## 💡 Conceptos Clave

- **Slope/Coef**: Pendiente de la línea de regresión
- **Intercept/Bias**: Intercepto con el eje Y
- **Overfitting**: Sobreajuste del modelo
- **Feature Scaling**: Normalización de características
- **Polynomial Features**: Generación de características polinómicas
