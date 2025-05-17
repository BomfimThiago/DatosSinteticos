# Proyecto: Generación de datos sintéticos para modelos de predicción de *bank default analysis*

---

## 🔹 FASE 1 — Limpieza y preparación del dataset original

**Objetivos:**

- Asegurar calidad del dataset real antes de usarlo como base del generador.

**Acciones:**

- Cargar y explorar el dataset (valores nulos, *outliers*, cardinalidad).
- Normalizar tipos de variables (convertir categóricas, binarizar si es necesario).
- Analizar balance de clases de la variable objetivo (moroso).
- Guardar versión limpia y documentada del dataset.

**Herramientas:**

- `pandas`, `seaborn`, `matplotlib`, `sklearn.preprocessing`

---

## 🔹 FASE 2 — Generación de datos sintéticos

**Objetivos:**

- Entrenar modelos generativos tabulares sobre el dataset limpio.
- Generar un dataset sintético equivalente.

**Modelos recomendados:**

- **CTGAN** → Muy sólido para mezcla de variables categóricas y numéricas.
- **TVAE** → *Autoencoder* variacional adaptado a tablas.
- **TabDDPM** → Difusor de última generación, ideal para mayor fidelidad.
- *(Opcional)* **GaussianCopula** → Modelo estadístico base para comparación.

**Acciones:**

- Crear script modular para entrenar cada generador.
- Configurar entrenamiento y *sampling* con cada modelo.
- Generar un dataset sintético de tamaño igual al original.
- Guardar los datasets sintéticos con trazabilidad del modelo usado.

**Herramientas:**

- `SDV`: `CTGANSynthesizer`, `TVAESynthesizer`, `GaussianCopulaSynthesizer`
- `TabDDPM`: para evaluar modelo *SOTA*

---

## 🔹 FASE 3 — Evaluación de calidad de los datos sintéticos

**Objetivos:**

- Comparar realismo, utilidad y preservación de estructuras estadísticas.

**Métricas a utilizar:**

- **Realismo estadístico**
  - Divergencia de Jensen-Shannon (JS)
  - Kolmogorov-Smirnov (KS test por variable)
  - Histogramas y correlaciones cruzadas

- **Preservación de correlaciones**
  - Pearson/Spearman
  - Matriz de correlación real vs sintética

- **Utilidad para ML**
  - Experimento TSTR (*Train on Synthetic, Test on Real*)
  - Comparar AUC, *accuracy*, F1 entre entrenamientos con datos reales y sintéticos

- **Privacidad (si aplica)**
  - *Nearest-neighbor overlap*
  - *Membership inference* (si se requiere robustez de privacidad)

**Herramientas:**

- `SDMetrics`
- `sklearn.metrics`, `xgboost`
- `matplotlib`, `seaborn`

---

## 🔹 FASE 4 — Reporte y visualización

**Objetivos:**

- Documentar resultados comparativos.
- Justificar elección del mejor modelo generador.
- Visualizar distribución y resultados predictivos.

**Acciones:**

- Comparar visualmente las distribuciones reales vs sintéticas.
- Crear matriz resumen de métricas por modelo.
- Redactar informe con conclusiones robustas: realismo, utilidad y viabilidad.

---

# Instrucciones para rodar la aplicación

1. Primero crea un ambiente virtual, puedes usar el nombre que quieras:  
   ```bash python -m venv venv```

2. Después activa el ambiente virtual:
   ```source venv/bin/activate```
    Para Windows haz:
    
    `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`
    `.\venv\Scripts\Activate.ps1`

3. Con el ambiente virtual activado, instala las librerías:
```pip install -r requirements.txt```

