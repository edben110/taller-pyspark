# ETL PySpark — Online Retail Dataset (UCI id=352)

Pipeline ETL en PySpark para el dataset *Online Retail* de la UCI
(Machine Learning Repository, id=352). El script normaliza los datos con
reglas de calidad por columna, genera un **segundo CSV limpio** y ejecuta
**10 consultas de negocio** sobre ese CSV procesado, además de ejercitar
funciones de ventana (`rank`, `row_number`) y un `join`.

---

## 1. Estructura del proyecto

```
online_retail/
├── etl_online_retail_md.py   # Script ETL (formato de fecha mes/dia: M/d/yyyy H:mm)
├── README.md
├── data/
│   ├── online_retail.csv            # CSV ORIGINAL (no se modifica)
│   └── online_retail_clean_md.csv   # SEGUNDO CSV procesado (generado por Fase A)
└── out/
    ├── 01_total_facturas.csv        # Resultados de las 10 consultas
    ├── 02_clientes_unicos.csv
    └── ... (hasta 10_pct_facturas_devoluciones.csv)
```

> Nota: existe una variante del script con formato de fecha día/mes
> (`d/M/yyyy H:mm`). En este repositorio se encuentra actualmente la variante
> con formato mes/día (`M/d/yyyy H:mm`).

---

## 2. Metodología

El proceso sigue el modelo **ELT ampliado** dividido en dos fases:

### FASE A — Normalizador por columna (Transform)
1. **EXTRACT**: lee el CSV local `data/online_retail.csv`. Si no existe, lo
   descarga automáticamente desde UCI con la API `ucimlrepo`
   (`fetch_ucirepo(id=352)`). El original **nunca se modifica**.
2. **TRANSFORM**: aplica reglas de validación por columna. Cada columna genera
   un **flag booleano `_ok_<columna>`** que indica si la fila cumple la regla.
3. **Integridad**: añade filtros de consistencia a nivel de *producto*
   (mismo precio) y de *factura* (mismo cliente y misma fecha).
4. **LOAD**: escribe el DataFrame resultante en un **segundo CSV procesado**
   (`data/online_retail_clean_md.csv`), limpio y listo para análisis.

### FASE B — Análisis (Load/Consume)
1. Lee sólamente el CSV procesado de la Fase A.
2. Crea columnas derivadas: `revenue = Quantity * UnitPrice`, año, mes,
   `year_month`, banderas de cancelación y devolución.
3. Ejecuta la **10 consultas de negocio**, las funciones de **ventana** y un
   **join**, guardando cada resultado en un CSV dentro de `out/`.

---

## 3. Filtros y validaciones aplicadas (Fase A)

Cada regla marca la fila como inválida si no se cumple; una fila solo pasa si
**cumple TODAS** las reglas.

### 3.1 Por columna

| Columna       | Regla aplicada                                                                 | Implementación                                                        |
|---------------|--------------------------------------------------------------------------------|----------------------------------------------------------------------|
| `InvoiceNo`   | Entero de **exactamente 6 dígitos**. Si empieza con `C` es cancelación → inválido (no es entero de 6 dígitos). Sin negativos, letras, símbolos ni nulos. Se permiten duplicados (varios artículos por factura). | `_valid_id("InvoiceNo", 6)` con regex `^[0-9]{6}$` |
| `StockCode`   | Entero de **exactamente 5 dígitos** (un id por producto).                 | `_valid_id("StockCode", 5)` con regex `^[0-9]{5}$` |
| `Description` | Sin caracteres especiales, salvo `_` o `-`. No nula ni vacía. **No debe empezar con un número (dígito)**. | regex `^[A-Za-z0-9 _\-]+$` + `NOT ^[0-9]` |
| `Quantity`    | Numérica, sin nulos ni letras y **mayor o igual a 0**.                     | regex `^[0-9]+(\.[0-9]+)?$` y `cast(double) >= 0`  |
| `UnitPrice`   | Numérica, sin nulos ni letras y **mayor o igual a 0**.                     | regex `^[0-9]+(\.[0-9]+)?$` y `cast(double) >= 0`  |
| `InvoiceDate` | Fecha válida en formato **`M/d/yyyy H:mm`**, dentro del rango **2010-12-01 a 2011-12-09**, con **hora entre 0 y 24** y **minutos entre 0 y 59**. | regex formato + `try_to_timestamp` + rango + validación hora/minuto |
| `CustomerID`  | Entero de **exactamente 5 dígitos** (un id asignado a un solo cliente).     | `_valid_id("CustomerID", 5)` con regex `^[0-9]{5}$` |
| `Country`     | País **especificado**: no nulo, no vacío y distinto de `Unspecified`.         | `_ok_country` con `lower(country) != "unspecified"` |

### 3.2 Consistencia de precio por producto
Si una misma `Description` (nombre de producto) aparece con **2 o más
`UnitPrice` distintos**, se descartan **todas** las filas de ese producto
(join `left_anti` por descripción).

### 3.3 Consistencia de factura
Si dos o más filas comparten `InvoiceNo` pero tienen **`CustomerID` distinto**
o **fecha distinta**, se descartan **todas** las filas de esa factura
(agrupación con `countDistinct` y join `left_anti`).

### 3.4 Deduplicación
Se aplica `dropDuplicates()` para eliminar filas idénticas repetidas.

### 3.5 Detalle técnico importante (Spark 4 / ANSI)
El script usa `F.try_to_timestamp` (y no `F.to_timestamp`) porque en Spark 4.x
el **modo ANSI está activado por defecto**: `to_timestamp` *lanza una
excepción* (`SparkDateTimeException`) ante una fecha malformada y aborta todo
el job. `try_to_timestamp` **devuelve `NULL`** y la fila se descarta
silenciosamente, que es el comportamiento deseado del normalizador.

Además, el formato debe pasarse con `F.lit("M/d/yyyy H:mm")`: en esta versión
de PySpark, `try_to_timestamp`/`to_timestamp` tratan todos los argumentos como
columnas; sin `F.lit` Spark intenta resolver el formato como una columna y
falla con `[UNRESOLVED_COLUMN]`.

---

## 4. Análisis de negocio (Fase B) — 10 consultas

1. **Q1. Número total de facturas** — `countDistinct(InvoiceNo)`.
2. **Q2. Clientes únicos** — `countDistinct(CustomerID)`.
3. **Q3. Ingreso total** — `sum(Quantity * UnitPrice)`.
4. **Q4. Producto más vendido en cantidad** — `groupBy + sum + orderBy desc`.
5. **Q5. Cliente con mayor volumen de compra** — `groupBy CustomerID + sum + orderBy desc`.
6. **Q6. Top 5 países fuera de Reino Unido** — `filter Country != 'United Kingdom' + agrupación + orderBy ingreso desc`.
7. **Q7. Ticket promedio por factura** — `avg(sum(revenue) por InvoiceNo)`.
8. **Q8. Min/max/promedio de productos por factura** — agregación sobre `n_productos`.
9. **Q9. Mes del año con más ventas** — `groupBy year_month + orderBy ingreso desc`.
10. **Q10. Porcentaje de facturas con devoluciones** — facturas con `Quantity < 0` entre el total.

### Funciones de ventana
- **`rank()`** y **`row_number()`** sobre `Window.orderBy(total_cantidad.desc())`
  (top productos) y sobre `Window.orderBy(total_gasto.desc())` (top clientes).

### Join
- **Facturas × Clientes**: `df_facturas` con `dim_clientes` por `CustomerID`
  (join `left`), ordenado por total de factura descendente.

### Ventas válidas (Fase B)
Se filtra además sobre el CSV procesado: no canceladas, `Quantity > 0` y
`UnitPrice >= 0`.

---

## 5. Salidas (LOAD)

Los resultados se escriben como **un único CSV** por consulta en `out/`:

```
01_total_facturas.csv              02_clientes_unicos.csv
03_ingreso_total.csv               04_producto_mas_vendido.csv
05_mejor_cliente.csv               06_top5_paises_fuera_uk.csv
07_ticket_promedio.csv             08_productos_por_factura_resumen.csv
09_mes_mas_ventas.csv              10_pct_facturas_devoluciones.csv
```

La escritura usa `df.coalesce(1)` + mover el `part-*.csv` a un archivo único
(helper `write_single_csv`), y se limpian los `out/*.csv` previos al inicio.

---

## 6. Requisitos y ejecución

- Python 3.10+ con **PySpark 4.x** (probado con PySpark 4.2.0).
- En Windows: el script configura `HADOOP_HOME` (`winutils`) automáticamente y
  apunta los workers de Python al intérprete del entorno actual.

```bash
python etl_online_retail_md.py
```

Al terminar, muestra el **resumen de respuestas** y la lista de CSV generados.

---

## 7. Notas finales

- El CSV original `data/online_retail.csv` permanece intacto; toda la limpieza
  se materializa en `data/online_retail_clean_md.csv`.
- Las reglas de longitud (6/5/5 dígitos) descartan automáticamente ids
  alfanuméricos (p. ej. `StockCode` como `85123A`) y cancelaciones `C...`.
- Si el CSV con el que se califica contiene fechas en formato **día/mes**, usar
  la variante que parsea con `d/M/yyyy H:mm`; si contiene fechas **mes/día**
  (como el UCI original), usar esta variante `M/d/yyyy H:mm`.