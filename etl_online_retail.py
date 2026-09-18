"""
ETL con PySpark - Online Retail Dataset (UCI Machine Learning Repository, id=352)

Extraccion  : Python API de UCI (ucimlrepo) -> fetch_ucirepo(id=352)
Transformar : operaciones clave de Spark (select, filter, groupBy, join, windows...)
Cargar      : resultados en CSV (carpeta out/)

Ejecutar:
    python etl_online_retail.py
"""

import os
import sys
import glob
import shutil

from ucimlrepo import fetch_ucirepo

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ----------------------------------------------------------------------------
# 0. Configuracion de directorios
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Limpiar salidas anteriores para que solo queden los CSV de las 10 consultas
for _f in glob.glob(os.path.join(OUT_DIR, "*.csv")):
    try:
        os.remove(_f)
    except OSError as _e:
        print(f"[AVISO] No se pudo borrar {os.path.basename(_f)} "
              f"(archivo abierto en otra aplicacion): {_e}", file=sys.stderr)
for _d in glob.glob(os.path.join(OUT_DIR, "_tmp_*")):
    shutil.rmtree(_d, ignore_errors=True)

CSV_FINAL = os.path.join(DATA_DIR, "online_retail.csv")
ID_DATASET = 352  # Online Retail en UCI


def _configure_winutils():
    """Windows: Hadoop requiere winutils para escribir archivos."""
    if os.name != "nt" or os.environ.get("HADOOP_HOME"):
        return
    for cand in (os.path.join(BASE_DIR, "winutils"),
                 os.path.join(os.path.dirname(BASE_DIR), "pyspark_ejemplo", "winutils")):
        if os.path.exists(os.path.join(cand, "bin", "winutils.exe")):
            os.environ["HADOOP_HOME"] = cand
            os.environ["PATH"] = os.pathsep.join(
                [os.path.join(cand, "bin"), os.environ.get("PATH", "")])
            return


_configure_winutils()

# Windows: apuntar los python workers al interprete del entorno actual
import sys  # noqa: E402

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def write_csv(df, name):
    """Escribe un DataFrame a un unico CSV en out/ (coalesce a 1 particion)."""
    tmp_dir = os.path.join(OUT_DIR, "_tmp_" + name)
    df.coalesce(1).write.mode("overwrite").format("csv").option("header", True).save(tmp_dir)
    part_file = glob.glob(os.path.join(tmp_dir, "part-*.csv"))[0]
    final_file = os.path.join(OUT_DIR, name + ".csv")
    if os.path.exists(final_file):
        os.remove(final_file)
    os.replace(part_file, final_file)
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ----------------------------------------------------------------------------
# 1. EXTRACT - Descargar el dataset con la Python API de UCI
# ----------------------------------------------------------------------------
print("=" * 70)
print("ETL - Online Retail Dataset (UCI id=352)")
print("=" * 70)

retail = fetch_ucirepo(id=ID_DATASET)

print("\n[EXTRACT] Python API de UCI (ucimlrepo)")
print("Nombre        :", retail.metadata["name"])
print("Fuente        :", retail.metadata["repository_url"])
print("Instancias    :", retail.metadata["num_instances"])
print("DOI           :", retail.metadata["dataset_doi"])
print("Columnas API  :", retail.metadata["index_col"], "+", list(retail.data.features.columns))

# La libreria ucimlrepo separa las columnas indice (InvoiceNo, StockCode);
# se recuperan del data_url que la propia API proporciona (CSV original completo).
import pandas as pd  # noqa: E402

raw = pd.read_csv(retail.metadata["data_url"], low_memory=False)
print("DataFrame pandas (desde data_url):", raw.shape)
print("Columnas    :", list(raw.columns))

raw.to_csv(CSV_FINAL, index=False)
print("Archivo local:", CSV_FINAL)

# ----------------------------------------------------------------------------
# 2. Lectura de datos con Spark (spark.read.format("csv"))
# ----------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("ETL_Online_Retail") \
    .master("local[*]") \
    .config("spark.pyspark.python", sys.executable) \
    .config("spark.pyspark.driver.python", sys.executable) \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.sql.shuffle.partitions", "24") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("quote", '"') \
    .option("escape", '"') \
    .load(CSV_FINAL)

print("\n[LECTURA] spark.read.format('csv')")
print("Registros  :", df.count())
print("Esquema    :")
df.printSchema()
print("Muestra    :")
df.show(5, truncate=False)

# ----------------------------------------------------------------------------
# 3. Seleccion de columnas (select) y exploracion
# ----------------------------------------------------------------------------
print("\n[SELECCION] select()")
df_claves = df.select("InvoiceNo", "StockCode", "Description", "Quantity",
                      "UnitPrice", "CustomerID", "Country")
df_claves.show(5, truncate=False)

# ----------------------------------------------------------------------------
# 4. Limpieza y columnas derivadas (withColumn)
# ----------------------------------------------------------------------------
print("\n[TRANSFORM] withColumn()")
df = df \
    .withColumn("InvoiceDateTs", F.to_timestamp("InvoiceDate", "M/d/yyyy H:mm")) \
    .withColumn("revenue", F.round(F.col("Quantity") * F.col("UnitPrice"), 2)) \
    .withColumn("anio", F.year("InvoiceDateTs")) \
    .withColumn("mes", F.month("InvoiceDateTs")) \
    .withColumn("mes_nombre", F.date_format("InvoiceDateTs", "MMMM")) \
    .withColumn("year_month", F.date_format("InvoiceDateTs", "yyyyMM").cast("int")) \
    .withColumn("es_cancelacion", F.col("InvoiceNo").startswith("C")) \
    .withColumn("es_devolucion", F.col("Quantity") < 0) \
    .cache()

df.count()  # forzar materializacion del DataFrame en memoria

df.show(5, truncate=False)

# ----------------------------------------------------------------------------
# 4.1 Limpieza: eliminar filas duplicadas, valores nulos y no numericos
#     para que los calculos y consultas solo operen sobre datos validos
# ----------------------------------------------------------------------------

# Criterio de datos validos (indicado por el usuario / LibreOffice):
#   * Filas identicas (duplicados) eliminadas
#   * Misma clave (InvoiceNo, StockCode) repetida -> solo la primera
#   * Valores repetidos de una columna / repetidos (bajo el criterio numerico)
#   * Numericos (Quantity, UnitPrice, CustomerID): NO negativos, SIN letras y NO nulos
df_limpio = df.dropDuplicates() \
    .dropDuplicates(["InvoiceNo", "StockCode"]) \
    .filter(
        F.col("InvoiceNo").isNotNull() & (F.trim(F.col("InvoiceNo")) != "") &
        F.col("StockCode").isNotNull() & (F.trim(F.col("StockCode")) != "") &
        F.col("Description").isNotNull() &
        F.col("Quantity").isNotNull() &
        (F.col("Quantity") >= 0) &
        F.col("Quantity").cast("double").isNotNull() &
        F.col("UnitPrice").isNotNull() &
        (F.col("UnitPrice") >= 0) &
        F.col("UnitPrice").cast("double").isNotNull() &
        F.col("CustomerID").isNotNull() &
        F.col("CustomerID").cast("long").isNotNull() &
        (F.col("CustomerID").cast("long") >= 0)
    ) \
    .cache()

n_original = df.count()
n_limpio = df_limpio.count()
print(f"Filas originales  : {n_original}")
print(f"Filas limpias     : {n_limpio}")
print(f"Filas eliminadas  : {n_original - n_limpio} "
      "(duplicados, nulos, no numericos y negativos)")

# Ventas validas (sobre datos limpios): no canceladas, cantidad positiva,
# precio no negativo
df_ventas = df_limpio.where(~F.col("es_cancelacion")) \
              .where(F.col("Quantity") > 0) \
              .where(F.col("UnitPrice") >= 0)

print("\n[FILTRADO] filter()/where() -> ventas validas (sobre df_limpio)")
print("Filas con ventas validas:", df_ventas.count())
print("Facturas validas distintas:", df_ventas.select("InvoiceNo").distinct().count())

# ----------------------------------------------------------------------------
# 5. Validaciones de calidad de datos
#    (valores negativos, ids repetidos, datos iguales,
#     nulos y claves inconsistentes) - solo se imprimen, no generan CSV
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDACIONES DE CALIDAD DE DATOS (resultados en consola)")
print("=" * 70)

TOTAL_FILAS = df.count()
COLUMNAS = df.columns


def _filas(expr):
    return df.filter(expr).count()


# --- V1. Valores negativos en facturas ----------------------------------------
n_qty_neg = _filas(F.col("Quantity") < 0)
n_price_neg = _filas(F.col("UnitPrice") < 0)
sum_qty_neg = df.filter(F.col("Quantity") < 0) \
    .agg(F.sum("Quantity").alias("s")).collect()[0]["s"]
n_facturas_con_dev = df.filter(F.col("Quantity") < 0).select("InvoiceNo").distinct().count()
n_facturas_total_neg = df.groupBy("InvoiceNo") \
    .agg(F.sum("revenue").alias("t")).filter(F.col("t") < 0).count()

df_negativos = spark.createDataFrame(
    [("Renglones con Quantity < 0", float(n_qty_neg)),
     ("Renglones con UnitPrice < 0", float(n_price_neg)),
     ("Unidades devueltas (sum Quantity < 0)", float(sum_qty_neg)),
     ("Facturas con al menos un renglon negativo", float(n_facturas_con_dev)),
     ("Facturas con total de factura negativo", float(n_facturas_total_neg))],
    ["validacion", "valor"])
print("\n[V1] Valores negativos en facturas")
df_negativos.show(truncate=False)
print("Muestra de valores negativos:")
df.filter((F.col("Quantity") < 0) | (F.col("UnitPrice") < 0)) \
    .select("InvoiceNo", "StockCode", "Quantity", "UnitPrice", "CustomerID") \
    .limit(8).show(truncate=False)

# --- V2. IDs repetidos --------------------------------------------------------
ids_check = {}
for _colid in ["InvoiceNo", "StockCode", "CustomerID"]:
    tot = df.select(F.count(_colid)).collect()[0][0]
    dist = df.select(F.countDistinct(_colid)).collect()[0][0]
    ids_check[_colid] = (tot, dist, tot - dist)

df_ids = spark.createDataFrame(
    [("InvoiceNo", *ids_check["InvoiceNo"]),
     ("StockCode", *ids_check["StockCode"]),
     ("CustomerID", *ids_check["CustomerID"])],
    ["id_columna", "total", "distintos", "repetidos"])
print("\n[V2] IDs repetidos (repetidos = total - distintos)")
df_ids.show(truncate=False)
print("Top InvoiceNo con mas lineas:")
df.groupBy("InvoiceNo").agg(F.count("*").alias("n")).filter(F.col("n") > 1) \
    .orderBy(F.col("n").desc()).limit(5).show(truncate=False)
print("Top StockCode con mas apariciones:")
df.groupBy("StockCode").agg(F.count("*").alias("n")).filter(F.col("n") > 1) \
    .orderBy(F.col("n").desc()).limit(5).show(truncate=False)
print("Nota: varias lineas por factura es normal; el duplicado anomalo es la")
print("      misma clave (InvoiceNo, StockCode) repetida (ver V3).")

# --- V3. Datos iguales (filas totalmente iguales / duplicados) -----------------
n_filas_unicas = df.dropDuplicates().count()
n_dup_exactas = TOTAL_FILAS - n_filas_unicas
df_dup_grupos = df.groupBy(df.columns).agg(F.count("*").alias("n_repetidos")) \
    .filter(F.col("n_repetidos") > 1)
n_grupos_dup = df_dup_grupos.count()
filas_en_grupos_dup = df_dup_grupos.agg(F.sum("n_repetidos").alias("s")).collect()[0]["s"]

df_dup_clave = df.groupBy("InvoiceNo", "StockCode") \
    .agg(F.count("*").alias("n_repetidos")).filter(F.col("n_repetidos") > 1)
n_dup_clave = df_dup_clave.count()

df_datos_iguales = spark.createDataFrame(
    [("Filas totalmente identicas (duplicados exactos)", float(n_dup_exactas)),
     ("Grupos de filas identicas", float(n_grupos_dup)),
     ("Filas involucradas en grupos duplicados", float(filas_en_grupos_dup)),
     ("Duplicados por (InvoiceNo, StockCode)", float(n_dup_clave))],
    ["validacion", "valor"])
print("\n[V3] Datos iguales / duplicados")
df_datos_iguales.show(truncate=False)
print("Muestra de filas identicas repetidas:")
df_dup_grupos.orderBy(F.col("n_repetidos").desc()) \
    .select(*df.columns, "n_repetidos").limit(5).show(truncate=False)

# --- V5. Valores nulos por columna ---------------------------------------------
nulos_expr = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in COLUMNAS]
df_nulos_row = df.select(*nulos_expr).collect()[0]

df_nulos = spark.createDataFrame(
    [(c, float(df_nulos_row[c])) for c in COLUMNAS], ["columna", "nulos"]) \
    .withColumn("pct", F.round(F.col("nulos") / TOTAL_FILAS * 100, 2)) \
    .orderBy(F.col("nulos").desc())
print("\n[V5] Valores nulos por columna")
df_nulos.show(truncate=False)

# --- V6. Otras validaciones auxiliares -----------------------------------------
n_cant_zero = _filas(F.col("Quantity") == 0)
n_precio_zero = _filas(F.col("UnitPrice") == 0)
n_fechas_null = _filas(F.col("InvoiceDateTs").isNull())
n_desc_null = _filas(F.col("Description").isNull())
n_es_cancelacion = _filas(F.col("es_cancelacion"))

df_stock_invalido = df.filter(
    F.col("StockCode").isNull() |
    (F.trim(F.col("StockCode")) == "") |
    ~F.col("StockCode").rlike("^[A-Za-z0-9]+$"))
n_stock_invalido = df_stock_invalido.count()
n_codigos_invalidos = df_stock_invalido.select("StockCode").distinct().count()

df_desc_inc = df.where(F.col("Description").isNotNull()) \
    .groupBy("StockCode") \
    .agg(F.countDistinct("Description").alias("n_descripciones")) \
    .filter(F.col("n_descripciones") > 1)
n_stock_inconsistentes = df_desc_inc.count()

df_aux = spark.createDataFrame(
    [("Cantidad == 0", float(n_cant_zero)),
     ("UnitPrice == 0", float(n_precio_zero)),
     ("Fechas no parseadas", float(n_fechas_null)),
     ("Description faltante", float(n_desc_null)),
     ("Facturas canceladas (InvoiceNo C*)", float(n_es_cancelacion)),
     ("Filas con StockCode invalido", float(n_stock_invalido)),
     ("StockCodes invalidos (distintos)", float(n_codigos_invalidos)),
     ("StockCodes con descripcion inconsistente", float(n_stock_inconsistentes))],
    ["validacion", "valor"])
print("\n[V6] Otras validaciones auxiliares")
df_aux.show(truncate=False)

# --- V7. Resumen consolidado de validaciones ----------------------------------
df_resumen_valid = spark.createDataFrame([
    ("Negativos", "Renglones con Quantity < 0", float(n_qty_neg)),
    ("Negativos", "Renglones con UnitPrice < 0", float(n_price_neg)),
    ("Negativos", "Facturas con total negativo", float(n_facturas_total_neg)),
    ("Duplicados", "Filas duplicadas exactas", float(n_dup_exactas)),
    ("Duplicados", "Duplicados por (InvoiceNo, StockCode)", float(n_dup_clave)),
    ("IDs", "InvoiceNo con ocurrencias extra", float(ids_check["InvoiceNo"][2])),
    ("IDs", "StockCode con ocurrencias extra", float(ids_check["StockCode"][2])),
    ("Nulos", "Filas sin CustomerID", float(df_nulos_row["CustomerID"])),
    ("Nulos", "Description faltante", float(n_desc_null)),
    ("Nulos", "Fechas no parseadas", float(n_fechas_null)),
    ("Claves", "StockCodes invalidos (distintos)", float(n_codigos_invalidos)),
    ("Claves", "StockCodes con descripcion inconsistente", float(n_stock_inconsistentes)),
    ("Registros", "Cancelaciones (InvoiceNo C*)", float(n_es_cancelacion)),
], ["tipo", "validacion", "resultado"])

print("\n[V7] Resumen consolidado de validaciones")
df_resumen_valid.show(truncate=False)

# ----------------------------------------------------------------------------
# 6. Preguntas de negocio
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALISIS (agregaciones, agrupaciones, ordenamiento)")
print("=" * 70)

# --- Q1. Numero total de facturas -------------------------------------------
facturas_totales = df_limpio.select(F.countDistinct("InvoiceNo").alias("total")).collect()[0]["total"]
facturas_validas = df_ventas.select(F.countDistinct("InvoiceNo").alias("total")).collect()[0]["total"]

q1 = spark.createDataFrame(
    [("Facturas totales (distintas)", facturas_totales),
     ("Facturas de ventas validas (sin cancelaciones)", facturas_validas)],
    ["concepto", "total_facturas"])
write_csv(q1, "01_total_facturas")
print("\n--- Q1. Numero total de facturas ---")
q1.show(truncate=False)

# --- Q2. Numero de clientes unicos ------------------------------------------
clientes_unicos = df_limpio.select(F.countDistinct("CustomerID").alias("total")).collect()[0]["total"]

q2 = spark.createDataFrame([("Clientes unicos (con CustomerID)", clientes_unicos)],
                           ["concepto", "total_clientes"])
write_csv(q2, "02_clientes_unicos")
print("\n--- Q2. Numero de clientes unicos ---")
q2.show(truncate=False)

# --- Q3. Ingreso total (Quantity * UnitPrice) -------------------------------
ingreso_bruto = df_limpio.agg(F.sum("revenue").alias("total")).collect()[0]["total"]
ingreso_valido = df_ventas.agg(F.sum("revenue").alias("total")).collect()[0]["total"]

q3 = spark.createDataFrame(
    [("Ingreso bruto (datos limpios, incl. devoluciones)", ingreso_bruto),
     ("Ingreso ventas validas", ingreso_valido)],
    ["concepto", "ingreso_total"])
write_csv(q3, "03_ingreso_total")
print("\n--- Q3. Ingreso total (Quantity * UnitPrice) ---")
q3.show(truncate=False)

# --- Q4. Producto mas vendido en cantidad (sum, orderBy) --------------------
df_productos = df_ventas.groupBy("StockCode", "Description") \
    .agg(F.sum("Quantity").alias("total_cantidad"),
         F.sum("revenue").alias("total_ingreso"))
df_productos = df_productos.orderBy(F.col("total_cantidad").desc())

q4 = df_productos.limit(1)
write_csv(q4, "04_producto_mas_vendido")
print("\n--- Q4. Producto mas vendido en cantidad ---")
q4.show(truncate=False)

# --- Q5. Cliente con mayor volumen de compra en dinero -----------------------
df_clientes = df_ventas.where(F.col("CustomerID").isNotNull()) \
    .groupBy("CustomerID") \
    .agg(F.sum("revenue").alias("total_gasto"),
         F.countDistinct("InvoiceNo").alias("n_facturas"))
df_clientes = df_clientes.orderBy(F.col("total_gasto").desc())

q5 = df_clientes.limit(1)
write_csv(q5, "05_mejor_cliente")
print("\n--- Q5. Cliente con mayor volumen de compra en dinero ---")
q5.show(truncate=False)

# --- Q6. Top 5 paises que mas compran fuera de Reino Unido -------------------
df_paises = df_ventas.where(F.col("Country") != "United Kingdom") \
    .groupBy("Country") \
    .agg(F.sum("revenue").alias("total_ingreso"),
         F.sum("Quantity").alias("total_cantidad")) \
    .orderBy(F.col("total_ingreso").desc())

q6 = df_paises.limit(5)
write_csv(q6, "06_top5_paises_fuera_uk")
print("\n--- Q6. Top 5 paises fuera de Reino Unido ---")
q6.show(truncate=False)

# --- Q7. Ticket promedio por factura (avg) -----------------------------------
df_facturas = df_ventas.groupBy("InvoiceNo") \
    .agg(F.sum("revenue").alias("total_factura"),
         F.sum("Quantity").alias("items_factura"),
         F.countDistinct("StockCode").alias("n_productos"),
         F.first("CustomerID").alias("CustomerID"))

ticket_promedio = df_facturas.agg(F.avg("total_factura").alias("ticket_promedio")) \
                             .collect()[0]["ticket_promedio"]

q7 = spark.createDataFrame([("Ticket promedio por factura", ticket_promedio)],
                           ["concepto", "ticket_promedio"])
write_csv(q7, "07_ticket_promedio")
print("\n--- Q7. Ticket promedio por factura ---")
q7.show(truncate=False)

# --- Q8. Min, max y promedio de productos por factura ------------------------
res_prod = df_facturas.agg(
    F.min("n_productos").alias("min_productos"),
    F.max("n_productos").alias("max_productos"),
    F.avg("n_productos").alias("avg_productos"),
    F.count("InvoiceNo").alias("n_facturas")).collect()[0]

q8 = spark.createDataFrame(
    [("min_productos", float(res_prod["min_productos"])),
     ("max_productos", float(res_prod["max_productos"])),
     ("avg_productos", float(res_prod["avg_productos"])),
     ("n_facturas", float(res_prod["n_facturas"]))],
    ["metrica", "valor"])
write_csv(q8, "08_productos_por_factura_resumen")
print("\n--- Q8. Mínimo, máximo y promedio de productos por factura ---")
q8.show(truncate=False)

# --- Q9. Mes del ano con mas ventas -------------------------------------------
df_meses = df_ventas.groupBy("year_month", "anio", "mes", "mes_nombre") \
    .agg(F.sum("revenue").alias("total_ingreso")) \
    .orderBy(F.col("total_ingreso").desc())

q9_mes = df_meses.limit(1)
write_csv(q9_mes, "09_mes_mas_ventas")
print("\n--- Q9. Ventas por mes (top = mes con mas ventas) ---")
df_meses.show(12, truncate=False)

# --- Q10. Porcentaje de facturas con devoluciones -----------------------------
facturas_con_dev = df_limpio.where(F.col("Quantity") < 0) \
                     .select(F.countDistinct("InvoiceNo").alias("total")).collect()[0]["total"]

pct_dev = (facturas_con_dev / facturas_totales) * 100

q10 = spark.createDataFrame(
    [("Total de facturas", float(facturas_totales)),
     ("Facturas con devoluciones (Quantity < 0)", float(facturas_con_dev)),
     ("Porcentaje de facturas con devoluciones (%)", round(pct_dev, 2))],
    ["concepto", "valor"])
write_csv(q10, "10_pct_facturas_devoluciones")
print("\n--- Q10. Porcentaje de facturas con devoluciones ---")
q10.show(truncate=False)

# ----------------------------------------------------------------------------
# 6. Funciones de ventana (window): rank() y row_number()
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FUNCIONES DE VENTANA: rank() y row_number()")
print("=" * 70)

# Ranking de productos por cantidad vendida
win_prod = Window.orderBy(F.col("total_cantidad").desc())
df_ranking_prod = df_productos \
    .withColumn("rank", F.rank().over(win_prod)) \
    .withColumn("row_number", F.row_number().over(win_prod)) \
    .orderBy("rank")
print("\nTop productos por cantidad vendida (rank):")
df_ranking_prod.limit(10).show(truncate=False)

# Ranking de clientes por dinero gastado
win_cli = Window.orderBy(F.col("total_gasto").desc())
df_ranking_clientes = df_clientes \
    .withColumn("rank", F.rank().over(win_cli)) \
    .withColumn("row_number", F.row_number().over(win_cli)) \
    .orderBy("rank")
print("\nTop clientes por gasto (rank):")
df_ranking_clientes.limit(5).show(truncate=False)

# ----------------------------------------------------------------------------
# 7. Union de DataFrames (join)
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("JOIN: facturas X clientes")
print("=" * 70)

# Resumen por cliente (dimension cliente)
dim_clientes = df_clientes.selectExpr("CustomerID as CustomerID",
                                      "total_gasto as gasto_total_cliente",
                                      "n_facturas as facturas_cliente")

# Facturas de ventas validas: unir con dimension de cliente por CustomerID
df_facturas_clientes = df_facturas.select("InvoiceNo", "CustomerID", "total_factura") \
    .join(dim_clientes, on="CustomerID", how="left") \
    .orderBy(F.col("total_factura").desc())

print("Ejemplo de join facturas-clientes:")
df_facturas_clientes.show(5, truncate=False)

# ----------------------------------------------------------------------------
# 8. Resumen final de respuestas y LOAD (write.csv)
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN DE RESPUESTAS")
print("=" * 70)

resumen = spark.createDataFrame([
    ("1", "Numero total de facturas", str(facturas_totales)),
    ("2", "Numero de clientes unicos", str(clientes_unicos)),
    ("3", "Ingreso total (Quantity * UnitPrice)", str(round(ingreso_bruto, 2))),
    ("4", "Producto mas vendido en cantidad", str(q4.collect()[0]["Description"])),
    ("5", "Cliente con mayor volumen de compra en dinero", str(q5.collect()[0]["CustomerID"])),
    ("6", "Top 5 paises que mas compran fuera de Reino Unido (Top1)",
     str(q6.collect()[0]["Country"])),
    ("7", "Ticket promedio por factura", str(round(ticket_promedio, 2))),
    ("8", "Productos por factura (min/max/prom)",
     f"{res_prod['min_productos']} / {res_prod['max_productos']} / {round(float(res_prod['avg_productos']), 2)}"),
    ("9", "Mes del ano con mas ventas", str(q9_mes.collect()[0]["year_month"])),
    ("10", "Porcentaje de facturas con devoluciones (%)", str(round(pct_dev, 2))),
], ["pregunta", "descripcion", "respuesta"])

resumen.show(truncate=False)

print("\nArchivos CSV generados en:", OUT_DIR)
for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.csv"))):
    print("  -", os.path.basename(f))

spark.stop()