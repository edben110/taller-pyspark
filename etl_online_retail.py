"""
ETL con PySpark - Online Retail Dataset (UCI Machine Learning Repository, id=352)

Extraccion  : usa el CSV local (data/online_retail.csv) si existe; si no, lo
              descarga con la Python API de UCI (ucimlrepo, fetch_ucirepo(id=352))
Transformar : NORMALIZADOR POR COLUMNA + operaciones clave de Spark
              (select, filter, groupBy, join, windows...)
Cargar      : resultados en CSV (carpeta out/)

Reglas del normalizador (una por columna):
  * InvoiceNo : solo ids de factura -> SOLO digitos, sin negativos, sin letras,
                sin caracteres especiales y sin nulos. Se permiten ids repetidos
                (varios articulos pertenecen a una misma factura).
  * StockCode : misma regla que InvoiceNo (solo ids de producto validos).
  * Description: sin caracteres especiales, salvo "_" o "-".
  * Quantity  : sin nulos, sin letras y no menor a 0 (>= 0).
  * UnitPrice : sin nulos, sin letras y no menor a 0 (>= 0).
  * InvoiceDate: fecha correcta y con el formato "M/d/yyyy H:mm" (usa "/").
  * CustomerID: misma regla que InvoiceNo (id numerico valido).
  * Consistencia de factura: si hay 2+ filas con la misma InvoiceNo pero con
    CustomerID diferente o fecha diferente, TODAS las filas de esa factura se
    ignoran en la lectura de datos.

Ejecutar:
    python etl_online_retail.py
"""

import os
import sys
import glob
import shutil

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
# 1. EXTRACT - CSV local si existe; si no, descargar con la Python API de UCI
# ----------------------------------------------------------------------------
print("=" * 70)
print("ETL - Online Retail Dataset (UCI id=352)")
print("=" * 70)

if not os.path.exists(CSV_FINAL):
    print("\n[EXTRACT] No existe CSV local -> descargando desde UCI (ucimlrepo)")
    from ucimlrepo import fetch_ucirepo  # noqa: E402

    retail = fetch_ucirepo(id=ID_DATASET)
    print("Nombre        :", retail.metadata["name"])
    print("Fuente        :", retail.metadata["repository_url"])
    print("Instancias    :", retail.metadata["num_instances"])
    print("DOI           :", retail.metadata["dataset_doi"])

    import pandas as pd  # noqa: E402

    raw = pd.read_csv(retail.metadata["data_url"], low_memory=False)
    print("DataFrame pandas (desde data_url):", raw.shape)
    raw.to_csv(CSV_FINAL, index=False)
else:
    print(f"\n[EXTRACT] Usando CSV local existente: {CSV_FINAL}")

print("Archivo local :", CSV_FINAL)
print("Tamanio       :", os.path.getsize(CSV_FINAL), "bytes")

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
# 3. NORMALIZADOR / VALIDADOR POR COLUMNA
#    Cada regla genera una columna booleana _ok_<columna>. Al final solo se
#    conservan las filas que cumplen TODAS las reglas.
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("NORMALIZADOR POR COLUMNA (validaciones individuales)")
print("=" * 70)


def _valid_id(col_name):
    """Id numerico: solo digitos, sin negativos, sin letras, sin caracteres
    especiales y sin nulos. Tolera el sufijo '.0' (p.ej. '17850.0')."""
    t = F.trim(F.col(col_name).cast("string"))
    num = F.regexp_replace(t, r"\.0+$", "")
    return (num.isNotNull() & (num != "") & num.rlike(r"^[0-9]+$") &
            (F.col(col_name).cast("long") >= 0))


def _valid_num(col_name):
    """Numerico: sin nulos, sin letras y no menor a 0."""
    t = F.trim(F.col(col_name).cast("string"))
    return (t.isNotNull() & (t != "") &
            t.rlike(r"^[0-9]+(\.[0-9]+)?$") &
            (F.col(col_name).cast("double") >= 0))


def _valid_date():
    """Fecha correcta con el formato M/d/yyyy H:mm (usa '/')."""
    raw = F.col("InvoiceDate").cast("string")
    ts = F.to_timestamp(raw, "M/d/yyyy H:mm")
    return raw.contains("/") & ts.isNotNull(), ts


_desc_trim = F.trim(F.col("Description").cast("string"))
_ok_description = (_desc_trim.isNotNull() &
                   (_desc_trim != "") &
                   _desc_trim.rlike(r"^[A-Za-z0-9 _\-]+$"))

_ok_invoice = _valid_id("InvoiceNo")
_ok_stock = _valid_id("StockCode")
_ok_customer = _valid_id("CustomerID")
_ok_quantity = _valid_num("Quantity")
_ok_price = _valid_num("UnitPrice")
_ok_date, _ts = _valid_date()

df_norm = df \
    .withColumn("_ok_invoice", _ok_invoice) \
    .withColumn("_ok_stock", _ok_stock) \
    .withColumn("_ok_description", _ok_description) \
    .withColumn("_ok_quantity", _ok_quantity) \
    .withColumn("_ok_price", _ok_price) \
    .withColumn("_ok_date", _ok_date) \
    .withColumn("_ok_customer", _ok_customer) \
    .withColumn("_date_raw", F.col("InvoiceDate").cast("string")) \
    .withColumn("_desc_trim", _desc_trim) \
    .withColumn("InvoiceDateTs", _ts)

# Reporte de cuantas filas incumplen cada regla individual
print("\nIncidencia por regla (filas que NO cumplen cada columna):")
df_norm.select(
    F.sum(F.when(~F.col("_ok_invoice"), 1).otherwise(0)).alias("InvoiceNo invalido"),
    F.sum(F.when(~F.col("_ok_stock"), 1).otherwise(0)).alias("StockCode invalido"),
    F.sum(F.when(~F.col("_ok_description"), 1).otherwise(0)).alias("Description invalida"),
    F.sum(F.when(~F.col("_ok_quantity"), 1).otherwise(0)).alias("Quantity invalida"),
    F.sum(F.when(~F.col("_ok_price"), 1).otherwise(0)).alias("UnitPrice invalido"),
    F.sum(F.when(~F.col("_ok_date"), 1).otherwise(0)).alias("InvoiceDate invalida"),
    F.sum(F.when(~F.col("_ok_customer"), 1).otherwise(0)).alias("CustomerID invalido"),
).show(truncate=False)

# 3.1 Fila valida = cumple TODAS las reglas de columna
df_fila_valida = df_norm.filter(
    F.col("_ok_invoice") &
    F.col("_ok_stock") &
    F.col("_ok_description") &
    F.col("_ok_quantity") &
    F.col("_ok_price") &
    F.col("_ok_date") &
    F.col("_ok_customer")
)

print("\nTras validar TODAS las columnas:")
print("  Filas totales de la fuente:", df_norm.count())
print("  Filas que cumplen todas    :", df_fila_valida.count())

# ----------------------------------------------------------------------------
# 4. Consistencia por factura: si 2+ filas con la misma InvoiceNo tienen
#    CustomerID diferente o fecha diferente, la factura se ignora por completo.
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CONSISTENCIA DE FACTURA (InvoiceNo -> mismo CustomerID y misma fecha)")
print("=" * 70)

df_inconsistentes = df_fila_valida.groupBy("InvoiceNo").agg(
    F.countDistinct("CustomerID").alias("n_clientes"),
    F.countDistinct("InvoiceDateTs").alias("n_fechas")) \
    .filter((F.col("n_clientes") > 1) | (F.col("n_fechas") > 1))

n_facturas_inconsistentes = df_inconsistentes.count()
print("Facturas inconsistentes (a descartar):", n_facturas_inconsistentes)

df_limpio = df_fila_valida \
    .join(df_inconsistentes.select("InvoiceNo"), on="InvoiceNo", how="left_anti") \
    .dropDuplicates() \
    .drop("_ok_invoice", "_ok_stock", "_ok_description", "_ok_quantity",
          "_ok_price", "_ok_date", "_ok_customer", "_date_raw", "_desc_trim")

n_original = df_norm.count()
n_limpio = df_limpio.count()
print(f"Filas originales  : {n_original}")
print(f"Filas limpias     : {n_limpio}")
print(f"Filas eliminadas  : {n_original - n_limpio}")

# ----------------------------------------------------------------------------
# 5. Seleccion de columnas (select) y columnas derivadas (withColumn)
# ----------------------------------------------------------------------------
print("\n[SELECCION] select() + columnas derivadas")
df_limpio = df_limpio \
    .withColumn("revenue", F.round(F.col("Quantity").cast("double") *
                                   F.col("UnitPrice").cast("double"), 2)) \
    .withColumn("anio", F.year("InvoiceDateTs")) \
    .withColumn("mes", F.month("InvoiceDateTs")) \
    .withColumn("mes_nombre", F.date_format("InvoiceDateTs", "MMMM")) \
    .withColumn("year_month", F.date_format("InvoiceDateTs", "yyyyMM").cast("int")) \
    .withColumn("es_cancelacion", F.col("InvoiceNo").cast("string").startswith("C")) \
    .withColumn("es_devolucion", F.col("Quantity").cast("double") < 0) \
    .cache()

df_limpio.count()  # forzar materializacion del DataFrame en memoria
df_limpio.show(5, truncate=False)

# Ventas validas (sobre datos limpios): no canceladas, cantidad positiva,
# precio no negativo
df_ventas = df_limpio.where(~F.col("es_cancelacion")) \
                     .where(F.col("Quantity").cast("double") > 0) \
                     .where(F.col("UnitPrice").cast("double") >= 0)

print("\n[FILTRADO] ventas validas (sobre df_limpio)")
print("Filas con ventas validas:", df_ventas.count())
print("Facturas validas distintas:", df_ventas.select("InvoiceNo").distinct().count())

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
    .agg(F.sum("Quantity").cast("double").alias("total_cantidad"),
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
         F.sum("Quantity").cast("double").alias("total_cantidad")) \
    .orderBy(F.col("total_ingreso").desc())

q6 = df_paises.limit(5)
write_csv(q6, "06_top5_paises_fuera_uk")
print("\n--- Q6. Top 5 paises fuera de Reino Unido ---")
q6.show(truncate=False)

# --- Q7. Ticket promedio por factura (avg) -----------------------------------
df_facturas = df_ventas.groupBy("InvoiceNo") \
    .agg(F.sum("revenue").alias("total_factura"),
         F.sum("Quantity").cast("double").alias("items_factura"),
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
facturas_con_dev = df_limpio.where(F.col("Quantity").cast("double") < 0) \
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
# 7. Funciones de ventana (window): rank() y row_number()
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
# 8. Union de DataFrames (join)
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
# 9. Resumen final de respuestas y LOAD (write.csv)
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