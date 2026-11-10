"""Build the self-contained experiment notebook from the current DT snapshot."""

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
original = json.loads((ROOT / "decision-tree.ipynb").read_text(encoding="utf-8"))
original_cells = {cell.get("id"): cell for cell in original["cells"]}
snapshot_ids = ["node-structure", "impurity-functions", "training-helpers", "c45-algorithm"]
snapshot_hash = hashlib.sha256(
    "".join("".join(original_cells[cell_id]["source"]) for cell_id in snapshot_ids).encode("utf-8")
).hexdigest()


def md(cell_id, source):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": dedent(source).lstrip("\n").splitlines(keepends=True)}


def code(cell_id, source):
    return {"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {},
            "outputs": [], "source": dedent(source).lstrip("\n").splitlines(keepends=True)}


cells = [
    md("presentacion", f"""
    # Pruebas de Decision Tree y Random Forest

    Este notebook ejecuta las pruebas acordadas para Heart Disease de UCI: las clases
    originales **0, 1, 2, 3 y 4**, filas completas y los mismos datos para ambos modelos.
    El Decision Tree es una copia exacta de las cuatro celdas del algoritmo de
    `decision-tree.ipynb` (SHA-256 de las fuentes: `{snapshot_hash}`). El Random Forest
    utiliza `RandomForestClassifier` directamente. El notebook de RF local contiene un
    ejemplo categórico. En `main` remoto verificado (`ff9469c`) ya utiliza UCI Heart
    Disease con 100 árboles y criterio Gini predeterminado; esta comparación usa
    entropía, según la configuración acordada. Las fuentes del DT remoto coinciden
    con la copia local.

    **Uso en Colab:** ejecutá las celdas en orden. Por defecto no se ejecutan las
    pruebas extensas ni la evaluación final. En la sección 6 activá cada operación
    cuando corresponda. La descarga del dataset requiere conexión a internet.

    ## Criterios metodológicos

    - Reserva estratificada del 20 % para la evaluación final; con el 80 % restante,
      validación cruzada estratificada de 5 particiones compartidas por los modelos.
    - La selección usa el **promedio del F1 macro por partición**. Entre configuraciones
      del mismo modelo que estén a 1 punto porcentual o menos del mejor promedio,
      se elige la de menor número total promedio de nodos y, ante empate, el mayor F1.
    - En el DT, `min_gain` controla la ganancia **local**. En RF,
      `min_impurity_decrease` considera una reducción ponderada por el tamaño del nodo:
      un mismo número no significa una restricción idéntica.
    - **Pendientes:** Gain Ratio y poda del DT; no están implementados en el algoritmo
      revisado y no se simulan en este notebook.
    - La salida guardada del DT original ya muestra un resultado para una partición de
      prueba con la misma semilla 55. La reserva se mantiene separada en esta ejecución,
      pero no debe presentarse como completamente inédita si se utilizó en decisiones
      anteriores. Las clases minoritarias son pequeñas; interpretar las diferencias con cautela.
    """),
    md("dependencias-texto", """
    ## 1. Dependencias

    En Colab se instalan solamente las bibliotecas que falten. También se muestran
    sus versiones para que las pruebas sean reproducibles.
    """),
    code("dependencias", """
    import importlib.util
    import subprocess
    import sys

    paquetes = {
        "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
        "matplotlib": "matplotlib", "ucimlrepo": "ucimlrepo",
    }
    faltantes = [paquete for modulo, paquete in paquetes.items()
                 if importlib.util.find_spec(modulo) is None]
    if faltantes:
        print("Instalando dependencias faltantes:", ", ".join(faltantes))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *faltantes])
    """),
    code("importaciones", """
    import hashlib
    import json
    import time
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import sklearn
    from IPython.display import display
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | "
          f"NumPy {np.__version__} | scikit-learn {sklearn.__version__}")
    """),
    md("datos-texto", """
    ## 2. Dataset y particiones

    El preprocesamiento reproduce las versiones revisadas: se combinan atributos y
    clase, se eliminan filas con algún faltante y se conserva `num` sin agrupar.
    Los índices de la reserva no se entregan a la validación ni a la selección.
    No se ajusta ningún imputador ni transformador usando datos de prueba.
    """),
    code("datos-funciones", """
    CLASES = [0, 1, 2, 3, 4]
    SEMILLA_RESERVA = 55
    SEMILLA_FOLDS = 42
    SEMILLA_RF = 42
    N_PARTICIONES = 5
    PROPORCION_RESERVA = 0.2


    def cargar_heart_disease():
        # Devuelve los registros completos del dataset 45, con target al final.
        from ucimlrepo import fetch_ucirepo

        conjunto = fetch_ucirepo(id=45)
        datos = pd.concat([conjunto.data.features, conjunto.data.targets], axis=1)
        datos = datos.rename(columns={"num": "target"}).dropna(axis=0, how="any")
        if "target" not in datos.columns:
            raise ValueError("No se encontró la columna objetivo 'num' en Heart Disease.")
        datos = datos[[col for col in datos.columns if col != "target"] + ["target"]]
        if set(datos["target"].unique()) != set(CLASES):
            raise ValueError("El dataset debe conservar las cinco clases originales 0 a 4.")
        datos.attrs["ids_uci"] = datos.index.tolist()
        return datos.reset_index(drop=True)


    def preparar_particiones(datos):
        # Reserva el 20 % y crea cinco particiones solo dentro del desarrollo.
        X = datos.drop(columns="target").reset_index(drop=True)
        y = datos["target"].reset_index(drop=True)
        X_des, X_reserva, y_des, y_reserva = train_test_split(
            X, y, test_size=PROPORCION_RESERVA, stratify=y, random_state=SEMILLA_RESERVA
        )
        ids = np.asarray(datos.attrs.get("ids_uci", datos.index.tolist()))
        ids_des, ids_reserva = ids[X_des.index], ids[X_reserva.index]
        X_des, X_reserva = X_des.reset_index(drop=True), X_reserva.reset_index(drop=True)
        X_des.attrs["ids_uci"], X_reserva.attrs["ids_uci"] = ids_des.tolist(), ids_reserva.tolist()
        y_des, y_reserva = y_des.reset_index(drop=True), y_reserva.reset_index(drop=True)
        if y_des.value_counts().min() < N_PARTICIONES:
            raise ValueError("No hay suficientes casos por clase para cinco particiones estratificadas.")
        cv = StratifiedKFold(n_splits=N_PARTICIONES, shuffle=True, random_state=SEMILLA_FOLDS)
        folds = [(ent, val) for ent, val in cv.split(X_des, y_des)]
        return X_des, X_reserva, y_des, y_reserva, folds


    def huella_datos(datos):
        # Incluye orden, columnas, valores e identidad original, sin depender del índice reiniciado.
        ids = datos.attrs.get("ids_uci", datos.index.tolist())
        contenido = datos.to_csv(index=False, float_format="%.17g", lineterminator="\\n")
        contenido += json.dumps([str(valor) for valor in ids], ensure_ascii=False)
        return hashlib.sha256(contenido.encode("utf-8")).hexdigest()


    def huella_experimento(datos, X_des, y_des, X_reserva, y_reserva, folds):
        # Vincula la selección con los datos y las asignaciones efectivamente evaluadas.
        partes = [huella_datos(datos)]
        for X, y in ((X_des, y_des), (X_reserva, y_reserva)):
            marco = X.copy()
            marco["target"] = np.asarray(y)
            partes.append(huella_datos(marco))
        partes.append([[[int(i) for i in ent], [int(i) for i in val]] for ent, val in folds])
        return hashlib.sha256(json.dumps(partes, sort_keys=True).encode("utf-8")).hexdigest()


    def exportar_particiones(carpeta, X_des, X_reserva, y_des, y_reserva, folds):
        filas = []
        for conjunto, X, y in (("desarrollo", X_des, y_des), ("reserva", X_reserva, y_reserva)):
            for indice, (id_uci, clase) in enumerate(zip(X.attrs["ids_uci"], y)):
                filas.append({"conjunto": conjunto, "indice": indice, "id_uci": id_uci,
                              "clase": int(clase)})
        pd.DataFrame(filas).to_csv(carpeta / "asignaciones_reserva.csv", index=False)
        filas = []
        for numero, (ent, val) in enumerate(folds, start=1):
            for rol, indices in (("entrenamiento", ent), ("validacion", val)):
                for indice in indices:
                    filas.append({"particion": numero, "rol": rol, "indice_desarrollo": int(indice),
                                  "id_uci": X_des.attrs["ids_uci"][indice]})
        pd.DataFrame(filas).to_csv(carpeta / "asignaciones_folds.csv", index=False)
    """),
    code("datos-carga", """
    datos = cargar_heart_disease()
    X_desarrollo, X_final, y_desarrollo, y_final, particiones = preparar_particiones(datos)
    print(f"Registros completos: {len(datos)} | Atributos: {X_desarrollo.shape[1]}")
    print(f"Desarrollo: {len(y_desarrollo)} | Reserva final: {len(y_final)}")
    distribucion = pd.DataFrame({
        "Datos completos": datos["target"].value_counts().reindex(CLASES, fill_value=0),
        "Desarrollo": y_desarrollo.value_counts().reindex(CLASES, fill_value=0),
        "Reserva final": y_final.value_counts().reindex(CLASES, fill_value=0),
    }).rename_axis("Clase")
    display(distribucion)
    """),
    md("dt-texto", """
    ## 3. Decision Tree del equipo

    Las siguientes cuatro celdas reproducen exactamente las definiciones de
    `decision-tree.ipynb`, sin ejecutar sus celdas de datos, entrenamiento ni prueba.
    Para evitar divergencias, si cambia el algoritmo original se debe actualizar esta
    copia y volver a verificar su procedencia. `threshold` dentro de un nodo es un
    corte numérico; el umbral mínimo de ganancia se configura con `min_gain`.
    """),
]

for source_id in snapshot_ids:
    source = "".join(original_cells[source_id]["source"])
    cells.append(code("snapshot-" + source_id, source))

cells.append(code("procedencia", f"HUELLA_DT = {snapshot_hash!r}\n"))

cells.extend([
    md("plan-texto", """
    ## 4. Configuración base y plan de pruebas

    Cada experimento vuelve a la base y modifica solo lo indicado. La tabla del plan
    puede repetir una configuración que aparece en más de una prueba; esa configuración
    se entrena una sola vez por partición y su resultado se reutiliza.
    La base del DT usa `1e-12`; `0` se evalúa aparte y permite divisiones con ganancia cero.
    """),
    code("plan-funciones", """
    BASE_DT = {"min_gain": 1e-12, "max_depth": None, "min_samples_split": 2}
    BASE_RF = {
        "criterion": "entropy", "n_estimators": 100, "max_depth": None,
        "min_samples_split": 2, "min_impurity_decrease": 0.0,
        "max_features": "sqrt", "bootstrap": True, "random_state": SEMILLA_RF,
    }
    MARGEN_F1_MACRO = 0.01  # Un punto porcentual para considerar alternativas similares.


    def identificador(modelo, parametros):
        descripcion = json.dumps({"modelo": modelo, "parametros": parametros},
                                 sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(descripcion.encode("utf-8")).hexdigest()[:16]


    def construir_plan():
        # Crea la tabla visible y un catálogo de configuraciones únicas.
        catalogo, filas = {}, []

        def agregar(prueba, modelo, parametro, valor, cambios):
            base = BASE_DT if modelo == "DT" else BASE_RF
            parametros = {**base, **cambios}
            clave = identificador(modelo, parametros)
            catalogo.setdefault(clave, {"modelo": modelo, "parametros": parametros})
            filas.append({"prueba": prueba, "modelo": modelo, "parametro": parametro,
                          "valor": "sin límite" if valor is None else str(valor),
                          "config_id": clave})

        for modelo in ("DT", "RF"):
            agregar("Referencia", modelo, "configuración base", "base", {})
        for modelo in ("DT", "RF"):
            nombre = "min_gain" if modelo == "DT" else "min_impurity_decrease"
            valores = [1e-12, 0.0, 0.001, 0.005, 0.01, 0.05] if modelo == "DT" else [0.0, 0.001, 0.005, 0.01, 0.05]
            for valor in valores:
                agregar("Ganancia mínima", modelo, nombre, valor, {nombre: valor})
        for modelo in ("DT", "RF"):
            for valor in [1, 2, 4, 6, 10, None]:
                agregar("Profundidad máxima", modelo, "max_depth", valor, {"max_depth": valor})
        for modelo in ("DT", "RF"):
            for valor in [2, 10, 25, 50, 100]:
                agregar("Mínimo para dividir", modelo, "min_samples_split", valor,
                        {"min_samples_split": valor})
        for modelo in ("DT", "RF"):
            for profundidad in [4, 6]:
                for minimo in [2, 10, 50]:
                    agregar("Profundidad + mínimo", modelo, "max_depth + min_samples_split",
                            f"{profundidad} + {minimo}",
                            {"max_depth": profundidad, "min_samples_split": minimo})
        for valor in [10, 50, 100, 200]:
            agregar("Cantidad de árboles", "RF", "n_estimators", valor,
                    {"n_estimators": valor})
        return pd.DataFrame(filas), catalogo
    """),
    code("plan-listado", """
    plan, catalogo = construir_plan()
    print(f"{len(catalogo)} configuraciones únicas × {N_PARTICIONES} particiones "
          f"= {len(catalogo) * N_PARTICIONES} entrenamientos de validación.")
    display(plan)
    """),
    md("evaluacion-texto", """
    ## 5. Funciones de entrenamiento y evaluación

    Cada entrenamiento usa únicamente las filas de entrenamiento de su partición.
    El tiempo mide el entrenamiento, sin incluir la preparación de filas ni la predicción.
    En DT la complejidad es la profundidad y los nodos del árbol; en RF se registran
    la profundidad y nodos promedio por árbol, y el total de nodos del bosque.
    El recall se calcula para cada una de las cinco clases, con `zero_division=0`.
    Accuracy, F1 y recall se muestran entre 0 y 1: por ejemplo, 0.80 equivale al 80 %.
    """),
    code("evaluacion-funciones", """
    def dimensiones_dt(nodo):
        # Cuenta nodos y profundidad real, con raíz en profundidad cero.
        if not nodo.children:
            return 1, 0
        hijos = [dimensiones_dt(hijo) for hijo in nodo.children.values()]
        return 1 + sum(n for n, _ in hijos), 1 + max(prof for _, prof in hijos)


    def entrenar_modelo(nombre, X_ent, y_ent, parametros):
        # Entrena el DT del equipo o el RF de sklearn y mide solo el entrenamiento.
        X_ent = np.asarray(X_ent)
        y_ent = np.asarray(y_ent)
        if nombre == "DT":
            filas = [list(fila) + [clase] for fila, clase in zip(X_ent.tolist(), y_ent.tolist())]
            atributos = list(range(X_ent.shape[1]))
            inicio = time.perf_counter()
            modelo = decision_tree(filas, atributos, **parametros)
        elif nombre == "RF":
            modelo = RandomForestClassifier(**parametros)
            inicio = time.perf_counter()
            modelo.fit(X_ent, y_ent)
        else:
            raise ValueError(f"Modelo desconocido: {nombre}")
        return modelo, time.perf_counter() - inicio


    def predecir_modelo(nombre, modelo, X):
        X = np.asarray(X)
        if nombre == "DT":
            return np.asarray([predict_one(modelo, fila) for fila in X.tolist()])
        if nombre == "RF":
            return modelo.predict(X)
        raise ValueError(f"Modelo desconocido: {nombre}")


    def complejidad_modelo(nombre, modelo):
        if nombre == "DT":
            nodos, profundidad = dimensiones_dt(modelo)
            return {"arboles": 1, "profundidad_media": float(profundidad),
                    "nodos_medios": float(nodos), "nodos_totales": int(nodos)}
        arboles = modelo.estimators_
        nodos = [arbol.tree_.node_count for arbol in arboles]
        profundidades = [arbol.get_depth() for arbol in arboles]
        return {"arboles": len(arboles), "profundidad_media": float(np.mean(profundidades)),
                "nodos_medios": float(np.mean(nodos)), "nodos_totales": int(sum(nodos))}


    def evaluar_configuracion(config_id, configuracion, X_des, y_des, folds):
        # Evalúa una configuración solo en desarrollo y conserva predicciones OOF.
        nombre, parametros = configuracion["modelo"], configuracion["parametros"]
        X, y = np.asarray(X_des), np.asarray(y_des)
        detalles, predicciones = [], []
        for numero, (idx_ent, idx_val) in enumerate(folds, start=1):
            modelo, segundos = entrenar_modelo(nombre, X[idx_ent], y[idx_ent], parametros)
            pred_ent = predecir_modelo(nombre, modelo, X[idx_ent])
            pred_val = predecir_modelo(nombre, modelo, X[idx_val])
            recall = recall_score(y[idx_val], pred_val, labels=CLASES, average=None, zero_division=0)
            fila = {
                "config_id": config_id, "modelo": nombre, "particion": numero,
                "registros_entrenamiento": len(idx_ent), "registros_validacion": len(idx_val),
                "accuracy_entrenamiento": accuracy_score(y[idx_ent], pred_ent),
                "accuracy_validacion": accuracy_score(y[idx_val], pred_val),
                "f1_macro_validacion": f1_score(y[idx_val], pred_val, labels=CLASES,
                                                 average="macro", zero_division=0),
                "tiempo_entrenamiento_s": segundos,
                **complejidad_modelo(nombre, modelo),
            }
            fila.update({f"recall_clase_{clase}": float(valor)
                         for clase, valor in zip(CLASES, recall)})
            detalles.append(fila)
            predicciones.extend({"config_id": config_id, "modelo": nombre,
                                  "particion": numero, "indice_desarrollo": int(indice),
                                  "clase_real": int(real), "clase_predicha": int(predicha)}
                                 for indice, real, predicha in zip(idx_val, y[idx_val], pred_val))
        return detalles, predicciones


    def resumir_configuracion(config_id, configuracion, detalles):
        marco = pd.DataFrame(detalles)
        resumen = {"config_id": config_id, "modelo": configuracion["modelo"],
                   "parametros": json.dumps(configuracion["parametros"], sort_keys=True)}
        medidas = ["accuracy_entrenamiento", "accuracy_validacion", "f1_macro_validacion",
                   "tiempo_entrenamiento_s", "profundidad_media", "nodos_medios", "nodos_totales"]
        medidas += [f"recall_clase_{clase}" for clase in CLASES]
        for medida in medidas:
            resumen[f"{medida}_media"] = float(marco[medida].mean())
            resumen[f"{medida}_desvio"] = float(marco[medida].std(ddof=1))
        resumen["arboles"] = int(marco["arboles"].iloc[0])
        return resumen


    def ejecutar_experimentos(X_des, y_des, folds, configuraciones, progreso=True, al_finalizar=None):
        # El catálogo ya deduplicado evita entrenar configuraciones repetidas.
        todos_detalles, todas_predicciones, resumenes = [], [], []
        total = len(configuraciones)
        for numero, (clave, config) in enumerate(configuraciones.items(), start=1):
            if progreso:
                print(f"[{numero}/{total}] {config['modelo']} {config['parametros']}", flush=True)
            detalles, predicciones = evaluar_configuracion(clave, config, X_des, y_des, folds)
            todos_detalles.extend(detalles)
            todas_predicciones.extend(predicciones)
            resumenes.append(resumir_configuracion(clave, config, detalles))
            if al_finalizar is not None:
                al_finalizar(todos_detalles, todas_predicciones, resumenes)
        return pd.DataFrame(todos_detalles), pd.DataFrame(todas_predicciones), pd.DataFrame(resumenes)


    def seleccionar_configuraciones(resumen, margen=MARGEN_F1_MACRO):
        # Por modelo, elige menos nodos entre opciones dentro de un punto del mejor F1.
        elegidas = []
        for nombre in ("DT", "RF"):
            opciones = resumen.loc[resumen["modelo"] == nombre].copy()
            if opciones.empty:
                raise ValueError(f"Faltan resultados de validación para {nombre}.")
            mejor_f1 = opciones["f1_macro_validacion_media"].max()
            similares = opciones.loc[opciones["f1_macro_validacion_media"] >= mejor_f1 - margen]
            elegida = similares.sort_values(
                ["nodos_totales_media", "f1_macro_validacion_media", "config_id"],
                ascending=[True, False, True], kind="stable"
            ).iloc[0].copy()
            elegida["mejor_f1_del_modelo"] = mejor_f1
            elegidas.append(elegida)
        return pd.DataFrame(elegidas).reset_index(drop=True)


    def evaluar_reserva(elegidas, catalogo, X_des, y_des, X_reserva, y_reserva):
        # Entrena solo en desarrollo y evalúa una configuración seleccionada por modelo.
        resumenes, predicciones = [], []
        for _, elegida in elegidas.iterrows():
            clave = elegida["config_id"]
            config = catalogo[clave]
            nombre = config["modelo"]
            modelo, segundos = entrenar_modelo(nombre, X_des, y_des, config["parametros"])
            pred = predecir_modelo(nombre, modelo, X_reserva)
            reales = np.asarray(y_reserva)
            recall = recall_score(reales, pred, labels=CLASES, average=None, zero_division=0)
            fila = {"modelo": nombre, "config_id": clave,
                    "parametros": json.dumps(config["parametros"], sort_keys=True),
                    "accuracy_final": accuracy_score(reales, pred),
                    "f1_macro_final": f1_score(reales, pred, labels=CLASES,
                                                average="macro", zero_division=0),
                    "tiempo_entrenamiento_s": segundos,
                    **complejidad_modelo(nombre, modelo)}
            fila.update({f"recall_clase_{clase}": float(valor)
                         for clase, valor in zip(CLASES, recall)})
            resumenes.append(fila)
            predicciones.extend({"modelo": nombre, "config_id": clave,
                                  "indice_reserva": indice, "clase_real": int(real),
                                  "clase_predicha": int(predicha)}
                                 for indice, (real, predicha) in enumerate(zip(reales, pred)))
        return pd.DataFrame(resumenes), pd.DataFrame(predicciones)
    """),
    md("ejecucion-texto", """
    ## 6. Ejecutar la validación

    1. Revisá el plan y las bases de la sección 4.
    2. Cambiá `EJECUTAR_PRUEBAS` a `True` y ejecutá la siguiente celda. Son
       200 entrenamientos de validación; el DT propio puede tardar varios minutos.
    3. Revisá las tablas y los gráficos. Las predicciones de validación no incluyen la reserva.
    4. Solo después de fijar las configuraciones, habilitá la evaluación final en la sección 8.

    Las salidas se guardan en una nueva subcarpeta de `resultados_notebook_pruebas`
    para conservar ejecuciones anteriores. Después de cada configuración completa
    se guardan CSV parciales. Si se interrumpe, esos archivos se conservan, aunque
    para completar todas las pruebas hay que volver a iniciar la ejecución.
    """),
    code("ejecucion", """
    EJECUTAR_PRUEBAS = False  # Cambiar a True para ejecutar las 40 configuraciones.
    RAIZ_RESULTADOS = Path("resultados_notebook_pruebas")

    if EJECUTAR_PRUEBAS:
        CARPETA_RESULTADOS = RAIZ_RESULTADOS / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=False)
        resultado_final, clave_resultado_final = None, None
        huella_validacion = huella_experimento(datos, X_desarrollo, y_desarrollo,
                                              X_final, y_final, particiones)
        exportar_particiones(CARPETA_RESULTADOS, X_desarrollo, X_final,
                             y_desarrollo, y_final, particiones)
        plan.to_csv(CARPETA_RESULTADOS / "plan_de_pruebas.csv", index=False)
        metadatos = {
            "dataset": "UCI Heart Disease, id 45", "clases": CLASES,
            "registros_completos": len(datos), "registros_desarrollo": len(y_desarrollo),
            "registros_reserva": len(y_final), "semilla_reserva": SEMILLA_RESERVA,
            "semilla_folds": SEMILLA_FOLDS, "particiones": N_PARTICIONES,
            "semilla_rf": SEMILLA_RF, "margen_f1_macro": MARGEN_F1_MACRO,
            "sha256_codigo_dt": HUELLA_DT, "sha256_datos": huella_datos(datos),
            "sha256_experimento": huella_validacion,
            "ids_originales": "Índice original UCI antes de eliminar faltantes (base cero)",
            "version_python": sys.version.split()[0], "version_numpy": np.__version__,
            "version_pandas": pd.__version__, "version_sklearn": sklearn.__version__,
            "evaluacion_reserva": "La partición con semilla 55 ya tuvo un resultado guardado en el DT original.",
        }
        (CARPETA_RESULTADOS / "metadatos.json").write_text(
            json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        def guardar_parciales(detalles, predicciones, resumenes):
            pd.DataFrame(detalles).to_csv(CARPETA_RESULTADOS / "resultados_parciales.csv", index=False)
            pd.DataFrame(predicciones).to_csv(CARPETA_RESULTADOS / "predicciones_parciales.csv", index=False)
            pd.DataFrame(resumenes).to_csv(CARPETA_RESULTADOS / "configuraciones_parciales.csv", index=False)

        detalle_cv, predicciones_cv, resumen_cv = ejecutar_experimentos(
            X_desarrollo, y_desarrollo, particiones, catalogo, al_finalizar=guardar_parciales
        )
        vista_cv = plan.merge(resumen_cv, on=["config_id", "modelo"], validate="many_to_one")
        elegidas = seleccionar_configuraciones(resumen_cv)
        detalle_cv.to_csv(CARPETA_RESULTADOS / "resultados_por_particion.csv", index=False)
        resumen_cv.to_csv(CARPETA_RESULTADOS / "resultados_por_configuracion.csv", index=False)
        vista_cv.to_csv(CARPETA_RESULTADOS / "resultados_por_prueba.csv", index=False)
        predicciones_cv.to_csv(CARPETA_RESULTADOS / "predicciones_validacion.csv", index=False)
        elegidas.to_csv(CARPETA_RESULTADOS / "configuraciones_seleccionadas.csv", index=False)
        print(f"Resultados guardados en {CARPETA_RESULTADOS.resolve()}")
        columnas_principales = [
            "prueba", "modelo", "parametro", "valor", "accuracy_entrenamiento_media",
            "accuracy_validacion_media", "f1_macro_validacion_media", "f1_macro_validacion_desvio",
            "profundidad_media_media", "nodos_totales_media", "arboles", "tiempo_entrenamiento_s_media",
        ]
        display(vista_cv[columnas_principales].round(4))
    else:
        print("Pruebas sin ejecutar. Cambiá EJECUTAR_PRUEBAS a True cuando estés listo.")
    """),
    md("lectura-texto", """
    ## 7. Comparación y gráficos de validación

    Las métricas mostradas son promedios por partición; la desviación describe su
    variación y no es un intervalo de confianza. Las matrices agrupan las predicciones
    de validación de las cinco particiones para cada configuración elegida.
    """),
    code("lectura-graficos", """
    def mostrar_graficos(vista):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for nombre in ("DT", "RF"):
            filas = vista[(vista["prueba"] == "Profundidad máxima") & (vista["modelo"] == nombre)]
            axes[0].errorbar(filas["valor"], filas["f1_macro_validacion_media"],
                             yerr=filas["f1_macro_validacion_desvio"], marker="o", capsize=3, label=nombre)
        axes[0].set(title="Profundidad y F1 macro", xlabel="Profundidad máxima", ylabel="F1 macro de validación")
        axes[0].legend()
        filas = vista[(vista["prueba"] == "Cantidad de árboles") & (vista["modelo"] == "RF")]
        axes[1].errorbar(filas["valor"], filas["f1_macro_validacion_media"],
                         yerr=filas["f1_macro_validacion_desvio"], marker="o", capsize=3)
        axes[1].set(title="Cantidad de árboles y F1 macro", xlabel="Árboles en RF", ylabel="F1 macro de validación")
        fig.tight_layout()
        plt.show()
        return fig


    def mostrar_matrices(predicciones, columna_id="config_id"):
        import matplotlib.pyplot as plt

        grupos = list(predicciones.groupby(["modelo", columna_id], sort=True))
        fig, axes = plt.subplots(1, len(grupos), figsize=(5 * len(grupos), 4), squeeze=False)
        for ax, ((nombre, clave), grupo) in zip(axes[0], grupos):
            matriz = confusion_matrix(grupo["clase_real"], grupo["clase_predicha"], labels=CLASES)
            ax.imshow(matriz, cmap="Blues")
            for fila in range(len(CLASES)):
                for columna in range(len(CLASES)):
                    color = "white" if matriz.max() > 0 and matriz[fila, columna] > matriz.max() / 2 else "black"
                    ax.text(columna, fila, str(matriz[fila, columna]), ha="center", va="center", color=color)
            ax.set(xticks=range(len(CLASES)), yticks=range(len(CLASES)),
                   xticklabels=CLASES, yticklabels=CLASES, xlabel="Clase predicha",
                   ylabel="Clase real", title=f"{nombre} ({clave})")
        fig.tight_layout()
        plt.show()
        return fig


    if "resumen_cv" in globals():
        print("Configuraciones seleccionadas con validación:")
        columnas_elegidas = ["modelo", "parametros", "accuracy_validacion_media",
                             "f1_macro_validacion_media", "f1_macro_validacion_desvio",
                             "mejor_f1_del_modelo", "nodos_totales_media", "tiempo_entrenamiento_s_media"]
        display(elegidas[columnas_elegidas].round(4))
        print("Recall promedio por clase en validación:")
        display(elegidas[["modelo"] + [f"recall_clase_{clase}_media" for clase in CLASES]].round(4))
        grafico = mostrar_graficos(vista_cv)
        grafico.savefig(CARPETA_RESULTADOS / "graficos_validacion.png", dpi=160, bbox_inches="tight")
        predicciones_elegidas = predicciones_cv[predicciones_cv["config_id"].isin(elegidas["config_id"])]
        matrices = mostrar_matrices(predicciones_elegidas)
        matrices.savefig(CARPETA_RESULTADOS / "matrices_validacion.png", dpi=160, bbox_inches="tight")
    else:
        print("Todavía no hay resultados de validación para graficar.")
    """),
    md("final-texto", """
    ## 8. Evaluación final de las configuraciones elegidas

    Una vez revisadas y fijadas las configuraciones con validación, cambiá el control
    siguiente a `True` y ejecutá la celda. Se entrena cada modelo con todo el desarrollo
    y se evalúa la reserva. No se utiliza la reserva para volver a elegir parámetros.
    La celda evita repetir la evaluación en la misma sesión. Una partición usada
    previamente para tomar decisiones no se transforma en inédita por volver a ejecutarla.
    """),
    code("final-control", """
    EVALUAR_PRUEBA_FINAL = False  # Cambiar a True solo después de seleccionar con validación.
    if "resultado_final" not in globals():
        resultado_final, clave_resultado_final = None, None

    if EVALUAR_PRUEBA_FINAL:
        if "elegidas" not in globals():
            raise RuntimeError("Primero ejecutá las pruebas de validación de la sección 6.")
        huella_actual = huella_experimento(datos, X_desarrollo, y_desarrollo,
                                           X_final, y_final, particiones)
        if globals().get("huella_validacion") != huella_actual:
            raise RuntimeError("Los datos o las particiones cambiaron desde la validación. "
                               "Volvé a ejecutar la validación antes de evaluar la reserva.")
        clave_actual = (str(CARPETA_RESULTADOS.resolve()), huella_actual, HUELLA_DT,
                        tuple(sorted((fila["modelo"], fila["config_id"],
                                      json.dumps(catalogo[fila["config_id"]]["parametros"], sort_keys=True))
                                     for _, fila in elegidas.iterrows())))
        if resultado_final is not None and globals().get("clave_resultado_final") != clave_actual:
            raise RuntimeError("La evaluación final pertenece a otra corrida o selección. "
                               "Volvé a ejecutar la validación antes de evaluar una selección nueva.")
        if resultado_final is not None:
            print("La reserva ya fue evaluada en esta sesión; se muestran los resultados existentes.")
        else:
            resultado_final, predicciones_finales = evaluar_reserva(
                elegidas, catalogo, X_desarrollo, y_desarrollo, X_final, y_final
            )
            clave_resultado_final = clave_actual
            CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
            resultado_final.to_csv(CARPETA_RESULTADOS / "resultados_prueba_final.csv", index=False)
            predicciones_finales.to_csv(CARPETA_RESULTADOS / "predicciones_prueba_final.csv", index=False)
        columnas_finales = ["modelo", "parametros", "accuracy_final", "f1_macro_final",
                            "arboles", "profundidad_media", "nodos_totales", "tiempo_entrenamiento_s"]
        display(resultado_final[columnas_finales].round(4))
        print("Recall por clase en la reserva:")
        display(resultado_final[["modelo"] + [f"recall_clase_{clase}" for clase in CLASES]].round(4))
        matrices_finales = mostrar_matrices(predicciones_finales)
        matrices_finales.savefig(CARPETA_RESULTADOS / "matrices_prueba_final.png", dpi=160, bbox_inches="tight")
    else:
        print("Evaluación final pendiente. No se utilizaron los resultados de la reserva para seleccionar.")
    """),
    md("referencias", """
    ## Referencias y pendientes

    - UCI Machine Learning Repository: [Heart Disease, dataset 45](https://archive.ics.uci.edu/dataset/45/heart+disease).
    - [RandomForestClassifier, Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
    - [Validación cruzada, Scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html).
    - Pendientes del algoritmo propio: implementar Gain Ratio y poda antes de incluir
      pruebas que dependan de esas funciones.
    """),
])

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
output = ROOT / "notebook-pruebas.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Escrito: {output} ({len(cells)} celdas; DT SHA-256: {snapshot_hash})")
