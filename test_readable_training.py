"""Comprueba la variante didáctica sin cambiar los contratos del árbol original."""

import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest

from test_decision_tree_training import execute
import test_enriched_selection as seleccion
import test_enriched_pruning as poda


ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "decision-tree-training-readable.ipynb"


def cargar(nombre=NOTEBOOK):
    notebook = json.loads(nombre.read_text())
    celdas = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    return notebook, execute(celdas)


class NotebookLegibleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, (cls.ns, cls.stdout, cls.stderr) = cargar()
        _, (cls.base, _, _) = cargar(ROOT / "decision-tree-training.ipynb")

    def test_notebook_autonomo_y_markdown_breve(self):
        self.assertEqual((self.stdout, self.stderr), ("", ""))
        original = json.loads((ROOT / "decision-tree-training.ipynb").read_text())
        textos = ["".join(c["source"]) for c in self.notebook["cells"]
                  if c["cell_type"] == "markdown"]
        originales = ["".join(c["source"]) for c in original["cells"]
                      if c["cell_type"] == "markdown"]
        self.assertLess(sum(map(len, textos)), sum(map(len, originales)) / 4)
        self.assertTrue(all(len(texto.splitlines()) <= 8 for texto in textos))
        for celda in self.notebook["cells"]:
            if celda["cell_type"] == "code":
                self.assertIsNone(celda["execution_count"])
                self.assertEqual(celda["outputs"], [])
                fuente = "".join(celda["source"])
                self.assertNotIn("read_text(", fuente)
                self.assertNotIn("exec(", fuente)
        # Las variantes usan demos distintos; compara los entrenadores con las mismas filas.
        arbol_base = self.base["entrenar_arbol_enriquecido"](self.ns["D"], self.ns["A"])
        self.assertEqual(self.ns["arbol_enriquecido"].resumen_poda, arbol_base.resumen_poda)

    def test_crecimiento_lineal_y_entrenamiento_en_cuatro_etapas(self):
        codigo = "\n".join("".join(c["source"]) for c in self.notebook["cells"]
                           if c["cell_type"] == "code")
        funciones = {n.name: n for n in ast.parse(codigo).body if isinstance(n, ast.FunctionDef)}
        crecer = funciones["_crecer_enriquecido"]
        self.assertTrue(all(not n.orelse for n in ast.walk(crecer) if isinstance(n, ast.If)))
        llamadas = [n.func.id for n in ast.walk(funciones["entrenar_arbol_enriquecido"])
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        etapas = ["_validar_entrenamiento", "_preparar_entrenamiento",
                  "_crecer_enriquecido", "_aplicar_poda"]
        self.assertEqual([n for n in llamadas if n in etapas], etapas)
        self.assertEqual(inspect.signature(self.ns["entrenar_arbol_enriquecido"]),
                         inspect.signature(self.base["entrenar_arbol_enriquecido"]))

    def test_dataframe_por_nombre_con_nulos_sin_mutar(self):
        pd = self.ns["pd"]
        datos = pd.DataFrame({"Clase": ["sí", "no", "sí", "no"],
                              "Ignorar": [9, 8, 7, 6],
                              "Edad": pd.Series([1, pd.NA, 3, 4], dtype="Int64"),
                              "Color": ["rojo", "?", None, float("nan")]})
        antes = datos.copy(deep=True)
        filas, atributos, continuos, nombres = self.ns["preparar_dataframe"](
            datos, "Clase", atributos=["Color", "Edad"], continuos=["Edad"])
        self.assertEqual(nombres, ["Color", "Edad", "Clase"])
        self.assertEqual(atributos, [0, 1])
        self.assertEqual(continuos, [1])
        self.assertEqual(filas, [["rojo", 1, "sí"], [None, None, "no"],
                                 [None, 3, "sí"], [None, 4, "no"]])
        pd.testing.assert_frame_equal(datos, antes)
        self.assertEqual(self.ns["preparar_dataframe"](datos, "Clase")[2], [])
        arbol = self.ns["entrenar_arbol_enriquecido"](
            filas, atributos, continuos=continuos, min_peso_rama=1, podar=False)
        self.assertEqual(arbol.total_weight, 4)
        self.assertEqual(self.ns["faltantes"].sum(), 0)
        self.assertEqual(self.ns["distribucion_clases"].to_dict(), {"no": 5, "yes": 9})

    def test_dataframe_invalido_rechazado_sin_modificar(self):
        pd = self.ns["pd"]
        casos = [
            (pd.DataFrame({"X": [1], "Y": [None]}), "Y", {}, "La clase debe ser conocida."),
            (pd.DataFrame({"X": [1], "Y": ["?"]}), "Y", {}, "La clase debe ser conocida."),
            (pd.DataFrame({"X": [1], "Y": ["sí"]}), "Z", {}, "La columna objetivo no existe."),
            (pd.DataFrame([[1, 2]], columns=["Y", "Y"]), "Y", {}, "Las columnas deben ser únicas."),
            (pd.DataFrame({"X": [], "Y": []}), "Y", {}, "Se requiere un DataFrame no vacío."),
            (pd.DataFrame({"X": [1], "Y": ["sí"]}), "Y", {"atributos": ["Y"]},
             "Los atributos deben existir, ser únicos y excluir la clase."),
            (pd.DataFrame({"X": [1], "Y": ["sí"]}), "Y", {"atributos": ["X", "X"]},
             "Los atributos deben existir, ser únicos y excluir la clase."),
            (pd.DataFrame({"X": [1], "Y": ["sí"]}), "Y", {"continuos": ["Z"]},
             "Los continuos deben ser atributos seleccionados y únicos."),
        ]
        for datos, objetivo, opciones, mensaje in casos:
            with self.subTest(mensaje=mensaje):
                antes = datos.copy(deep=True)
                with self.assertRaises(ValueError) as error:
                    self.ns["preparar_dataframe"](datos, objetivo, **opciones)
                self.assertEqual(str(error.exception), mensaje)
                pd.testing.assert_frame_equal(datos, antes)

    def test_equivalencia_de_arbol_y_predicciones(self):
        filas = [[1, "a", "no"], [2, "a", "no"], [3, "b", "sí"],
                 [4, "b", "sí"], [None, "a", "sí"], ["?", "b", "no"]]
        pesos = [1, .5, 2, 1, .25, .75]
        def estado(nodo):
            return (nodo.attribute, nodo.threshold, nodo.prediction, nodo.class_counts,
                    nodo.class_probabilities, nodo.branch_probabilities,
                    [(rama, estado(hijo)) for rama, hijo in nodo.children.items()])
        for metodo in ("ninguna", "pesimista", "reduced_error", "cost_complexity"):
            for criterio in ("gain", "gain_ratio"):
                opciones = dict(continuos=[0], pesos=pesos, min_peso_rama=.25,
                                criterio=criterio, correccion_mdl=True, metodo_poda=metodo)
                if metodo == "reduced_error":
                    opciones["filas_validacion"] = deepcopy(filas)
                antes = deepcopy((filas, opciones))
                with self.subTest(metodo=metodo, criterio=criterio):
                    nuevo = self.ns["entrenar_arbol_enriquecido"](filas, [0, 1], **opciones)
                    previo = self.base["entrenar_arbol_enriquecido"](filas, [0, 1], **opciones)
                    self.assertEqual(estado(nuevo), estado(previo))
                    self.assertEqual(nuevo.resumen_poda, previo.resumen_poda)
                    for fila in filas + [[None, "nueva"], [2.5, "?"]]:
                        self.assertEqual(self.ns["predecir_probabilidades_enriquecido"](nuevo, fila),
                                         self.base["predecir_probabilidades_enriquecido"](previo, fila))
                    self.assertEqual((filas, opciones), antes)


class SeleccionLegibleTests(seleccion.SeleccionEnriquecidaTests):
    @classmethod
    def setUpClass(cls):
        _, (cls.ns, cls.stdout, cls.stderr) = cargar()


class PodaLegibleTests(poda.PodaEnriquecidaTests):
    @classmethod
    def setUpClass(cls):
        _, (cls.ns, cls.stdout, cls.stderr) = cargar()


if __name__ == "__main__":
    unittest.main()
