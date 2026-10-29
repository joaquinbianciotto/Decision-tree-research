"""Poda enriquecida: riesgo 0-1 global, validación explícita y compatibilidad."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_decision_tree_training import execute


class PodaEnriquecidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(Path(__file__).with_name("decision-tree-training.ipynb").read_text())
        cls.ns, cls.stdout, cls.stderr = execute(
            [c for c in notebook["cells"] if c["cell_type"] == "code"])

    def train(self, filas, atributos=(0,), **kwargs):
        antes = repr((filas, atributos, kwargs))
        arbol = self.ns["entrenar_arbol_enriquecido"](filas, atributos, **kwargs)
        self.assertEqual(repr((filas, atributos, kwargs)), antes)
        return arbol

    def estado(self, nodo):
        return (nodo.prediction, nodo.attribute, nodo.threshold, nodo.class_counts,
                nodo.total_weight, nodo.class_probabilities, nodo.branch_probabilities,
                [(rama, self.estado(hijo)) for rama, hijo in nodo.children.items()])

    def riesgo(self, arbol, filas, pesos):
        return sum(p for f, p in zip(filas, pesos)
                   if self.ns["predecir_enriquecido"](arbol, f) != f[-1])

    def test_metodos_compatibles_y_argumentos_solo_por_nombre(self):
        filas = [["a", "no"], ["b", "si"]]
        parametros = list(inspect.signature(self.ns["entrenar_arbol_enriquecido"]).parameters.values())
        self.assertEqual(len([p for p in parametros if p.kind == p.POSITIONAL_OR_KEYWORD]), 10)
        for nombre in ("metodo_poda", "subtree_raising", "ccp_alpha",
                       "filas_validacion", "pesos_validacion"):
            self.assertEqual(next(p for p in parametros if p.name == nombre).kind,
                             inspect.Parameter.KEYWORD_ONLY)
        crecido = self.train(filas, podar=False)
        self.assertEqual(self.estado(crecido), self.estado(self.train(filas, metodo_poda="ninguna")))
        self.assertEqual(self.estado(crecido), self.estado(self.train(
            filas, podar=False, metodo_poda="ninguna")))
        self.assertEqual(self.estado(self.train(filas)), self.estado(self.train(
            filas, metodo_poda="pesimista")))
        posicional = self.ns["entrenar_arbol_enriquecido"](
            filas, [0], (), None, 1e-12, 2.0, .25, False, None, 2)
        self.assertEqual(self.estado(crecido), self.estado(posicional))

    def test_validacion_de_configuracion_antes_de_crecer(self):
        filas = [["a", "no"], ["b", "si"]]
        invalidos = [dict(metodo_poda="otro"), dict(metodo_poda=[]), dict(podar=1),
                     dict(subtree_raising=0), dict(subtree_raising=None),
                     dict(metodo_poda="reduced_error"),
                     dict(pesos_validacion=[1]), dict(filas_validacion=[])]
        invalidos += [dict(ccp_alpha=a) for a in (-1, float("nan"), float("inf"), True, "1", None)]
        invalidos += [dict(podar=False, metodo_poda=m)
                      for m in ("pesimista", "reduced_error", "cost_complexity")]
        invalidos += [dict(metodo_poda=m, filas_validacion=[["a", "no"]])
                      for m in ("pesimista", "cost_complexity", "ninguna")]
        with patch.dict(self.ns, {"_crecer_enriquecido": lambda *a, **k: self.fail("Creció antes de validar")}):
            for kwargs in invalidos:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.train(filas, **kwargs)

    def test_reduced_error_exige_poblacion_independiente_pesos_y_ancho_compatibles(self):
        filas = [["a", "no"], ["b", "si"]]
        invalidos = [dict(filas_validacion=filas), dict(filas_validacion=[]),
                     dict(filas_validacion=[["no"]]),
                     dict(filas_validacion=[["a", "x", "no"]]),
                     dict(filas_validacion=[["a", "no"], ["si"]])]
        invalidos += [dict(filas_validacion=[["a", "no"]], pesos_validacion=p)
                      for p in ([], [0], [-1], [float("nan")], [float("inf")], [1, 2])]
        with patch.dict(self.ns, {"_crecer_enriquecido": lambda *a, **k: self.fail("Creció sin validación válida")}):
            for kwargs in invalidos:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.train(filas, metodo_poda="reduced_error", **kwargs)

    def test_subtree_raising_desactivado_conserva_reemplazo_por_hoja(self):
        filas = [["main", 0, "no"]] * 4 + [["main", 1, "si"]] * 4
        filas += [[str(i), v, c] for i, (v, c) in enumerate(
            [(0, "no"), (0, "si"), (1, "no"), (1, "si")] * 2)]
        self.assertEqual(self.train(filas, [0, 1]).attribute, 1)
        sin_elevar = self.train(filas, [0, 1], subtree_raising=False)
        self.assertEqual(sin_elevar.attribute, 0)
        reemplazo = [["a", "no"]] * 6 + [["b", "no"]] * 9 + [["c", "si"]]
        self.assertIsNone(self.train(reemplazo, subtree_raising=False).attribute)
        # El estimador aprobado sigue dando los valores del ejemplo de Quinlan.
        self.assertAlmostEqual(self.ns["errores_estimados"](16, 1) / 16, .157, places=3)

    def test_reduced_error_pesado_empate_y_estadisticas_solo_de_entrenamiento(self):
        filas = [["a", "no"], ["b", "si"]]
        validacion = [["b", "si"], ["b", "no"]]
        comun = dict(metodo_poda="reduced_error", filas_validacion=validacion, pesos=[3, 1])
        conservar = self.train(filas, pesos_validacion=[2, 1], **comun)
        podado = self.train(filas, pesos_validacion=[1, 2], **comun)
        empate = self.train(filas, pesos_validacion=[.5, .5], **comun)
        self.assertEqual(conservar.attribute, 0)
        for arbol in (podado, empate):
            self.assertIsNone(arbol.attribute)
            self.assertEqual(arbol.prediction, "no")
            self.assertEqual(arbol.class_counts, {"no": 3, "si": 1})
            self.assertEqual(arbol.class_probabilities, {"no": .75, "si": .25})
        self.assertEqual(validacion, [["b", "si"], ["b", "no"]])

    def test_riesgo_es_argmax_no_uno_menos_probabilidad_verdadera(self):
        filas = [["a", "no"], ["b", "no"], ["b", "si"]]
        validacion = [["a", "si"], ["b", "si"]]
        arbol = self.train(filas, pesos=[2, 4, 6], metodo_poda="reduced_error",
                           filas_validacion=validacion)
        # Hoja: error 0-1 = 2 frente a 1 del árbol. En cambio 1-p daría 1 frente a 1.4.
        self.assertEqual(arbol.attribute, 0)
        self.assertEqual(self.riesgo(arbol, validacion, [1, 1]), 1)

    def plantilla_con_mezcla(self):
        Nodo = self.ns["NodoEnriquecido"]
        izquierdo = Nodo([["no"], ["si"]], [6, 4])
        izquierdo.attribute = 1
        izquierdo.children = {"u": Nodo([["no"]], [6]), "v": Nodo([["si"]], [4])}
        izquierdo.branch_probabilities = {"u": .6, "v": .4}
        derecho = Nodo([["no"], ["si"]], [3.75, 11.25])
        raiz = Nodo([["no"], ["si"]], [9.75, 15.25])
        raiz.attribute = 0
        raiz.children = {"a": izquierdo, "b": derecho}
        raiz.branch_probabilities = {"a": .4, "b": .6}
        return raiz

    def test_mezclas_desconocidas_rechazan_poda_local_que_empeora_riesgo_global(self):
        plantilla = self.plantilla_con_mezcla()
        antes = deepcopy(self.estado(plantilla))
        filas = [[None, "u", "no"], ["?", "u", "no"], [float("nan"), "u", "no"]]
        for alpha in (0, .1):
            podado = self.ns["_podar_por_riesgo_enriquecido"](plantilla, filas, [.25, .5, .75], alpha)
            # El hijo a clasifica no antes y después: un criterio local aceptaría el empate.
            # La mezcla global pasa de P(no)=.55 a .39, por eso debe rechazarse.
            self.assertEqual(podado.children["a"].attribute, 1)
            self.assertEqual(self.riesgo(podado, filas, [.25, .5, .75]), 0)
            self.assertEqual(self.estado(plantilla), antes)
            self.assertIsNot(podado, plantilla)
            self.assertIsNot(podado.children["a"], plantilla.children["a"])

    def test_cost_complexity_alpha_fijo_error_crudo_y_escala_de_pesos(self):
        filas = [["a", "no"], ["b", "si"]]
        for pesos, alpha, hoja in (([3, 1], 0, False), ([3, 1], .99, False),
                                   ([3, 1], 1, True), ([3, 1], 100, True),
                                   ([30, 10], 1, False), ([30, 10], 10, True)):
            with self.subTest(pesos=pesos, alpha=alpha):
                arbol = self.train(filas, pesos=pesos, metodo_poda="cost_complexity", ccp_alpha=alpha)
                self.assertEqual(arbol.attribute is None, hoja)
                self.assertEqual(arbol.class_counts, {"no": pesos[0], "si": pesos[1]})

    def test_reduced_error_recorrido_ascendente_poda_hijo_sin_perder_raiz(self):
        filas = [["a", "u", "no"], ["a", "v", "si"], ["b", "u", "si"]]
        validacion = [["a", "v", "no"], ["b", "u", "si"]]
        arbol = self.train(filas, [0, 1], pesos=[6, 2, 10], metodo_poda="reduced_error",
                           filas_validacion=validacion)
        self.assertEqual(arbol.attribute, 0)
        self.assertIsNone(arbol.children["a"].attribute)
        self.assertEqual(self.riesgo(arbol, validacion, [1, 1]), 0)

    def test_ambos_metodos_soportan_continuos_nulos_y_categorias_nuevas(self):
        filas = [[1, "no"], [2, "no"], [4, "si"], [5, "si"], [None, "si"]]
        pesos = [1.5, .5, 2, 4, .5]
        validacion = [[2, "no"], [3, "si"], [None, "si"]]
        for metodo in ("cost_complexity", "reduced_error"):
            kwargs = dict(filas_validacion=validacion, pesos_validacion=[.5, 1, 2]) if metodo == "reduced_error" else {}
            arbol = self.train(filas, continuos=[0], pesos=pesos, metodo_poda=metodo, **kwargs)
            self.assertEqual(arbol.threshold, 2)
            self.assertEqual(self.riesgo(arbol, validacion, [.5, 1, 2]), 0)
        categorico = self.train([["a", "no"], ["b", "si"]], pesos=[1, 3],
                                 metodo_poda="reduced_error", filas_validacion=[["nueva", "si"]])
        self.assertEqual(self.ns["predecir_enriquecido"](categorico, ["nueva"]), "si")

    def test_resumen_y_estadisticas_reutilizables_no_reentrenan(self):
        filas = [["a", "no"], ["b", "si"]]
        arbol = self.train(filas, pesos=[3, 1], metodo_poda="cost_complexity", ccp_alpha=1)
        resumen = arbol.resumen_poda
        self.assertEqual(resumen, {"metodo": "cost_complexity",
                                   "antes": {"profundidad": 1, "nodos": 3, "hojas": 2},
                                   "despues": {"profundidad": 0, "nodos": 1, "hojas": 1}})
        with patch.dict(self.ns, {"_crecer_enriquecido": lambda *a, **k: self.fail("Reentrenó")}):
            self.assertEqual(self.ns["estadisticas_arbol_enriquecido"](arbol), resumen["despues"])
        self.assertEqual(self.ns["estadisticas_arbol_enriquecido"](self.plantilla_con_mezcla()),
                         {"profundidad": 2, "nodos": 5, "hojas": 3})
        sin_poda = self.train(filas, metodo_poda="ninguna")
        self.assertEqual(sin_poda.resumen_poda["antes"], sin_poda.resumen_poda["despues"])
        self.assertEqual((self.stdout, self.stderr), ("", ""))

    def test_empate_a_alpha_cero_y_resumen_deterministas(self):
        filas = [["a", "no"], ["a", "si"], ["b", "no"], ["b", "si"]]
        crecido = self.train(filas, pesos=[9, 1, 6, 4], podar=False)
        self.assertEqual(crecido.attribute, 0)
        for metodo in ("cost_complexity", "reduced_error"):
            kwargs = dict(filas_validacion=deepcopy(filas), pesos_validacion=[9, 1, 6, 4]) if metodo == "reduced_error" else {}
            primero = self.train(filas, pesos=[9, 1, 6, 4], metodo_poda=metodo, **kwargs)
            segundo = self.train(filas, pesos=[9, 1, 6, 4], metodo_poda=metodo, **kwargs)
            self.assertIsNone(primero.attribute)
            self.assertEqual(self.estado(primero), self.estado(segundo))
            self.assertEqual(primero.resumen_poda["metodo"], metodo)
            self.assertEqual(primero.resumen_poda["antes"], {"profundidad": 1, "nodos": 3, "hojas": 2})

    def test_ninguna_devuelve_el_mismo_arbol_crecido_con_solo_metadatos(self):
        plantilla = self.plantilla_con_mezcla()
        antes = deepcopy(self.estado(plantilla))
        with patch.dict(self.ns, {"_crecer_enriquecido": lambda *a, **k: plantilla}):
            arbol = self.train([["a", "u", "no"]], [0, 1], metodo_poda="ninguna")
        self.assertIs(arbol, plantilla)
        self.assertEqual(self.estado(arbol), antes)
        self.assertEqual(arbol.resumen_poda["metodo"], "ninguna")
        self.assertFalse(hasattr(arbol.children["a"], "resumen_poda"))

    def test_alternativas_copian_sin_recalibrar_ni_usar_estimador_pesimista(self):
        plantilla = self.plantilla_con_mezcla()
        antes = deepcopy(self.estado(plantilla))
        def prohibido(*args, **kwargs):
            self.fail("La poda alternativa llamó al estimador o recalibró con validación")
        with patch.dict(self.ns, {"errores_estimados": prohibido,
                                  "_recalibrar_enriquecido": prohibido,
                                  "_crecer_enriquecido": lambda *a, **k: plantilla}):
            for metodo, kwargs in (("cost_complexity", {"ccp_alpha": 100}),
                                    ("reduced_error", {"filas_validacion": [["a", "u", "si"]]})):
                arbol = self.train([["a", "u", "no"]], [0, 1], metodo_poda=metodo, **kwargs)
                self.assertIsNot(arbol, plantilla)
                self.assertEqual(self.estado(plantilla), antes)
                self.assertEqual(arbol.class_counts, plantilla.class_counts)


if __name__ == "__main__":
    unittest.main()
