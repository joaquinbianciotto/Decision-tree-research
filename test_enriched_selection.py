"""Selección C4.5 enriquecida: ganancia, factibilidad de hojas y MDL."""

from copy import deepcopy
import inspect
import json
from math import log2
from pathlib import Path
import unittest

from test_decision_tree_training import execute


class SeleccionEnriquecidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(Path(__file__).with_name("decision-tree-training.ipynb").read_text())
        cls.ns, cls.stdout, cls.stderr = execute(
            [c for c in notebook["cells"] if c["cell_type"] == "code"])

    def entrenar(self, filas, atributos, **opciones):
        antes = deepcopy((filas, atributos, opciones))
        arbol = self.ns["entrenar_arbol_enriquecido"](
            filas, atributos, podar=False, **opciones)
        self.assertEqual((filas, atributos, opciones), antes)
        self.assertEqual((self.stdout, self.stderr), ("", ""))
        return arbol

    def test_gain_ratio_evita_identificadores_y_empates_son_estables(self):
        filas = [[i, i // 4, "no" if i < 4 else "sí"] for i in range(8)]
        self.assertEqual(self.entrenar(filas, [0, 1]).attribute, 0)
        self.assertEqual(self.entrenar(filas, [0, 1], criterio="gain_ratio").attribute, 1)
        empatadas = [[0, 0, "no"], [1, 1, "sí"]]
        for criterio in ("gain", "gain_ratio"):
            self.assertEqual(self.entrenar(empatadas, [1, 0], criterio=criterio).attribute, 1)

    def test_filtro_promedio_descarta_ratio_alto_de_ganancia_pequena(self):
        clases = ["no"] * 6 + ["sí"] * 2 + ["no"] * 2 + ["sí"] * 6
        filas = [[int(i == 6), i // 8, clase] for i, clase in enumerate(clases)]
        sin_filtro = self.entrenar(filas, [0, 1], criterio="gain_ratio",
                                   filtro_ganancia_promedio=False)
        self.assertEqual(sin_filtro.attribute, 0)
        self.assertEqual(self.entrenar(filas, [0, 1], criterio="gain_ratio").attribute, 1)
        self.assertEqual(self.entrenar(filas, [0, 1], criterio="gain").attribute, 1)

    def test_promedio_incluye_representantes_factibles_con_gain_cero(self):
        clases = ["no"] * 6 + ["sí"] * 2 + ["no"] * 2 + ["sí"] * 6
        filas = [[int(i == 6), i // 8, i % 2, 1 - i % 2, "constante", None, clase]
                 for i, clase in enumerate(clases)]
        for atributo in (2, 3):
            self.assertEqual(self.ns["evaluar_prueba"](filas, atributo)["gain"], 0)
        self.assertEqual(self.entrenar(filas, [0, 1, 4, 5], criterio="gain_ratio").attribute, 1)
        self.assertEqual(self.entrenar(filas, [0, 1, 2, 3], criterio="gain_ratio").attribute, 0)
        self.assertIsNone(self.entrenar(filas, [2, 3, 4, 5], criterio="gain_ratio").attribute)

    def test_min_gain_ratio_independiente_en_ambos_criterios(self):
        filas = [["a", "no"], ["b", "sí"], [None, "sí"]]
        pesos = [2, 6, 4]
        ratio = self.ns["evaluar_prueba"](filas, 0, pesos)["gain_ratio"]
        for criterio in ("gain", "gain_ratio"):
            for minimo, esperado in ((ratio, 0), (ratio + .001, None)):
                arbol = self.entrenar(filas, [0], pesos=pesos, criterio=criterio,
                                      min_gain_ratio=minimo)
                self.assertEqual(arbol.attribute, esperado)

    def test_min_gain_sigue_crudo_global_no_ratio_ni_mdl(self):
        filas = [["grande", "a", "no"], ["grande", "b", "no"],
                 ["pequeño", "a", "sí"], ["pequeño", "b", "no"]]
        for minimo, esperado in ((.1, 1), (.1001, None)):
            arbol = self.entrenar(filas, [0, 1], pesos=[45, 45, 5, 5],
                                  criterio="gain_ratio", min_gain=minimo)
            self.assertEqual(arbol.attribute, 0)
            self.assertEqual(arbol.children["pequeño"].attribute, esperado)
        numericas = [[i, "no" if i <= 3 else "sí"] for i in range(1, 7)]
        arbol = self.entrenar(numericas, [0], continuos=[0], correccion_mdl=True, min_gain=1)
        self.assertEqual(arbol.threshold, 3)

    def test_umbral_representante_por_gain_crudo_antes_de_ratio(self):
        filas = [[i, clase] for i, clase in enumerate(
            ["no", "no", "sí", "no", "sí", "sí", "sí", "sí"], 1)]
        candidatos = self.ns["evaluar_umbrales_continuos"](filas, 0, min_branch_weight=1)
        self.assertEqual(max(candidatos, key=lambda c: c["gain"])["threshold"], 4)
        self.assertEqual(max(candidatos, key=lambda c: c["gain_ratio"])["threshold"], 2)
        for mdl in (False, True):
            arbol = self.entrenar(filas, [0], continuos=[0], min_peso_rama=1,
                                  criterio="gain_ratio", correccion_mdl=mdl)
            self.assertEqual(arbol.threshold, 4)

    def test_promedio_usa_un_representante_por_atributo_y_gain_ajustada(self):
        filas = [[i, int(i > 3), "no" if i <= 3 else "sí"] for i in range(1, 7)]
        mejor = self.ns["_seleccionar_prueba_enriquecida"](
            filas, [0, 1], [1.] * 6, {0}, {0: list(range(1, 7))}, 2.,
            criterio="gain_ratio", correccion_mdl=True)
        self.assertEqual(mejor[0], 1)
        self.assertEqual(mejor[1]["gain_ajustada"], 1)
        self.assertEqual(mejor[1]["mdl_penalty"], 0)

    def test_atributo_inviable_no_oculta_otra_division(self):
        filas = [[i, i // 3, "no" if i < 3 else "sí"] for i in range(6)]
        for opciones in ({"min_samples_leaf": 2}, {"min_weight_leaf": 2.}):
            arbol = self.entrenar(filas, [0, 1], **opciones)
            self.assertEqual(arbol.attribute, 1)
            self.assertEqual(len(arbol.children), 2)

    def test_continuo_rechaza_mejor_corte_inviable_antes_de_elegir(self):
        filas = [[i, "sí" if i == 1 else "no"] for i in range(1, 7)]
        self.assertEqual(self.entrenar(filas, [0], continuos=[0], min_peso_rama=1).threshold, 1)
        for opciones in ({"min_samples_leaf": 2}, {"min_weight_leaf": 2.}):
            arbol = self.entrenar(filas, [0], continuos=[0], min_peso_rama=1, **opciones)
            self.assertEqual(arbol.threshold, 2)

    def test_conteo_solo_originales_positivos_conocidos_y_peso_total_repartido(self):
        filas = [["a", "no"], ["b", "sí"], [None, "sí"], ["a", "no"]]
        pesos = [2, 6, 4, 0]
        self.assertIsNone(self.entrenar(filas, [0], pesos=pesos, min_samples_leaf=2).attribute)
        arbol = self.entrenar(filas, [0], pesos=pesos, min_samples_leaf=1, min_weight_leaf=3)
        self.assertEqual([h.total_weight for h in arbol.children.values()], [3, 9])
        self.assertIsNone(self.entrenar(filas, [0], pesos=pesos, min_weight_leaf=3.01).attribute)
        # Dos observaciones livianas cuentan como dos, no como su masa .2.
        livianas = [["a", "no"], ["a", "no"], ["b", "sí"], ["b", "sí"]]
        self.assertEqual(self.entrenar(livianas, [0], pesos=[.1] * 4,
                                      min_samples_leaf=2).attribute, 0)

    def test_fracciones_ancestrales_nunca_recuperan_conteo_original(self):
        filas = ([["x", "u", "no"]] * 2 + [["x", "v", "sí"]]
                 + [["y", "u", "sí"]] * 3 + [["y", "v", "sí"]] * 3
                 + [[None, "v", "sí"]])
        libre = self.entrenar(filas, [0, 1])
        self.assertEqual((libre.attribute, libre.children["x"].attribute), (0, 1))
        limitado = self.entrenar(filas, [0, 1], min_samples_leaf=2)
        self.assertEqual(limitado.attribute, 0)
        self.assertIsNone(limitado.children["x"].attribute)
        self.assertAlmostEqual(sum(h.total_weight for h in limitado.children.values()), len(filas))

    def test_min_samples_split_y_min_peso_rama_siguen_independientes(self):
        filas = [[1, "no"], [1, "no"], [2, "sí"], [2, "sí"], [None, "sí"]]
        self.assertEqual(self.entrenar(filas, [0], continuos=[0], min_samples_leaf=2,
                                      min_weight_leaf=2.5, min_samples_split=5).threshold, 1)
        self.assertIsNone(self.entrenar(filas, [0], continuos=[0], min_samples_split=6).attribute)
        self.assertIsNone(self.entrenar(filas, [0], continuos=[0], min_peso_rama=2.01,
                                       min_weight_leaf=2.5).attribute)

    def test_mdl_cuenta_cortes_admisibles_sin_filtrar_cambios_de_clase(self):
        filas = [[i, "no" if i <= 3 else "sí"] for i in range(1, 7)]
        evaluar = self.ns["evaluar_umbrales_continuos"]
        crudos = evaluar(filas, 0)
        corregidos = evaluar(filas, 0, correccion_mdl=True)
        self.assertEqual([c["threshold"] for c in corregidos], [2, 3, 4])
        for crudo, ajustado in zip(crudos, corregidos):
            self.assertEqual(crudo["gain"], ajustado["gain"])
            self.assertEqual(crudo["gain_ajustada"], crudo["gain"])
            self.assertEqual(crudo["mdl_penalty"], 0)
            self.assertAlmostEqual(ajustado["mdl_penalty"], log2(3) / 6)
            self.assertAlmostEqual(ajustado["gain_ajustada"], crudo["gain"] - log2(3) / 6)
            self.assertAlmostEqual(ajustado["gain_ratio"], ajustado["gain_ajustada"] / ajustado["split_info"])
        restringidos = evaluar(filas, 0, correccion_mdl=True, min_samples_leaf=3)
        self.assertEqual([c["threshold"] for c in restringidos], [3])
        self.assertEqual(restringidos[0]["mdl_penalty"], 0)

    def test_mdl_pesos_y_nulos_sin_segunda_fraccion_conocida(self):
        filas = [[i, "no" if i <= 3 else "sí"] for i in range(1, 7)] + [[None, "sí"]]
        antes = deepcopy(filas)
        for escala in (1, 2):
            pesos = [escala] * 6 + [2 * escala]
            cortes = self.ns["evaluar_umbrales_continuos"](
                filas, 0, pesos, min_branch_weight=2 * escala, correccion_mdl=True)
            medio = cortes[1]
            self.assertAlmostEqual(medio["gain"], .75)
            self.assertAlmostEqual(medio["mdl_penalty"], log2(3) / (8 * escala))
            self.assertAlmostEqual(medio["gain_ajustada"], .75 - log2(3) / (8 * escala))
            self.assertEqual(pesos, [escala] * 6 + [2 * escala])
        self.assertEqual(filas, antes)

    def test_mdl_sin_cortes_o_gain_ajustada_no_positiva(self):
        evaluar = self.ns["evaluar_umbrales_continuos"]
        for filas in ([], [[None, "sí"]], [[1, "sí"], [1, "no"]], [[1, "sí"], [2, "no"]]):
            self.assertEqual(evaluar(filas, 0, correccion_mdl=True), [])
        filas = [[i, "sí" if i % 2 else "no"] for i in range(1, 7)]
        self.assertIsNotNone(self.entrenar(filas, [0], continuos=[0], min_gain=0).attribute)
        self.assertIsNone(self.entrenar(filas, [0], continuos=[0], correccion_mdl=True,
                                       min_gain=0).attribute)

    def test_compatibilidad_posicional_y_opciones_solo_por_nombre(self):
        entrenar = self.ns["entrenar_arbol_enriquecido"]
        parametros = list(inspect.signature(entrenar).parameters.values())
        self.assertEqual([p.name for p in parametros[:10]], [
            "filas", "atributos", "continuos", "pesos", "min_gain", "min_peso_rama",
            "confianza", "podar", "max_depth", "min_samples_split"])
        nuevos = {p.name: p for p in parametros[10:]}
        esperados = dict(criterio="gain", filtro_ganancia_promedio=True, min_gain_ratio=None,
                         correccion_mdl=False, min_samples_leaf=0, min_weight_leaf=0.)
        for nombre, defecto in esperados.items():
            self.assertEqual(nuevos[nombre].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(nuevos[nombre].default, defecto)
        filas = [[1, "no"], [1, "no"], [2, "sí"], [2, "sí"]]
        arbol = entrenar(filas, [0], [0], [1.] * 4, 1., 2., .25, False, None, 2)
        self.assertEqual(arbol.threshold, 1)

    def test_parametros_invalidos_se_rechazan_incluso_en_nodo_puro(self):
        invalidos = [dict(criterio="otro"), dict(min_samples_leaf=-1),
                     dict(min_samples_leaf=1.5), dict(min_samples_leaf=True),
                     dict(min_weight_leaf=-1), dict(min_weight_leaf=float("inf")),
                     dict(min_weight_leaf=float("nan")), dict(min_gain_ratio=-1),
                     dict(min_gain_ratio=float("nan")), dict(min_gain_ratio=float("inf")),
                     dict(correccion_mdl="sí"), dict(filtro_ganancia_promedio="sí")]
        for opciones in invalidos:
            with self.subTest(opciones=opciones), self.assertRaises(ValueError):
                self.entrenar([[1, "sí"]], [0], **opciones)
        self.assertIsNone(self.entrenar([[None, "no"], [None, "sí"]], [0],
                                       criterio="gain_ratio").attribute)
        self.assertIsNone(self.entrenar([[1, "no"], [1, "sí"]], [0],
                                       criterio="gain_ratio").attribute)


if __name__ == "__main__":
    unittest.main()
