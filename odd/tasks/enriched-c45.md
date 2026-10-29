# Enriched tree: C4.5 selection and pruning
Objective: extend only the enriched training notebook. Branch: feat/enriched-c45; base: b20785e. Runner: Python unittest and sequential cell execution.
Delivery: feature-branch-chain (user selected); local reviewable work-unit commits only, no publishing. Forecast: ~1500 authored changed lines, split by behavior; coherent units may exceed the ~400-line heuristic.

## Specs

### S1

## 1. Gain Ratio como criterio de selección

Actualmente `_seleccionar_prueba_enriquecida()` selecciona el atributo con mayor ganancia de información, aunque `evaluar_prueba()` ya calcula `gain_ratio` y `split_info`.

Implementar un criterio configurable:

`criterio = "gain" | "gain_ratio"`

Para `gain_ratio`, aplicar la estrategia de C4.5:

1. Evaluar todas las pruebas candidatas válidas.
2. Calcular la ganancia de información de cada atributo.
3. Calcular la ganancia promedio de los atributos elegibles.
4. Descartar para la selección por Gain Ratio aquellos atributos cuya ganancia sea inferior al promedio.
5. Entre los atributos restantes, seleccionar el de mayor Gain Ratio.
6. Conservar un criterio de desempate determinista.
7. Evitar divisiones por cero cuando `split_info = 0`.

Agregar parámetros:

- `criterio`: criterio de selección.
- `filtro_ganancia_promedio`: activar/desactivar el filtro.
- `min_gain_ratio`: umbral mínimo opcional para aceptar una división.

El cálculo debe ser compatible con atributos continuos, valores desconocidos y pesos fraccionarios.

Mantener `criterio="gain"` como comportamiento predeterminado para garantizar compatibilidad con la implementación actual.

**Importante:** distinguir claramente el valor utilizado para elegir un atributo del utilizado para determinar si una división supera `min_gain`.

### S2

## 2. Mínimo de observaciones y peso por hoja

Actualmente `min_samples_split` controla la cantidad mínima de filas necesaria para dividir un nodo, pero no garantiza un mínimo de observaciones en las hojas resultantes.

Implementar:

- `min_samples_leaf`: mínimo de observaciones para aceptar cada rama resultante.
- `min_weight_leaf`: peso efectivo mínimo que debe recibir cada rama.

Requisitos:

1. Aplicar las restricciones a atributos continuos y categóricos.
2. Evaluarlas antes de aceptar una división.
3. Conservar `min_samples_split` como restricción independiente.
4. Mantener la compatibilidad con `min_peso_rama`, documentando las diferencias.
5. Evitar que una fila con valor desconocido, distribuida entre varias ramas, se contabilice incorrectamente como múltiples observaciones originales.
6. Diferenciar explícitamente conteo de observaciones y suma de pesos fraccionarios.
7. Rechazar divisiones que produzcan ramas insuficientes, en vez de generar estructuras inválidas.
8. Conservar las reglas de parada existentes.

Definir y documentar una semántica inequívoca para el conteo de observaciones con valores desconocidos. Por ejemplo, exigir `min_samples_leaf` sobre observaciones con valor conocido y `min_weight_leaf` sobre el peso total posterior al reparto fraccionario.

Agregar casos de prueba para variables categóricas con muchas categorías, ramas muy pequeñas y valores faltantes.

### S3

## 3. Penalización MDL para atributos continuos

Actualmente `evaluar_umbrales_continuos()` evalúa umbrales y selecciona candidatos según su ganancia, pero no incorpora una penalización por la cantidad de cortes posibles.

Implementar una corrección basada en *Minimum Description Length*, siguiendo el tratamiento de atributos continuos de C4.5.

Agregar:

`correccion_mdl = True | False`

La corrección deberá considerar la cantidad de puntos de corte candidatos disponibles en el nodo.

Como referencia, para datos completos sin pesos:

`ganancia_ajustada = ganancia - log2(K) / N`

Donde `K` representa el número de posibles cortes y `N` el número de observaciones correspondiente.

Requisitos:

1. Verificar la formulación original de Quinlan antes de implementarla.
2. Separar la ganancia original de la ganancia ajustada.
3. Documentar cómo se adapta el cálculo a pesos y valores desconocidos.
4. Evitar aplicar penalizaciones cuando no existen cortes válidos.
5. No alterar el cálculo original de la entropía.
6. Permitir comparar resultados con MDL activado y desactivado.
7. Mantener una distinción entre la selección del mejor umbral de cada atributo y la comparación final entre atributos.
8. Hacer que la corrección pueda combinarse con Gain Ratio sin penalizar dos veces la misma decisión.

Conservar en los resultados de los candidatos, cuando corresponda, los campos `gain`, `gain_ajustada`, `gain_ratio`, `threshold` y `mdl_penalty`.

La corrección MDL debe quedar desactivada por defecto para conservar los resultados anteriores.

### S4

## 4. Métodos alternativos de poda

Actualmente existe poda pesimista mediante `_podar_enriquecido()`.

No eliminarla ni reemplazarla. Incorporar una selección de métodos:

`metodo_poda = "pesimista" | "reduced_error" | "cost_complexity" | "ninguna"`

### 4.1 Poda pesimista

Conservar la lógica existente de C4.5, incluyendo sus estimaciones de error.

Agregar:

`subtree_raising = True | False`

Este parámetro debe permitir activar o desactivar la elevación de subárboles sin desactivar el reemplazo de subárboles por hojas.

Mantener el comportamiento actual como predeterminado.

### 4.2 Reduced-error pruning

Implementar poda mediante un conjunto de validación independiente.

Requisitos:

- Admitir datos de validación explícitos.
- No utilizar los datos de prueba final para podar.
- Recorrer el árbol de abajo hacia arriba.
- Comparar el error del subárbol con el error de reemplazarlo por una hoja.
- Conservar una regla de desempate determinista.
- Soportar valores faltantes y pesos.
- Documentar cómo se evalúan las predicciones probabilísticas durante la poda.

No mezclar las estimaciones pesimistas con los errores de validación de este método.

### 4.3 Cost-complexity pruning

Implementar una alternativa inspirada en CART, utilizando un parámetro de complejidad:

`ccp_alpha >= 0`

Considerar un objetivo de la forma:

`R_alpha(T) = R(T) + alpha * hojas(T)`

Donde `R(T)` representa el error de clasificación ponderado del árbol.

Requisitos:

- Permitir configurar `ccp_alpha`.
- Comparar el costo de conservar un subárbol con el de reemplazarlo por una hoja.
- Utilizar la misma definición de riesgo en todas las alternativas.
- No alterar los modelos originales.
- Documentar cualquier diferencia respecto del procedimiento completo de weakest-link pruning de CART.

### 4.4 Desactivar poda

El modo `"ninguna"` debe devolver el árbol crecido sin modificaciones posteriores.

Conservar el argumento existente `podar` para compatibilidad. Definir claramente su interacción con `metodo_poda`, evitando combinaciones contradictorias o silenciosamente ambiguas.

### S5

## 5. Integración con ipywidgets

Ampliar `crear_panel_arbol_enriquecido()` para incluir los nuevos parámetros.

Organizar visualmente los controles en cuatro grupos:

**División**

- Criterio: Gain / Gain Ratio.
- Filtro de ganancia promedio.
- Gain Ratio mínimo.
- Corrección MDL.
- Selección de atributos continuos.

**Crecimiento**

- Ganancia mínima.
- Profundidad máxima.
- Mínimo de observaciones para dividir.
- Mínimo de observaciones por hoja.
- Peso mínimo por hoja.
- Peso mínimo por lado para continuos.

**Poda**

- Método de poda.
- Factor de confianza.
- Habilitar elevación de subárboles.
- Parámetro `ccp_alpha`.
- Parámetros necesarios para reduced-error pruning.

**Resultados**

- Árbol generado.
- Profundidad alcanzada.
- Cantidad de nodos.
- Cantidad de hojas.
- Parámetros utilizados.
- Comparación estructural antes y después de podar.

Los controles deben habilitarse o deshabilitarse según el método seleccionado.

Por ejemplo, `ccp_alpha` solamente debe habilitarse cuando se selecciona cost-complexity pruning.

Mantener el funcionamiento dentro de Jupyter/Google Colab, sin requerir una aplicación web externa.

### S6

## 6. Compatibilidad y arquitectura

Estas restricciones son obligatorias:

1. No modificar las funciones del algoritmo original `decision_tree`, `Node`, `predict_one` ni su entrenamiento.
2. Conservar las APIs existentes del árbol enriquecido.
3. Añadir nuevos parámetros sin romper las llamadas posicionales actuales; preferir argumentos nuevos exclusivamente por nombre.
4. No introducir dependencias innecesarias.
5. Mantener los nombres de las funciones existentes en español.
6. Documentar qué funcionalidades pertenecen a C4.5 y cuáles son extensiones inspiradas en CART u otros algoritmos.
7. No alterar las matrices originales ni los pesos proporcionados por el usuario.
8. Mantener resultados reproducibles y desempates deterministas.
9. Preservar los ejemplos y demostraciones ya existentes.
10. No reestructurar las celdas originales salvo que sea estrictamente necesario.

Si algún parámetro nuevo requiere datos que no están disponibles, como un conjunto de validación, generar un error claro en lugar de utilizar implícitamente los datos de entrenamiento o prueba.

### S7

## 7. Criterios de finalización

La tarea se considera completa cuando:

- Las cuatro mejoras estén implementadas y conectadas al entrenador enriquecido.
- Los nuevos parámetros puedan configurarse desde el panel.
- Los comportamientos predeterminados anteriores se mantengan.
- Las nuevas funcionalidades tengan pruebas específicas.
- La suite de tests pase completamente.
- El notebook pueda ejecutarse secuencialmente sin errores.
- La documentación Markdown explique las nuevas fórmulas y decisiones.
- No existan cambios accidentales en el algoritmo original.

Al finalizar, proporcionar un resumen de los archivos modificados, nuevas APIs, decisiones técnicas, tests ejecutados y cualquier diferencia pendiente respecto de C4.5 original.

**Implementá los cambios en el repositorio. No te limites a proponer pseudocódigo o describir las modificaciones.**

## Tasks

- T1 | S1-S3,S6 | delegated: selection worker; parallel independent behavior, isolated worktree | done | commit: fe33b86; 17 dedicated + combined74 + full143 GREEN; unchanged46-case SHA; native medium, reviewDue=false.
- T2 | S4,S6 | delegated: pruning worker; parallel independent behavior, isolated worktree | done | commit: 515008d; 14 dedicated plus 35 existing tests GREEN; native medium, reviewDue=false.
- T3 | S6-S7 | inline: parent; baseline test drift and executable notebook isolation | done | commit: 2edd3e3; 108 tests GREEN, original notebook byte-for-byte unchanged.
- T4 | S5-S6 | inline: parent; widget integration depends on T1/T2 | in_progress (8 panel tests and actual kernel GREEN; committing final slice) | commit: pending; four control groups and structural results.
- T5 | S1-S7 | inline: parent functional/seam checks; independent verify only for named high risk | pending (functional proof complete; final delivery readback next) | commit: pending; complete suite143 and actual sequential Jupyter7cells + live widget GREEN.

## Log

### L1 — Original request (verbatim)

Archivo principal: `decision-tree-training.ipynb`

Necesito ampliar la implementación del árbol de decisión enriquecido, tomando como referencia el algoritmo C4.5 de J. Ross Quinlan y las implementaciones existentes de Weka J48.

Quiero implementar **cuatro mejoras adicionales**, integrándolas exclusivamente al árbol enriquecido.

## 1. Gain Ratio como criterio de selección

Actualmente `_seleccionar_prueba_enriquecida()` selecciona el atributo con mayor ganancia de información, aunque `evaluar_prueba()` ya calcula `gain_ratio` y `split_info`.

Implementar un criterio configurable:

`criterio = "gain" | "gain_ratio"`

Para `gain_ratio`, aplicar la estrategia de C4.5:

1. Evaluar todas las pruebas candidatas válidas.
2. Calcular la ganancia de información de cada atributo.
3. Calcular la ganancia promedio de los atributos elegibles.
4. Descartar para la selección por Gain Ratio aquellos atributos cuya ganancia sea inferior al promedio.
5. Entre los atributos restantes, seleccionar el de mayor Gain Ratio.
6. Conservar un criterio de desempate determinista.
7. Evitar divisiones por cero cuando `split_info = 0`.

Agregar parámetros:

- `criterio`: criterio de selección.
- `filtro_ganancia_promedio`: activar/desactivar el filtro.
- `min_gain_ratio`: umbral mínimo opcional para aceptar una división.

El cálculo debe ser compatible con atributos continuos, valores desconocidos y pesos fraccionarios.

Mantener `criterio="gain"` como comportamiento predeterminado para garantizar compatibilidad con la implementación actual.

**Importante:** distinguir claramente el valor utilizado para elegir un atributo del utilizado para determinar si una división supera `min_gain`.

## 2. Mínimo de observaciones y peso por hoja

Actualmente `min_samples_split` controla la cantidad mínima de filas necesaria para dividir un nodo, pero no garantiza un mínimo de observaciones en las hojas resultantes.

Implementar:

- `min_samples_leaf`: mínimo de observaciones para aceptar cada rama resultante.
- `min_weight_leaf`: peso efectivo mínimo que debe recibir cada rama.

Requisitos:

1. Aplicar las restricciones a atributos continuos y categóricos.
2. Evaluarlas antes de aceptar una división.
3. Conservar `min_samples_split` como restricción independiente.
4. Mantener la compatibilidad con `min_peso_rama`, documentando las diferencias.
5. Evitar que una fila con valor desconocido, distribuida entre varias ramas, se contabilice incorrectamente como múltiples observaciones originales.
6. Diferenciar explícitamente conteo de observaciones y suma de pesos fraccionarios.
7. Rechazar divisiones que produzcan ramas insuficientes, en vez de generar estructuras inválidas.
8. Conservar las reglas de parada existentes.

Definir y documentar una semántica inequívoca para el conteo de observaciones con valores desconocidos. Por ejemplo, exigir `min_samples_leaf` sobre observaciones con valor conocido y `min_weight_leaf` sobre el peso total posterior al reparto fraccionario.

Agregar casos de prueba para variables categóricas con muchas categorías, ramas muy pequeñas y valores faltantes.

## 3. Penalización MDL para atributos continuos

Actualmente `evaluar_umbrales_continuos()` evalúa umbrales y selecciona candidatos según su ganancia, pero no incorpora una penalización por la cantidad de cortes posibles.

Implementar una corrección basada en *Minimum Description Length*, siguiendo el tratamiento de atributos continuos de C4.5.

Agregar:

`correccion_mdl = True | False`

La corrección deberá considerar la cantidad de puntos de corte candidatos disponibles en el nodo.

Como referencia, para datos completos sin pesos:

`ganancia_ajustada = ganancia - log2(K) / N`

Donde `K` representa el número de posibles cortes y `N` el número de observaciones correspondiente.

Requisitos:

1. Verificar la formulación original de Quinlan antes de implementarla.
2. Separar la ganancia original de la ganancia ajustada.
3. Documentar cómo se adapta el cálculo a pesos y valores desconocidos.
4. Evitar aplicar penalizaciones cuando no existen cortes válidos.
5. No alterar el cálculo original de la entropía.
6. Permitir comparar resultados con MDL activado y desactivado.
7. Mantener una distinción entre la selección del mejor umbral de cada atributo y la comparación final entre atributos.
8. Hacer que la corrección pueda combinarse con Gain Ratio sin penalizar dos veces la misma decisión.

Conservar en los resultados de los candidatos, cuando corresponda, los campos `gain`, `gain_ajustada`, `gain_ratio`, `threshold` y `mdl_penalty`.

La corrección MDL debe quedar desactivada por defecto para conservar los resultados anteriores.

## 4. Métodos alternativos de poda

Actualmente existe poda pesimista mediante `_podar_enriquecido()`.

No eliminarla ni reemplazarla. Incorporar una selección de métodos:

`metodo_poda = "pesimista" | "reduced_error" | "cost_complexity" | "ninguna"`

### 4.1 Poda pesimista

Conservar la lógica existente de C4.5, incluyendo sus estimaciones de error.

Agregar:

`subtree_raising = True | False`

Este parámetro debe permitir activar o desactivar la elevación de subárboles sin desactivar el reemplazo de subárboles por hojas.

Mantener el comportamiento actual como predeterminado.

### 4.2 Reduced-error pruning

Implementar poda mediante un conjunto de validación independiente.

Requisitos:

- Admitir datos de validación explícitos.
- No utilizar los datos de prueba final para podar.
- Recorrer el árbol de abajo hacia arriba.
- Comparar el error del subárbol con el error de reemplazarlo por una hoja.
- Conservar una regla de desempate determinista.
- Soportar valores faltantes y pesos.
- Documentar cómo se evalúan las predicciones probabilísticas durante la poda.

No mezclar las estimaciones pesimistas con los errores de validación de este método.

### 4.3 Cost-complexity pruning

Implementar una alternativa inspirada en CART, utilizando un parámetro de complejidad:

`ccp_alpha >= 0`

Considerar un objetivo de la forma:

`R_alpha(T) = R(T) + alpha * hojas(T)`

Donde `R(T)` representa el error de clasificación ponderado del árbol.

Requisitos:

- Permitir configurar `ccp_alpha`.
- Comparar el costo de conservar un subárbol con el de reemplazarlo por una hoja.
- Utilizar la misma definición de riesgo en todas las alternativas.
- No alterar los modelos originales.
- Documentar cualquier diferencia respecto del procedimiento completo de weakest-link pruning de CART.

### 4.4 Desactivar poda

El modo `"ninguna"` debe devolver el árbol crecido sin modificaciones posteriores.

Conservar el argumento existente `podar` para compatibilidad. Definir claramente su interacción con `metodo_poda`, evitando combinaciones contradictorias o silenciosamente ambiguas.

## 5. Integración con ipywidgets

Ampliar `crear_panel_arbol_enriquecido()` para incluir los nuevos parámetros.

Organizar visualmente los controles en cuatro grupos:

**División**

- Criterio: Gain / Gain Ratio.
- Filtro de ganancia promedio.
- Gain Ratio mínimo.
- Corrección MDL.
- Selección de atributos continuos.

**Crecimiento**

- Ganancia mínima.
- Profundidad máxima.
- Mínimo de observaciones para dividir.
- Mínimo de observaciones por hoja.
- Peso mínimo por hoja.
- Peso mínimo por lado para continuos.

**Poda**

- Método de poda.
- Factor de confianza.
- Habilitar elevación de subárboles.
- Parámetro `ccp_alpha`.
- Parámetros necesarios para reduced-error pruning.

**Resultados**

- Árbol generado.
- Profundidad alcanzada.
- Cantidad de nodos.
- Cantidad de hojas.
- Parámetros utilizados.
- Comparación estructural antes y después de podar.

Los controles deben habilitarse o deshabilitarse según el método seleccionado.

Por ejemplo, `ccp_alpha` solamente debe habilitarse cuando se selecciona cost-complexity pruning.

Mantener el funcionamiento dentro de Jupyter/Google Colab, sin requerir una aplicación web externa.

## 6. Compatibilidad y arquitectura

Estas restricciones son obligatorias:

1. No modificar las funciones del algoritmo original `decision_tree`, `Node`, `predict_one` ni su entrenamiento.
2. Conservar las APIs existentes del árbol enriquecido.
3. Añadir nuevos parámetros sin romper las llamadas posicionales actuales; preferir argumentos nuevos exclusivamente por nombre.
4. No introducir dependencias innecesarias.
5. Mantener los nombres de las funciones existentes en español.
6. Documentar qué funcionalidades pertenecen a C4.5 y cuáles son extensiones inspiradas en CART u otros algoritmos.
7. No alterar las matrices originales ni los pesos proporcionados por el usuario.
8. Mantener resultados reproducibles y desempates deterministas.
9. Preservar los ejemplos y demostraciones ya existentes.
10. No reestructurar las celdas originales salvo que sea estrictamente necesario.

Si algún parámetro nuevo requiere datos que no están disponibles, como un conjunto de validación, generar un error claro en lugar de utilizar implícitamente los datos de entrenamiento o prueba.

## 7. Criterios de finalización

La tarea se considera completa cuando:

- Las cuatro mejoras estén implementadas y conectadas al entrenador enriquecido.
- Los nuevos parámetros puedan configurarse desde el panel.
- Los comportamientos predeterminados anteriores se mantengan.
- Las nuevas funcionalidades tengan pruebas específicas.
- La suite de tests pase completamente.
- El notebook pueda ejecutarse secuencialmente sin errores.
- La documentación Markdown explique las nuevas fórmulas y decisiones.
- No existan cambios accidentales en el algoritmo original.

Al finalizar, proporcionar un resumen de los archivos modificados, nuevas APIs, decisiones técnicas, tests ejecutados y cualquier diferencia pendiente respecto de C4.5 original.

**Implementá los cambios en el repositorio. No te limites a proponer pseudocódigo o describir las modificaciones.**

### L2 — Baseline and routing evidence

Clean main at b20785e. Training notebook has no original algorithm/training cell after that commit; do NOT restore it. Initial .venv unittest discovery: 102 tests, 1 failure and 7 errors (stale tests expecting original bindings; sklearn missing). System Python has sklearn; .venv has ipywidgets. Research: Quinlan (1996), Improved Use of Continuous Attributes in C4.5, pp. 79–80 (PDF 3–4) confirms log2(Ndistinct-1)/|D| and best threshold by gain before comparing attributes by adjusted gain ratio. Weka C45Split.java counts eligible numeric cut positions and divides by total node weight. No class-boundary-only reduction of K. Weka model selection adds a high-cardinality heuristic not required here.

### L3 — User delivery choice (verbatim)

Cadena sobre una rama de feature (recomendado)

### L4 — T3 verification

Committed 2edd3e3: existing suite 108/108 GREEN using `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(/usr/bin/python3 -m site --user-site)" .venv/bin/python -m unittest discover -v`. Restored original 46-case behavioral SHA without changing its expected value. Original models remain separate and unchanged. No dependency installation; system user-site supplies existing sklearn. Native committed-range assessment unavailable (generated untracked pycache); no review started, no named high-risk item in this test/docs unit. Own generated untracked caches removed and future runs disable bytecode. Rollback boundary: explicit original-model test fixture and normalized notebook outputs/introduction; no algorithm change.


### L5 — Pruning delivery and slice boundaries

T2 committed as 515008d5f2ee23e6e878c350155066443b137b93 after parent spot-check: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest -v test_enriched_pruning test_decision_tree_training` = 49 GREEN. Native committed assessment from 2edd3e3: medium, 417 changed lines, reviewDue=false/under_budget, self-verification sufficient; no independent verifier or review started. Risk: none (only requested enriched paths; original algorithms and helpers unchanged). Rollback: new pruning dispatch/helpers/stats and test_enriched_pruning.py. Fixed-alpha whole-tree greedy objective explicitly documented, not full weakest-link CART.

Feature-branch-chain delivery remains local only. Slice 0 (test-fixture/bootstrap): b20785e→2edd3e3, includes generated verbatim ODD specs/log; slice 1 (pruning): 2edd3e3→515008d, 417 authored changed lines including tests/docs. This is one coherent behavior, close to the advisory400 target. Next selection slice starts at515008d; UI follows separately. No push or PR authorized.

### L6 — Integration routing

Selection worker returned partial solely for stale-baseline test expectations, with17 dedicated GREEN and10 focused existing GREEN. Runtime context backstop fired; bounded worker mvty5pgv-4-9lzt integrates T1 by function boundaries into parent, preserves pruning/UI, reconciles intentional helper-name expectations, and runs combined feature tests. Parent will run full discovery and real Jupyter kernel once integration returns.


### L7 — Selection integration and final functional proof

Integration worker mvty5pgv-4-9lzt completed: combined feature tests74/74 GREEN, including8 real-widget tests and unchanged46-case SHA; no pruning/panel implementation changes. Its generic TDD-not-active text is not lifecycle evidence: original selection/pruning writers and parent UI observed RED then GREEN; integration reused that implementation without new behavior. T1 committed as fe33b86da7ead83785cac862a237ae40d4434da3; slice2 starts515008d→fe33b86,440 authored changed lines (gain/MDL/leaf feasibility are one coherent selection pipeline with tests/docs). Native committed assessment medium, reviewDue=false/under_budget; no reviewer/independent verifier required. Risk:none; original models and info/log2 unchanged.

Parent full discovery: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(/usr/bin/python3 -m site --user-site)" .venv/bin/python -m unittest discover -v` =143/143 GREEN. `git diff --exit-code b20785e -- decision-tree.ipynb random-forest.ipynb requirements.txt` =0 (no original-model/dependency edits). `git diff --check` clean. Actual Jupyter KernelManager job2 ran all7 code cells sequentially and clicked the live train button; all passed, with expected structural summary and no original model imported. Non-failing environment warning: local kernel TCP transport is unencrypted; no code execution errors. No browser screenshot or hosted Colab run performed; real widgets+local kernel prove execution, Colab instructions documented.

T1 rollback: selection helpers/growth/trainer kwargs plus test_enriched_selection.py and associated docs. Slice3 (UI) starts atfe33b86; parent now commits already-tested panel code, tests and docs. No publishing.
