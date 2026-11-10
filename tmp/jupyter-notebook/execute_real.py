"""Execute the existing notebook in a fresh local kernel, preserving source defaults."""
import json
import os
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[2]
output = root / 'output' / 'jupyter-notebook'
output.mkdir(parents=True, exist_ok=True)
kernels = root / 'tmp' / 'jupyter-notebook' / 'kernels' / 'notebook-pruebas'
kernels.mkdir(parents=True, exist_ok=True)
(kernels / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Notebook pruebas local', 'language': 'python'}), encoding='utf8')
os.environ['JUPYTER_PATH'] = str(kernels.parent.parent)
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(root / 'tmp' / 'jupyter-notebook' / 'matplotlib')
notebook = nbformat.read(root / 'notebook-pruebas.ipynb', as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    if cell.id == 'ejecucion':
        cell.source = cell.source.replace('EJECUTAR_PRUEBAS = False', 'EJECUTAR_PRUEBAS = True')
        cell.source = cell.source.replace('Path("resultados_notebook_pruebas")', 'Path("output/jupyter-notebook/resultados")')
    if cell.id == 'final-control':
        cell.source = cell.source.replace('EVALUAR_PRUEBA_FINAL = False', 'EVALUAR_PRUEBA_FINAL = True')
start = time.perf_counter()
client = NotebookClient(notebook, timeout=1800, kernel_name='notebook-pruebas', resources={'metadata': {'path': str(root)}})
client.on_cell_start = lambda cell, cell_index: print(f'Cell {cell_index}: {cell.get("id")}', flush=True)
try:
    client.execute()
finally:
    nbformat.write(notebook, output / 'notebook-pruebas-ejecutado.ipynb')
    print(f'Elapsed seconds: {time.perf_counter() - start:.3f}', flush=True)
nbformat.validate(notebook)
print('Execution and nbformat validation succeeded', flush=True)
