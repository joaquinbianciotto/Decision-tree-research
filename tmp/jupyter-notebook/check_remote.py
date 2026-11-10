import json, subprocess, urllib.request
from pathlib import Path
origin=subprocess.check_output(['git','remote','get-url','origin'],text=True).strip()
if origin.startswith('git@github.com:'): repo=origin.split(':',1)[1]
elif origin.startswith('https://github.com/'): repo=origin[len('https://github.com/'):]
else: raise RuntimeError('Origin is not a recognized GitHub URL; no remote request made')
repo=repo.removesuffix('.git')
sha='ff9469c9df58abae89bce321cac08aae64e8938b'
for name in ['decision-tree.ipynb','random-forest.ipynb']:
 data=urllib.request.urlopen(f'https://raw.githubusercontent.com/{repo}/{sha}/{name}',timeout=30).read()
 p=Path('tmp/jupyter-notebook')/('remote-'+name);p.write_bytes(data)
 remote=json.loads(data); local=json.loads(Path(name).read_text(encoding='utf8'))
 remote_code={c.get('id'):''.join(c.get('source',[])) for c in remote['cells'] if c['cell_type']=='code'}
 local_code={c.get('id'):''.join(c.get('source',[])) for c in local['cells'] if c['cell_type']=='code'}
 print(name, 'same_bytes=', data==Path(name).read_bytes(), 'remote_cells=',len(remote['cells']))
 print('Changed code IDs:', [k for k in set(remote_code)|set(local_code) if remote_code.get(k)!=local_code.get(k)])
 if name.startswith('random'):
  for k,v in remote_code.items():
   if any(t in v for t in ['fetch_ucirepo','RandomForestClassifier','train_test_split','dropna']):print(k,v[:550])
