"""Execute Package_Damage_Detection.ipynb and export it to PDF (via HTML + Microsoft Edge headless)."""
import os, subprocess, pathlib
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

HERE = pathlib.Path(__file__).resolve().parent
NB = HERE / 'Package_Damage_Detection.ipynb'
HTML = HERE / 'Package_Damage_Detection.html'
PDF = HERE / 'Package_Damage_Detection.pdf'

nb = nbformat.read(NB, as_version=4)
NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(HERE)}}).execute()
nbformat.write(nb, NB)
print('Executed notebook')

body, _ = HTMLExporter(template_name='classic').from_notebook_node(nb)
body = body.replace('</head>', '<style>@page{size:A4;margin:12mm} body{-webkit-print-color-adjust:exact}'
                                '.prompt{min-width:0!important;font-size:9px}</style></head>')
HTML.write_text(body, encoding='utf-8')
print('Wrote', HTML.name)

edge = next(p for p in [r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'] if os.path.exists(p))
subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                f'--print-to-pdf={PDF}', HTML.as_uri()], check=True, timeout=180)
print('Wrote', PDF.name)
