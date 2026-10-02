"""Render linked offline README/manual previews with embedded showcase videos.

Run ``python -m scripts.reports.render_docs_preview --output-dir /path/to/preview``.
Requires markdown-it-py, Pygments, Matplotlib and Pillow. Generated previews use
browser color preferences, optional theme overrides and local copy controls.
"""

from pathlib import Path
import base64,io,re
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=root/'outputs/readme_preview')
output=parser.parse_args().output_dir
output.mkdir(parents=True, exist_ok=True)
source=(root/'README.md').read_text()
def encoded(path,mime):
 return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()
def equation(match):
 formula=' '.join(match.group(1).split()).replace('\\qquad','\\quad')
 if '\\begin{aligned}' in formula:
  formula=formula.replace('\\begin{aligned}','').replace('\\end{aligned}','').replace('&','')
  rows=[row.strip() for row in formula.split('\\\\') if row.strip()]
 else:
  rows=[formula]
 fig=plt.figure(figsize=(9,max(.4,.32*len(rows))))
 labels=[fig.text(.5,1-(i+.5)/len(rows),'$'+row+'$',ha='center',va='center',fontsize=12) for i,row in enumerate(rows)]
 fig.canvas.draw()
 from matplotlib.transforms import Bbox
 bounds=Bbox.union([label.get_window_extent(fig.canvas.get_renderer()) for label in labels]).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.08)
 buf=io.BytesIO();fig.savefig(buf,format='svg',bbox_inches=bounds,pad_inches=.02,transparent=True);plt.close(fig)
 return '<div class="equation"><img alt="Guiding-center equations, wall conditions, and BDF2 time step" src="data:image/svg+xml;base64,'+base64.b64encode(buf.getvalue()).decode()+'"></div>'
source=re.sub(r'\$\$(.*?)\$\$',equation,source,flags=re.S)
def code(text,language,*args):
 try:return '<pre class="highlight"><code>'+highlight(text,get_lexer_by_name(language),HtmlFormatter(nowrap=True))+'</code></pre>'
 except Exception:return ''
def render_markdown(source):
 parser=MarkdownIt('commonmark',{'html':True,'highlight':code}).enable('table').enable('strikethrough')
 tokens=parser.parse(source)
 used={}
 for i,token in enumerate(tokens):
  if token.type=='heading_open':
   title=tokens[i+1].content
   slug=re.sub(r'[^\w -]','',title.lower()).replace(' ','-')
   count=used.get(slug,0);used[slug]=count+1
   token.attrSet('id',slug+('-'+str(count) if count else ''))
 return parser.renderer.render(tokens,parser.options,{})
body=render_markdown(source)
body=re.sub(r'(<pre\b.*?</pre>)',r'<div class="code-block"><button class="copy-code" type="button" aria-label="Copy code">Copy</button>\1</div>',body,flags=re.S)
for asset,run in [('vortex_gas','signed_200mb'),('positive_density','positive_200mb')]:
 media=root/'docs/getting_started/media'
 movie=media/f'{asset}.mp4'
 from PIL import Image
 with Image.open(media/f'{asset}.gif') as frame:
  buffer=io.BytesIO();frame.convert('RGB').save(buffer,format='PNG')
 poster_url='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
 player=('<div class="simulation-player"><video id="'+asset+'" controls playsinline preload="none" poster="'+poster_url+'" src="'+encoded(movie,'video/mp4')+'"></video>'
 '<button class="simulation-play" aria-label="Play simulation" onclick="this.previousElementSibling.play();this.hidden=true">▶</button></div>')
 target=f'docs/getting_started/media/{asset}.gif'
 body=re.sub(r'<img src="'+re.escape(target)+r'"[^>]*>',lambda _:player,body)
 body=re.sub(r'<p><a href="docs/getting_started/media/'+asset+r'\.mp4">.*?</a></p>','',body)
style='''body{margin:0;background:#f6f8fa;color:#24292f;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}main{max-width:1040px;margin:28px auto;padding:32px 42px;background:white;border:1px solid #d0d7de;border-radius:8px}h1,h2{line-height:1.25;border-bottom:1px solid #d8dee4;padding-bottom:.35em}h2{margin-top:32px}a{color:#0969da;text-decoration:none}a:hover{text-decoration:underline}code{font-size:85%;background:#eff1f3;padding:2px 5px;border-radius:4px}pre{overflow:auto;padding:16px;background:#f6f8fa;border-radius:6px;line-height:1.5}pre code{padding:0;background:none}table{border-collapse:collapse;width:100%;font-size:94%}td,th{border:1px solid #d0d7de;padding:8px 12px;text-align:left}tr:nth-child(even){background:#f6f8fa}img{max-width:100%}.equation{text-align:center;padding:8px 0}.equation img{width:auto;height:auto;max-width:100%}.code-block{position:relative;margin:16px 0}.code-block pre{margin:0;padding:12px 20px 12px 20px;white-space:pre;font:16px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}.code-block pre code{font:inherit}.copy-code{position:absolute;top:7px;right:7px;padding:3px 7px;border:1px solid #d0d7de;border-radius:5px;background:white;color:#57606a;font-size:11px;cursor:pointer}.copy-code:hover{color:#24292f;background:#eef1f4}.simulation-player{position:relative;display:block}.simulation-player video{display:block;width:100%}.simulation-play{position:absolute;bottom:48px;right:20px;border:1px solid #aaa;border-radius:50%;background:#ffffffc9;color:#333;width:34px;height:34px;cursor:pointer;font-size:13px}.simulation-play[hidden]{display:none}@media(max-width:700px){main{margin:0;padding:18px;border:0}table{display:block;overflow:auto}}'''+HtmlFormatter(style='friendly').get_style_defs('html[data-theme="light"] .highlight')+HtmlFormatter(style='github-dark').get_style_defs('html[data-theme="dark"] .highlight')

style += """
@media(max-width:700px){.code-block pre{font-size:13px;padding-left:12px;padding-right:12px}}
.preview-nav{display:flex;gap:18px;font-size:14px;border-bottom:1px solid #d0d7de;padding-bottom:10px;margin-bottom:22px}html[data-theme="dark"] .preview-nav{border-color:#30363d}h2,h3{scroll-margin-top:20px}details.manual-contents{padding:12px 16px;border:1px solid #d0d7de;border-radius:6px;margin:20px 0}details.manual-contents summary{cursor:pointer}html[data-theme="dark"] details.manual-contents{border-color:#30363d}
html{color-scheme:light}html[data-theme="dark"]{color-scheme:dark}
.theme-bar{position:fixed;right:14px;top:14px;z-index:10}.theme-toggle{border:1px solid #d0d7de;border-radius:6px;width:34px;height:34px;padding:0;background:#f6f8fa;color:#57606a;cursor:pointer;font-size:18px;box-shadow:0 1px 5px #0002}
.code-block pre{padding-top:18px;padding-bottom:18px;line-height:1.6;border:1px solid #e5e9ef}
html[data-theme="light"] .highlight{background:#f6f8fa;color:#24292f}
html[data-theme="light"] .highlight .k,html[data-theme="light"] .highlight .kn,html[data-theme="light"] .highlight .ow{color:#cf222e;font-weight:600}
html[data-theme="light"] .highlight .s,html[data-theme="light"] .highlight .s1,html[data-theme="light"] .highlight .s2,html[data-theme="light"] .highlight .sa{color:#0a3069}
html[data-theme="light"] .highlight .nf,html[data-theme="light"] .highlight .nc,html[data-theme="light"] .highlight .nb{color:#8250df}
html[data-theme="light"] .highlight .mi,html[data-theme="light"] .highlight .mf,html[data-theme="light"] .highlight .o{color:#0550ae}
html[data-theme="light"] .highlight .c,html[data-theme="light"] .highlight .c1,html[data-theme="light"] .highlight .cm{color:#6e7781;font-style:normal}
html[data-theme="dark"] body{background:#0d1117;color:#e6edf3}
html[data-theme="dark"] main{background:#161b22;border-color:#30363d}
html[data-theme="dark"] h1,html[data-theme="dark"] h2{border-color:#30363d}
html[data-theme="dark"] a{color:#79c0ff}
html[data-theme="dark"] code{background:#292f38;color:#e6edf3}
html[data-theme="dark"] .code-block pre{background:#0d1117;color:#e6edf3;border-color:#30363d}
html[data-theme="dark"] pre code{background:none;color:inherit}
html[data-theme="dark"] td,html[data-theme="dark"] th{border-color:#30363d}
html[data-theme="dark"] tr:nth-child(even){background:#1c2129}
html[data-theme="dark"] .copy-code,html[data-theme="dark"] .theme-toggle{background:#21262d;color:#c9d1d9;border-color:#484f58}
html[data-theme="dark"] .copy-code:hover,html[data-theme="dark"] .theme-toggle:hover{background:#30363d;color:white}
html[data-theme="dark"] .equation img{filter:invert(1)}
html[data-theme="dark"] .highlight .c,html[data-theme="dark"] .highlight .c1,html[data-theme="dark"] .highlight .cm{color:#8b949e}
"""
script = """<script>
const themeButton=document.getElementById('theme-toggle');
const preference=window.matchMedia('(prefers-color-scheme: dark)');
let mode='auto';
try{const saved=localStorage.getItem('hdgfem-preview-theme');if(['auto','light','dark'].includes(saved))mode=saved;}catch(e){}
function setTheme(){const theme=mode==='auto'?(preference.matches?'dark':'light'):mode;document.documentElement.dataset.theme=theme;themeButton.textContent=mode==='auto'?'◐':mode==='dark'?'☾':'☀';themeButton.title='Theme: '+mode+'. Click to cycle Auto → Light → Dark.';themeButton.setAttribute('aria-label',themeButton.title);themeButton.setAttribute('aria-pressed',String(mode!=='auto'));}
setTheme();
themeButton.addEventListener('click',()=>{mode={auto:'light',light:'dark',dark:'auto'}[mode];try{localStorage.setItem('hdgfem-preview-theme',mode);}catch(e){}setTheme();});
if(preference.addEventListener)preference.addEventListener('change',()=>{if(mode==='auto')setTheme();});else preference.addListener(()=>{if(mode==='auto')setTheme();});
function fallbackCopy(text){const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();const ok=document.execCommand('copy');area.remove();if(!ok)throw Error('Copy failed');}
document.querySelectorAll('.copy-code').forEach(button=>button.addEventListener('click',async()=>{const text=button.parentElement.querySelector('pre').textContent;try{try{if(!navigator.clipboard)throw Error('Clipboard unavailable');await navigator.clipboard.writeText(text);}catch(e){fallbackCopy(text);}button.textContent='Copied';}catch(e){button.textContent='Select to copy';const selection=window.getSelection();const range=document.createRange();range.selectNodeContents(button.parentElement.querySelector('pre'));selection.removeAllRanges();selection.addRange(range);}setTimeout(()=>button.textContent='Copy',1800);}));
</script>"""
startup="<script>try{const m=localStorage.getItem('hdgfem-preview-theme');document.documentElement.dataset.theme=(m==='light'||m==='dark')?m:(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');}catch(e){document.documentElement.dataset.theme=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}</script>"
from urllib.parse import urlsplit,unquote
import html as html_module

def local_links(body):
 def link(match):
  href=html_module.unescape(match.group(1));parts=urlsplit(href)
  if parts.scheme or not parts.path:return match.group(0)
  if parts.path in ('README.md','MANUAL.md'):
   target='hdgfem-readme-portable.html' if parts.path=='README.md' else 'hdgfem-manual.html'
   return 'href="'+target+('#'+parts.fragment if parts.fragment else '')+'"'
  target=root/unquote(parts.path)
  if target.exists():return 'href="'+target.as_uri()+('#'+parts.fragment if parts.fragment else '')+'"'
  return match.group(0)
 return re.sub(r'href="([^"]+)"',link,body)

nav='<nav class="preview-nav" aria-label="Documentation"><a href="hdgfem-readme-portable.html">README</a><a href="hdgfem-manual.html">Manual</a></nav>'
def document(body,title,theme='light'):
 return '<!doctype html><html lang="en" data-theme="'+theme+'"><head><meta charset="utf-8">'+startup+'<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+style+'</style></head><body><main><div class="theme-bar"><button id="theme-toggle" class="theme-toggle" type="button" aria-label="Choose color theme">◐</button></div>'+nav+local_links(body)+'</main>'+script+'</body></html>'

manual_source=(root/'MANUAL.md').read_text()
manual_source=re.sub(r'\$\$(.*?)\$\$',equation,manual_source,flags=re.S)
manual_body=render_markdown(manual_source)
manual_body=re.sub(r'(<pre\b.*?</pre>)',r'<div class="code-block"><button class="copy-code" type="button" aria-label="Copy code">Copy</button>\1</div>',manual_body,flags=re.S)
contents=[]
for anchor,title in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>',manual_body):
 contents.append('<li><a href="#'+anchor+'">'+title+'</a></li>')
manual_body=re.sub(r'(</h1>)',lambda m:m.group(1)+'<details class="manual-contents"><summary>Contents</summary><ul>'+''.join(contents)+'</ul></details>',manual_body,count=1)
for name,title,content,theme in [
 ('hdgfem-readme-portable.html','hdgfem README',body,'light'),
 ('hdgfem-readme-preview.html','hdgfem README',body,'light'),
 ('hdgfem-readme-dark.html','hdgfem README',body,'dark'),
 ('hdgfem-manual.html','hdgfem Manual',manual_body,'light')]:
 rendered=document(content,title,theme)
 for directory in (output,):
  path=directory/name;temporary=path.with_suffix('.html.tmp');temporary.write_text(rendered);temporary.replace(path)
  print(str(path),f'{path.stat().st_size/1e6:.2f} MB')
