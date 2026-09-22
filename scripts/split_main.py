import re
with open(r'n:\AERIS-3D-main\backend\main.py', 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find('HTML_PAGE = """<!DOCTYPE html>')
if idx != -1:
    python_part = content[:idx]
    with open(r'n:\AERIS-3D-main\backend\python_part.py', 'w', encoding='utf-8') as f2:
        f2.write(python_part)
    
    html_part = content[idx + len('HTML_PAGE = """'): -3]
    with open(r'n:\AERIS-3D-main\backend\viewer.html', 'w', encoding='utf-8') as f3:
        f3.write('<!DOCTYPE html>' + html_part)
    print('Split successful')
else:
    print('Could not find HTML_PAGE')
