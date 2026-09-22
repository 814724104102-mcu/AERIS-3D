with open(r'n:\AERIS-3D-main\backend\python_part.py', 'r', encoding='utf-8') as f:
    python_part = f.read()

with open(r'n:\AERIS-3D-main\backend\viewer.html', 'r', encoding='utf-8') as f:
    html_part = f.read()
    
if html_part.startswith('<!DOCTYPE html>'):
    html_part = html_part[len('<!DOCTYPE html>'):]

new_content = python_part + 'HTML_PAGE = """<!DOCTYPE html>' + html_part + '"""\n\n'
new_content += '@app.get("/", response_class=HTMLResponse)\n'
new_content += 'def index():\n'
new_content += '    return HTML_PAGE\n\n'
new_content += 'if __name__ == "__main__":\n'
new_content += '    import uvicorn\n'
new_content += '    uvicorn.run(app, host="127.0.0.1", port=8000)\n'

with open(r'n:\AERIS-3D-main\backend\main.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
    
print('Successfully assembled main.py')
