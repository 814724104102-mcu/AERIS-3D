import re
with open(r'n:\AERIS-3D-main\backend\main.py', 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find('</html>')
if idx != -1:
    clean_html = content[:idx + len('</html>')]
    # find where HTML_PAGE starts
    start_idx = clean_html.find('HTML_PAGE = """')
    if start_idx != -1:
        # just append the python footer properly
        new_content = clean_html + '\n"""\n\n'
        new_content += '@app.get("/", response_class=HTMLResponse)\n'
        new_content += 'def index():\n'
        new_content += '    return HTML_PAGE\n\n'
        new_content += 'if __name__ == "__main__":\n'
        new_content += '    import uvicorn\n'
        new_content += '    uvicorn.run(app, host="127.0.0.1", port=8000)\n'
        
        with open(r'n:\AERIS-3D-main\backend\main.py', 'w', encoding='utf-8') as fw:
            fw.write(new_content)
        print('Fixed main.py')
    else:
        print('Could not find HTML_PAGE start')
else:
    print('Could not find </html>')
