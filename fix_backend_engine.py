import re

with open("backend/main.py", "r") as f:
    content = f.read()

# Add global engine initialization
init_engine_str = """
from core.depth_engine import DepthEngine
depth_engine = None

@app.on_event("startup")
async def startup_event():
    \"\"\"Load configuration once at startup.\"\"\"
    global main_loop, depth_engine
    main_loop = asyncio.get_running_loop()
    depth_engine = DepthEngine(cfg)
"""

content = re.sub(
    r'@app\.on_event\("startup"\)\nasync def startup_event\(\):\n\s+"""Load configuration once at startup\."""\n\s+global main_loop\n\s+main_loop = asyncio\.get_running_loop\(\)\n',
    init_engine_str,
    content,
    flags=re.MULTILINE
)

# Update _run_pipeline_job to use the global engine
replace_engine_str = """
        global depth_engine
        engine = depth_engine
        depth_result = engine.estimate(
"""
content = re.sub(
    r'\s+from core\.depth_engine import DepthEngine\n+\s+engine = DepthEngine\(cfg\)\n\s+depth_result = engine\.estimate\(',
    replace_engine_str,
    content
)

with open("backend/main.py", "w") as f:
    f.write(content)
