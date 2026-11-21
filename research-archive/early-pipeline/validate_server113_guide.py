from pathlib import Path
import re, subprocess, json
text=Path('/mnt/d/00/工作/SMPL-X/00_部署与训练指南.md').read_text()
blocks=re.findall(r'```bash\n(.*?)```',text,re.S)
for i,block in enumerate(blocks):
    r=subprocess.run(['bash','-n'],input=block,text=True,capture_output=True)
    assert r.returncode==0,(i,r.stderr)
print(json.dumps(dict(shell_blocks_syntax_passed=len(blocks),commands_executed=False)))
