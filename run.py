"""Recognize license plates using the packaged offline runtime."""
from pathlib import Path
import runpy,sys
if __name__=='__main__':
    root=Path(__file__).resolve().parent/'solution'
    sys.path.insert(0,str(root))
    runpy.run_path(str(root/'run.py'),run_name='__main__')
