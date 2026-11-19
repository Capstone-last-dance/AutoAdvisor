"""스크래치 스크립트 실행기: python -I run.py <script.py> [args...]
-I(격리 모드)는 스크립트 폴더를 sys.path에 넣지 않으므로 여기서 넣고 실행한다."""
import runpy
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(str(here / script), run_name="__main__")
