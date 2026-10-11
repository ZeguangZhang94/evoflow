"""测试公共配置：未安装包时直接从 src 布局导入，保证裸环境可跑测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
