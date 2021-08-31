import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

NYUv2_PATH = os.path.join(ROOT, 'data/NYUv2')
NYUv2_EXISTS = (
    os.path.exists(NYUv2_PATH)
    and len(os.listdir(NYUv2_PATH)) > 0
)
