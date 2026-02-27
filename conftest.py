import os
import sys


def pytest_configure():
    # Ensure the project root is on sys.path so tests can import `agent`, `prompts`, etc.
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

