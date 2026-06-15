import importlib.util
from pathlib import Path


def test_streamlit_app_imports() -> None:
    path = Path("app/streamlit_app.py")
    spec = importlib.util.spec_from_file_location("streamlit_app", path)

    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
