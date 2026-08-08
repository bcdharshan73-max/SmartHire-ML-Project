import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

if __name__ == "__main__":
    import streamlit.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "APPS" / "streamlit_app.py"),
        "--server.port=8000",
        "--server.headless=true",
    ]

    sys.exit(stcli.main())
