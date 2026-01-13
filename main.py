"""Entry point for LiverSeg 3D (Modern UI, English).

Run:
  python main.py

Note: Place model weights in ./checkpoints according to config.py.
"""

import os
import sys
import logging

# Ensure local imports work when launched directly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

log_path = os.path.join(BASE_DIR, "liverseg3d.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
)
logger = logging.getLogger("liverseg3d")


def main():
    try:
        from PyQt6.QtWidgets import QApplication

        from gui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("LiverSeg 3D")
        app.setOrganizationName("LiverSeg Team")

        window = MainWindow()
        window.show()

        sys.exit(app.exec())

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        raise


if __name__ == "__main__":
    main()
