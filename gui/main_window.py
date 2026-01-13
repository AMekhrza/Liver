"""Modern English UI for LiverSeg 3D.

This UI is intentionally different in layout from the reference project:
- Left navigation rail (Data / Segmentation / Results / Logs)
- Dashboard-like pages with cards
- Tabs for 2D + 3D views

Runs on PyQt6 (fallback PyQt5).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, pyqtSlot, QSettings
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QFileDialog,
    QProgressBar,
    QSplitter,
    QStackedWidget,
    QFrame,
    QTabWidget,
    QTextEdit,
    QSlider,
    QCheckBox,
    QMessageBox,
    QToolBar,
    QSizePolicy,
    QGroupBox,
    QFormLayout,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception as e:
    pv = None
    QtInteractor = None

try:
    from skimage import measure
except Exception:
    measure = None

try:
    from gui.workers import DicomLoaderWorker, SegmentationWorker, VolumeCalculationWorker, TotalSegmentatorWorker
    from utils.dicom_loader import DicomLoader
    from config import get_model_path, validate_model_paths
except ImportError:
    # If executed as a package
    from .workers import DicomLoaderWorker, SegmentationWorker, VolumeCalculationWorker, TotalSegmentatorWorker
    from ..utils.dicom_loader import DicomLoader
    from ..config import get_model_path, validate_model_paths


DARK_QSS = """
QMainWindow { background: #0f172a; }
QWidget { color: #e5e7eb; font-family: Segoe UI; font-size: 11pt; }
QLineEdit, QComboBox, QListWidget, QTextEdit {
  background: #111827; border: 1px solid #24324a; border-radius: 10px; padding: 8px;
}
QGroupBox {
  border: 1px solid #24324a; border-radius: 12px; margin-top: 10px; padding: 10px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #93c5fd; }
QPushButton {
  background: #2563eb; border: 0; border-radius: 10px; padding: 8px 12px; font-weight: 600;
}
QPushButton:disabled { background: #334155; color: #9ca3af; }
QPushButton#secondary {
  background: #111827; border: 1px solid #24324a;
}
QPushButton#danger { background: #b91c1c; }
QProgressBar { border: 1px solid #24324a; border-radius: 10px; text-align: center; background: #111827; }
QProgressBar::chunk { background: #22c55e; border-radius: 10px; }
QToolBar { background: #0b1220; border: 0; }
QToolBar QToolButton { background: transparent; border-radius: 10px; padding: 6px 10px; }
QFrame#nav { background: #0b1220; border-right: 1px solid #24324a; }
QListWidget#navList { background: transparent; border: 0; padding: 8px; }
QListWidget#navList::item { padding: 10px 10px; border-radius: 10px; }
QListWidget#navList::item:selected { background: #1f2937; border: 1px solid #24324a; }
"""


@dataclass
class SeriesSelection:
    folder: str = ""
    series_id: str = ""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LiverSeg 3D — Liver Segmentation")
        self.setMinimumSize(1280, 760)

        self.selection = SeriesSelection()
        self.current_image: Optional[np.ndarray] = None
        self.sitk_image = None  # Store SimpleITK image for TotalSegmentator
        self.current_mask: Optional[np.ndarray] = None
        self.current_voxel_volume_mm3: float = 1.0
        self.last_volume_ml: float = 0.0

        self.dicom_loader = DicomLoader()

        self.settings = QSettings("LiverSeg3D", "Settings")
        self._model_settings_keys = {
            "YOLOv11": "model_path_yolov11",
            "U-Net": "model_path_unet",
            "nnU-Net": "model_path_nnunet",
        }

        self.dicom_worker: Optional[DicomLoaderWorker] = None
        self.seg_worker: Optional[SegmentationWorker] = None
        self.vol_worker: Optional[VolumeCalculationWorker] = None

        self._build_toolbar()
        self._build_layout()
        self._apply_theme()

        self._log("Ready.")
        self._refresh_model_status()

    def _get_configured_model_path(self, model_name: str) -> str:
        key = self._model_settings_keys.get(model_name)
        if not key:
            return ""
        value = self.settings.value(key, "", type=str)
        return value or ""

    def _set_configured_model_path(self, model_name: str, path: str) -> None:
        key = self._model_settings_keys.get(model_name)
        if not key:
            return
        self.settings.setValue(key, path)

    def _get_effective_model_path(self, model_name: str) -> Optional[str]:
        configured = self._get_configured_model_path(model_name)
        if configured and os.path.exists(configured):
            return configured
        return get_model_path(model_name)

    def _apply_theme(self):
        self.setStyleSheet(DARK_QSS)

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self.title_label = QLabel("LiverSeg 3D")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: 800; color: #e5e7eb; padding: 4px 10px;")
        toolbar.addWidget(self.title_label)

        toolbar.addSeparator()

        action_select = QAction("Open DICOM Folder", self)
        action_select.triggered.connect(self.select_folder)
        toolbar.addAction(action_select)

        action_clear = QAction("Clear", self)
        action_clear.triggered.connect(self.clear_session)
        toolbar.addAction(action_clear)

        toolbar.addSeparator()

        action_about = QAction("About", self)
        action_about.triggered.connect(self.show_about)
        toolbar.addAction(action_about)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.status_chip = QLabel("GPU: checking…")
        self.status_chip.setStyleSheet(
            "background: #111827; border: 1px solid #24324a; border-radius: 999px; padding: 6px 10px;"
        )
        toolbar.addWidget(self.status_chip)

    def _build_layout(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # Navigation rail
        nav = QFrame()
        nav.setObjectName("nav")
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(10, 10, 10, 10)
        nav_layout.setSpacing(10)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        self.nav_list.setFixedWidth(220)
        for name in ["Data", "Segmentation", "Results", "Logs"]:
            item = QListWidgetItem(name)
            self.nav_list.addItem(item)
        self.nav_list.currentRowChanged.connect(self.on_nav_changed)
        nav_layout.addWidget(self.nav_list)

        nav_layout.addStretch(1)
        self.quick_run_btn = QPushButton("Run Segmentation")
        self.quick_run_btn.setEnabled(False)
        self.quick_run_btn.clicked.connect(self.start_segmentation)
        nav_layout.addWidget(self.quick_run_btn)

        main.addWidget(nav)

        # Pages
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_page_data())
        self.pages.addWidget(self._build_page_segmentation())
        self.pages.addWidget(self._build_page_results())
        self.pages.addWidget(self._build_page_logs())
        main.addWidget(self.pages)

        # Select default page after pages exist (avoids early signal crash)
        self.nav_list.setCurrentRow(0)

    def _build_card(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        return box

    def _build_page_data(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QLabel("Data Import")
        header.setStyleSheet("font-size: 18pt; font-weight: 800; color: #e5e7eb;")
        layout.addWidget(header)

        top = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select a folder that contains DICOM series…")
        btn_browse = QPushButton("Browse")
        btn_browse.setObjectName("secondary")
        btn_browse.clicked.connect(self.select_folder)
        top.addWidget(self.folder_edit, 1)
        top.addWidget(btn_browse)
        layout.addLayout(top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        # Series list
        series_card = self._build_card("DICOM Series")
        series_layout = QVBoxLayout(series_card)
        self.series_list = QListWidget()
        self.series_list.itemSelectionChanged.connect(self.on_series_selected)
        series_layout.addWidget(self.series_list)
        grid.addWidget(series_card, 0, 0, 1, 2)

        # Selection summary
        summary = self._build_card("Selection")
        form = QFormLayout(summary)
        self.sel_series = QLabel("—")
        self.sel_slices = QLabel("—")
        self.sel_shape = QLabel("—")
        form.addRow("Series ID:", self.sel_series)
        form.addRow("Slices:", self.sel_slices)
        form.addRow("Array shape:", self.sel_shape)
        grid.addWidget(summary, 1, 0)

        # Model availability
        models = self._build_card("Models")
        mlay = QVBoxLayout(models)

        hint = QLabel("If models are missing, click Browse to locate the weight files.")
        hint.setStyleSheet("color: #9ca3af;")
        hint.setWordWrap(True)
        mlay.addWidget(hint)

        self.models_status = QLabel("—")
        self.models_status.setWordWrap(True)
        mlay.addWidget(self.models_status)

        self.model_path_edits: Dict[str, QLineEdit] = {}
        for model_name in ["YOLOv11", "U-Net", "nnU-Net"]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText(f"{model_name} weights path…")
            self.model_path_edits[model_name] = edit

            btn = QPushButton("Browse")
            btn.setObjectName("secondary")
            btn.clicked.connect(lambda _checked=False, m=model_name: self.browse_model_path(m))

            row_layout.addWidget(edit, 1)
            row_layout.addWidget(btn)
            mlay.addWidget(QLabel(model_name))
            mlay.addWidget(row)

        grid.addWidget(models, 1, 1)

        layout.addLayout(grid)

        self.data_progress = QProgressBar()
        self.data_progress.setRange(0, 100)
        self.data_progress.setValue(0)
        layout.addWidget(self.data_progress)

        layout.addStretch(1)
        return page

    def _build_page_segmentation(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QLabel("Segmentation")
        header.setStyleSheet("font-size: 18pt; font-weight: 800; color: #e5e7eb;")
        layout.addWidget(header)

        row = QHBoxLayout()

        settings = self._build_card("Settings")
        sform = QFormLayout(settings)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TotalSegmentator", "YOLOv11", "U-Net", "nnU-Net"])
        sform.addRow("Model:", self.model_combo)

        self.btn_start = QPushButton("Start")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_segmentation)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_workers)

        sform.addRow(self.btn_start, self.btn_stop)
        row.addWidget(settings, 0)

        stats = self._build_card("Run Status")
        v = QVBoxLayout(stats)
        self.seg_status = QLabel("Waiting for data…")
        self.seg_status.setWordWrap(True)
        v.addWidget(self.seg_status)
        self.seg_progress = QProgressBar()
        self.seg_progress.setRange(0, 100)
        self.seg_progress.setValue(0)
        v.addWidget(self.seg_progress)

        self.volume_label = QLabel("Liver volume: —")
        self.volume_label.setStyleSheet("font-size: 13pt; font-weight: 800; color: #a7f3d0;")
        v.addWidget(self.volume_label)

        row.addWidget(stats, 1)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _build_page_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QLabel("Results")
        header.setStyleSheet("font-size: 18pt; font-weight: 800; color: #e5e7eb;")
        layout.addWidget(header)

        self.results_tabs = QTabWidget()

        # 2D tab
        tab2d = QWidget()
        l2d = QVBoxLayout(tab2d)

        viewer_card = self._build_card("2D Slice Viewer")
        vlay = QVBoxLayout(viewer_card)
        self.slice_view = QLabel("Load a series to view slices")
        self.slice_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slice_view.setStyleSheet("background: #060b16; border: 1px solid #24324a; border-radius: 12px;")
        self.slice_view.setMinimumHeight(420)
        vlay.addWidget(self.slice_view)

        controls = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        self.slice_label = QLabel("Slice: —")

        self.chk_overlay = QCheckBox("Overlay mask")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setEnabled(False)
        self.chk_overlay.stateChanged.connect(self.redraw_slice)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.valueChanged.connect(self.redraw_slice)

        self.btn_go_liver = QPushButton("Go to Liver")
        self.btn_go_liver.setObjectName("secondary")
        self.btn_go_liver.setEnabled(False)
        self.btn_go_liver.clicked.connect(self.go_to_liver_slice)

        controls.addWidget(self.slice_slider, 1)
        controls.addWidget(self.slice_label)
        controls.addWidget(self.btn_go_liver)
        controls.addWidget(self.chk_overlay)
        controls.addWidget(QLabel("Opacity"))
        controls.addWidget(self.opacity_slider)
        vlay.addLayout(controls)

        l2d.addWidget(viewer_card)
        self.results_tabs.addTab(tab2d, "2D")

        # 3D tab
        tab3d = QWidget()
        l3d = QVBoxLayout(tab3d)

        render_card = self._build_card("3D Viewer")
        rlay = QVBoxLayout(render_card)

        if QtInteractor is None or pv is None:
            self.pv_widget = QLabel("3D viewer requires 'pyvista' and 'pyvistaqt'.")
            self.pv_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.pv_widget = QtInteractor(render_card)
            self.pv_widget.set_background("#060b16")

        rlay.addWidget(self.pv_widget)

        # 3D Controls
        controls_3d = QHBoxLayout()
        
        # Show CT checkbox
        self.chk_show_ct = QCheckBox("Show CT Volume")
        self.chk_show_ct.setChecked(False)
        self.chk_show_ct.stateChanged.connect(self.update_3d_display)
        controls_3d.addWidget(self.chk_show_ct)
        
        # Show Liver checkbox
        self.chk_show_liver = QCheckBox("Show Liver Surface")
        self.chk_show_liver.setChecked(True)
        self.chk_show_liver.stateChanged.connect(self.update_3d_display)
        controls_3d.addWidget(self.chk_show_liver)
        
        controls_3d.addStretch(1)
        rlay.addLayout(controls_3d)
        
        # Window/Level controls
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("Window Center:"))
        self.wc_slider = QSlider(Qt.Orientation.Horizontal)
        self.wc_slider.setRange(-1000, 1000)
        self.wc_slider.setValue(40)
        self.wc_slider.valueChanged.connect(self.on_window_changed)
        wl_layout.addWidget(self.wc_slider)
        self.wc_label = QLabel("40")
        wl_layout.addWidget(self.wc_label)
        
        wl_layout.addWidget(QLabel("Width:"))
        self.ww_slider = QSlider(Qt.Orientation.Horizontal)
        self.ww_slider.setRange(1, 2000)
        self.ww_slider.setValue(400)
        self.ww_slider.valueChanged.connect(self.on_window_changed)
        wl_layout.addWidget(self.ww_slider)
        self.ww_label = QLabel("400")
        wl_layout.addWidget(self.ww_label)
        rlay.addLayout(wl_layout)
        
        # Liver opacity control
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Liver Opacity:"))
        self.liver_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.liver_opacity_slider.setRange(10, 100)
        self.liver_opacity_slider.setValue(85)
        self.liver_opacity_slider.valueChanged.connect(self.update_3d_display)
        opacity_layout.addWidget(self.liver_opacity_slider)
        
        opacity_layout.addWidget(QLabel("CT Opacity:"))
        self.ct_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.ct_opacity_slider.setRange(1, 100)
        self.ct_opacity_slider.setValue(30)  # Default low opacity for CT
        self.ct_opacity_slider.valueChanged.connect(self.update_3d_display)
        opacity_layout.addWidget(self.ct_opacity_slider)
        rlay.addLayout(opacity_layout)

        btns = QHBoxLayout()
        self.btn_render_3d = QPushButton("Render Liver Surface")
        self.btn_render_3d.setEnabled(False)
        self.btn_render_3d.clicked.connect(self.render_3d)

        self.btn_clear_3d = QPushButton("Clear 3D")
        self.btn_clear_3d.setObjectName("secondary")
        self.btn_clear_3d.clicked.connect(self.clear_3d)

        btns.addWidget(self.btn_render_3d)
        btns.addWidget(self.btn_clear_3d)
        btns.addStretch(1)
        rlay.addLayout(btns)

        l3d.addWidget(render_card)
        self.results_tabs.addTab(tab3d, "3D")

        layout.addWidget(self.results_tabs)
        return page

    def _build_page_logs(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QLabel("Logs")
        header.setStyleSheet("font-size: 18pt; font-weight: 800; color: #e5e7eb;")
        layout.addWidget(header)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        layout.addWidget(self.logs)

        return page

    @pyqtSlot(int)
    def on_nav_changed(self, idx: int):
        if not hasattr(self, "pages"):
            return
        if 0 <= idx < self.pages.count():
            self.pages.setCurrentIndex(idx)

    def _log(self, msg: str):
        if hasattr(self, "logs"):
            self.logs.append(msg)

    def show_about(self):
        QMessageBox.information(
            self,
            "About",
            "LiverSeg 3D\n\nA application for liver segmentation from CT DICOM series.\n",
        )

    def _refresh_model_status(self):
        try:
            # Build status from effective paths (configured path overrides default)
            lines = []
            
            # TotalSegmentator is always available (downloads weights automatically)
            lines.append("• TotalSegmentator: ✓ Ready (auto-download)")
            
            for name in ["YOLOv11", "U-Net", "nnU-Net"]:
                effective_path = self._get_effective_model_path(name) or ""
                exists = bool(effective_path) and os.path.exists(effective_path)
                state = "Available" if exists else "Missing"
                lines.append(f"• {name}: {state}")
                if hasattr(self, "model_path_edits") and name in self.model_path_edits:
                    configured = self._get_configured_model_path(name)
                    self.model_path_edits[name].setText(configured or effective_path)

            self.models_status.setText("\n".join(lines))
        except Exception as e:
            self.models_status.setText(f"Model status unavailable: {e}")

        # Very lightweight device display
        try:
            import torch

            gpu = "CUDA" if torch.cuda.is_available() else "CPU"
            self.status_chip.setText(f"Device: {gpu}")
        except Exception:
            self.status_chip.setText("Device: Unknown")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder", "")
        if not folder:
            return
        self.selection.folder = folder
        self.folder_edit.setText(folder)
        self._log(f"Selected folder: {folder}")
        self.load_series_list()

    def load_series_list(self):
        self.series_list.clear()
        self.selection.series_id = ""

        if not self.selection.folder:
            return

        try:
            series_ids = self.dicom_loader.get_dicom_series(self.selection.folder)
            if not series_ids:
                QMessageBox.warning(self, "No DICOM", "No DICOM series were found in this folder.")
                return

            for sid in series_ids:
                info = self.dicom_loader.get_series_info(self.selection.folder, sid)
                item = QListWidgetItem(f"{sid}  —  {info.get('num_slices', '?')} slices")
                item.setData(Qt.ItemDataRole.UserRole, sid)
                self.series_list.addItem(item)

            self._log(f"Found {len(series_ids)} series.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to scan folder: {e}")

    @pyqtSlot()
    def on_series_selected(self):
        item = self.series_list.currentItem()
        if not item:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if not sid:
            return

        self.selection.series_id = sid
        self._log(f"Selected series: {sid}")
        self.load_selected_series()

    def load_selected_series(self):
        if not self.selection.folder or not self.selection.series_id:
            return

        self.data_progress.setValue(0)

        self.dicom_worker = DicomLoaderWorker()
        self.dicom_worker.progress_updated.connect(self.data_progress.setValue)
        self.dicom_worker.series_loaded.connect(self.on_series_loaded)
        self.dicom_worker.error_occurred.connect(self.on_error)
        self.dicom_worker.load_series(self.selection.folder, self.selection.series_id)
        self.dicom_worker.start()

        self._log("Loading series in background…")

    @pyqtSlot(dict)
    def on_series_loaded(self, payload: Dict[str, Any]):
        try:
            self.current_image = payload["image_array"]
            self.sitk_image = payload.get("sitk_image")  # Store for TotalSegmentator
            self.current_voxel_volume_mm3 = float(payload.get("voxel_volume", 1.0))

            sid = payload.get("series_id", "—")
            info = payload.get("series_info", {})

            self.sel_series.setText(str(sid))
            self.sel_slices.setText(str(info.get("num_slices", "—")))
            self.sel_shape.setText(str(getattr(self.current_image, "shape", "—")))

            self._log(f"Series loaded. Shape: {self.current_image.shape}")

            # Enable segmentation
            self.btn_start.setEnabled(True)
            self.quick_run_btn.setEnabled(True)

            # Setup 2D viewer
            if self.current_image is not None and self.current_image.ndim == 3:
                self.slice_slider.setEnabled(True)
                self.slice_slider.setRange(0, self.current_image.shape[0] - 1)
                self.slice_slider.setValue(0)
                self.chk_overlay.setEnabled(True)
                self.opacity_slider.setEnabled(True)
                self.redraw_slice()

        except Exception as e:
            self.on_error(f"Failed to process loaded series: {e}")

    def start_segmentation(self):
        if self.current_image is None:
            QMessageBox.warning(self, "No Data", "Load a DICOM series first.")
            return

        model_type = self.model_combo.currentText()
        
        # TotalSegmentator doesn't need a model file - it downloads weights automatically
        if model_type == "TotalSegmentator":
            self._run_totalsegmentator()
            return
        
        model_path = self._get_effective_model_path(model_type)
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(
                self,
                "Model Missing",
                f"Model file not found for {model_type}.\nExpected: {model_path}",
            )
            return

        self.seg_progress.setValue(0)
        self.seg_status.setText(f"Running {model_type}…")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.quick_run_btn.setEnabled(False)

        self.seg_worker = SegmentationWorker()
        self.seg_worker.progress_updated.connect(self.seg_progress.setValue)
        self.seg_worker.segmentation_completed.connect(self.on_segmentation_done)
        self.seg_worker.error_occurred.connect(self.on_error)
        self.seg_worker.segment_image(self.current_image, model_type, model_path)
        self.seg_worker.start()

        # Switch to Segmentation page
        self.nav_list.setCurrentRow(1)

    def _run_totalsegmentator(self):
        """Run TotalSegmentator segmentation."""
        if not hasattr(self, 'sitk_image') or self.sitk_image is None:
            QMessageBox.warning(self, "No Data", "Please reload the DICOM series.")
            return
        
        self.seg_progress.setValue(0)
        self.seg_status.setText("Running TotalSegmentator (downloading weights if needed)…")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.quick_run_btn.setEnabled(False)
        
        # Switch to Segmentation page
        self.nav_list.setCurrentRow(1)
        
        # Import and start the TotalSegmentator worker
        from gui.workers import TotalSegmentatorWorker
        
        self.ts_worker = TotalSegmentatorWorker()
        self.ts_worker.progress_updated.connect(self.seg_progress.setValue)
        self.ts_worker.segmentation_completed.connect(self.on_segmentation_done)
        self.ts_worker.error_occurred.connect(self.on_error)
        self.ts_worker.set_image(self.sitk_image)
        self.ts_worker.start()

    @pyqtSlot(object)
    def on_segmentation_done(self, result: object):
        try:
            if isinstance(result, dict):
                mask = result.get("mask")
                model_type = result.get("model_type", "")
            else:
                mask = result
                model_type = ""

            self.current_mask = mask
            self.seg_status.setText(f"Completed {model_type}.")
            self.btn_stop.setEnabled(False)
            self.btn_start.setEnabled(True)
            self.quick_run_btn.setEnabled(True)

            self.btn_render_3d.setEnabled(self.current_mask is not None)
            self.btn_go_liver.setEnabled(self.current_mask is not None)
            self._log("Segmentation completed.")

            self.redraw_slice()
            self.calculate_volume()

            # Switch to Results
            self.nav_list.setCurrentRow(2)
        except Exception as e:
            self.on_error(f"Failed to handle segmentation result: {e}")

    def calculate_volume(self):
        if self.current_mask is None:
            return

        self.vol_worker = VolumeCalculationWorker()
        self.vol_worker.volume_calculated.connect(self.on_volume)
        self.vol_worker.error_occurred.connect(self.on_error)
        self.vol_worker.calculate_volume(self.current_mask, self.current_voxel_volume_mm3)
        self.vol_worker.start()

    @pyqtSlot(float)
    def on_volume(self, volume_ml: float):
        self.last_volume_ml = float(volume_ml)
        self.volume_label.setText(f"Liver volume: {volume_ml:.1f} mL")
        self._log(f"Volume computed: {volume_ml:.1f} mL")

    def stop_workers(self):
        for worker in [self.dicom_worker, self.seg_worker, self.vol_worker]:
            if worker and worker.isRunning():
                try:
                    worker.stop()
                except Exception:
                    pass
                worker.wait(500)

        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(self.current_image is not None)
        self.quick_run_btn.setEnabled(self.current_image is not None)
        self.seg_status.setText("Stopped.")
        self._log("Stopped all running tasks.")

    @pyqtSlot(str)
    def on_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)

        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(self.current_image is not None)
        self.quick_run_btn.setEnabled(self.current_image is not None)

    def clear_session(self):
        self.stop_workers()
        self.current_image = None
        self.sitk_image = None
        self.current_mask = None
        self.last_volume_ml = 0.0

        self.series_list.clear()
        self.folder_edit.clear()
        self.selection = SeriesSelection()

        self.sel_series.setText("—")
        self.sel_slices.setText("—")
        self.sel_shape.setText("—")

        self.data_progress.setValue(0)
        self.seg_progress.setValue(0)
        self.seg_status.setText("Waiting for data…")
        self.volume_label.setText("Liver volume: —")

        self.slice_view.setText("Load a series to view slices")
        self.slice_slider.setEnabled(False)
        self.chk_overlay.setEnabled(False)
        self.opacity_slider.setEnabled(False)
        self.btn_render_3d.setEnabled(False)
        self.clear_3d()

        self.btn_start.setEnabled(False)
        self.quick_run_btn.setEnabled(False)

        self._log("Session cleared.")

    def browse_model_path(self, model_name: str) -> None:
        """Let the user pick a weight file for a model and persist the choice."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select weights for {model_name}",
            "",
            "Weights (*.pt *.pth);;All Files (*)",
        )
        if not file_path:
            return

        self._set_configured_model_path(model_name, file_path)
        self._log(f"Configured {model_name} weights: {file_path}")
        self._refresh_model_status()

    def go_to_liver_slice(self):
        """Jump to the middle slice that contains the liver."""
        if self.current_mask is None:
            QMessageBox.information(self, "No Mask", "Run segmentation first to locate the liver.")
            return
        
        # Find slices that contain liver (non-zero mask values)
        liver_slices = []
        for i in range(self.current_mask.shape[0]):
            if np.any(self.current_mask[i] > 0):
                liver_slices.append(i)
        
        if not liver_slices:
            QMessageBox.information(self, "No Liver Found", "No liver detected in the segmentation mask.")
            return
        
        # Go to the middle of the liver
        middle_idx = liver_slices[len(liver_slices) // 2]
        self.slice_slider.setValue(middle_idx)
        self._log(f"Jumped to liver slice {middle_idx} (liver spans slices {liver_slices[0]}-{liver_slices[-1]})")

    def on_slice_changed(self, idx: int):
        self.slice_label.setText(f"Slice: {idx}")
        self.redraw_slice()

    def redraw_slice(self):
        if self.current_image is None or self.current_image.ndim != 3:
            return

        idx = int(self.slice_slider.value())
        slice_img = self.current_image[idx]

        # Get window/level from sliders (use defaults if sliders don't exist yet)
        wc = getattr(self, 'wc_slider', None)
        ww = getattr(self, 'ww_slider', None)
        if wc and ww:
            wc = wc.value()
            ww = ww.value()
        else:
            wc, ww = 40, 400
            
        lo = wc - ww / 2
        hi = wc + ww / 2
        slice_win = np.clip(slice_img, lo, hi)
        slice_norm = (slice_win - lo) / max(hi - lo, 1e-6)
        base = (slice_norm * 255).astype(np.uint8)

        rgb = np.stack([base] * 3, axis=-1)

        if self.current_mask is not None and self.chk_overlay.isChecked():
            # Ensure mask is properly sized
            mask = self.current_mask
            
            # Handle shape mismatch - resize mask if needed
            if mask.shape != self.current_image.shape:
                from scipy.ndimage import zoom
                zoom_factors = tuple(t / m for t, m in zip(self.current_image.shape, mask.shape))
                mask = zoom(mask, zoom_factors, order=0)
                self._log(f"Resized mask from {self.current_mask.shape} to {mask.shape}")
            
            if mask.shape == self.current_image.shape:
                m = mask[idx].astype(bool)
                alpha = self.opacity_slider.value() / 100.0
                
                # Create colored overlay (red for liver)
                overlay = rgb.copy()
                overlay[m, 0] = 255  # Red
                overlay[m, 1] = 80   # Some green
                overlay[m, 2] = 80   # Some blue
                
                # Blend original and overlay
                rgb = (rgb.astype(float) * (1 - alpha) + overlay.astype(float) * alpha).astype(np.uint8)

        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.slice_view.width(),
            self.slice_view.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.slice_view.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep slice view crisp on resize
        if self.current_image is not None:
            self.redraw_slice()

    def clear_3d(self):
        if QtInteractor is None or pv is None:
            return
        try:
            self.pv_widget.clear()
            self.pv_widget.reset_camera()
        except Exception:
            pass

    def render_3d(self):
        if self.current_mask is None:
            QMessageBox.warning(self, "No Mask", "Run segmentation first.")
            return
        if QtInteractor is None or pv is None:
            QMessageBox.warning(self, "3D Missing", "Install pyvista and pyvistaqt.")
            return
        if measure is None:
            QMessageBox.warning(self, "Missing Dependency", "Install scikit-image for 3D surface extraction.")
            return

        self._log("Rendering 3D view…")

        try:
            self.pv_widget.clear()
            
            # Add CT volume if checkbox is checked
            if self.chk_show_ct.isChecked():
                self._add_ct_volume()
            
            # Add liver surface
            if self.chk_show_liver.isChecked():
                self._add_liver_surface()
            
            self.pv_widget.reset_camera()
            self.pv_widget.add_axes()

            self._log("3D render complete.")
        except Exception as e:
            self.on_error(f"3D render failed: {e}")

    def on_window_changed(self, _value: int):
        """Update window/level labels and redraw."""
        self.wc_label.setText(str(self.wc_slider.value()))
        self.ww_label.setText(str(self.ww_slider.value()))
        # Update 2D slice view
        self.redraw_slice()

    def update_3d_display(self):
        """Update 3D display based on checkboxes."""
        if QtInteractor is None or pv is None:
            return
        if self.current_image is None:
            return
        
        try:
            self.pv_widget.clear()
            
            # Show CT Volume if checked
            if self.chk_show_ct.isChecked() and self.current_image is not None:
                self._add_ct_volume()
            
            # Show Liver Surface if checked
            if self.chk_show_liver.isChecked() and self.current_mask is not None:
                self._add_liver_surface()
            
            self.pv_widget.reset_camera()
            
        except Exception as e:
            self._log(f"3D update error: {e}")

    def _add_ct_volume(self):
        """Add CT volume to 3D viewer."""
        if self.current_image is None:
            return
        
        try:
            # Apply window/level
            wc = self.wc_slider.value()
            ww = self.ww_slider.value()
            lo = wc - ww / 2
            hi = wc + ww / 2
            
            volume = np.clip(self.current_image, lo, hi)
            volume = ((volume - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
            
            # Create PyVista grid
            grid = pv.ImageData()
            grid.dimensions = np.array(volume.shape) + 1
            grid.spacing = (1.0, 1.0, 1.0)
            grid.origin = (0, 0, 0)
            grid.cell_data["values"] = volume.ravel(order="F")
            
            # Get CT opacity from slider
            ct_opacity = self.ct_opacity_slider.value() / 100.0
            
            # Create custom opacity transfer function
            opacity_str = f"sigmoid_{int(ct_opacity * 10)}"
            
            # Add volume with gray colormap and adjustable transparency
            self.pv_widget.add_volume(
                grid, 
                cmap="gray", 
                opacity=[0, ct_opacity * 0.1, ct_opacity * 0.3, ct_opacity * 0.5, ct_opacity],
                name="ct_volume",
                show_scalar_bar=False
            )
            
        except Exception as e:
            self._log(f"CT volume render error: {e}")

    def _add_liver_surface(self):
        """Add liver surface mesh to 3D viewer."""
        if self.current_mask is None or measure is None:
            return
        
        try:
            # Handle mask size mismatch
            mask = self.current_mask
            if mask.shape != self.current_image.shape:
                from scipy.ndimage import zoom
                zoom_factors = tuple(t / m for t, m in zip(self.current_image.shape, mask.shape))
                mask = zoom(mask, zoom_factors, order=0)
            
            mask = (mask > 0).astype(np.uint8)
            
            if mask.sum() == 0:
                self._log("No liver voxels found in mask")
                return
            
            # Extract surface using marching cubes
            verts, faces, _normals, _values = measure.marching_cubes(mask, level=0.5)
            faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
            mesh = pv.PolyData(verts, faces_pv)
            
            # Smooth the mesh for better appearance
            mesh = mesh.smooth(n_iter=50, relaxation_factor=0.1)
            
            # Get opacity from slider
            opacity = self.liver_opacity_slider.value() / 100.0
            
            # Add liver mesh with orange/red color
            self.pv_widget.add_mesh(
                mesh, 
                color="#f97316",  # Orange color
                opacity=opacity,
                smooth_shading=True,
                name="liver_surface"
            )
            
            self.pv_widget.add_text("Liver Surface", position="upper_left", font_size=12, color="white")
            
        except Exception as e:
            self._log(f"Liver surface render error: {e}")
