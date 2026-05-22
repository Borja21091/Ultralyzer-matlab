import cv2
import numpy as np
from pathlib import Path
from definitions import IMAGE_FORMATS, METRIC_DICTIONARY, SEG_DIR, IM_DIR
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QMessageBox, QComboBox, QTextEdit, 
    QProgressDialog, QApplication, QCompleter, QToolButton,
    QMenu, QWidgetAction, QSlider, QStyle
)
from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtGui import QAction, QPainter, QPen, QIcon, QPixmap, QPalette
from frontend.widgets.widget_s2 import SegmentationWidget
from backend.models.database import DatabaseManager

class MainWindow(QMainWindow):
    """
    Main application window - displays selected processing step
    """
    
    def __init__(self):
        super().__init__()
        
        self._image_folder = None
        self._mask_folder = None
        self._db_manager = DatabaseManager()
        self._image_list = []
        self._all_image_paths = []
        self.worker = None
        self.widget: SegmentationWidget
        self.status_refresh_button: QToolButton | None = None
        self.status_zoom_button: QToolButton | None = None
        self.status_opacity_button: QToolButton | None = None
        self.status_opacity_slider: QSlider | None = None
        self.status_opacity_label: QLabel | None = None
        self.action_batch_segment: QAction | None = None
        self.action_batch_metrics: QAction | None = None
        
        self._init_ui()
    
    ############ PROPERTIES ############
    
    @property
    def image_folder(self):
        """Get current image folder"""
        return self._image_folder
    
    @property
    def db_manager(self):
        """Get database manager"""
        return self._db_manager
    
    @property
    def image_list(self):
        return self._image_list
    
    @image_list.setter
    def image_list(self, files: list):
        self._image_list = files
    
    ############ UI ############
    
    def _init_ui(self):
        """Initialize main window UI"""
        
        # Initialize window
        self.setWindowTitle("Ultralyzer - Retinal Image Processing Pipeline")
        self.showMaximized()
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top: Folder label & image dropdown
        top_layout = QHBoxLayout()
        
        # Folder label
        self.folder_label = QLabel("No image folder loaded")
        self.folder_label.setWordWrap(True)
        top_layout.addWidget(self.folder_label, 1)

        self.image_filter = QComboBox()
        self.image_filter.addItems(["All images", "Unreviewed", "No segmentation", "No metrics", "Rejected"])
        self.image_filter.setToolTip("Filter the image list by review and processing status")
        self.image_filter.currentTextChanged.connect(self._on_image_filter_changed)
        top_layout.addWidget(self.image_filter)
        
        # Dropdown with image names in folder
        self.image_dropdown = QComboBox()
        self.image_dropdown.setEditable(True)
        self.image_dropdown.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.image_dropdown.setEnabled(False)
        self.image_dropdown.setPlaceholderText("Select an image")
        self.image_dropdown.setToolTip("Search or select an image filename")
        self.image_dropdown.activated.connect(self._on_select_image)
        line_edit = self.image_dropdown.lineEdit()
        if line_edit:
            line_edit.editingFinished.connect(self._on_select_image_text)
        top_layout.addWidget(self.image_dropdown)

        main_layout.addLayout(top_layout)
        
        # Create step-specific widget
        self.widget = self._create_widget()
        self.widget.index_changed.connect(self.image_dropdown.setCurrentIndex)
        main_layout.addWidget(self.widget, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        self._create_status_bar_utilities()
        
        # Create top menu bar
        self._create_menu_bar()

    def _create_status_bar_utilities(self):
        """Add compact zoom and opacity controls to the bottom status bar."""
        status_bar = self.statusBar()

        button_style = """
            QToolButton {
                border: none;
                padding: 2px 6px;
            }
            QToolButton:hover {
                background: rgba(100, 116, 139, 35);
                border-radius: 4px;
            }
            QToolButton::menu-indicator {
                image: none;
                width: 0px;
            }
        """

        self.status_refresh_button = QToolButton(self)
        self.status_refresh_button.setObjectName("statusRefreshButton")
        self.status_refresh_button.setAutoRaise(True)
        self.status_refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_refresh_button.setIcon(self._create_status_icon("refresh"))
        self.status_refresh_button.setIconSize(QSize(16, 16))
        self.status_refresh_button.setToolTip("Refresh current image and mask")
        self.status_refresh_button.setStyleSheet(button_style)
        self.status_refresh_button.clicked.connect(self.widget._on_refresh_display)
        status_bar.addPermanentWidget(self.status_refresh_button)

        self.status_zoom_button = QToolButton(self)
        self.status_zoom_button.setObjectName("statusZoomButton")
        self.status_zoom_button.setAutoRaise(True)
        self.status_zoom_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_zoom_button.setIcon(self._create_status_icon("zoom"))
        self.status_zoom_button.setIconSize(QSize(16, 16))
        self.status_zoom_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.status_zoom_button.setToolTip("Zoom options")
        self.status_zoom_button.setStyleSheet(button_style)

        zoom_menu = QMenu(self)
        zoom_menu.addAction("Fit", self.widget._on_fit_view)
        zoom_menu.addAction("100%", self.widget._on_actual_size)
        zoom_menu.addAction("Center", self.widget._on_center_view)
        self.status_zoom_button.setMenu(zoom_menu)
        status_bar.addPermanentWidget(self.status_zoom_button)

        self.status_opacity_button = QToolButton(self)
        self.status_opacity_button.setObjectName("statusOpacityButton")
        self.status_opacity_button.setAutoRaise(True)
        self.status_opacity_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_opacity_button.setIcon(self._create_status_icon("opacity"))
        self.status_opacity_button.setIconSize(QSize(16, 16))
        self.status_opacity_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.status_opacity_button.setStyleSheet(button_style)

        opacity_menu = QMenu(self)
        opacity_widget = QWidget(opacity_menu)
        opacity_layout = QHBoxLayout(opacity_widget)
        opacity_layout.setContentsMargins(10, 8, 10, 8)
        opacity_layout.setSpacing(8)

        opacity_label_title = QLabel("Opacity", opacity_widget)
        self.status_opacity_slider = QSlider(Qt.Orientation.Horizontal, opacity_widget)
        self.status_opacity_slider.setRange(0, 100)
        self.status_opacity_slider.setMinimumWidth(120)
        self.status_opacity_slider.setMaximumWidth(160)
        self.status_opacity_label = QLabel("75%", opacity_widget)
        self.status_opacity_label.setMinimumWidth(32)

        opacity_layout.addWidget(opacity_label_title)
        opacity_layout.addWidget(self.status_opacity_slider)
        opacity_layout.addWidget(self.status_opacity_label)

        opacity_action = QWidgetAction(opacity_menu)
        opacity_action.setDefaultWidget(opacity_widget)
        opacity_menu.addAction(opacity_action)
        self.status_opacity_button.setMenu(opacity_menu)
        self.status_opacity_button.setToolTip("Mask opacity: 75%")
        self.status_opacity_slider.valueChanged.connect(
            lambda value: self.status_opacity_button.setToolTip(f"Mask opacity: {value}%")
        )
        status_bar.addPermanentWidget(self.status_opacity_button)

        self.widget.bind_opacity_controls(self.status_opacity_slider, self.status_opacity_label)

    def _create_status_icon(self, kind: str) -> QIcon:
        """Draw small status-bar icons for popover controls."""
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)

        palette = self.statusBar().palette()
        primary = palette.color(QPalette.ColorRole.WindowText)
        secondary = palette.color(QPalette.ColorRole.Mid)
        if secondary == primary:
            secondary = primary.darker(140) if primary.lightness() > 128 else primary.lighter(140)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if kind == "refresh":
            painter.end()
            return self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        if kind == "zoom":
            pen = QPen(primary, 1.6)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(2.5, 2.5, 7.0, 7.0))
            painter.drawLine(8.5, 8.5, 13.0, 13.0)
        else:
            rect = QRectF(3.0, 3.5, 10.0, 9.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(secondary.lighter(135) if secondary.lightness() < 128 else secondary)
            painter.drawRoundedRect(QRectF(rect.x(), rect.y(), rect.width() / 2, rect.height()), 2.0, 2.0)
            painter.setBrush(secondary.darker(140) if secondary.lightness() > 128 else secondary)
            painter.drawRoundedRect(QRectF(rect.x() + rect.width() / 2, rect.y(), rect.width() / 2, rect.height()), 2.0, 2.0)
            painter.setPen(QPen(primary, 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, 2.0, 2.0)

        painter.end()
        return QIcon(pixmap)
    
    def _create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        action_open_image_folder = QAction("Open Image Folder...", self)
        action_open_image_folder.triggered.connect(self._on_select_image_folder)
        file_menu.addAction(action_open_image_folder)
        
        action_load_mask_folder = QAction("Load Segmentation Folder...", self)
        action_load_mask_folder.triggered.connect(self._on_select_mask_folder)
        file_menu.addAction(action_load_mask_folder)
        
        file_menu.addSeparator()
        
        import_menu = file_menu.addMenu("Import")

        action_import_psd = QAction("PSD Segmentations...", self)
        action_import_psd.triggered.connect(self._on_import_from_psd)
        import_menu.addAction(action_import_psd)
        
        action_dicom_converter = QAction("DICOM Images...", self)
        action_dicom_converter.triggered.connect(self._on_dicom_converter)
        import_menu.addAction(action_dicom_converter)

        export_menu = file_menu.addMenu("Export")

        action_export_metrics = QAction("Results Workbook", self)
        action_export_metrics.triggered.connect(self._on_export_metrics)
        export_menu.addAction(action_export_metrics)

        export_menu.addSeparator()

        export_psd_menu = export_menu.addMenu("PSD Segmentations")

        action_export_psd_current = QAction("Current Image", self)
        action_export_psd_current.triggered.connect(self._on_export_to_psd_current)
        export_psd_menu.addAction(action_export_psd_current)

        action_export_psd_all = QAction("All Images", self)
        action_export_psd_all.triggered.connect(self._on_export_to_psd_all)
        export_psd_menu.addAction(action_export_psd_all)

        edit_menu = menubar.addMenu("Edit")

        action_undo_edits = QAction("Undo", self)
        action_undo_edits.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        action_undo_edits.triggered.connect(self.widget._on_undo)
        edit_menu.addAction(action_undo_edits)

        action_redo_edits = QAction("Redo", self)
        action_redo_edits.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        action_redo_edits.triggered.connect(self.widget._on_redo)
        edit_menu.addAction(action_redo_edits)

        edit_menu.addSeparator()

        action_save_edits = QAction("Save Edits", self)
        action_save_edits.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        action_save_edits.triggered.connect(self.widget._save_edits)
        edit_menu.addAction(action_save_edits)

        action_reset_edits = QAction("Reset Edits", self)
        action_reset_edits.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        action_reset_edits.triggered.connect(self.widget._on_reset_edits)
        edit_menu.addAction(action_reset_edits)

        image_menu = menubar.addMenu("Image")

        action_prev_image = QAction("Previous Image", self)
        action_prev_image.triggered.connect(self.widget._on_prev)
        image_menu.addAction(action_prev_image)

        action_next_image = QAction("Next Image", self)
        action_next_image.triggered.connect(self.widget._on_next)
        image_menu.addAction(action_next_image)

        image_menu.addSeparator()

        action_refresh_display = QAction("Refresh Current Image", self)
        action_refresh_display.triggered.connect(self.widget._on_refresh_display)
        image_menu.addAction(action_refresh_display)

        view_menu = image_menu.addMenu("View")

        action_fit_view = QAction("Fit to Window", self)
        action_fit_view.triggered.connect(self.widget._on_fit_view)
        view_menu.addAction(action_fit_view)

        action_actual_size = QAction("Actual Size", self)
        action_actual_size.triggered.connect(self.widget._on_actual_size)
        view_menu.addAction(action_actual_size)

        action_center_view = QAction("Center Image", self)
        action_center_view.triggered.connect(self.widget._on_center_view)
        view_menu.addAction(action_center_view)

        image_menu.addSeparator()

        qc_assign_all_menu = image_menu.addMenu("Assign QC to All")

        action_qc_pass_all = QAction("Pass", self)
        action_qc_pass_all.triggered.connect(lambda: self._on_assign_qc_all("pass"))
        qc_assign_all_menu.addAction(action_qc_pass_all)

        action_qc_borderline_all = QAction("Borderline", self)
        action_qc_borderline_all.triggered.connect(lambda: self._on_assign_qc_all("borderline"))
        qc_assign_all_menu.addAction(action_qc_borderline_all)

        action_qc_reject_all = QAction("Reject", self)
        action_qc_reject_all.triggered.connect(lambda: self._on_assign_qc_all("reject"))
        qc_assign_all_menu.addAction(action_qc_reject_all)

        analyze_menu = menubar.addMenu("Analyze")

        current_image_menu = analyze_menu.addMenu("Current Image")

        action_segment_current = QAction("Segment", self)
        action_segment_current.triggered.connect(self.widget._on_segment_current_image)
        current_image_menu.addAction(action_segment_current)

        action_metrics_current = QAction("Calculate Metrics", self)
        action_metrics_current.triggered.connect(self.widget._on_metrics_current_image)
        current_image_menu.addAction(action_metrics_current)

        pending_images_menu = analyze_menu.addMenu("Pending Images")

        self.action_batch_segment = QAction("Segment Pending Images", self)
        self.action_batch_segment.triggered.connect(self.widget._on_segment_all)
        pending_images_menu.addAction(self.action_batch_segment)

        self.action_batch_metrics = QAction("Calculate Pending Metrics", self)
        self.action_batch_metrics.triggered.connect(self.widget._on_calculate_metrics_all)
        pending_images_menu.addAction(self.action_batch_metrics)

        analyze_menu.addSeparator()

        segmentation_menu = analyze_menu.addMenu("Segmentation")

        action_av_segment = QAction("A/V Segment", self)
        action_av_segment.triggered.connect(self._on_av_segment)
        segmentation_menu.addAction(action_av_segment)
        
        action_disc_segment = QAction("Disc Segment", self)
        action_disc_segment.triggered.connect(self._on_disc_segment)
        segmentation_menu.addAction(action_disc_segment)
        
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        action_metric_definitions = QAction("Metric Definitions", self)
        action_metric_definitions.triggered.connect(self._on_metric_definitions)
        help_menu.addAction(action_metric_definitions)
        
        action_about = QAction("About Ultralyzer", self)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)

        self.widget.bind_batch_actions(self.action_batch_segment, self.action_batch_metrics)
    
    ############ PUBLIC METHODS ############
    
    ############ PRIVATE METHODS ############
    
    def _create_widget(self) -> SegmentationWidget:
        """Create the appropriate widget for the selected step"""
        widget = SegmentationWidget(self._db_manager)
        widget.request_open_folder.connect(self._on_select_image_folder)
        widget.decision_made.connect(self._on_qc_decision)
        widget.status_text.connect(self.statusBar().showMessage)
        return widget
    
    def _load_images(self, folder: Path):
        """
        Load images from the specified folder into database & combobox widget
        """
        if not folder.is_dir():
            raise ValueError(f"Invalid folder: {folder}")
        
        self.widget.load_images(folder)
        self._all_image_paths = list(self.widget.image_paths)
        image_files = [p.name for p in self._all_image_paths]
        self.image_list = image_files
        
        if not self.image_list:
            self._refresh_image_dropdown([])
            self.widget.show_empty_state(
                "No supported images found in this folder.",
                "Choose another folder containing supported image files."
            )
            return False
        
        self._apply_image_filter(self.image_filter.currentText())
        
        return True

    def _refresh_image_dropdown(self, image_files: list[str]):
        """Refresh the image selector and search completer."""
        self.image_dropdown.blockSignals(True)
        self.image_dropdown.clear()
        self.image_dropdown.addItems(image_files)
        self.image_dropdown.setEnabled(bool(image_files))
        self.image_dropdown.blockSignals(False)

        self.image_completer = QCompleter(image_files, self)
        self.image_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.image_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.image_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.image_dropdown.setCompleter(self.image_completer)

    def _apply_image_filter(self, filter_text: str):
        """Apply the image list filter without reloading metadata."""
        if not self._all_image_paths:
            self._refresh_image_dropdown([])
            return

        filtered_paths = [p for p in self._all_image_paths if self._image_matches_filter(p, filter_text)]
        self.widget.image_paths = filtered_paths
        self.widget.index = 0
        self.image_list = [p.name for p in filtered_paths]
        self._refresh_image_dropdown(self.image_list)

        if not filtered_paths:
            self.widget.show_empty_state(
                f"No images match '{filter_text}'.",
                "Change the filter or load another image folder."
            )
            self.statusBar().showMessage(f"No images match filter: {filter_text}")
            return

        self.image_dropdown.setCurrentIndex(0)
        self.widget.display_image()
        self.statusBar().showMessage(f"Showing {len(filtered_paths)} / {len(self._all_image_paths)} images")

    def _image_matches_filter(self, image_path: Path, filter_text: str) -> bool:
        """Return whether an image belongs in the selected filter."""
        name = image_path.stem
        if filter_text == "Unreviewed":
            return self._db_manager.get_qc_result(name) is None
        if filter_text == "Rejected":
            qc_result = self._db_manager.get_qc_result(name)
            return bool(qc_result and qc_result.decision.value == "reject")
        if filter_text == "No segmentation":
            return self._db_manager.get_segmentation_by_filename(name) is None
        if filter_text == "No metrics":
            return self._db_manager.get_metrics_by_filename(name) is None
        return True
    
    def _load_mask_info_to_db(self, mask_folder: Path):
        """Load mask information from folder into database"""
        mask_files = list(mask_folder.glob("*"))
        mask_files = [f for f in mask_files if f.suffix.lower() in IMAGE_FORMATS]
        for mask_file in mask_files:
            mask_folder = mask_file.parent
            mask_name = mask_file.name
            mask_suffix = mask_file.suffix.lower()
            if mask_suffix not in IMAGE_FORMATS:
                continue
            meta = self._db_manager.get_metadata_by_filename(mask_name)
            if not meta:
                continue
            self._db_manager.set_mask_info(meta.id, mask_folder, mask_suffix)
    
    def _save_empty_mask(self, image_path: Path, seg_path: Path):
        """Save an empty segmentation mask to the specified path"""
        image = cv2.imread(str(image_path))
        empty_mask = np.zeros(image.shape[:2] + (3,), dtype=np.uint8)
        cv2.imwrite(str(seg_path), empty_mask)
    
    ############ ACTIONS ############
    
    def _on_about(self):
        """Show about dialog"""
        QMessageBox.information(
            self,
            "About Ultralyzer",
            "Ultralyzer - Retinal Image Processing Pipeline\n\nVersion 1.0"
        )
    
    def _on_metric_definitions(self):
        """Show metric definitions dialog"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Metric Definitions")
        dialog.setText("Metric Definitions")
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMarkdown(''.join(f"<p><b>{key}</b>: {value}</p>" for key, value in METRIC_DICTIONARY.items()))
        text_edit.setMinimumWidth(800)
        text_edit.setMinimumHeight(600)
        
        dialog.layout().addWidget(text_edit, 0, 1)
        dialog.exec()
    
    def _on_qc_decision(self, filename: str, decision: str):
        """Handle quality control decision"""
        status = f"Decided: {filename} → {decision.upper()}"
        self.statusBar().showMessage(status)
        if self.image_filter.currentText() in {"Unreviewed", "Rejected"}:
            self._apply_image_filter(self.image_filter.currentText())
    
    def _on_select_image(self, img_idx: int):
        """Handle image selection from dropdown"""        
        if not self._image_folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select an image folder first.")
            return
        if img_idx < 0 or img_idx >= len(self.widget.image_paths):
            return
        
        self.widget.index = img_idx
        self.widget.display_image()
        
        image_name = self.image_dropdown.itemText(img_idx)
        image_path = self._image_folder / image_name
        if not image_path.exists():
            QMessageBox.warning(self, "Image Not Found", f"The selected image does not exist: {image_path}")
            return

    def _on_select_image_text(self):
        """Jump to an image after typing in the searchable selector."""
        image_name = self.image_dropdown.currentText().strip()
        if not image_name or image_name not in self.image_list:
            return

        self._on_select_image(self.image_list.index(image_name))

    def _on_image_filter_changed(self, filter_text: str):
        """Handle quick image list filter changes."""
        self._apply_image_filter(filter_text)
    
    def _on_select_image_folder(self):
        """Select image folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder"
        )
        
        if not folder:
            return
        
        self._image_folder = Path(folder)
        self.folder_label.setText(f"📂 {self._image_folder.name}")
        self.statusBar().showMessage(f"Loaded folder: {self._image_folder}")
        
        # Load images in the appropriate widget
        self._load_images(self._image_folder)
    
    def _on_select_mask_folder(self):
        """Select mask folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Mask Folder"
        )
        
        if not folder:
            return
        self._mask_folder = Path(folder)
        self._load_mask_info_to_db(self._mask_folder)
        if self._all_image_paths:
            self._apply_image_filter(self.image_filter.currentText())
        self.statusBar().showMessage(f"Loaded masks from: {self._mask_folder}")
    
    def _on_av_segment(self):
        """Handle A/V segmentation action"""
        from backend.utils.threads import AVSegmentationWorker
        
        # Get current image
        img_name = self.image_dropdown.currentText()
        if not img_name:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to segment.")
            return
        
        # Find in database
        meta = self._db_manager.get_metadata_by_filename(img_name)
        if not meta:
            QMessageBox.warning(self, "Image Not in Database", f"The selected image is not in the database: {img_name}")
            return
        image_path = Path(meta.folder) / Path(meta.name + meta.extension)
        
        # Find segmentation mask details
        seg_meta = self._db_manager.get_segmentation_result_by_id(meta.id)
        if not seg_meta:
            # Save new segmentation entry
            seg_meta = {
                "id": meta.id,
                "extension": ".png",
                "seg_folder": SEG_DIR,
                "model_name": "uwf_av_segmentor",
                "model_version": "1.0"
            }
            self._db_manager.save_segmentation_result(**seg_meta)
            # Save empty segmentation mask
            seg_path = Path(SEG_DIR) / Path(meta.name + seg_meta["extension"])
            self._save_empty_mask(image_path, seg_path)
        else:
            seg_path = Path(seg_meta.seg_folder) / Path(meta.name + seg_meta.extension)
        
        # Perform segmentation
        try:
            self.statusBar().showMessage(f"Segmenting A/V for image: {img_name}")
            self.worker = AVSegmentationWorker(self.widget.step_seg, image_path, seg_path)
            self.worker.finished.connect(lambda success: self.statusBar().showMessage(
                f"A/V Segmentation {'succeeded' if success else 'failed'} for image: {img_name}"
            ))
            self.worker.finished.connect(self.widget.display_image)
            self.worker.start()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error during A/V segmentation: {str(e)}")
            
        # Update display
        self.widget.display_image()
    
    def _on_disc_segment(self):
        """Handle Disc segmentation action"""
        from backend.utils.threads import DiscSegmentationWorker
        
        # Get current image
        img_name = self.image_dropdown.currentText()
        if not img_name:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to segment.")
            return
        
        # Find in database
        meta = self._db_manager.get_metadata_by_filename(img_name)
        if not meta:
            QMessageBox.warning(self, "Image Not in Database", f"The selected image is not in the database: {img_name}")
            return
        image_path = Path(meta.folder) / Path(meta.name + meta.extension)
        
        # Find segmentation mask details
        seg_meta = self._db_manager.get_segmentation_result_by_id(meta.id)
        if not seg_meta:
            # Save new segmentation entry
            seg_meta = {
                "id": meta.id,
                "extension": ".png",
                "seg_folder": SEG_DIR,
                "model_name": "uwf_disc_full_seg",
                "model_version": "1.0"
            }
            self._db_manager.save_segmentation_result(**seg_meta)
            # Save empty segmentation mask
            seg_path = Path(SEG_DIR) / Path(meta.name + seg_meta["extension"])
            self._save_empty_mask(image_path, seg_path)
        else:
            seg_path = Path(seg_meta.seg_folder) / Path(meta.name + seg_meta.extension)
        
        # Perform segmentation
        try:
            self.statusBar().showMessage(f"Segmenting Disc for image: {img_name}")
            self.worker = DiscSegmentationWorker(self.widget.step_seg, image_path, seg_path)
            self.worker.finished.connect(lambda success: self.statusBar().showMessage(
                f"Disc Segmentation {'succeeded' if success else 'failed'} for image: {img_name}"
            ))
            self.worker.finished.connect(self.widget.display_image)
            self.worker.start()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error during disc segmentation: {str(e)}")
    
    def _on_assign_qc_all(self, decision: str):
        """Assign QC decision to all images in the current list"""
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "No images loaded to assign QC.")
            return
            
        # Create message box explicitly to better control focus/default
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Batch Assignment")
        msg_box.setText(f"Are you sure you want to assign '{decision.upper()}' to all {len(self.image_list)} images?")
        msg_box.setIcon(QMessageBox.Icon.Question)
        
        yes_btn = msg_box.addButton(QMessageBox.StandardButton.Yes)
        no_btn = msg_box.addButton(QMessageBox.StandardButton.No)
        
        msg_box.setDefaultButton(no_btn)
        
        msg_box.exec()
        
        if msg_box.clickedButton() == yes_btn:
            count = 0
            for image_name in self.image_list:
                name = Path(image_name).stem
                if self._db_manager.save_qc_result(name, decision):
                    count += 1
            
            self.statusBar().showMessage(f"Assigned {decision.upper()} to {count} / {len(self.image_list)} images.")
            
            # Refresh current image display to show new status
            self.widget.display_image()

    def _on_export_metrics(self):
        """Export metrics from database"""
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results Workbook",
            "ultralyzer_results.xlsx",
            "Excel Workbooks (*.xlsx);;All Files (*)"
        )
        
        if not save_path:
            return
        
        try:
            save_path = Path(save_path)
            if save_path.suffix.lower() not in {".xlsx", ".xlsm"}:
                save_path = save_path.with_suffix(".xlsx")
            self._db_manager.export_metrics_results(save_path)
            self.statusBar().showMessage(f"Results workbook exported to: {save_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error exporting results workbook: {str(e)}")
            
    def _on_export_to_psd_all(self):
        """Export all segmentations to PSD"""
        import photoshopapi as psapi

        # Get image list from dropdown
        image_names = self.image_list
        
        if not image_names:
            QMessageBox.warning(self, "No Images", "No images loaded to export.")
            return
        
        save_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save PSD Files")
        
        # Setup Progress Dialog
        progress = QProgressDialog("Initializing export...", "Cancel", 0, len(image_names), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        k = 0
        for i, img_name in enumerate(image_names):
            
            # Check if user clicked cancel
            if progress.wasCanceled():
                break
                
            progress.setLabelText(f"Exporting {img_name}...")
            
            name = Path(img_name).stem
            
            # Find mask in database
            meta = self._db_manager.get_metadata_by_filename(img_name)
            if not meta:
                QMessageBox.warning(self, "Image Not in Database", f"The selected image is not in the database: {name}")
                return
            seg_meta = self._db_manager.get_segmentation_result_by_id(meta.id)
            if not seg_meta:
                QMessageBox.warning(self, "No Segmentation Found", f"No segmentation found for image: {name}")
                return
            img_path = Path(meta.folder) / Path(meta.name + meta.extension)
            seg_path = Path(seg_meta.seg_folder) / Path(meta.name + seg_meta.extension)
            
            # Check if files exist
            if not img_path.exists() or not seg_path.exists():
                QMessageBox.warning(self, "Files Not Found", f"Image or segmentation file not found for: {name}")
                continue
            
            # Read image & mask
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            green = cv2.cvtColor(image[:,:,1], cv2.COLOR_GRAY2RGB)
            mask = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            w, h = image.shape[1], image.shape[0]

            arteries = np.zeros((h, w, 4), dtype=np.uint8)
            arteries[mask[:,:,0] > 0] = [255, 0, 0, 255]
            veins = np.zeros((h, w, 4), dtype=np.uint8)
            veins[mask[:,:,2] > 0] = [0, 0, 255, 255]
            disc = np.zeros((h, w, 4), dtype=np.uint8)
            disc[mask[:,:,1] > 0] = [0, 255, 0, 255]
            
            # Transpose to (C, H, W) for psapi
            image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
            green = np.ascontiguousarray(np.transpose(green, (2, 0, 1)))
            arteries = np.ascontiguousarray(np.transpose(arteries, (2, 0, 1)))
            veins = np.ascontiguousarray(np.transpose(veins, (2, 0, 1)))
            disc = np.ascontiguousarray(np.transpose(disc, (2, 0, 1)))
            
            # Prepare PSD file
            color_mode = psapi.enum.ColorMode.rgb
            psd = psapi.LayeredFile_8bit(color_mode, w, h)
            
            # Center the layers
            cx, cy = w / 2, h / 2
            
            psd.add_layer(psapi.ImageLayer_8bit(arteries, "Arteries", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
            psd.add_layer(psapi.ImageLayer_8bit(veins, "Veins", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
            psd.add_layer(psapi.ImageLayer_8bit(disc, "Optic Disc", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
            psd.add_layer(psapi.ImageLayer_8bit(green, "Green Channel", width=w, height=h, is_visible=True, is_locked=True, pos_x=cx, pos_y=cy))
            psd.add_layer(psapi.ImageLayer_8bit(image, "Color Image", width=w, height=h, is_visible=False, is_locked=True, pos_x=cx, pos_y=cy))
            
            
            # Save PSD file
            psd.write(Path(save_folder) / Path(name + ".psd"))
            k += 1
            
            # Update progress
            progress.setValue(i + 1)
            QApplication.processEvents()
            
        progress.setValue(len(image_names))        
        self.statusBar().showMessage(f"Exported {k} / {len(image_names)} PSD files to: {save_folder}")            
    
    def _on_export_to_psd_current(self):
        """Export current segmentation to PSD"""
        import photoshopapi as psapi

        img_name = self.image_dropdown.currentText()
        name = Path(img_name).stem
        
        if not img_name:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to export.")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PSD File",
            name + ".psd",
            "PSD Files (*.psd);;All Files (*)"
        )
        
        if not save_path:
            return
        
        # Find mask in database
        meta = self._db_manager.get_metadata_by_filename(name)
        if not meta:
            QMessageBox.warning(self, "Image Not in Database", f"The selected image is not in the database: {name}")
            return
        seg_meta = self._db_manager.get_segmentation_result_by_id(meta.id)
        if not seg_meta:
            QMessageBox.warning(self, "No Segmentation Found", f"No segmentation found for image: {name}")
            return
        img_path = Path(meta.folder) / Path(meta.name + meta.extension)
        seg_path = Path(seg_meta.seg_folder) / Path(meta.name + seg_meta.extension)
        
        # Check if files exist
        if not img_path.exists() or not seg_path.exists():
            QMessageBox.warning(self, "Files Not Found", f"Image or segmentation file not found for: {name}")
            return
        
        # Read image & mask
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        green = cv2.cvtColor(image[:,:,1], cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        w, h = image.shape[1], image.shape[0]

        arteries = np.zeros((h, w, 4), dtype=np.uint8)
        arteries[mask[:,:,0] > 0] = [255, 0, 0, 255]
        veins = np.zeros((h, w, 4), dtype=np.uint8)
        veins[mask[:,:,2] > 0] = [0, 0, 255, 255]
        disc = np.zeros((h, w, 4), dtype=np.uint8)
        disc[mask[:,:,1] > 0] = [0, 255, 0, 255]
        
        # Transpose to (C, H, W) for psapi
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        green = np.ascontiguousarray(np.transpose(green, (2, 0, 1)))
        arteries = np.ascontiguousarray(np.transpose(arteries, (2, 0, 1)))
        veins = np.ascontiguousarray(np.transpose(veins, (2, 0, 1)))
        disc = np.ascontiguousarray(np.transpose(disc, (2, 0, 1)))
        
        # Prepare PSD file
        w, h = image.shape[2], image.shape[1]
        color_mode = psapi.enum.ColorMode.rgb
        psd = psapi.LayeredFile_8bit(color_mode, w, h)
        
        # Center the layers
        cx, cy = w / 2, h / 2
        
        psd.add_layer(psapi.ImageLayer_8bit(arteries, "Arteries", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
        psd.add_layer(psapi.ImageLayer_8bit(veins, "Veins", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
        psd.add_layer(psapi.ImageLayer_8bit(disc, "Optic Disc", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
        psd.add_layer(psapi.ImageLayer_8bit(green, "Green Channel", width=w, height=h, is_visible=True, is_locked=True, pos_x=cx, pos_y=cy))
        psd.add_layer(psapi.ImageLayer_8bit(image, "Color Image", width=w, height=h, is_visible=False, is_locked=True, pos_x=cx, pos_y=cy))
        
        # Save PSD file
        print(save_path)
        psd.write(save_path)
        
        self.statusBar().showMessage(f"Exported PSD file to: {save_path}")

    def _on_import_from_psd(self):
        """Import segmentations from a folder containing PSD files"""
        from backend.utils.psd_handler import process_psd_file

        psd_path, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PSD File(s) to Import From",
            "",
            "PSD Files (*.psd);;All Files (*)"
        )
        
        if not psd_path:
            return
        
        # Remove non-PSD files (just in case)
        psd_path = [p for p in psd_path if Path(p).suffix.lower() == ".psd"]
        
        # Setup Progress Dialog
        progress = QProgressDialog("Importing selected PSD files...", "Cancel", 0, len(psd_path), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        k = 0
        for i, path_str in enumerate(psd_path):
            
            filename = Path(path_str).stem
            mask, image = process_psd_file(path_str)
            
            # Check if image exists in database
            meta = self._db_manager.get_metadata_by_filename(filename)
            
            if not meta: # Image not in database, save it and QC it borderline
                try:
                    # Save image
                    folder = Path(self.image_folder) if self.image_folder else Path(IM_DIR)
                    folder.mkdir(parents=True, exist_ok=True)
                    extension = ".tif"
                    cv2.imwrite(str(folder / Path(filename + extension)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    self.statusBar().showMessage(f"Error saving imported image {filename}: {str(e)}")
                    continue
                # Update database
                meta = {'filename': filename, 'extension': extension, 'folder': str(folder)}
                succ = self._db_manager.save_image_metadata(**meta)
                if not succ:
                    self.statusBar().showMessage(f"Error saving metadata for imported image: {filename}")
                    continue
                # Update QC on database
                succ = self._db_manager.save_qc_result(name=filename, 
                                                        decision="borderline", 
                                                        notes="Imported from PSD, needs review")
                if not succ:
                    self.statusBar().showMessage(f"Error saving QC for imported image: {filename}")
                    continue
                meta = self._db_manager.get_metadata_by_filename(filename)
                
            if meta:
                # Save segmentation mask
                seg_folder = Path(SEG_DIR)
                seg_folder.mkdir(parents=True, exist_ok=True)
                seg_extension = ".png"
                seg_path = seg_folder / Path(filename + seg_extension)
                cv2.imwrite(str(seg_path), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
                
                # Update database
                succ = self._db_manager.save_segmentation_result(id = meta.id, 
                                                                 extension = seg_extension, 
                                                                 seg_folder = str(seg_folder), 
                                                                 model_name = "Imported from PSD", 
                                                                 model_version = "N/A")
                if not succ:
                    self.statusBar().showMessage(f"Error saving segmentation for imported image: {filename}")
                    continue
                self.statusBar().showMessage(f"Imported segmentation for image: {filename}")
                
            k += 1
            
            # Update progress
            progress.setValue(i + 1)
            QApplication.processEvents()
            
        progress.setValue(len(psd_path))        
        self.statusBar().showMessage(f"Imported {k} / {len(psd_path)} PSD files.")  
                    
    def _on_dicom_converter(self):
        """Convert DICOM files to TIFF images"""
        from backend.utils.dicom_handler import dicom_to_image
        
        dicom_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DICOM File(s) to Convert",
            "",
            "DICOM Files (*.dcm *.dicom);;All Files (*)"
        )
        
        if not dicom_paths:
            return
        
        # Remove non-DICOM files (just in case)
        dicom_paths = [p for p in dicom_paths if Path(p).suffix.lower() in [".dcm", ".dicom"]]
        
        # Setup Progress Dialog
        progress = QProgressDialog("Extracting images from selected DICOM files...", "Cancel", 0, len(dicom_paths), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        k = 0
        for i, path_str in enumerate(dicom_paths):
            
            filename = Path(path_str).stem
            try:
                succ, image, metadata = dicom_to_image(path_str)
                if not succ:
                    self.statusBar().showMessage(f"Failed to read DICOM file: {filename}")
                    continue
            except Exception as e:
                self.statusBar().showMessage(f"Error extracting metadata from DICOM file {filename}: {str(e)}")
                continue
            
            # Save image
            try:
                # Compose image name
                image_name = f"{metadata['PatientID']}_{metadata['Laterality']}_{metadata['Timestamp']}"
                folder = Path(self._image_folder) if self._image_folder else Path(IM_DIR)
                folder.mkdir(parents=True, exist_ok=True)
                extension = ".tif"
                cv2.imwrite(str(folder / Path(image_name + extension)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self.statusBar().showMessage(f"Converted DICOM to image: {image_name + extension}")
            except Exception as e:
                self.statusBar().showMessage(f"Error saving converted image {filename}: {str(e)}")
                continue
            
            k += 1
            
            # Update progress
            progress.setValue(i + 1)
            QApplication.processEvents()
            
        progress.setValue(len(dicom_paths))        
        self.statusBar().showMessage(f"Converted {k} / {len(dicom_paths)} DICOM files to images.")
        
                  
