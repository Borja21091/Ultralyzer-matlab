import numpy as np
from PIL import Image
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsView, 
    QProgressBar, QComboBox, QSlider, QSplitter, 
    QLabel, QMessageBox, QWidget, QTextEdit, QFrame, QCheckBox, QStyle,
    QScrollArea, QStackedWidget
)
from frontend.widgets.canvas import Canvas
from PySide6.QtCore import Qt, Signal, QEvent, QPoint, QThread
from frontend.widgets.widget_base import BaseWidget
from backend.models.database import DatabaseManager
from backend.services.geometry_adapter import GeometryTrafficLight, get_default_geometry_adapter
from backend.steps.segmentation import SegmentationStep
from frontend.widgets.canvas import ImageLayer, OverlayLayer
from definitions import IMAGE_CHANNEL_MAP, OVERLAY_MAP, BLANK_STATE
from PySide6.QtGui import QShortcut, QKeySequence, QCursor, QPixmap, QPainter, QColor, QAction

from backend.utils.threads import SingleMetricsWorker, BatchMetricsWorker
from backend.utils.threads import SingleSegmentationWorker, BatchSegmentationWorker
from backend.services.roi_masks import ROIMaskService, SegmentationBundle

if TYPE_CHECKING:
    from backend.steps.metrics import MetricsStep
    from backend.models.segmentor import Segmentor



class SegmentationWidget(BaseWidget):
    """Interactive segmentation, visualization and correction widget"""
    
    status_text = Signal(str)
    decision_made = Signal(str, str) # filename, decision
    request_open_folder = Signal()
    
    def __init__(self, 
                 db_manager: DatabaseManager | None = None, 
                 av_segmentor: "Segmentor | None" = None, 
                 disc_segmentor: "Segmentor | None" = None, 
                 fovea_segmentor: "Segmentor | None" = None):
        super().__init__(db_manager)
        self.step_seg = SegmentationStep(av_segmentor=av_segmentor, 
                                         disc_segmentor=disc_segmentor, 
                                         fovea_segmentor=fovea_segmentor,
                                         db_manager=self.db_manager)
        self.step_metrics: "MetricsStep | None" = None
        
        # Track segmentation worker thread to prevent garbage collection
        self._worker_thread = None
        
        # Overlay opacity
        self._overlay_opacity = 0.75
        
        # QC state
        self.state = BLANK_STATE.copy()
        
        # Edit mode state
        self._edit_mode = False
        self._overlay_layer: OverlayLayer | None = None
        self._active_tool = None
        self._brush_size = 5
        self.MIN_BRUSH_SIZE = 1
        self.MAX_BRUSH_SIZE = 500
        self._stroke_started = False
        self._has_unsaved_changes = False
        self.opacity_slider: QSlider | None = None
        self.opacity_label: QLabel | None = None
        self.geometry_status_indicator: QFrame | None = None
        self.geometry_adapter = get_default_geometry_adapter()
        
        # Track which keys are currently pressed
        self._keys_pressed = set()

        # ROI metric selection state
        self._roi_combo_index_by_codes = {}
        self._roi_definitions_by_code = {}
        self._roi_code_order = ("full", "mid_periphery", "central")
        self._persistent_roi_codes: tuple[str, ...] = tuple(self._roi_code_order)
        self._loading_roi_selection = False
        self._last_progress_message: str | None = None
        self._last_geometry_summary: str | None = None
        self._last_geometry_traffic_light: str | None = None
        self._roi_outlines_visible = False
        self._current_image_shape: tuple[int, int] | None = None
        self._current_image_array: np.ndarray | None = None
        self._current_overlay_array: np.ndarray | None = None
        self.roi_mask_service = ROIMaskService(db_manager=self.db_manager)
        self.roi_outline_colors = {
            "full": QColor(16, 185, 129, 220),
            "central": QColor(96, 165, 250, 220),
            "mid_periphery": QColor(245, 158, 11, 220),
        }
        self.btn_prev: QPushButton | None = None
        self.btn_next: QPushButton | None = None
        self.view_panel_widget: QWidget | None = None
        self.workflow_panel_widget: QWidget | None = None
        self.batch_segment_action: QAction | None = None
        self.batch_metrics_action: QAction | None = None
        self.workflow_scroll_area: QScrollArea | None = None
        self.workflow_mode_stack: QStackedWidget | None = None
        self.review_mode_widget: QWidget | None = None
        self.edit_mode_page_widget: QWidget | None = None
        self.btn_exit_edit_mode: QPushButton | None = None
        self.canvas_prev_overlay: QWidget | None = None
        self.canvas_next_overlay: QWidget | None = None
        self._canvas_overlays_enabled = False
        
        self._init_ui()
        self._refresh_geometry_readiness()

    def _get_step_metrics(self) -> "MetricsStep":
        if self.step_metrics is None:
            from backend.steps.metrics import MetricsStep

            self.step_metrics = MetricsStep(db_manager=self.db_manager)
        return self.step_metrics
    
    ############ PROPERTIES ############
    
    @property
    def state(self) -> dict:
        """Get current widget state"""
        return self._state

    @state.setter
    def state(self, value: dict):
        """Set current widget state"""
        self._state = value
    
    @property
    def edit_mode(self) -> bool:
        """Get edit mode state"""
        return self._edit_mode
    
    @edit_mode.setter
    def edit_mode(self, value: bool):
        """Set edit mode state"""
        self._edit_mode = value
    
    @property
    def active_tool(self):
        """Get active editing tool"""
        return self._active_tool
    
    @active_tool.setter
    def active_tool(self, tool: str):
        """Set active editing tool"""
        if self._active_tool == tool:
            # Deselect tool
            self._active_tool = None
            self.btn_brush.setChecked(False)
            self.btn_smart_paint.setChecked(False)
            self.btn_eraser.setChecked(False)
            self.btn_change.setChecked(False)
            self.btn_fovea_location.setChecked(False)
            self.canvas.set_tool(None)
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # Select tool
            self._active_tool = tool
            self.btn_brush.setChecked(tool == "brush")
            self.btn_smart_paint.setChecked(tool == "smart_paint")
            self.btn_eraser.setChecked(tool == "eraser")
            self.btn_change.setChecked(tool == "change")
            self.btn_fovea_location.setChecked(tool == "fovea_location")
            self.canvas.set_tool(tool)
            if tool in ["brush", "smart_paint", "eraser"]:
                cursor = self._create_brush_cursor(self.brush_size // 2)
                self.canvas.setCursor(cursor)
            else:
                self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    @property
    def channel(self) -> str:
        """Get current display channel"""
        return self.channel_combo.currentData() or self.channel_combo.currentText().lower()
    
    @channel.setter
    def channel(self, value: str):
        """Set current display channel"""
        if value.lower() in IMAGE_CHANNEL_MAP.keys():
            index = self.channel_combo.findData(value.lower())
            if index < 0:
                index = list(IMAGE_CHANNEL_MAP.keys()).index(value.lower())
            self.channel_combo.setCurrentIndex(index)
    
    @property
    def overlay(self) -> str:
        """Get current segmentation overlay option"""
        return self.overlay_combo.currentData() or self.overlay_combo.currentText().lower()
    
    @overlay.setter
    def overlay(self, value: str):
        """Set current segmentation overlay option"""
        if value.lower() in OVERLAY_MAP.keys():
            index = self.overlay_combo.findData(value.lower())
            if index < 0:
                index = list(OVERLAY_MAP.keys()).index(value.lower())
            self.overlay_combo.setCurrentIndex(index)

    @property
    def segmentation_mask_path(self) -> Path:
        """Get segmentation mask path for current image"""
        name = str(self.image_path.stem)
        return self.db_manager.get_segmentation_mask_path(name)

    @property
    def brush_size(self) -> int:
        """Get brush size"""
        return self._brush_size
    
    @brush_size.setter
    def brush_size(self, size: int):
        """Set brush size"""
        self._brush_size = min(max(1, size), 500)

    @property
    def has_unsaved_changes(self) -> bool:
        """Get unsaved changes state"""
        return bool(self._overlay_layer and self._overlay_layer._is_dirty)

    ############ UI ############
    
    def _init_ui(self):
        """Initialize User Interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Top section: Image info
        info_layout = QHBoxLayout()
        
        # Use base widget's top layout
        top_info, self.image_counter_label, self.image_name_label = self._create_top_info_layout()
        info_layout.addLayout(top_info)
        info_layout.addSpacing(10)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(12)
        info_layout.addWidget(self.progress_bar)

        self.progress_status_label = QLabel("Ready")
        self.progress_status_label.setMinimumWidth(220)
        self.progress_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_layout.addWidget(self.progress_status_label)
        
        self.image_status_label = QLabel("Load a folder to begin")
        self.image_status_label.setMinimumWidth(360)
        self.image_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_layout.addWidget(self.image_status_label)
        
        info_layout.addSpacing(10)
        layout.addLayout(info_layout)
        
        # Main content area: left view sidebar, center canvas, right workflow sidebar.
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.view_panel_widget = self._create_view_sidebar()
        self.view_panel_widget.setMinimumWidth(220)
        self.view_panel_widget.setMaximumWidth(260)
        self.main_splitter.addWidget(self.view_panel_widget)
        
        # Center: Canvas container
        self.canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        
        # Initialize Canvas immediately
        self.empty_state_widget = self._create_empty_state()
        canvas_layout.addWidget(self.empty_state_widget)

        self.canvas = Canvas()
        self.canvas.setVisible(False)
        self.canvas.signal_zoom_changed.connect(self._update_brush_cursor)
        self.canvas.signal_fovea_selected.connect(self._on_fovea_location_selected)
        self.canvas.signal_opacity_changed.connect(self._on_canvas_opacity_delta)
        self.canvas.signal_brush_radius_changed.connect(lambda delta: self._on_brush_size_changed(max(self.brush_size + delta, 1)))
        self.canvas.viewport().setMouseTracking(True)
        self.canvas.viewport().installEventFilter(self)
        canvas_layout.addWidget(self.canvas)
        self.canvas_prev_overlay = self._create_canvas_nav_button(self.canvas.viewport(), "prev")
        self.canvas_next_overlay = self._create_canvas_nav_button(self.canvas.viewport(), "next")
        self._set_canvas_overlay_visible(False)
        self.main_splitter.addWidget(self.canvas_container)
        
        self.workflow_panel_widget = self._create_workflow_sidebar()
        self.workflow_panel_widget.setMinimumWidth(280)
        self.workflow_panel_widget.setMaximumWidth(340)
        self.main_splitter.addWidget(self.workflow_panel_widget)
        
        # Set splitter proportions (canvas takes most space)
        self.main_splitter.setSizes([230, 1040, 320])
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, False)
        
        layout.addWidget(self.main_splitter, 1)
        
        # Shortcuts
        self._setup_shortcuts()

    def _create_canvas_nav_button(self, parent: QWidget | None = None, direction: str = "next") -> QWidget:
        """Create one hover-revealed navigation button over canvas."""
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        container.setStyleSheet("""
            QWidget {
                background: rgba(21, 24, 30, 175);
                border: 1px solid rgba(107, 114, 128, 140);
                border-radius: 10px;
            }
        """)

        button = QPushButton(container)
        button.setFixedSize(42, 72)
        container.setMouseTracking(True)
        button.setMouseTracking(True)
        if direction == "prev":
            button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
            button.setToolTip("Previous image")
            button.clicked.connect(self._on_prev)
            self.btn_prev = button
        else:
            button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
            button.setToolTip("Next image")
            button.clicked.connect(self._on_next)
            self.btn_next = button

        button.setIconSize(button.iconSize().expandedTo(button.size() * 0.45))
        button.setStyleSheet(self.button_styles["nav"]["normal"])
        layout.addWidget(button)

        container.installEventFilter(self)
        button.installEventFilter(self)

        container.adjustSize()
        return container

    def _set_canvas_overlay_visible(self, visible: bool):
        """Show or hide hover-based canvas navigation controls."""
        self._canvas_overlays_enabled = visible
        self._update_canvas_nav_visibility(None)

    def _update_canvas_nav_visibility(self, local_pos: QPoint | None):
        """Show nav buttons only when hovering near canvas side edges."""
        left_visible = False
        right_visible = False

        if self._canvas_overlays_enabled and self.canvas:
            viewport = self.canvas.viewport()
            pos = local_pos if local_pos is not None else viewport.mapFromGlobal(QCursor.pos())
            rect = viewport.rect()
            if rect.contains(pos):
                side_band = min(96, max(72, rect.width() // 8))
                left_visible = pos.x() <= side_band
                right_visible = pos.x() >= rect.width() - side_band

        if self.canvas_prev_overlay:
            self.canvas_prev_overlay.setVisible(left_visible)
        if self.canvas_next_overlay:
            self.canvas_next_overlay.setVisible(right_visible)

    def _position_canvas_overlays(self):
        """Keep hover navigation controls pinned to the canvas edges."""
        if not self.canvas:
            return

        viewport = self.canvas.viewport()
        width = viewport.width()
        height = viewport.height()
        margin = 12

        center_y = max(margin, (height // 2) - 44)

        if self.canvas_prev_overlay:
            self.canvas_prev_overlay.adjustSize()
            self.canvas_prev_overlay.move(margin, center_y)
            self.canvas_prev_overlay.raise_()

        if self.canvas_next_overlay:
            self.canvas_next_overlay.adjustSize()
            self.canvas_next_overlay.move(
                max(margin, width - self.canvas_next_overlay.width() - margin),
                center_y,
            )
            self.canvas_next_overlay.raise_()

    def eventFilter(self, watched: object, event: QEvent):
        """Reposition canvas overlays when viewport changes size."""
        prev_button = getattr(self, "btn_prev", None)
        next_button = getattr(self, "btn_next", None)

        if self.canvas and watched is self.canvas.viewport():
            if event.type() in {QEvent.Type.Resize, QEvent.Type.Show}:
                self._position_canvas_overlays()
                self._update_canvas_nav_visibility(None)
            elif event.type() == QEvent.Type.MouseMove:
                self._update_canvas_nav_visibility(event.pos())
            elif event.type() == QEvent.Type.Enter:
                self._update_canvas_nav_visibility(None)
            elif event.type() == QEvent.Type.Leave:
                self._update_canvas_nav_visibility(None)
        elif watched in {self.canvas_prev_overlay, prev_button}:
            if event.type() in {QEvent.Type.Enter, QEvent.Type.MouseMove, QEvent.Type.Show}:
                if self.canvas_prev_overlay:
                    self.canvas_prev_overlay.setVisible(self._canvas_overlays_enabled)
            elif event.type() == QEvent.Type.Leave:
                self._update_canvas_nav_visibility(None)
        elif watched in {self.canvas_next_overlay, next_button}:
            if event.type() in {QEvent.Type.Enter, QEvent.Type.MouseMove, QEvent.Type.Show}:
                if self.canvas_next_overlay:
                    self.canvas_next_overlay.setVisible(self._canvas_overlays_enabled)
            elif event.type() == QEvent.Type.Leave:
                self._update_canvas_nav_visibility(None)
        return super().eventFilter(watched, event)

    def _create_edit_toolbar(self) -> QVBoxLayout:
        """Create compact edit tools shown in the dedicated edit sidebar mode."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        tool_button_style = """
            QPushButton {
                background: #3f3f46;
                color: #f3f4f6;
                border: 1px solid #52525b;
                border-radius: 5px;
                padding: 6px 10px;
            }
            QPushButton:checked {
                background: #1f2937;
                border-color: #60a5fa;
            }
        """

        def create_tool_button(text: str, tool_name: str) -> QPushButton:
            button = QPushButton(text)
            button.setCheckable(True)
            button.setMinimumHeight(30)
            button.setStyleSheet(tool_button_style)
            button.clicked.connect(lambda _checked=False, name=tool_name: setattr(self, "active_tool", name))
            layout.addWidget(button)
            return button

        self.btn_brush = create_tool_button("Brush", "brush")
        self.btn_smart_paint = create_tool_button("Smart paint", "smart_paint")
        self.btn_eraser = create_tool_button("Eraser", "eraser")
        self.btn_change = create_tool_button("Swap", "change")
        self.btn_fovea_location = create_tool_button("Fovea", "fovea_location")

        size_label = QLabel("Brush size")
        size_label.setObjectName("sidebarField")
        layout.addWidget(size_label)

        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(self.MIN_BRUSH_SIZE)
        self.brush_size_slider.setMaximum(self.MAX_BRUSH_SIZE)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        layout.addWidget(self.brush_size_slider)

        self.brush_size_label = QLabel(f"{self.brush_size}px")
        self.brush_size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.brush_size_label)

        layout.addStretch()
        return layout

    def _setup_shortcuts(self):
        """Register keyboard shortcuts for overlays and edit tools."""
        QShortcut(QKeySequence("1"), self).activated.connect(lambda: self._set_overlay_by_key("red"))
        QShortcut(QKeySequence("2"), self).activated.connect(lambda: self._set_overlay_by_key("green"))
        QShortcut(QKeySequence("3"), self).activated.connect(lambda: self._set_overlay_by_key("blue"))
        QShortcut(QKeySequence("4"), self).activated.connect(lambda: self._set_overlay_by_key("vessels"))
        QShortcut(QKeySequence("5"), self).activated.connect(lambda: self._set_overlay_by_key("all"))
        QShortcut(QKeySequence("6"), self).activated.connect(lambda: self._set_overlay_by_key("none"))
        QShortcut(QKeySequence("B"), self).activated.connect(lambda: setattr(self, "active_tool", "brush"))
        QShortcut(QKeySequence("Shift+B"), self).activated.connect(lambda: setattr(self, "active_tool", "smart_paint"))
        QShortcut(QKeySequence("E"), self).activated.connect(lambda: setattr(self, "active_tool", "eraser"))
        QShortcut(QKeySequence("C"), self).activated.connect(lambda: setattr(self, "active_tool", "change"))
        QShortcut(QKeySequence("F"), self).activated.connect(lambda: setattr(self, "active_tool", "fovea_location"))
        QShortcut(QKeySequence("-"), self).activated.connect(self._on_decrease_brush_size)
        QShortcut(QKeySequence("="), self).activated.connect(self._on_increase_brush_size)
        QShortcut(QKeySequence.StandardKey.Save, self).activated.connect(self._on_save)
        QShortcut(QKeySequence.StandardKey.Undo, self).activated.connect(self._on_undo)
        QShortcut(QKeySequence.StandardKey.Redo, self).activated.connect(self._on_redo)
        shortcut_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        shortcut_prev.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut_prev.activated.connect(self._on_prev)
        shortcut_next = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        shortcut_next.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut_next.activated.connect(self._on_next)

    def _create_empty_state(self) -> QWidget:
        """Create first-run guidance for the canvas area."""
        panel = QFrame()
        panel.setObjectName("emptyStatePanel")
        panel.setStyleSheet("""
            QFrame#emptyStatePanel {
                background: #2f3133;
                border: 1px solid #45484c;
                border-radius: 4px;
            }
            QLabel#emptyTitle {
                color: #f3f4f6;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#emptyBody {
                color: #cbd5e1;
                font-size: 13px;
            }
            QPushButton#emptyOpenButton {
                background: #4b5563;
                color: #f9fafb;
                border: 1px solid #64748b;
                border-radius: 4px;
                padding: 6px 14px;
            }
            QPushButton#emptyOpenButton:hover {
                background: #5b6675;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        layout.addStretch()

        title = QLabel("No image folder loaded")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state_title_label = title
        layout.addWidget(title)

        body = QLabel("Open a folder to begin reviewing retinal images.")
        body.setObjectName("emptyBody")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state_body_label = body
        layout.addWidget(body)

        open_button = QPushButton("Open image folder")
        open_button.setObjectName("emptyOpenButton")
        open_button.setMinimumHeight(34)
        open_button.setMaximumWidth(180)
        open_button.clicked.connect(self.request_open_folder.emit)
        open_button.setAccessibleName("Open image folder")
        layout.addWidget(open_button, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        return panel

    def show_empty_state(self, title: str | None = None, message: str | None = None):
        """Show the canvas empty state with optional copy."""
        if title:
            self.empty_state_title_label.setText(title)
        if message:
            self.empty_state_body_label.setText(message)

        self._current_image_shape = None
        self._overlay_layer = None
        self.canvas.setVisible(False)
        self.empty_state_widget.setVisible(True)
        self.canvas.clear_roi_outlines()
        self._set_canvas_overlay_visible(False)
        self.image_counter_label.setText("0 / 0")
        self.image_name_label.setText("No image loaded")
        self.progress_bar.setValue(0)
        self.progress_status_label.setText("Ready")
        self.image_status_label.setText("Load a folder to begin")

    def _show_canvas(self):
        """Show the image canvas and hide first-run guidance."""
        self.empty_state_widget.setVisible(False)
        self.canvas.setVisible(True)
        self._set_canvas_overlay_visible(True)
        self._position_canvas_overlays()

    def bind_opacity_controls(self, slider: QSlider, label: QLabel | None = None):
        """Bind external opacity controls to the widget state."""
        self.opacity_slider = slider
        self.opacity_label = label

        self.opacity_slider.blockSignals(True)
        try:
            self.opacity_slider.setValue(int(self._overlay_opacity * 100))
        finally:
            self.opacity_slider.blockSignals(False)

        self.opacity_slider.valueChanged.connect(lambda value: self._on_opacity_changed(value / 100.0))
        self._on_opacity_changed(self._overlay_opacity)

    def bind_batch_actions(self, segment_action: QAction | None = None, metrics_action: QAction | None = None):
        """Bind external menu actions for batch processing state."""
        self.batch_segment_action = segment_action
        self.batch_metrics_action = metrics_action

    def _create_sidebar_panel(self, object_name: str) -> tuple[QWidget, QVBoxLayout]:
        """Create a styled sidebar container and its layout."""
        panel = QWidget()
        panel.setObjectName(object_name)
        panel.setStyleSheet("""
            QFrame#sidebarCard {
                background: #1d2024;
                border: 1px solid #343a42;
                border-radius: 10px;
            }
            QLabel#sidebarTitle {
                color: #e5e7eb;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#sidebarHint {
                color: #94a3b8;
                font-size: 11px;
            }
            QLabel#sidebarField {
                color: #a1a1aa;
                font-size: 11px;
                font-weight: 600;
            }
            QCheckBox#sidebarToggle {
                color: #e5e7eb;
                spacing: 8px;
            }
            QCheckBox#sidebarToggle::indicator {
                width: 30px;
                height: 16px;
                border-radius: 8px;
                border: 1px solid #6b7280;
                background: #4b5563;
                image: none;
            }
            QCheckBox#sidebarToggle::indicator:checked {
                background: #0f766e;
                border-color: #14b8a6;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        return panel, layout

    def _create_sidebar_card(
        self,
        title: str,
        description: str | None = None,
        header_trailing: QWidget | None = None,
    ) -> tuple[QFrame, QVBoxLayout]:
        """Create a titled card for either sidebar."""
        card = QFrame()
        card.setObjectName("sidebarCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("sidebarTitle")
        title_layout.addWidget(title_label)
        title_layout.addStretch(1)
        if header_trailing is not None:
            title_layout.addWidget(
                header_trailing,
                0,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
        layout.addLayout(title_layout)

        if description:
            description_label = QLabel(description)
            description_label.setObjectName("sidebarHint")
            description_label.setWordWrap(True)
            layout.addWidget(description_label)

        return card, layout

    def _create_labeled_control(self, title: str, control: QWidget) -> QWidget:
        """Stack a field label above a control."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        label = QLabel(title)
        label.setObjectName("sidebarField")
        layout.addWidget(label)
        layout.addWidget(control)
        return container

    def _create_view_sidebar(self) -> QWidget:
        """Create the left sidebar with all view-state controls."""
        panel, panel_layout = self._create_sidebar_panel("viewSidebar")

        view_card, view_layout = self._create_sidebar_card(
            "View",
            "Adjust how the current image and measurements are displayed.",
        )

        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumHeight(32)
        self.channel_combo.addItem("Color image", "color")
        self.channel_combo.addItem("Red channel", "red")
        self.channel_combo.addItem("Green channel", "green")
        self.channel_combo.addItem("Blue channel", "blue")
        self.channel_combo.setCurrentIndex(0)
        self.channel_combo.currentTextChanged.connect(lambda _: self._on_image_channel_changed(self.channel))
        view_layout.addWidget(self._create_labeled_control("Image", self.channel_combo))

        self.overlay_combo = QComboBox()
        self.overlay_combo.setMinimumHeight(32)
        self.overlay_combo.addItem("Arteries", "red")
        self.overlay_combo.addItem("Optic disc", "green")
        self.overlay_combo.addItem("Veins", "blue")
        self.overlay_combo.addItem("All vessels", "vessels")
        self.overlay_combo.addItem("All masks", "all")
        self.overlay_combo.addItem("No mask", "none")
        self.overlay_combo.setCurrentIndex(4)
        self.overlay_combo.currentTextChanged.connect(lambda _: self._on_overlay_channel_changed(self.overlay))
        view_layout.addWidget(self._create_labeled_control("Mask overlay", self.overlay_combo))

        self.roi_combo = QComboBox()
        self.roi_combo.setMinimumHeight(32)
        self.roi_combo.setMinimumContentsLength(10)
        self.roi_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.roi_combo.setStyleSheet("""
            QComboBox {
                padding: 4px 10px;
            }
        """)
        for roi_definition in self.db_manager.get_roi_definitions():
            code = str(roi_definition.code)
            self._roi_definitions_by_code[code] = roi_definition
        self._populate_roi_combo()
        self._persistent_roi_codes = self._resolved_roi_codes(self._persistent_roi_codes)
        self._set_selected_roi_codes(self._persistent_roi_codes)
        self._update_roi_combo_tooltip()
        self.roi_combo.currentIndexChanged.connect(self._on_roi_combo_changed)
        view_layout.addWidget(self._create_labeled_control("Metric ROI", self.roi_combo))

        toggle_title = QLabel("Canvas overlays")
        toggle_title.setObjectName("sidebarField")
        view_layout.addWidget(toggle_title)

        self.chk_roi_outlines = QCheckBox("Show ROI edges")
        self.chk_roi_outlines.setObjectName("sidebarToggle")
        self.chk_roi_outlines.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_roi_outlines.setToolTip("Show or hide ROI edges on canvas")
        self.chk_roi_outlines.toggled.connect(self._on_toggle_roi_outlines)
        view_layout.addWidget(self.chk_roi_outlines)

        self.chk_fovea = QCheckBox("Show fovea")
        self.chk_fovea.setObjectName("sidebarToggle")
        self.chk_fovea.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_fovea.setToolTip("Show or hide fovea marker on canvas")
        self.chk_fovea.toggled.connect(self._on_toggle_fovea)
        view_layout.addWidget(self.chk_fovea)

        panel_layout.addWidget(view_card)
        panel_layout.addStretch(1)
        return panel

    def _create_workflow_sidebar(self) -> QWidget:
        """Create a scrollable right sidebar that switches between review and edit modes."""
        panel, panel_layout = self._create_sidebar_panel("workflowSidebar")

        self.workflow_scroll_area = QScrollArea()
        self.workflow_scroll_area.setWidgetResizable(True)
        self.workflow_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.workflow_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.workflow_scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """)

        self.workflow_mode_stack = QStackedWidget()
        self.review_mode_widget = self._create_review_mode_panel()
        self.edit_mode_page_widget = self._create_edit_mode_panel()
        self.workflow_mode_stack.addWidget(self.review_mode_widget)
        self.workflow_mode_stack.addWidget(self.edit_mode_page_widget)
        self.workflow_scroll_area.setWidget(self.workflow_mode_stack)
        panel_layout.addWidget(self.workflow_scroll_area)

        self._apply_processing_button_defaults()
        self._set_workflow_mode("review")
        return panel

    def _create_review_mode_panel(self) -> QWidget:
        """Create the review-state sidebar page."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)

        review_card, review_layout = self._create_sidebar_card("Review")
        qc_row = QHBoxLayout()
        qc_row.setContentsMargins(0, 0, 0, 0)
        qc_row.setSpacing(8)

        self.btn_pass = QPushButton("Pass")
        self.btn_pass.setMinimumHeight(36)
        self.btn_pass.setStyleSheet(self._qc_button_style("pass"))
        self.btn_pass.clicked.connect(lambda: self._on_qc_decision("pass"))
        qc_row.addWidget(self.btn_pass)

        self.btn_borderline = QPushButton("Borderline")
        self.btn_borderline.setMinimumHeight(36)
        self.btn_borderline.setStyleSheet(self._qc_button_style("borderline"))
        self.btn_borderline.clicked.connect(lambda: self._on_qc_decision("borderline"))
        qc_row.addWidget(self.btn_borderline)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setMinimumHeight(36)
        self.btn_reject.setStyleSheet(self._qc_button_style("reject"))
        self.btn_reject.clicked.connect(lambda: self._on_qc_decision("reject"))
        qc_row.addWidget(self.btn_reject)

        review_layout.addLayout(qc_row)
        page_layout.addWidget(review_card)

        notes_card, notes_layout = self._create_sidebar_card("Notes")
        self.notes_text = QTextEdit()
        self.notes_text.setPlaceholderText("Add a review note if needed...")
        self.notes_text.setFixedHeight(110)
        self.notes_text.setStyleSheet("""
            QTextEdit {
                background: #161616;
                color: #f3f4f6;
                border: 1px solid #3a3a40;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        notes_layout.addWidget(self.notes_text)
        page_layout.addWidget(notes_card)

        self.geometry_status_indicator = QFrame()
        self.geometry_status_indicator.setFixedSize(14, 14)
        self.geometry_status_indicator.setCursor(Qt.CursorShape.PointingHandCursor)
        self.geometry_status_indicator.setStyleSheet(
            "background: #9ca3af; border-radius: 7px; border: 1px solid rgba(255, 255, 255, 0.18);"
        )
        self.geometry_status_indicator.setToolTip("Geometry readiness is being checked.")

        actions_card, actions_layout = self._create_sidebar_card(
            "Current image",
            header_trailing=self.geometry_status_indicator,
        )

        self.btn_segment_current = QPushButton("Segment current image")
        self.btn_segment_current.setMinimumHeight(36)
        self.btn_segment_current.clicked.connect(self._on_segment_current_image)
        actions_layout.addWidget(self.btn_segment_current)

        self.btn_metrics_current = QPushButton("Calculate current metrics")
        self.btn_metrics_current.setMinimumHeight(36)
        self.btn_metrics_current.clicked.connect(self._on_metrics_current_image)
        actions_layout.addWidget(self.btn_metrics_current)

        self.btn_edit = QPushButton("Edit mask")
        self.btn_edit.setMinimumHeight(36)
        self.btn_edit.clicked.connect(self._on_edit_mode_toggle)
        actions_layout.addWidget(self.btn_edit)
        page_layout.addWidget(actions_card)

        page_layout.addStretch(1)
        return page

    def _create_edit_mode_panel(self) -> QWidget:
        """Create the dedicated edit-state sidebar page."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)

        edit_mode_card, edit_mode_layout = self._create_sidebar_card(
            "Mask editing",
            "Use the Edit menu or shortcuts for save, undo, redo and reset.",
        )
        self.btn_exit_edit_mode = QPushButton("Exit edit mode")
        self.btn_exit_edit_mode.setMinimumHeight(36)
        self.btn_exit_edit_mode.clicked.connect(self._on_edit_mode_toggle)
        edit_mode_layout.addWidget(self.btn_exit_edit_mode)
        page_layout.addWidget(edit_mode_card)

        self.edit_toolbar_widget, edit_tools_layout = self._create_sidebar_card("Tools")
        edit_tools_layout.addLayout(self._create_edit_toolbar())
        page_layout.addWidget(self.edit_toolbar_widget)

        page_layout.addStretch(1)
        return page

    def _set_workflow_mode(self, mode: str):
        """Switch the right sidebar between review and edit pages."""
        if self.workflow_mode_stack is None:
            return

        target = self.edit_mode_page_widget if mode == "edit" else self.review_mode_widget
        if target is None:
            return

        self.workflow_mode_stack.setCurrentWidget(target)
        self.workflow_mode_stack.setMinimumHeight(target.sizeHint().height())
        if self.workflow_scroll_area is not None:
            self.workflow_scroll_area.verticalScrollBar().setValue(0)

    ############ WRAPPER METHODS ############

    def _set_channel_by_key(self, channel: str):
        """Wrapper for channel property setter."""
        self.channel = channel

    def _set_overlay_by_key(self, channel: str):
        """Wrapper for overlay property setter."""
        self.overlay = channel

    def display_image(self):
        """Display current image in canvas."""
        if not self.image_paths:
            self.show_empty_state()
            return

        self.progress_bar.setValue(self.get_progress_percentage())
        self._display_new_image()

    def _on_image_channel_changed(self, channel_text: str):
        """Update displayed image channel on the canvas."""
        if self.canvas and self.canvas.image_layer is not None:
            self.canvas.set_image_channel(channel_text)

    def _on_overlay_channel_changed(self, channel_text: str):
        """Update displayed overlay channel on the canvas."""
        if self.canvas and self.canvas.overlay_layer is not None:
            self.canvas.set_overlay_channel(channel_text)
            if channel_text in {"red", "green", "blue"}:
                self.canvas.set_brush_channel(channel_text)

    ############ PRIVATE METHODS ############

    def _display_new_image(self):
        """Display new image with selected channel and segmentation overlay."""
        if not self.image_paths:
            self.show_empty_state()
            self.canvas.clear_roi_outlines()
            return

        name = self.image_path.stem
        image_array = np.array(Image.open(self.image_path).convert("RGB"))
        self._current_image_array = image_array
        self._current_image_shape = image_array.shape[:2]
        image_layer = ImageLayer(image_array)

        overlay_array = np.zeros_like(image_array)
        seg_result = self.db_manager.get_segmentation_by_filename(name)
        if seg_result:
            mask_path = Path(str(seg_result.seg_folder)) / f"{name}{seg_result.extension}"
            if mask_path.is_file():
                overlay_array = np.array(Image.open(mask_path).convert("RGB"))
        self._current_overlay_array = overlay_array

        overlay_layer = OverlayLayer(overlay_array)
        overlay_layer.opacity = self._overlay_opacity
        self._overlay_layer = overlay_layer if seg_result else None

        self._show_canvas()
        self.canvas.reset_layers(image_layer, overlay_layer)
        self.canvas.set_overlay_opacity(self._overlay_opacity)

        self.image_counter_label.setText(f"{self.index + 1} / {len(self.image_paths)}")
        self.image_name_label.setText(name)

        qc_result = self.db_manager.get_qc_result(name)
        if qc_result:
            self._load_state(qc_result)
            self.notes_text.setPlainText(str(qc_result.notes or ""))
            qc_decision = qc_result.decision.value
            self._highlight_decision(qc_decision)
            self.canvas_color = qc_decision
        else:
            self.notes_text.clear()
            self._clear_decision_highlight()
            self.state["filename"] = name
            self.state["decision"] = None
            self.state["notes"] = ""
            self.canvas_color = "default"

        self.canvas.set_fovea_visibility(False)
        landmark_metrics = self.db_manager.get_landmark_metrics_by_filename(name)
        if (
            landmark_metrics
            and landmark_metrics.fovea_center_x is not None
            and landmark_metrics.fovea_center_y is not None
        ):
            self.canvas.update_fovea(
                float(str(landmark_metrics.fovea_center_x)),
                float(str(landmark_metrics.fovea_center_y)),
            )
            self.canvas.set_fovea_visibility(self.chk_fovea.isChecked())

        self.canvas.fit_to_view()
        self.canvas.set_image_channel(self.channel)
        self.canvas.set_overlay_channel(self.overlay)
        self._update_image_status(name)

        if self.edit_mode and self.active_tool:
            self.canvas.set_tool(self.active_tool)
            self._update_brush_cursor()

        self._load_roi_selection()

    def _update_image_status(self, name: str):
        """Update compact per-image workflow status."""
        qc_result = self.db_manager.get_qc_result(name)
        qc_status = qc_result.decision.value.upper() if qc_result else "UNREVIEWED"
        seg_status = "present" if self.db_manager.get_segmentation_by_filename(name) else "missing"
        metrics_status = "ready" if self.db_manager.get_metrics_by_filename(name) else "pending"
        self.image_status_label.setText(
            f"QC: {qc_status} | Segmentation: {seg_status} | Metrics: {metrics_status}"
        )

    def _highlight_decision(self, decision: str):
        """Highlight decision button"""
        self._clear_decision_highlight()

        if decision == "pass":
            self.btn_pass.setStyleSheet(self._qc_button_style("pass", active=True))
        elif decision == "borderline":
            self.btn_borderline.setStyleSheet(self._qc_button_style("borderline", active=True))
        elif decision == "reject":
            self.btn_reject.setStyleSheet(self._qc_button_style("reject", active=True))
    
    def _clear_decision_highlight(self):
        """Clear all decision highlights"""
        self.btn_pass.setStyleSheet(self._qc_button_style("pass"))
        self.btn_borderline.setStyleSheet(self._qc_button_style("borderline"))
        self.btn_reject.setStyleSheet(self._qc_button_style("reject"))

    def _qc_button_style(self, decision: str, active: bool = False) -> str:
        """Return calmer default styles for QC buttons, stronger only when selected."""
        if active:
            return self.button_styles[decision]["highlighted"]

        neutral_styles = {
            "pass": self._create_button_stylesheet(
                "#173126", "#86efac", "#1d3b2f", "#14271f", border_color="#28563f"
            ),
            "borderline": self._create_button_stylesheet(
                "#342812", "#fbbf24", "#433318", "#2a1f0f", border_color="#7c5a16"
            ),
            "reject": self._create_button_stylesheet(
                "#351724", "#fda4af", "#431d2d", "#2a121d", border_color="#7f1d35"
            ),
        }
        return neutral_styles[decision]

    def _set_batch_actions_enabled(self, enabled: bool):
        """Enable or disable any bound batch menu actions."""
        for action in (self.batch_segment_action, self.batch_metrics_action):
            if action is not None:
                action.setEnabled(enabled)

    def _set_active_worker(self, worker: QThread | None):
        """Keep a stable reference to the currently running worker."""
        self._worker_thread = worker
        self.worker_thread = worker

    def _has_running_worker(self) -> bool:
        """Return whether the widget still owns a running worker thread."""
        worker = self._worker_thread
        return bool(worker and worker.isRunning())

    def _default_processing_style(self, button: QPushButton | None) -> str:
        """Return default button style based on action priority and scope."""
        return self.button_styles["segment"]["normal"]

    def _apply_processing_button_defaults(self):
        """Reset the visible current-image process buttons."""
        for button_name in (
            "btn_segment_current",
            "btn_metrics_current",
            "btn_edit",
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setStyleSheet(self._default_processing_style(button))
    
    def _prompt_save_changes(self):
        """Prompt user to save unsaved edits"""
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved edits. Do you want to save them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._save_state()
            self._save_edits()
        elif reply == QMessageBox.StandardButton.No:
            return False
        
        return True
    
    def _create_brush_cursor(self, radius: float) -> QCursor:
        """Create a circular brush cursor with transparent fill"""
        # Create pixmap for cursor
        size = int(2 * radius)
        cursor_pixmap = QPixmap(size, size)
        cursor_pixmap.fill(Qt.GlobalColor.transparent)
        
        # Draw white circle outline
        painter = QPainter(cursor_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawEllipse(0, 0, size - 1, size - 1)
        painter.end()
        
        # Create cursor from pixmap
        cursor = QCursor(cursor_pixmap)
        return cursor
    
    def _reset_buttons_state(self):
        """Reset button styles to default"""
        # QC decision buttons
        self._clear_decision_highlight()
        self.notes_text.clear()
        self._apply_processing_button_defaults()

    def _available_roi_codes(self) -> tuple[str, ...]:
        """Return ROI codes in a stable display order."""
        ordered_codes = [code for code in self._roi_code_order if code in self._roi_definitions_by_code]
        for code in self._roi_definitions_by_code:
            if code not in ordered_codes:
                ordered_codes.append(code)
        return tuple(ordered_codes)

    def _roi_display_name(self, roi_code: str) -> str:
        """Return compact ROI labels for the combo box."""
        short_names = {
            "full": "Full",
            "mid_periphery": "Mid",
            "central": "Central",
        }
        if roi_code in short_names:
            return short_names[roi_code]
        roi_definition = self._roi_definitions_by_code.get(roi_code)
        if roi_definition:
            return str(roi_definition.name)
        return roi_code.replace("_", " ").title()

    def _roi_tooltip_name(self, roi_code: str) -> str:
        """Return full ROI name for tooltips."""
        roi_definition = self._roi_definitions_by_code.get(roi_code)
        if roi_definition:
            return str(roi_definition.name)
        return self._roi_display_name(roi_code)

    def _roi_combo_label(self, selected_codes: tuple[str, ...], available_codes: tuple[str, ...]) -> str:
        """Return short combo label for one ROI combination."""
        if len(selected_codes) == len(available_codes):
            return "All ROIs"
        return " + ".join(self._roi_display_name(code) for code in selected_codes)

    def _populate_roi_combo(self):
        """Populate the ROI combo with all non-empty ROI combinations."""
        if not hasattr(self, "roi_combo"):
            return

        self._roi_combo_index_by_codes.clear()
        available_codes = self._available_roi_codes()

        self.roi_combo.blockSignals(True)
        try:
            self.roi_combo.clear()
            for size in range(1, len(available_codes) + 1):
                for selected_codes in combinations(available_codes, size):
                    label = self._roi_combo_label(selected_codes, available_codes)
                    self.roi_combo.addItem(label, selected_codes)
                    self._roi_combo_index_by_codes[selected_codes] = self.roi_combo.count() - 1
        finally:
            self.roi_combo.blockSignals(False)

    def _resolved_roi_codes(self, selected_codes: tuple[str, ...] | None = None) -> tuple[str, ...]:
        """Normalize ROI codes against the available definitions, defaulting to all."""
        available_codes = self._available_roi_codes()
        selected_lookup = set(selected_codes or ())
        normalized_codes = tuple(code for code in available_codes if code in selected_lookup)
        return normalized_codes or available_codes

    def _persist_roi_selection(self, image_id: int, selected_codes: tuple[str, ...]) -> bool:
        """Persist one ROI combination for the given image."""
        normalized_codes = self._resolved_roi_codes(selected_codes)
        selected_lookup = set(normalized_codes)
        selected_by_code = {
            str(image_roi.roi_definition.code): bool(image_roi.selected_for_metrics)
            for image_roi in self.db_manager.get_image_rois_by_image_id(image_id)
        }

        for roi_code in self._available_roi_codes():
            should_select = roi_code in selected_lookup
            if roi_code not in selected_by_code:
                self.db_manager.ensure_image_roi(
                    image_id=image_id,
                    roi_code=roi_code,
                    selected_for_metrics=should_select,
                )
                continue

            if selected_by_code[roi_code] != should_select:
                if not self.db_manager.select_image_roi(image_id, roi_code, should_select):
                    return False

        return True

    def _persist_roi_selection_for_metadata(self, metadata: list, selected_codes: tuple[str, ...]) -> tuple[str, ...]:
        """Persist the selected ROI combination for each metadata row."""
        failed_names = []
        normalized_codes = self._resolved_roi_codes(selected_codes)

        for meta in metadata:
            if not self._persist_roi_selection(int(str(meta.id)), normalized_codes):
                failed_names.append(str(meta.name))

        return tuple(failed_names)

    def _get_selected_roi_codes(self) -> tuple[str, ...]:
        """Return the currently selected ROI combination."""
        if not hasattr(self, "roi_combo") or self.roi_combo.currentIndex() < 0:
            return ()

        selected_codes = self.roi_combo.currentData()
        if not selected_codes:
            return ()
        return tuple(str(code) for code in selected_codes)

    def _set_selected_roi_codes(self, selected_codes: tuple[str, ...]):
        """Update the combo box without emitting persistence signals."""
        normalized_codes = self._resolved_roi_codes(selected_codes)
        index = self._roi_combo_index_by_codes.get(normalized_codes)
        if index is None:
            index = 0 if self.roi_combo.count() else -1

        self._loading_roi_selection = True
        try:
            self.roi_combo.setCurrentIndex(index)
        finally:
            self._loading_roi_selection = False

    def _update_roi_combo_tooltip(self):
        """Keep the ROI combo tooltip aligned with the selected combination."""
        if not hasattr(self, "roi_combo"):
            return

        selected_codes = self._get_selected_roi_codes()
        if not selected_codes:
            self.roi_combo.setToolTip("Choose metric ROI combination")
            return

        tooltip = "\n".join(self._roi_tooltip_name(code) for code in selected_codes)
        self.roi_combo.setToolTip(tooltip)

    def _build_display_roi_bundle(
        self,
        image_id: int,
        name: str,
        shape: tuple[int, int],
    ) -> SegmentationBundle:
        """Build a segmentation bundle for ROI display, even without saved masks."""
        if (
            self._current_image_array is not None
            and self._current_overlay_array is not None
            and self._current_image_array.shape[:2] == shape
            and self._current_overlay_array.shape[:2] == shape
        ):
            overlay_array = self._current_overlay_array
            a_mask = overlay_array[:, :, 0] > 0
            od_mask = overlay_array[:, :, 1] > 0
            v_mask = overlay_array[:, :, 2] > 0
            return SegmentationBundle(
                image_id=int(image_id),
                name=name,
                extension=str(self.image_path.suffix) if self.image_path else ".png",
                seg_mask_path=self.image_path or Path(name),
                shape=shape,
                a_mask=a_mask,
                od_mask=od_mask,
                v_mask=v_mask,
                vessel_mask=a_mask | v_mask,
                source_image=self._current_image_array,
            )

        try:
            return self.roi_mask_service.load_segmentation_bundle(name)
        except Exception:
            empty_mask = np.zeros(shape, dtype=bool)
            return SegmentationBundle(
                image_id=int(image_id),
                name=name,
                extension=str(self.image_path.suffix) if self.image_path else ".png",
                seg_mask_path=self.image_path or Path(name),
                shape=shape,
                a_mask=empty_mask,
                od_mask=empty_mask,
                v_mask=empty_mask,
                vessel_mask=empty_mask,
                source_image=self._current_image_array,
            )

    def _update_roi_outlines(self):
        """Render selected RoI outlines on the canvas."""
        if not self.canvas:
            return

        if not self._roi_outlines_visible:
            self.canvas.clear_roi_outlines()
            return

        if not self.image_paths or not self._current_image_shape:
            self.canvas.clear_roi_outlines()
            return

        selected_codes = list(self._get_selected_roi_codes())
        if not selected_codes:
            self.canvas.clear_roi_outlines()
            return

        metadata = self.db_manager.get_metadata_by_filename(self.image_path.stem)
        if not metadata:
            self.canvas.clear_roi_outlines()
            return

        bundle = self._build_display_roi_bundle(
            int(str(metadata.id)),
            self.image_path.stem,
            self._current_image_shape,
        )
        landmarks = self.roi_mask_service.load_landmark_context(self.image_path.stem)

        roi_masks = {}
        for code in selected_codes:
            roi_definition = self._roi_definitions_by_code.get(code)
            if not roi_definition:
                continue
            try:
                roi_context = self.roi_mask_service.build_roi_context(bundle, roi_definition, landmarks)
            except Exception:
                continue
            roi_masks[code] = roi_context.mask

        self.canvas.set_roi_outlines(roi_masks, self.roi_outline_colors)

    def _load_roi_selection(self):
        """Load selected metric RoIs for the current image."""
        if not self.image_paths or not self._roi_combo_index_by_codes:
            return

        metadata = self.db_manager.get_metadata_by_filename(self.image_path.stem)
        if not metadata:
            self.canvas.clear_roi_outlines()
            return

        image_id = int(str(metadata.id))
        selected_codes = self._resolved_roi_codes(self._persistent_roi_codes)
        self._persistent_roi_codes = selected_codes
        self._persist_roi_selection(image_id, selected_codes)

        self._set_selected_roi_codes(selected_codes)
        self._update_roi_combo_tooltip()
        self._update_roi_outlines()

    def _on_roi_combo_changed(self, _: int):
        """Persist the selected ROI combination for the current image."""
        if self._loading_roi_selection or not self.image_paths:
            return

        metadata = self.db_manager.get_metadata_by_filename(self.image_path.stem)
        if not metadata:
            return

        image_id = int(str(metadata.id))
        selected_codes = self._resolved_roi_codes(self._get_selected_roi_codes())
        all_saved = self._persist_roi_selection(image_id, selected_codes)

        if all_saved:
            self._persistent_roi_codes = selected_codes
            self.status_text.emit(f"Metric ROI set to {self.roi_combo.currentText()}")
            self._update_roi_combo_tooltip()
            self._update_roi_outlines()
        else:
            self.status_text.emit("Could not save metric ROI selection")
            self._load_roi_selection()

    def _on_toggle_roi_outlines(self, checked: bool):
        """Toggle ROI outline visibility on canvas."""
        self._roi_outlines_visible = checked
        self._update_roi_outlines()
    
    def _update_brush_cursor(self):
        """Update brush cursor based on current zoom and brush size"""
        if self._active_tool in ["brush", "smart_paint", "eraser"] and self.canvas:
            # Scale brush radius by canvas zoom level
            scaled_radius = (self.brush_size / 2) * self.canvas._zoom_level
            cursor = self._create_brush_cursor(scaled_radius)
            self.canvas.setCursor(cursor)
    
    def _load_state(self, qc_result):
        """Load QC data into state"""
        if not self.image_paths:
            return
        
        self.state['filename'] = qc_result.name
        self.state['decision'] = qc_result.decision.value
        self.state['notes'] = qc_result.notes
    
    def _save_state(self):
        """Save current image's QC data"""
        if not self.image_paths:
            return

        self.state['notes'] = self.notes_text.toPlainText()
        if not self.state.get('decision'):
            return True

        try:
            state = self.state
            self.db_manager.save_qc_result(
                state['filename'],
                state['decision'],
                state['notes']
                )
        except Exception as e:
            print(f"Error saving QC result: {e}")
            return False
        return True
        
    def _save_edits(self):
        """Save current overlay edits"""
        if not self.canvas.overlay_layer:
            self.status_text.emit("No segmentation overlay loaded")
            return
        
        # Get the RGB array from overlay
        overlay_array = self.canvas.overlay_layer.get_array()
        
        # Get id of current image
        name = str(self.image_path.stem)
        qc_result = self.db_manager.get_qc_result(name)
        id = int(str(qc_result.id))
        
        # Get segmentation results
        seg_result = self.db_manager.get_segmentation_result_by_id(id)
        if seg_result is None:
            self.status_text.emit("No segmentation result found in database")
            return
        
        extension = str(seg_result.extension)
        seg_path = Path(str(seg_result.seg_folder))
        
        # Save to file
        seg_path.mkdir(parents=True, exist_ok=True)
        
        mask_path = seg_path / Path(name + extension)
        Image.fromarray(overlay_array).save(mask_path)
        
        # Mark overlay as saved
        self.canvas.overlay_layer.mark_saved()
        
        self.status_text.emit(f"Edits saved to {mask_path}")
        
    ############ ACTIONS ############
    
    def _on_qc_decision(self, decision: str):
        """Handle decision"""
        if not self.image_paths:
            return
        
        notes = self.notes_text.toPlainText()
        
        self._highlight_decision(decision)
        self.canvas_color = decision

        # Update state
        self.state['filename'] = self.image_path.stem
        self.state['decision'] = decision
        self.state['notes'] = notes

        # Save to database & emit signal
        if self._save_state():
            self.decision_made.emit(self.state['filename'], decision)
    
    def _on_next(self):
        """Move to next image"""
        if self.has_unsaved_changes:
            self._prompt_save_changes()
        
        if self.next_image():
            self._save_state()
            self._reset_buttons_state()
            self.display_image()
    
    def _on_prev(self):
        """Move to previous image"""
        if self.has_unsaved_changes:
            self._prompt_save_changes()
        
        if self.previous_image():
            self._save_state()
            self._reset_buttons_state()
            self.display_image()
    
    def _on_refresh_display(self):
        """Refresh image display"""
        self._display_new_image()

    def _on_fit_view(self):
        """Fit the current image to the canvas."""
        if self.canvas:
            self.canvas.fit_to_view()

    def _on_actual_size(self):
        """Show the current image at actual size."""
        if self.canvas:
            self.canvas.show_actual_size()

    def _on_center_view(self):
        """Center the current image in the canvas."""
        if self.canvas:
            self.canvas.center_image()
    
    def _on_segment_all(self):
        """Start segmentation"""
        pending = self.step_seg.get_pending_images()
        
        if not pending:
            self.status_text.emit(("No images to segment"))
            return
        
        self.status_text.emit(f"Segmenting {len(pending)} images...")
        self.progress_status_label.setText(f"Segmenting 0 / {len(pending)}")
        self.progress_bar.setValue(0)
        self._set_batch_actions_enabled(False)
        
        try:
            worker = BatchSegmentationWorker(self.step_seg, pending)
            self._set_active_worker(worker)
            worker.progress.connect(self._on_progress)
            worker.finished.connect(self._on_batch_segment_finished)
            worker.finished.connect(lambda _success, worker=worker: self._on_worker_finished(worker))
            worker.start()
        except Exception as e:
            self.status_text.emit(f"Error: {str(e)}")
            self._set_active_worker(None)
            self._set_batch_actions_enabled(True)
    
    def _on_segment_current_image(self):
        """Segment only the currently displayed image"""
        if not self.image_path:
            self.status_text.emit("No image loaded")
            return
        
        self.status_text.emit(f"Segmenting {self.image_path.name}...")
        self.progress_status_label.setText("Segmenting current image")
        self.progress_bar.setValue(0)
        self.btn_segment_current.setEnabled(False)
        self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["highlighted"])
        
        try:
            metadata = self.db_manager.get_metadata_by_filename(self.image_path.stem)
            worker = SingleSegmentationWorker(self.step_seg, self.image_path, int(str(metadata.id)))
            self._set_active_worker(worker)
            worker.finished.connect(self._on_single_segment_finished)
            worker.finished.connect(lambda _success, worker=worker: self._on_worker_finished(worker))
            worker.start()
            
            self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["highlighted"])
        
        except Exception as e:
            self.status_text.emit(f"Error: {str(e)}")
            self._set_active_worker(None)
            self.btn_segment_current.setStyleSheet(self._default_processing_style(self.btn_segment_current))
            self.btn_segment_current.setEnabled(True)
    
    def _on_batch_segment_finished(self, success: bool):
        """Handle completion"""
        self._set_batch_actions_enabled(True)
        self.status_text.emit("Complete!" if success else "Failed!")
        self.progress_status_label.setText("Segmentation complete" if success else "Segmentation failed")
        self.progress_bar.setValue(100 if success else 0)
        
    def _on_single_segment_finished(self, success: bool):
        """Handle single image segmentation completion"""
        try:
            if success:
                self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["finished"])
                self.status_text.emit(f"{self.image_path.name}: ✓")
                self.progress_status_label.setText("Segmentation complete")
                self.progress_bar.setValue(100)
                self._display_new_image()
            else:
                self.btn_segment_current.setStyleSheet(self._default_processing_style(self.btn_segment_current))
                self.status_text.emit(f"{self.image_path.name}: ✗")
                self.progress_status_label.setText("Segmentation failed")
                self.progress_bar.setValue(0)
        finally:
            self.btn_segment_current.setEnabled(True)
    
    def _on_edit_mode_toggle(self):
        """Toggle edit mode on/off"""
        if self.edit_mode:
            # Exit edit mode
            self.edit_mode = False
            self.canvas.set_edit_mode(False)
            self._set_workflow_mode("review")
            self.btn_edit.setStyleSheet(self._default_processing_style(self.btn_edit))
            self.btn_next.setEnabled(True)
            self.btn_prev.setEnabled(True)
            self.canvas.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.canvas.set_tool(None)
            self._active_tool = None
        else:
            # Enter edit mode
            if not self._overlay_layer:
                self.status_text.emit("No overlay to edit")
                return
            
            self.edit_mode = True
            self.canvas.set_edit_mode(True)
            self._set_workflow_mode("edit")
            self.btn_edit.setStyleSheet(self.button_styles["segment"]["highlighted"])
            self.btn_next.setEnabled(False)
            self.btn_prev.setEnabled(False)
            self.canvas.setDragMode(QGraphicsView.DragMode.NoDrag)
    
    def _on_brush_size_changed(self, size: int):
        """Handle brush size slider change"""
        self._brush_size = size
        self.brush_size_label.setText(f"{size}px")
        self.canvas.set_brush_radius(size / 2)
        
        # Update cursor if a tool is active
        if self._active_tool:
            self._update_brush_cursor()
    
    def _on_increase_brush_size(self):
        """Increase brush size"""
        new_size = min(self.MAX_BRUSH_SIZE, self._brush_size + 1)
        self.brush_size_slider.setValue(new_size)
        self._on_brush_size_changed(new_size)
    
    def _on_decrease_brush_size(self):
        """Decrease brush size"""
        new_size = max(self.MIN_BRUSH_SIZE, self._brush_size - 1)
        self.brush_size_slider.setValue(new_size)
        self._on_brush_size_changed(new_size)
    
    def _on_undo(self):
        """Undo last operation"""
        if self._overlay_layer:
            self.canvas.undo()
    
    def _on_redo(self):
        """Redo last undone operation"""
        if self._overlay_layer:
            self.canvas.redo()

    def _on_reset_edits(self):
        """Reset all edits"""
        reply = QMessageBox.question(
            self,
            "Reset Edits",
            "Are you sure you want to discard all edits?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes and self._overlay_layer:
            self._overlay_layer.reset()
            self.canvas.update_overlay_display()

    def _on_save(self):
        """Handle save action"""
        self._save_edits()
        self._save_state()

    def _on_opacity_changed(self, value: float):
        """Handle opacity slider changes"""
        # Prepare value
        value = min(max(value, 0), 1.0)
        
        # Store opacity for future images
        self._overlay_opacity = value
        
        # Update label with percentage
        if self.opacity_label is not None:
            self.opacity_label.setText(f"{int(value * 100)}%")
        
        # Convert to 0.0-1.0 range and update canvas
        opacity_normalized = value
        if self.canvas is not None and self.canvas.overlay_layer is not None:
            self.canvas.set_overlay_opacity(opacity_normalized)

    def _on_canvas_opacity_delta(self, value: float):
        """Adjust opacity from canvas shortcuts and sync any bound controls."""
        opacity = min(max(self._overlay_opacity + value, 0.0), 1.0)
        if self.opacity_slider is not None:
            self.opacity_slider.setValue(int(opacity * 100))
        else:
            self._on_opacity_changed(opacity)

    def _on_worker_finished(self, worker: QThread | None):
        """Clean up worker thread"""
        if worker is None:
            return

        try:
            worker.quit()
            worker.wait()
        finally:
            if self._worker_thread is worker:
                self._worker_thread = None
            if getattr(self, "worker_thread", None) is worker:
                self.worker_thread = None
            worker.deleteLater()
    
    def _on_metrics_current_image(self):
        """Calculate metrics for currently displayed image"""
        if not self.image_path:
            self.status_text.emit("No image loaded")
            return
        
        if self.canvas.overlay_layer and self.canvas.overlay_layer.has_changes():
            QMessageBox.warning(
                self,
                "Unsaved Edits",
                "Please save your edits (Ctrl/Command + S) before calculating metrics."
            )
            return
        
        self.status_text.emit(f"Calculating metrics for {self.image_path.name}...")
        self._last_progress_message = None
        self.progress_status_label.setText("Calculating current metrics")
        self.progress_bar.setValue(0)
        self.btn_metrics_current.setEnabled(False)
        self.btn_metrics_current.setStyleSheet(self.button_styles["segment"]["highlighted"])
        
        try:
            step_metrics = self._get_step_metrics()
            metadata = self.db_manager.get_metadata_by_filename(self.image_path.stem)
            worker = SingleMetricsWorker(step_metrics, self.image_path, int(str(metadata.id)))
            self._set_active_worker(worker)
            worker.progress.connect(self._on_progress)
            worker.finished.connect(self._on_single_metrics_finished)
            worker.finished.connect(lambda _success, worker=worker: self._on_worker_finished(worker))
            worker.start()
        except Exception as e:
            self.status_text.emit(f"Error: {str(e)}")
            self._set_active_worker(None)
            self.btn_metrics_current.setStyleSheet(self._default_processing_style(self.btn_metrics_current))
            self.btn_metrics_current.setEnabled(True)
    
    def _on_single_metrics_finished(self, success: bool):
        """Handle single image metrics calculation completion"""
        try:
            if success:
                self.btn_metrics_current.setStyleSheet(self.button_styles["segment"]["finished"])
                partial = self._last_geometry_traffic_light in {
                    GeometryTrafficLight.RED.value,
                    GeometryTrafficLight.YELLOW.value,
                }
                if partial:
                    self.status_text.emit(
                        f"{self.image_path.name}: Metrics calculated; some geometry-dependent metrics may be skipped"
                    )
                    self.progress_status_label.setText("Metrics complete (partial)")
                else:
                    self.status_text.emit(f"{self.image_path.name}: Metrics calculated ✓")
                    self.progress_status_label.setText("Metrics complete")
                self.progress_bar.setValue(100)
                self._update_image_status(self.image_path.stem)
            else:
                self.btn_metrics_current.setStyleSheet(self._default_processing_style(self.btn_metrics_current))
                self.status_text.emit(self._last_progress_message or f"{self.image_path.name}: Metrics calculation failed ✗")
                self.progress_status_label.setText("Metrics failed")
                self.progress_bar.setValue(0)
        finally:
            self.btn_metrics_current.setEnabled(True)
    
    def _on_calculate_metrics_all(self):
        """Calculate metrics for pending images"""
        step_metrics = self._get_step_metrics()
        pending = step_metrics.get_pending_images()
        
        if not pending:
            self.status_text.emit(("No images to calculate metrics for"))
            return
        
        if self.canvas.overlay_layer and self.canvas.overlay_layer.has_changes():
            QMessageBox.warning(
                self,
                "Unsaved Edits",
                "Please save your edits (Ctrl/Command + S) before calculating metrics."
            )
            return

        selected_codes = self._resolved_roi_codes(self._get_selected_roi_codes() or self._persistent_roi_codes)
        failed_roi_names = self._persist_roi_selection_for_metadata(pending, selected_codes)
        if failed_roi_names:
            preview = ", ".join(failed_roi_names[:3])
            suffix = "" if len(failed_roi_names) <= 3 else f" and {len(failed_roi_names) - 3} more"
            self.status_text.emit(f"Could not save metric ROI selection for {preview}{suffix}")
            return

        self._persistent_roi_codes = selected_codes
        
        self.status_text.emit(f"Calculating metrics for {len(pending)} images...")
        self._last_progress_message = None
        self.progress_status_label.setText(f"Calculating 0 / {len(pending)}")
        self.progress_bar.setValue(0)
        self._set_batch_actions_enabled(False)
        
        try:
            worker = BatchMetricsWorker(step_metrics, pending)
            self._set_active_worker(worker)
            worker.progress.connect(self._on_progress)
            worker.finished.connect(self._on_batch_metrics_finished)
            worker.finished.connect(lambda _success, worker=worker: self._on_worker_finished(worker))
            worker.start()
        except Exception as e:
            self.status_text.emit(f"Error: {str(e)}")
            self._set_active_worker(None)
            self._set_batch_actions_enabled(True)
    
    def _on_progress(self, progress: int, message: str):
        """Handle metrics calculation progress"""
        self._last_progress_message = message
        self.progress_bar.setValue(progress)
        self.progress_status_label.setText(message)
        self.status_text.emit(message)
    
    def _on_batch_metrics_finished(self, success: bool):
        """Handle completion of batch metrics calculation"""
        self._set_batch_actions_enabled(True)
        if success:
            if self._last_geometry_traffic_light in {
                GeometryTrafficLight.RED.value,
                GeometryTrafficLight.YELLOW.value,
            }:
                self.status_text.emit("Metrics calculation complete; some geometry-dependent metrics may be skipped")
            else:
                self.status_text.emit("Metrics calculation complete!")
        else:
            self.status_text.emit(self._last_progress_message or "Metrics calculation failed!")
        self.progress_status_label.setText("Metrics complete" if success else "Metrics failed")
        self.progress_bar.setValue(100 if success else 0)

    def _apply_geometry_readiness(self, readiness) -> None:
        self._last_geometry_summary = readiness.summary()
        self._last_geometry_traffic_light = readiness.traffic_light.value
        if self.geometry_status_indicator is None:
            return

        color_map = {
            GeometryTrafficLight.RED.value: "#e11d48",
            GeometryTrafficLight.YELLOW.value: "#d97706",
            GeometryTrafficLight.GREEN.value: "#059669",
        }
        self.geometry_status_indicator.setStyleSheet(
            "border-radius: 7px; border: 1px solid rgba(255, 255, 255, 0.18);"
            f"background: {color_map.get(readiness.traffic_light.value, '#9ca3af')};"
        )
        self.geometry_status_indicator.setToolTip(
            f"Geometry readiness: {readiness.traffic_light.value.upper()}\n{readiness.summary()}"
        )

    def _refresh_geometry_readiness(self) -> None:
        self._apply_geometry_readiness(self.geometry_adapter.readiness())
    
    def _on_toggle_fovea(self, checked: bool):
        """Toggle fovea visibility on canvas"""
        if self.canvas:
            self.canvas.set_fovea_visibility(checked)
    
    def _on_fovea_location_selected(self, x: int, y: int):
        """Handle fovea location selection"""
        if not self.image_paths:
            return
        
        name = self.image_path.stem
        
        try:
            self.db_manager.save_metrics_fovea_by_name(name, x, y)
            self.canvas.update_fovea(float(x), float(y))
            self.chk_fovea.setChecked(True)
            if not self._has_running_worker() and self.btn_metrics_current is not None:
                self.btn_metrics_current.setEnabled(True)
                self.btn_metrics_current.setStyleSheet(self._default_processing_style(self.btn_metrics_current))
            self.status_text.emit(
                f"Fovea location saved at ({x}, {y}) for {name}. Re-run metric calculation."
            )
        except Exception as e:
            self.status_text.emit(f"Error saving fovea location: {e}")
    