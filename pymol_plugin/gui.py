"""FlexAID∆S PyMOL GUI Panel.

Qt-based interface for docking result visualization and analysis.
"""

import os
from pathlib import Path

try:
    from pymol.Qt import QtWidgets, QtCore
    from pymol import cmd
except ImportError:
    raise ImportError("PyMOL Qt bindings not available")


class FlexAIDSPanel(QtWidgets.QDialog):
    """Main GUI panel for FlexAID∆S plugin."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FlexAID∆S: Entropy-Driven Docking")
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)
        
        self._setup_ui()
        self._connect_signals()
        
        # Data state
        self.binding_modes = []
        self.current_mode_index = 0
    
    def _setup_ui(self):
        """Construct GUI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # ─── File loading section ───
        file_group = QtWidgets.QGroupBox("Load Docking Results")
        file_layout = QtWidgets.QHBoxLayout()
        
        self.file_path_edit = QtWidgets.QLineEdit()
        self.file_path_edit.setPlaceholderText("Select FlexAID output directory...")
        
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setEnabled(False)
        
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)
        file_layout.addWidget(self.load_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # ─── Binding mode list ───
        mode_group = QtWidgets.QGroupBox("Binding Modes")
        mode_layout = QtWidgets.QVBoxLayout()
        
        self.mode_list = QtWidgets.QListWidget()
        self.mode_list.setSelectionMode(QtWidgets.QListWidget.SingleSelection)
        mode_layout.addWidget(self.mode_list)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # ─── Thermodynamic properties display ───
        thermo_group = QtWidgets.QGroupBox("Thermodynamics")
        thermo_layout = QtWidgets.QFormLayout()
        
        self.free_energy_label = QtWidgets.QLabel("-")
        self.enthalpy_label = QtWidgets.QLabel("-")
        self.entropy_label = QtWidgets.QLabel("-")
        self.entropy_term_label = QtWidgets.QLabel("-")
        self.n_poses_label = QtWidgets.QLabel("-")
        
        thermo_layout.addRow("ΔG (kcal/mol):", self.free_energy_label)
        thermo_layout.addRow("ΔH (kcal/mol):", self.enthalpy_label)
        thermo_layout.addRow("S (kcal/mol·K):", self.entropy_label)
        thermo_layout.addRow("TΔS (kcal/mol):", self.entropy_term_label)
        thermo_layout.addRow("# Poses:", self.n_poses_label)
        
        thermo_group.setLayout(thermo_layout)
        layout.addWidget(thermo_group)
        
        # ─── Visualization controls ───
        viz_group = QtWidgets.QGroupBox("Visualization")
        viz_layout = QtWidgets.QVBoxLayout()
        
        self.show_ensemble_btn = QtWidgets.QPushButton("Show Pose Ensemble")
        self.show_ensemble_btn.setEnabled(False)
        
        self.color_boltzmann_btn = QtWidgets.QPushButton("Color by Boltzmann Weight")
        self.color_boltzmann_btn.setEnabled(False)
        
        self.show_representative_btn = QtWidgets.QPushButton("Show Representative Only")
        self.show_representative_btn.setEnabled(False)
        
        viz_layout.addWidget(self.show_ensemble_btn)
        viz_layout.addWidget(self.color_boltzmann_btn)
        viz_layout.addWidget(self.show_representative_btn)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # ─── NRGSuite integration ───
        nrg_group = QtWidgets.QGroupBox("NRGSuite Integration")
        nrg_layout = QtWidgets.QVBoxLayout()
        
        self.launch_nrgsuite_btn = QtWidgets.QPushButton("Launch NRGSuite")
        self.export_to_nrg_btn = QtWidgets.QPushButton("Export to NRGSuite Format")
        self.export_to_nrg_btn.setEnabled(False)
        
        nrg_layout.addWidget(self.launch_nrgsuite_btn)
        nrg_layout.addWidget(self.export_to_nrg_btn)
        
        nrg_group.setLayout(nrg_layout)
        layout.addWidget(nrg_group)
        
        # ─── Close button ───
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def _connect_signals(self):
        """Wire up button click handlers."""
        self.browse_btn.clicked.connect(self._browse_directory)
        self.load_btn.clicked.connect(self._load_results)
        self.mode_list.itemSelectionChanged.connect(self._on_mode_selected)
        self.show_ensemble_btn.clicked.connect(self._show_pose_ensemble)
        self.color_boltzmann_btn.clicked.connect(self._color_by_boltzmann)
        self.show_representative_btn.clicked.connect(self._show_representative)
        self.launch_nrgsuite_btn.clicked.connect(self._launch_nrgsuite)
        self.export_to_nrg_btn.clicked.connect(self._export_to_nrgsuite)
    
    def _browse_directory(self):
        """Open file dialog to select FlexAID output directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select FlexAID Output Directory"
        )
        if directory:
            self.file_path_edit.setText(directory)
            self.load_btn.setEnabled(True)
    
    def _load_results(self):
        """Load docking results from selected directory."""
        output_dir = Path(self.file_path_edit.text())
        if not output_dir.exists():
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Directory not found: {output_dir}"
            )
            return
        
        # TODO: Implement FlexAID output parser
        # For now, show placeholder
        self.mode_list.clear()
        
        # Placeholder: fake binding modes
        for i in range(5):
            item_text = f"Mode {i+1}: ΔG = {-10.5 + i*0.5:.2f} kcal/mol"
            self.mode_list.addItem(item_text)
        
        self.show_ensemble_btn.setEnabled(True)
        self.color_boltzmann_btn.setEnabled(True)
        self.show_representative_btn.setEnabled(True)
        self.export_to_nrg_btn.setEnabled(True)
        
        QtWidgets.QMessageBox.information(
            self, "Success", 
            f"Loaded {self.mode_list.count()} binding modes from {output_dir.name}"
        )
    
    def _on_mode_selected(self):
        """Update thermodynamics display when binding mode is selected."""
        selected_items = self.mode_list.selectedItems()
        if not selected_items:
            return
        
        # TODO: Load actual thermodynamics from BindingMode object
        # Placeholder values
        self.free_energy_label.setText("-10.5")
        self.enthalpy_label.setText("-12.3")
        self.entropy_label.setText("0.006")
        self.entropy_term_label.setText("1.8")
        self.n_poses_label.setText("45")
    
    def _show_pose_ensemble(self):
        """Render all poses in selected binding mode."""
        # TODO: Implement PyMOL rendering
        cmd.do("print 'Show pose ensemble (not yet implemented)'")
    
    def _color_by_boltzmann(self):
        """Color poses by Boltzmann weight (blue=low, red=high)."""
        # TODO: Implement Boltzmann-weighted coloring
        cmd.do("print 'Color by Boltzmann weight (not yet implemented)'")
    
    def _show_representative(self):
        """Show only the representative pose (highest weight)."""
        # TODO: Implement representative pose display
        cmd.do("print 'Show representative pose (not yet implemented)'")
    
    def _launch_nrgsuite(self):
        """Launch NRGSuite plugin (if installed)."""
        try:
            cmd.do("nrgsuite")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", 
                f"NRGSuite not available: {e}\n\n"
                "Install from: https://github.com/NRGlab/NRGsuite"
            )
    
    def _export_to_nrgsuite(self):
        """Export binding modes to NRGSuite-compatible format."""
        # TODO: Implement NRGSuite export
        QtWidgets.QMessageBox.information(
            self, "Export", "NRGSuite export (not yet implemented)"
        )
