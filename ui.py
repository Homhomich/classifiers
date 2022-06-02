import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QFileDialog, QPushButton, QVBoxLayout, QPlainTextEdit, QGridLayout, \
    QRadioButton, QHBoxLayout, QCheckBox, QButtonGroup, QLabel
from PyQt5 import QtCore

from classifiers import mlp, random_forest, random_forest_test_and_train


class Example(QWidget):
    file_name = ''
    selected_classifier = 'MLP'
    features = set()
    features.add('Ширина')

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 870, 870)
        self.setWindowTitle('UI')
        vbox = QVBoxLayout()
        layout_buttons = QHBoxLayout()

        # Создание текстовой области
        self.text_area = QPlainTextEdit(self)
        self.text_area.resize(500, 400)
        self.text_area.move(20, 300)
        vbox.addWidget(self.text_area)

        # Запуск классификатора
        btn = QPushButton('Start', self)
        btn.setSizePolicy(300, 300)
        btn.move(100, 100)
        layout_buttons.addWidget(btn)
        btn.clicked.connect(self.start_classifier)

        # Выбор датасета
        btn = QPushButton('Get dataset', self)
        btn.setSizePolicy(300, 300)
        btn.move(20, 20)
        layout_buttons.addWidget(btn)
        btn.clicked.connect(self.open_file_name_dialog)

        # Выбор модели
        btn = QPushButton('Get model', self)
        btn.setSizePolicy(300, 300)
        btn.move(100, 100)
        layout_buttons.addWidget(btn)
        btn.clicked.connect(self.start_classifier)

        # Сохранение модели
        btn = QPushButton('Save model', self)
        btn.setSizePolicy(300, 300)
        btn.move(100, 100)
        layout_buttons.addWidget(btn)
        btn.clicked.connect(self.start_classifier)

        # Выбор классификатора
        layout_classifiers = QHBoxLayout()
        bg1 = QButtonGroup(self)
        self.radio_mlp = QRadioButton("MLP")
        self.radio_mlp.setChecked(True)
        self.radio_mlp.setFixedSize(100, 40)
        self.radio_mlp.toggled.connect(lambda: self.radio_button_clicked(self.radio_mlp))
        layout_classifiers.addWidget(self.radio_mlp, alignment=QtCore.Qt.AlignTop)
        bg1.addButton(self.radio_mlp)

        self.radio_rf = QRadioButton("RF")
        self.radio_rf.setFixedSize(100, 40)
        self.radio_rf.toggled.connect(lambda: self.radio_button_clicked(self.radio_rf))
        layout_classifiers.addWidget(self.radio_rf, alignment=QtCore.Qt.AlignTop)
        bg1.addButton(self.radio_rf)

        # Выбор процесса
        layout_process = QHBoxLayout()
        bg2 = QButtonGroup(self)
        self.radio_train = QRadioButton("Train")
        self.radio_train.setChecked(True)
        self.radio_train.setFixedSize(100, 40)
        self.radio_train.toggled.connect(lambda: self.radio_button_clicked(self.radio_train))
        layout_process.addWidget(self.radio_train, alignment=QtCore.Qt.AlignTop)
        bg2.addButton(self.radio_train)

        self.radio_test = QRadioButton("Test")
        self.radio_test.setFixedSize(100, 40)
        self.radio_test.toggled.connect(lambda: self.radio_button_clicked(self.radio_test))
        layout_process.addWidget(self.radio_test, alignment=QtCore.Qt.AlignTop)
        bg2.addButton(self.radio_test)

        # Подпись параметров MLP
        layout_parameters_title_mlp = QHBoxLayout()
        self.label_title_mlp = QLabel(self)
        self.label_title_mlp.setFont(QFont('Arial font', 9))
        self.label_title_mlp.setFixedSize(300, 85)
        self.label_title_mlp.setText("Настройка параметров MLP")
        layout_parameters_title_mlp.addWidget(self.label_title_mlp, alignment=QtCore.Qt.AlignTop)

        # Параметры MLP
        layout_parameters_mlp = QHBoxLayout()
        self.text_mlp_neurons = QPlainTextEdit(self)
        self.text_mlp_neurons.setFixedSize(70, 30)

        self.label_mlp_neurons = QLabel(self)
        self.label_mlp_neurons.setText("Количество нейронов")

        self.text_mlp_layers = QPlainTextEdit(self)
        self.text_mlp_layers.setFixedSize(70, 30)

        self.label_mlp_layers = QLabel(self)
        self.label_mlp_layers.setText("Количество скрытых слоев")

        layout_parameters_mlp.addWidget(self.label_mlp_neurons, alignment=QtCore.Qt.AlignLeft)
        layout_parameters_mlp.addWidget(self.text_mlp_neurons, alignment=QtCore.Qt.AlignLeft)

        layout_parameters_mlp.addWidget(self.label_mlp_layers, alignment=QtCore.Qt.AlignLeft)
        layout_parameters_mlp.addWidget(self.text_mlp_layers, alignment=QtCore.Qt.AlignLeft)

        # Подпись параметров RF
        layout_parameters_title_rf = QHBoxLayout()
        self.label_title_rf = QLabel(self)
        self.label_title_rf.setFont(QFont('Arial font', 9))
        self.label_title_rf.setFixedSize(300, 85)
        self.label_title_rf.setText("Настройка параметров RF")
        layout_parameters_title_rf.addWidget(self.label_title_rf, alignment=QtCore.Qt.AlignTop)

        # Параметры RF
        layout_parameters_rf = QHBoxLayout()

        self.text_rf_trees = QPlainTextEdit(self)
        self.text_rf_trees.setFixedSize(50, 30)
        self.label_rf_trees = QLabel(self)
        self.label_rf_trees.setText("Количество деревьев")

        self.text_rf_depth = QPlainTextEdit(self)
        self.text_rf_depth.setFixedSize(50, 30)
        self.label_rf_depth = QLabel(self)
        self.label_rf_depth.setText("Максимальная глубина")

        self.text_rf_leafs = QPlainTextEdit(self)
        self.text_rf_leafs.setFixedSize(50, 30)
        self.label_rf_leafs = QLabel(self)
        self.label_rf_leafs.setText("Максимальное количество листьев")

        layout_parameters_rf.addWidget(self.label_rf_trees, alignment=QtCore.Qt.AlignLeft)
        layout_parameters_rf.addWidget(self.text_rf_trees, alignment=QtCore.Qt.AlignLeft)

        layout_parameters_rf.addWidget(self.label_rf_leafs, alignment=QtCore.Qt.AlignLeft)
        layout_parameters_rf.addWidget(self.text_rf_leafs, alignment=QtCore.Qt.AlignLeft)

        layout_parameters_rf.addWidget(self.label_rf_depth, alignment=QtCore.Qt.AlignLeft)
        layout_parameters_rf.addWidget(self.text_rf_depth, alignment=QtCore.Qt.AlignLeft)

        # Выбор признаков
        layout_features = QHBoxLayout()
        self.b1 = QCheckBox("Ширина")
        self.b1.setChecked(True)
        self.b1.setFixedSize(120, 120)
        self.b1.stateChanged.connect(lambda: self.handle_checkbox_clicked(self.b1))
        layout_features.addWidget(self.b1, alignment=QtCore.Qt.AlignTop)

        self.b2 = QCheckBox("Высота")
        self.b2.setFixedSize(120, 120)
        self.b2.toggled.connect(lambda: self.handle_checkbox_clicked(self.b2))
        layout_features.addWidget(self.b2, alignment=QtCore.Qt.AlignTop)

        self.b3 = QCheckBox("Площадь")
        self.b3.setFixedSize(120, 120)
        self.b3.toggled.connect(lambda: self.handle_checkbox_clicked(self.b3))
        layout_features.addWidget(self.b3, alignment=QtCore.Qt.AlignTop)

        self.b4 = QCheckBox("Периметр")
        self.b4.setFixedSize(120, 120)
        self.b4.toggled.connect(lambda: self.handle_checkbox_clicked(self.b4))
        layout_features.addWidget(self.b4, alignment=QtCore.Qt.AlignTop)

        self.b5 = QCheckBox("Эксцентриситет")
        self.b5.setFixedSize(120, 120)
        self.b5.toggled.connect(lambda: self.handle_checkbox_clicked(self.b5))
        layout_features.addWidget(self.b5, alignment=QtCore.Qt.AlignTop)

        self.b6 = QCheckBox("Моменты")
        self.b6.setFixedSize(120, 120)
        self.b6.toggled.connect(lambda: self.handle_checkbox_clicked(self.b6))
        layout_features.addWidget(self.b6, alignment=QtCore.Qt.AlignTop)

        # установка layout
        vbox.addLayout(layout_buttons)
        vbox.addLayout(layout_classifiers)
        vbox.addLayout(layout_process)
        vbox.addLayout(layout_features)
        vbox.addLayout(layout_parameters_title_mlp)
        vbox.addLayout(layout_parameters_mlp)
        vbox.addLayout(layout_parameters_title_rf)
        vbox.addLayout(layout_parameters_rf)

        self.setLayout(vbox)

        # запуск
        self.show()

    def handle_checkbox_clicked(self, b):
        if not b.isChecked():
            self.features.discard(b.text())
            print(b.text())
        else:
            self.features.add(b.text())
            print(b.text())
        print(self.features)

    def radio_button_clicked(self, b):
        if b.text() == "MLP" and b.isChecked():
            self.selected_classifier = 'MLP'
            print(self.selected_classifier)

        if b.text() == "RF" and b.isChecked():
            self.selected_classifier = 'RF'
            print(self.selected_classifier)

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        chosen_file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                          "All Files (*);;Python Files (*.py)", options=options)
        if chosen_file_name:
            self.file_name = chosen_file_name

    def start_classifier(self):
        features_arr = list(self.features)
        if self.selected_classifier == 'MLP':
            result = mlp(self.file_name, features_arr)
        else:
            result = random_forest(self.file_name, features_arr)
        self.text_area.clear()
        self.text_area.insertPlainText(result)
