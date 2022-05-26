import sys
from PyQt5.QtWidgets import QWidget, QFileDialog, QPushButton, QVBoxLayout, QPlainTextEdit, QGridLayout, \
    QRadioButton, QHBoxLayout, QCheckBox, QButtonGroup
from PyQt5 import QtCore

from classifiers import mlp, random_forest


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

        # Создание текстовой области
        self.text_area = QPlainTextEdit(self)
        self.text_area.resize(500, 400)
        self.text_area.move(20, 300)
        vbox.addWidget(self.text_area)

        # Выбор файла
        btn = QPushButton('Get dataset', self)
        btn.setSizePolicy(300, 300)
        btn.move(20, 20)
        vbox.addWidget(btn)
        btn.clicked.connect(self.open_file_name_dialog)

        # Запуск классификатора
        btn = QPushButton('Start', self)
        btn.setSizePolicy(300, 300)
        btn.move(100, 100)
        vbox.addWidget(btn)
        btn.clicked.connect(self.start_classifier)

        # Сохранение модели
        btn = QPushButton('Save', self)
        btn.setSizePolicy(300, 300)
        btn.move(100, 100)
        vbox.addWidget(btn)
        btn.clicked.connect(self.start_classifier)

        # Выбор классификатора
        layout1 = QHBoxLayout()
        bg1 = QButtonGroup(self)
        self.radio_mlp = QRadioButton("MLP")
        self.radio_mlp.setChecked(True)
        self.radio_mlp.setFixedSize(100, 40)
        self.radio_mlp.toggled.connect(lambda: self.radio_button_clicked(self.radio_mlp))
        layout1.addWidget(self.radio_mlp, alignment=QtCore.Qt.AlignTop)
        bg1.addButton(self.radio_mlp)

        self.radio_rf = QRadioButton("RF")
        self.radio_rf.setFixedSize(100, 40)
        self.radio_rf.toggled.connect(lambda: self.radio_button_clicked(self.radio_rf))
        layout1.addWidget(self.radio_rf, alignment=QtCore.Qt.AlignTop)
        bg1.addButton(self.radio_rf)

        # Выбор процесса
        layout3 = QHBoxLayout()
        bg2 = QButtonGroup(self)
        self.radio_train = QRadioButton("Train")
        self.radio_train.setChecked(True)
        self.radio_train.setFixedSize(100, 40)
        self.radio_train.toggled.connect(lambda: self.radio_button_clicked(self.radio_train))
        layout3.addWidget(self.radio_train, alignment=QtCore.Qt.AlignTop)
        bg2.addButton(self.radio_train)

        self.radio_test = QRadioButton("Test")
        self.radio_test.setFixedSize(100, 40)
        self.radio_test.toggled.connect(lambda: self.radio_button_clicked(self.radio_test))
        layout3.addWidget(self.radio_test, alignment=QtCore.Qt.AlignTop)
        bg2.addButton(self.radio_test)

        # Выбор признаков
        layout2 = QHBoxLayout()

        self.b1 = QCheckBox("Ширина")
        self.b1.setChecked(True)
        self.b1.setFixedSize(120, 120)
        self.b1.stateChanged.connect(lambda: self.handle_checkbox_clicked(self.b1))
        layout2.addWidget(self.b1, alignment=QtCore.Qt.AlignTop)

        self.b2 = QCheckBox("Высота")
        self.b2.setFixedSize(120, 120)
        self.b2.toggled.connect(lambda: self.handle_checkbox_clicked(self.b2))
        layout2.addWidget(self.b2, alignment=QtCore.Qt.AlignTop)

        self.b3 = QCheckBox("Площадь")
        self.b3.setFixedSize(120, 120)
        self.b3.toggled.connect(lambda: self.handle_checkbox_clicked(self.b3))
        layout2.addWidget(self.b3, alignment=QtCore.Qt.AlignTop)

        self.b4 = QCheckBox("Периметр")
        self.b4.setFixedSize(120, 120)
        self.b4.toggled.connect(lambda: self.handle_checkbox_clicked(self.b4))
        layout2.addWidget(self.b4, alignment=QtCore.Qt.AlignTop)

        self.b5 = QCheckBox("Эксцентриситет")
        self.b5.setFixedSize(120, 120)
        self.b5.toggled.connect(lambda: self.handle_checkbox_clicked(self.b5))
        layout2.addWidget(self.b5, alignment=QtCore.Qt.AlignTop)

        self.b6 = QCheckBox("Моменты")
        self.b6.setFixedSize(120, 120)
        self.b6.toggled.connect(lambda: self.handle_checkbox_clicked(self.b6))
        layout2.addWidget(self.b6, alignment=QtCore.Qt.AlignTop)

        # установка layout
        vbox.addLayout(layout1)
        vbox.addLayout(layout3)
        vbox.addLayout(layout2)
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