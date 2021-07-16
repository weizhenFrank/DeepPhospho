import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QLineEdit


def dialog():
    mbox = QMessageBox()

    mbox.setText("Your allegiance has been noted")
    mbox.setDetailedText("You are now a disciple and subject of the all-knowing Guru")
    mbox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

    mbox.exec_()


def set_config():
    config_w = QWidget()
    config_w.resize(600, 600)
    config_w.setWindowTitle('Parameters')

    line_edit = QLineEdit("GeeksforGeeks", )

    # setting geometry
    line_edit.setGeometry(80, 80, 150, 40)

    # creating a label
    label = QLabel("GfG", self)

    # setting geometry to the label
    label.setGeometry(80, 150, 120, 60)

    # setting word wrap property of label
    label.setWordWrap(True)

    # adding action to the line edit when enter key is pressed
    line_edit.returnPressed.connect(lambda: do_action())

    # method to do action
    def do_action():
        # getting text from the line edit
        value = line_edit.text()

        # setting text to the label
        label.setText(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(300, 300)
    w.setWindowTitle('DeepPhospho')

    label = QLabel(w)
    label.setText("Begin to run DeepPhospho")
    label.move(80, 130)
    label.show()

    line = QLineEdit(w)

    btn = QPushButton(w)
    btn.setText('Set config for pipline')
    btn.move(70, 150)
    btn.show()
    btn.clicked.connect(dialog)

    w.show()
    sys.exit(app.exec_())