from PyQt5.QtWidgets import (
    QPushButton,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QApplication,
    QPlainTextEdit,
    QTextEdit,
    QLabel,
    QFileDialog,
)
from PyQt5.QtGui import (
    QColor,
    QTextCharFormat,
    QFont,
    QFontDatabase,
    QSyntaxHighlighter,
    QTextCursor,
)
from pathlib import Path
from PyQt5.QtCore import QRegExp, QObject, pyqtSignal
from AMPS.parsers import process_experiment
from PyBiodesy.DataStructures import read_detail, subtract_blank_fluorescence
import yaml
import sys


templatestr = r"""#script template for parsing AMPS experiments
directories:
    input: "path/to/directory"
    output: "path/to/directory"

experiments:
    blank: "blank_file.xlsx"
    experiment_01: "experiment_file_01-Result.xlsx"
    experiment_02: "experiment_file_02-Result.xlsx"


# serial dilution pattern
# 1 refers to 100% labeled protein
# the last number is in the seqeuence refers to 100% unlabeled protein
# Each row designates a single column with top-to-bottom order.
# top-left number is located at the given 'origin'
patterns:
    simple: !!python/list [
        [1,7],
        [2,6],
        [3,5],
        [4,4],
        [5,3],
        [6,2],
        [7,1],
    ]


# Map configuration for each experiment listed above. except blank
# blocks with multiple origins will be combined into one file
experiment_01:
    "protein_R52C_apo":
        origins: A1,A3
        pattern: simple

    "protein_Q65C_apo":
        origins: A5,A7
        pattern: simple

experiment_02:
    "protein_R52C + drug1":
        origins: A1,A3
        pattern: simple

    "protein_Q65C + drug1":
        origins: A5,A7
        pattern: simple

"""


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(text)

    def flush(self):
        pass


def format(color, style="", background=None):
    """
    Return a QTextCharFormat with the given attributes.
    """
    _color = QColor()
    if type(color) is not str:
        _color.setRgb(color[0], color[1], color[2])
    else:
        _color.setNamedColor(color)

    _format = QTextCharFormat()
    _format.setForeground(_color)

    if background is not None:
        _bgcolor = QColor()
        if type(background) is not str:
            _bgcolor.setRgb(background[0], background[1], background[2])
        else:
            _bgcolor.setNamedColor(background)
        _format.setBackground(_bgcolor)

    if "bold" in style:
        _format.setFontWeight(QFont.Bold)
    if "italic" in style:
        _format.setFontItalic(True)
    return _format


STYLES = {
    "toplevel": format("blue", "bold"),
    "sublevel": format([255, 128, 0], "bold"),
    "operator": format([150, 150, 150]),
    "brace": format("darkGray"),
    "string": format([117, 163, 163]),
    "string2": format([117, 163, 163]),
    "comment": format("gray", "italic"),
    "numbers": format([100, 150, 190]),
    "spaces": format("gray", background=[230, 230, 230]),
    "tabs": format("gray", background=[255, 0, 0]),
    "special": format("green", "bold"),
    "reserved": format([219, 42, 101]),
}


class ScriptHighlighter(QSyntaxHighlighter):
    toplevel = ["directories", "experiments", "patterns"]
    sublevel = ["input", "output", "blank", "origins", "pattern", "exclude"]
    operators = [":"]
    spaces = [r"\s{4}"]
    tabs = [r"\t"]
    reserved = [r"!!python/list"]

    braces = [
        r"\[",
        r"\]",
    ]

    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)

        self.tri_single = (QRegExp("'''"), 1, STYLES["string2"])
        self.tri_double = (QRegExp('"""'), 2, STYLES["string2"])

        rules = []

        # Keyword, operator, and brace rules
        rules += [
            (r"\b{:s}\b".format(w), 0, STYLES["toplevel"])
            for w in ScriptHighlighter.toplevel
        ]

        rules += [
            (r"\b{:s}\b".format(w), 0, STYLES["sublevel"])
            for w in ScriptHighlighter.sublevel
        ]

        rules += [
            (r"{:s}".format(o), 0, STYLES["operator"])
            for o in ScriptHighlighter.operators
        ]

        rules += [
            (r"{:s}".format(b), 0, STYLES["brace"]) for b in ScriptHighlighter.braces
        ]

        rules += [
            (r"{:s}".format(b), 0, STYLES["spaces"]) for b in ScriptHighlighter.spaces
        ]

        rules += [
            (r"{:s}".format(b), 0, STYLES["tabs"]) for b in ScriptHighlighter.tabs
        ]

        rules += [
            (r"{:s}".format(s), 0, STYLES["reserved"])
            for s in ScriptHighlighter.reserved
        ]

        # strings
        rules += [
            # double quoted strings
            (r'"[^"/]*(/.[^"/]*)*"', 0, STYLES["string"]),
            # single quoted strings
            (r"'[^'/]*(/.[^'/]*)*'", 0, STYLES["string"]),
            # comment
            (r"#[^\n]*", 0, STYLES["comment"]),
        ]

        self.rules = [(QRegExp(pat), index, fmt) for (pat, index, fmt) in rules]

        self.wellrx = QRegExp(r"(?:origins:)(?:\s*)((\s?[A-P][1-9][0-9]?)(?:,)?)+")
        self.wellfmt = STYLES["special"]

    def highlightBlock(self, text):
        # Do other syntax formatting
        for expression, nth, qfmt in self.rules:
            index = expression.indexIn(text, 0)

            while index >= 0:
                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, qfmt)
                # search for the next occurence of expression in text
                index = expression.indexIn(text, index + length)

        # for well formatting
        wellstrpos = self.wellrx.indexIn(text, 0)
        subtext = self.wellrx.cap(0)

        while self.wellrx.indexIn(subtext, 0) >= 0:
            # thin down subtext for recursive search
            subtext = subtext[0 : self.wellrx.pos(1)]
            wellstr = self.wellrx.cap(2)
            strstart = wellstrpos + len(subtext)
            strlen = len(wellstr)
            colnum = int(wellstr.strip()[1:])
            if 1 <= colnum <= 24:
                self.setFormat(strstart, strlen, self.wellfmt)


class ScriptWriterDialog(QDialog):
    def __init__(self, parent=None):
        super(ScriptWriterDialog, self).__init__(parent)

        self.setGeometry(200, 100, 650, 700)

        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setFixedPitch(True)
        font.setPointSize(11)

        self.editor = QPlainTextEdit()
        self.editor.setFont(font)
        self.editor.setPlainText(templatestr)
        self.editor.setTabStopWidth(self.editor.fontMetrics().width(" ") * 4)
        # access the 'text' of QPlainTextEdit by
        self.highlight = ScriptHighlighter(self.editor.document())

        # replace system stdout with custom object to hijack it
        sys.stdout = Stream(newText=self.onUpdateText)

        self.stdoutbox = QTextEdit()
        self.stdoutbox.moveCursor(QTextCursor.Start)
        self.stdoutbox.setFixedHeight(150)

        self.loadScriptButton = QPushButton("Load script")
        self.checkSpacesButton = QPushButton("Convert tabs")
        self.saveScriptButton = QPushButton("Save script")
        self.runScriptButton = QPushButton("Run Script")

        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.addWidget(self.loadScriptButton)
        self.buttonsLayout.addWidget(self.checkSpacesButton)
        self.buttonsLayout.addWidget(self.saveScriptButton)
        self.buttonsLayout.addWidget(self.runScriptButton)

        self.layout = QVBoxLayout()

        self.currentfilelabel = QLabel("No scripts loaded")
        self.layout.addWidget(self.currentfilelabel)
        self.layout.addWidget(self.editor)
        self.layout.addLayout(self.buttonsLayout)
        self.layout.addWidget(QLabel("status message:"))
        self.layout.addWidget(self.stdoutbox)

        self.setLayout(self.layout)

        self.setWindowTitle("easyAMPS Script editor v0.1")

        self.setup_button_behavior()

    def setup_button_behavior(self):
        self.loadScriptButton.clicked.connect(self.loadscript)
        self.checkSpacesButton.clicked.connect(self.converttabs)
        self.saveScriptButton.clicked.connect(self.savescript)
        self.runScriptButton.clicked.connect(self.runscript)

    def loadscript(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, fileType = QFileDialog.getOpenFileName(
            self,
            "Open AMPS yaml configuration script",
            "",
            "YAML file (*.yaml)",
            options=options,
        )

        if fileName:

            self.currentfilepath = Path(fileName)
            self.currentfilelabel.setText(str(self.currentfilepath))

            # raw text is loaded too
            with open(self.currentfilepath, "rt") as fhd:
                scriptbody = "".join(fhd.readlines())

            self.editor.setPlainText(scriptbody)

            print("Script is loaded.")

        else:
            return False
        pass

    def converttabs(self):
        scriptbody = self.editor.toPlainText()
        scriptbody = scriptbody.replace("\t", "    ")

        print("Tabs have been replaced by spaces, check the script")

        # get current cursor position
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()

        # load the replaced script
        self.editor.setPlainText(scriptbody)

        cursor.endEditBlock()

    def savescript(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, fileType = QFileDialog.getSaveFileName(
            self,
            "Save AMPS yaml configuration script",
            "",
            "YAML file (*.yaml)",
            options=options,
        )

        if fileName:
            targetfile = Path(fileName).with_suffix(".yaml")

            # get text body
            scriptbody = self.editor.toPlainText()

            # replace tabs with spaces
            scriptbody = scriptbody.replace("\t", "    ")

            with open(targetfile, "wt") as fhd:
                fhd.write(scriptbody)

            print(f"Saved to {targetfile}")

    def runscript(self):
        """ Execute script from editor """
        scriptbody = self.editor.toPlainText()
        scriptbody = scriptbody.replace("\t", "    ")
        scriptconfig = yaml.load(scriptbody, Loader=yaml.FullLoader)

        print("Running script  ...")

        inputdir = Path(scriptconfig["directories"]["input"])
        outdir = Path(scriptconfig["directories"]["output"])

        print(f"Reading input files from : {inputdir}")
        print(f"Processed files will be saved to : {outdir}")

        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)

        experiments = {
            name: Path(inputdir / filename)
            for name, filename in scriptconfig["experiments"].items()
        }

        # load blank file
        blank = read_detail(experiments["blank"])

        # go through each experiment
        for expt_key in experiments.keys():
            if expt_key != "blank":
                print(f"Experiment : {expt_key}")

                if expt_key not in scriptconfig.keys():
                    # skip experiment that has no configurations
                    print(
                        f"Could not find configuration for '{expt_key}' experiment. Skipping it.'"
                    )
                    continue
                else:
                    alldata = read_detail(experiments[expt_key], label_rows=False)
                    timepoints = alldata["Read #"].unique().tolist()
                    lastread = timepoints[-1]

                    print(f"Found {len(timepoints)} read # :")
                    print(timepoints)
                    print("Using the last one : ", lastread)

                    data = alldata

                    subtract_blank_fluorescence(data, blank)
                    datadict = process_experiment(
                        data, scriptconfig, expt_key, transpose_block=False
                    )

                    for name, dataframe in datadict.items():
                        output_filename = f"{name}.csv"
                        output_target = outdir / output_filename
                        dataframe.to_csv(f"{str(output_target)}", index=False)

    def onUpdateText(self, text):
        cursor = self.stdoutbox.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.stdoutbox.setTextCursor(cursor)
        self.stdoutbox.ensureCursorVisible()

    def __del__(self):
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    import sys

    sys.path.append("C:\\Users\\delna\\Apps\\easyAMPS\\src\\main\\python")
    print("Running as main.... ")
    # to test the dialog box by via executing as python script
    app = QApplication(sys.argv)

    print("Instantiating ScriptWriterDialog()")
    form = ScriptWriterDialog()
    form.show()
    print("Event loop started.")
    sys.exit(app.exec_())
