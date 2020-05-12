from PyQt5.QtWidgets import (
    QDialog,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from PyQt5.QtGui import (
    QFontDatabase,
    QColor,
    QTextCharFormat,
    QFont,
    QSyntaxHighlighter,
)

from PyQt5.QtCore import QRegExp


def format_(color, style="", background=None):
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
    "keyword": format_("blue"),
    "operator": format_("red"),
    "brace": format_("darkGray"),
    "defclass": format_("black", "bold"),
    "string": format_("magenta"),
    "string2": format_("darkMagenta"),
    "comment": format_("darkGreen", "italic"),
    "self": format_("black", "italic"),
    "numbers": format_("brown"),
}


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the Python language.
     """

    # Python keywords
    keywords = [
        "and",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "exec",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "not",
        "or",
        "pass",
        "print",
        "raise",
        "return",
        "try",
        "while",
        "yield",
        "None",
        "True",
        "False",
    ]

    # Python operators
    operators = [
        "=",
        # Comparison
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        # Arithmetic
        "\+",
        "-",
        "\*",
        "/",
        "//",
        "\%",
        "\*\*",
        # In-place
        "\+=",
        "-=",
        "\*=",
        "/=",
        "\%=",
        # Bitwise
        "\^",
        "\|",
        "\&",
        "\~",
        ">>",
        "<<",
    ]

    # Python braces
    braces = [
        "\{",
        "\}",
        "\(",
        "\)",
        "\[",
        "\]",
    ]

    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)

        # Multi-line strings (expression, flag, style)
        # FIXME: The triple-quotes in these two lines will mess up the
        # syntax highlighting from this point onward
        self.tri_single = (QRegExp("'''"), 1, STYLES["string2"])
        self.tri_double = (QRegExp('"""'), 2, STYLES["string2"])

        rules = []

        # Keyword, operator, and brace rules
        rules += [
            (r"\b%s\b" % w, 0, STYLES["keyword"])
            for w in PythonHighlighter.keywords
        ]
        rules += [
            (r"%s" % o, 0, STYLES["operator"])
            for o in PythonHighlighter.operators
        ]
        rules += [
            (r"%s" % b, 0, STYLES["brace"]) for b in PythonHighlighter.braces
        ]

        # All other rules
        rules += [
            # 'self'
            (r"\bself\b", 0, STYLES["self"]),
            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES["string"]),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES["string"]),
            # 'def' followed by an identifier
            (r"\bdef\b\s*(\w+)", 1, STYLES["defclass"]),
            # 'class' followed by an identifier
            (r"\bclass\b\s*(\w+)", 1, STYLES["defclass"]),
            # From '#' until a newline
            (r"#[^\n]*", 0, STYLES["comment"]),
            # Numeric literals
            (r"\b[+-]?[0-9]+[lL]?\b", 0, STYLES["numbers"]),
            (r"\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b", 0, STYLES["numbers"]),
            (
                r"\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b",
                0,
                STYLES["numbers"],
            ),
        ]

        # Build a QRegExp for each pattern
        self.rules = [(QRegExp(pat), index, fmt) for (pat, index, fmt) in rules]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text.
         """
        # Do other syntax formatting
        for expression, nth, qformat in self.rules:
            index = expression.indexIn(text, 0)

            while index >= 0:
                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, qformat)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
         ``QRegExp`` for triple-single-quotes or triple-double-quotes, and
         ``in_state`` should be a unique integer to represent the corresponding
         state changes when inside those strings. Returns True if we're still
         inside a multi-line string when this function is finished.
         """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            start = delimiter.indexIn(text)
            # Move past this match
            add = delimiter.matchedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            end = delimiter.indexIn(text, start + add)
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + delimiter.matchedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = text.length() - start + add
            # Apply formatting
            self.setFormat(start, length, style)
            # Look for the next match
            start = delimiter.indexIn(text, start + length)

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False


# custom scripting interface for debugging
class ScriptWindow(QDialog):
    def __init__(self, parent=None):
        super(ScriptWindow, self).__init__(parent)

        self.setGeometry(100, 100, 800, 600)
        self.parent = parent
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setFixedPitch(True)
        font.setPointSize(10)

        self.editor = QPlainTextEdit()
        self.editor.setFont(font)
        self.editor.setTabStopWidth(self.editor.fontMetrics().width(" ") * 4)
        self.highlight = PythonHighlighter(self.editor.document())
        self.editor.setPlainText("# self.parent is the main GUI handle")
        self.runScriptButton = QPushButton("Run script")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.editor)
        self.layout.addWidget(self.runScriptButton)

        self.setLayout(self.layout)
        self.setWindowTitle("Script editor")

        self.setup_button_behavior()

        self.show()

    def setup_button_behavior(self):
        self.runScriptButton.clicked.connect(self.runscript)

    def runscript(self):
        print("Script is executed.")
        print("Testing parent data accessibility")

        text2execute = self.editor.toPlainText()

        exec(text2execute)
