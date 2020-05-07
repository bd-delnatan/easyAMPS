# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'easyAMPS_maingui.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(1019, 774)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(819, 767))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        MainWindow.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(800, 600))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tab.sizePolicy().hasHeightForWidth())
        self.tab.setSizePolicy(sizePolicy)
        self.tab.setObjectName("tab")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.tableWidget = DataFrameWidget(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setObjectName("tableWidget")
        self.verticalLayout_3.addWidget(self.tableWidget)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.visualchecksWidget = visAMPSWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.visualchecksWidget.sizePolicy().hasHeightForWidth())
        self.visualchecksWidget.setSizePolicy(sizePolicy)
        self.visualchecksWidget.setMinimumSize(QtCore.QSize(600, 0))
        self.visualchecksWidget.setObjectName("visualchecksWidget")
        self.horizontalLayout_2.addWidget(self.visualchecksWidget)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox = QtWidgets.QGroupBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(275, 0))
        self.groupBox.setObjectName("groupBox")
        self.phasePspinBox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.phasePspinBox.setGeometry(QtCore.QRect(100, 20, 101, 31))
        self.phasePspinBox.setMaximum(180.0)
        self.phasePspinBox.setProperty("value", 60.0)
        self.phasePspinBox.setObjectName("phasePspinBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(40, 30, 47, 14))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.backgroundPspinBox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.backgroundPspinBox.setGeometry(QtCore.QRect(100, 70, 151, 31))
        self.backgroundPspinBox.setMaximum(10000000.0)
        self.backgroundPspinBox.setObjectName("backgroundPspinBox")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 80, 71, 16))
        self.label_3.setObjectName("label_3")
        self.Pphase_sigma_label = QtWidgets.QLabel(self.groupBox)
        self.Pphase_sigma_label.setGeometry(QtCore.QRect(210, 30, 47, 14))
        self.Pphase_sigma_label.setText("")
        self.Pphase_sigma_label.setObjectName("Pphase_sigma_label")
        self.PinflectionSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.PinflectionSpinBox.setGeometry(QtCore.QRect(100, 120, 151, 31))
        self.PinflectionSpinBox.setMaximum(100000000.0)
        self.PinflectionSpinBox.setObjectName("PinflectionSpinBox")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(30, 120, 61, 31))
        self.label_5.setObjectName("label_5")
        self.verticalLayout_4.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName("groupBox_2")
        self.phaseSspinBox = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.phaseSspinBox.setGeometry(QtCore.QRect(100, 20, 101, 31))
        self.phaseSspinBox.setMaximum(180.0)
        self.phaseSspinBox.setProperty("value", 60.0)
        self.phaseSspinBox.setObjectName("phaseSspinBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 47, 14))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.backgroundSspinBox = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.backgroundSspinBox.setGeometry(QtCore.QRect(100, 70, 151, 31))
        self.backgroundSspinBox.setMaximum(10000000.0)
        self.backgroundSspinBox.setObjectName("backgroundSspinBox")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(10, 80, 71, 16))
        self.label_4.setObjectName("label_4")
        self.Sphase_sigma_label = QtWidgets.QLabel(self.groupBox_2)
        self.Sphase_sigma_label.setGeometry(QtCore.QRect(210, 30, 47, 14))
        self.Sphase_sigma_label.setText("")
        self.Sphase_sigma_label.setObjectName("Sphase_sigma_label")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(20, 120, 61, 31))
        self.label_6.setObjectName("label_6")
        self.SinflectionSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.SinflectionSpinBox.setGeometry(QtCore.QRect(100, 120, 151, 31))
        self.SinflectionSpinBox.setMaximum(100000000.0)
        self.SinflectionSpinBox.setObjectName("SinflectionSpinBox")
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setObjectName("groupBox_3")
        self.correctSHGButton = QtWidgets.QPushButton(self.groupBox_3)
        self.correctSHGButton.setGeometry(QtCore.QRect(60, 30, 171, 31))
        self.correctSHGButton.setAutoDefault(False)
        self.correctSHGButton.setDefault(False)
        self.correctSHGButton.setFlat(False)
        self.correctSHGButton.setObjectName("correctSHGButton")
        self.verticalLayout_4.addWidget(self.groupBox_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab_4)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.calculatorTableWidget = DataFrameWidget(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calculatorTableWidget.sizePolicy().hasHeightForWidth())
        self.calculatorTableWidget.setSizePolicy(sizePolicy)
        self.calculatorTableWidget.setMinimumSize(QtCore.QSize(350, 100))
        self.calculatorTableWidget.setMaximumSize(QtCore.QSize(500, 16777215))
        self.calculatorTableWidget.setObjectName("calculatorTableWidget")
        self.horizontalLayout.addWidget(self.calculatorTableWidget)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.anglecalc_mplwidget = matplotlibWidget(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.anglecalc_mplwidget.sizePolicy().hasHeightForWidth())
        self.anglecalc_mplwidget.setSizePolicy(sizePolicy)
        self.anglecalc_mplwidget.setMinimumSize(QtCore.QSize(300, 250))
        self.anglecalc_mplwidget.setObjectName("anglecalc_mplwidget")
        self.verticalLayout.addWidget(self.anglecalc_mplwidget)
        self.tab4_groupBox = QtWidgets.QGroupBox(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab4_groupBox.sizePolicy().hasHeightForWidth())
        self.tab4_groupBox.setSizePolicy(sizePolicy)
        self.tab4_groupBox.setMinimumSize(QtCore.QSize(300, 200))
        self.tab4_groupBox.setObjectName("tab4_groupBox")
        self.computeAnglesButton = QtWidgets.QPushButton(self.tab4_groupBox)
        self.computeAnglesButton.setGeometry(QtCore.QRect(20, 30, 200, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.computeAnglesButton.sizePolicy().hasHeightForWidth())
        self.computeAnglesButton.setSizePolicy(sizePolicy)
        self.computeAnglesButton.setMinimumSize(QtCore.QSize(200, 50))
        self.computeAnglesButton.setObjectName("computeAnglesButton")
        self.checkSolutionsButton = QtWidgets.QPushButton(self.tab4_groupBox)
        self.checkSolutionsButton.setGeometry(QtCore.QRect(20, 80, 200, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkSolutionsButton.sizePolicy().hasHeightForWidth())
        self.checkSolutionsButton.setSizePolicy(sizePolicy)
        self.checkSolutionsButton.setMinimumSize(QtCore.QSize(200, 50))
        self.checkSolutionsButton.setObjectName("checkSolutionsButton")
        self.verticalLayout.addWidget(self.tab4_groupBox)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tab_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.groupBox_4.setMinimumSize(QtCore.QSize(250, 110))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.label_7 = QtWidgets.QLabel(self.groupBox_4)
        self.label_7.setGeometry(QtCore.QRect(10, 30, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox_4)
        self.label_8.setGeometry(QtCore.QRect(10, 70, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.referenceDistributionSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_4)
        self.referenceDistributionSpinBox.setGeometry(QtCore.QRect(150, 70, 68, 24))
        self.referenceDistributionSpinBox.setMinimum(2.0)
        self.referenceDistributionSpinBox.setMaximum(70.0)
        self.referenceDistributionSpinBox.setProperty("value", 25.0)
        self.referenceDistributionSpinBox.setObjectName("referenceDistributionSpinBox")
        self.referenceTiltSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_4)
        self.referenceTiltSpinBox.setGeometry(QtCore.QRect(150, 30, 68, 24))
        self.referenceTiltSpinBox.setMinimum(0.0)
        self.referenceTiltSpinBox.setMaximum(90.0)
        self.referenceTiltSpinBox.setProperty("value", 45.0)
        self.referenceTiltSpinBox.setObjectName("referenceTiltSpinBox")
        self.label_8.raise_()
        self.label_7.raise_()
        self.referenceDistributionSpinBox.raise_()
        self.referenceTiltSpinBox.raise_()
        self.verticalLayout_5.addWidget(self.groupBox_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy)
        self.groupBox_5.setMinimumSize(QtCore.QSize(250, 110))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.label_9 = QtWidgets.QLabel(self.groupBox_5)
        self.label_9.setGeometry(QtCore.QRect(10, 30, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox_5)
        self.label_10.setGeometry(QtCore.QRect(10, 70, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.targetDistributionSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.targetDistributionSpinBox.setGeometry(QtCore.QRect(150, 70, 68, 24))
        self.targetDistributionSpinBox.setMinimum(2.0)
        self.targetDistributionSpinBox.setMaximum(70.0)
        self.targetDistributionSpinBox.setProperty("value", 25.0)
        self.targetDistributionSpinBox.setObjectName("targetDistributionSpinBox")
        self.targetTiltSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.targetTiltSpinBox.setGeometry(QtCore.QRect(150, 30, 68, 24))
        self.targetTiltSpinBox.setMinimum(0.0)
        self.targetTiltSpinBox.setMaximum(90.0)
        self.targetTiltSpinBox.setProperty("value", 45.0)
        self.targetTiltSpinBox.setObjectName("targetTiltSpinBox")
        self.verticalLayout_5.addWidget(self.groupBox_5)
        self.predictSignalButton = QtWidgets.QPushButton(self.tab_3)
        self.predictSignalButton.setMinimumSize(QtCore.QSize(90, 0))
        self.predictSignalButton.setMaximumSize(QtCore.QSize(120, 16777215))
        self.predictSignalButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.predictSignalButton.setStyleSheet("background-color: #76c472")
        self.predictSignalButton.setDefault(False)
        self.predictSignalButton.setFlat(False)
        self.predictSignalButton.setObjectName("predictSignalButton")
        self.verticalLayout_5.addWidget(self.predictSignalButton, 0, QtCore.Qt.AlignHCenter)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.predictedSignalWidget = matplotlibWidget(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictedSignalWidget.sizePolicy().hasHeightForWidth())
        self.predictedSignalWidget.setSizePolicy(sizePolicy)
        self.predictedSignalWidget.setMinimumSize(QtCore.QSize(480, 350))
        self.predictedSignalWidget.setObjectName("predictedSignalWidget")
        self.horizontalLayout_3.addWidget(self.predictedSignalWidget)
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1019, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSaveCSV = QtWidgets.QAction(MainWindow)
        self.actionSaveCSV.setObjectName("actionSaveCSV")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionAMPS_script_editor = QtWidgets.QAction(MainWindow)
        self.actionAMPS_script_editor.setObjectName("actionAMPS_script_editor")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSaveCSV)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuTools.addAction(self.actionAMPS_script_editor)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Data Table"))
        self.groupBox.setTitle(_translate("MainWindow", "P-polarization"))
        self.label.setText(_translate("MainWindow", "Δφ, °"))
        self.label_3.setText(_translate("MainWindow", "Background"))
        self.label_5.setText(_translate("MainWindow", "Inflection\n"
"point"))
        self.groupBox_2.setTitle(_translate("MainWindow", "S-polarization"))
        self.label_2.setText(_translate("MainWindow", "Δφ, °"))
        self.label_4.setText(_translate("MainWindow", "Background"))
        self.label_6.setText(_translate("MainWindow", "Inflection\n"
"point"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Actions"))
        self.correctSHGButton.setText(_translate("MainWindow", "Apply SHG corrections"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "SHG phase determination"))
        self.tab4_groupBox.setTitle(_translate("MainWindow", "actions"))
        self.computeAnglesButton.setToolTip(_translate("MainWindow", "Solve for Gaussian distribution (angle, distribution) from ratios"))
        self.computeAnglesButton.setText(_translate("MainWindow", "compute angles"))
        self.checkSolutionsButton.setToolTip(_translate("MainWindow", "Visualize selected point(s) from the table within all AMPS solution space"))
        self.checkSolutionsButton.setText(_translate("MainWindow", "check (selected) solutions"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Angle calculator"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Initial state (reference)"))
        self.label_7.setText(_translate("MainWindow", "Probe angle, ˚"))
        self.label_8.setText(_translate("MainWindow", "Probe distribution, ˚"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Final state"))
        self.label_9.setText(_translate("MainWindow", "Probe angle, ˚"))
        self.label_10.setText(_translate("MainWindow", "Probe distribution, ˚"))
        self.predictSignalButton.setText(_translate("MainWindow", "Do it"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Predict 4-ch signal"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.actionOpen.setText(_translate("MainWindow", "Open CSV file ..."))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSaveCSV.setText(_translate("MainWindow", "Save CSV file ..."))
        self.actionExit.setText(_translate("MainWindow", " &Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionAMPS_script_editor.setText(_translate("MainWindow", "AMPS script editor"))
from CustomTable import DataFrameWidget
from MplWidgets import matplotlibWidget, visAMPSWidget
