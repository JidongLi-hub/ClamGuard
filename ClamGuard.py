import re

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from MainWindow import *
from ScanWindow import *
from UpdateWindow import *
from ViewLogs import *
from DeepLearningWindow import *
from DL.DL_Detect import *
from DL.Compress import *
from DL.File_to_Img import *

import sys
import os
from datetime import datetime
import glob
import shutil


class ViewLogsWindow(QDialog):
    def __init__(self, parent=None):
        super(ViewLogsWindow, self).__init__(parent)
        self.ui = Ui_LogsWindow()
        self.ui.setupUi(self)
        self.setModal(True)
        self.WorkingDirectory = os.getcwd()
        self.QuarantineDirectory = self.WorkingDirectory + "/Quarantine/"
        self.LogsDirectory = self.WorkingDirectory + "/Logs/"

        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['File', 'Infection', 'Status'])
        self.ui.trvLogDetails.setModel(self.model)

        self.ui.cmbLogs.currentTextChanged.connect(self.GetLogDetails)
        self.ui.btnDeleteLog.clicked.connect(self.DeleteLog)
        self.virusfile = []
        self.infection = []
        self.virusnames = []

        # Create context menus for QTreeviews
        self.actionDelete = QAction("Delete file")
        self.actionDelete.setIcon(QIcon(":/clamguard/images/delete_virus32.png"))
        self.actionDelete.setIconVisibleInMenu(True)
        self.actionDelete.triggered.connect(self.DeleteFile)

        self.actionQuarantine = QAction(QIcon(":/clamguard/images/dl.png"), "Quarantine file")
        self.actionQuarantine.setIconVisibleInMenu(True)
        self.actionQuarantine.triggered.connect(self.QuarantineFile)

        self.ui.trvLogDetails.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.trvLogDetails.customContextMenuRequested.connect(self.showContextMenuLogDetails)
        self.contextMenuLogDetails = QMenu()
        self.contextMenuLogDetails.addAction(self.actionDelete)
        self.contextMenuLogDetails.addAction(self.actionQuarantine)

        self.ui.btnClose.setIcon(QIcon(":/clamguard/images/close32.png"))

    def showContextMenuLogDetails(self, position):
        self.contextMenuLogDetails.exec_(self.ui.trvLogDetails.viewport().mapToGlobal(position))

    def QuarantineFile(self):
        try:
            # Get file from treeview
            model = self.ui.trvLogDetails.model()
            index = self.ui.trvLogDetails.currentIndex()
            file = model.data(model.index(index.row(), 0))
            dst = self.QuarantineDirectory

            shutil.copy(file, dst) # Move file to quarantine
            os.remove(file)
            self.GetLogDetails() # Refresh treeview

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Information")
            msg.setInformativeText('File was moved to Quarantine folder !')
            msg.setWindowTitle("Information")
            msg.exec_()
        except Exception as e:
            print("ERROR: " + str(e))

    def DeleteFile(self):
        try:
            # Get file from treeview
            model = self.ui.trvLogDetails.model()
            index = self.ui.trvLogDetails.currentIndex()
            file = model.data(model.index(index.row(), 0))

            os.remove(file) # Delete file
            self.GetLogDetails() # Refresh treeview

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Information")
            msg.setInformativeText('File was deleted !')
            msg.setWindowTitle("Information")
            msg.exec_()
        except Exception as e:
            print("ERROR: " + str(e))

    def DeleteLog(self):
        # Get current log from combo
        if self.ui.cmbLogs.count() > 0:
            try:
                os.remove(str(self.ui.cmbLogs.currentText())) # delete file

                model = self.ui.trvLogDetails.model()
                # Empty model
                if model.hasChildren():
                    model.removeRows(0, model.rowCount())

                self.ListLogs() # refresh combo
            except Exception as e:
                print("ERROR: " + str(e))

    def ListLogs(self):
        # Get log files from Log directory
        lst = glob.glob(self.LogsDirectory + "*.log")
        self.ui.cmbLogs.clear() # Reset list
        if len(lst) > 0:
            self.ui.cmbLogs.addItems(lst)

    def GetLogDetails(self):
        if self.ui.cmbLogs.count() > 0:
            # selectedLogIndex = self.ui.cmbLogs.currentIndex()
            selectedLogText = self.ui.cmbLogs.currentText()

            self.virusfile.clear()
            self.infection.clear()

            pattern = "FOUND"
            file = open(selectedLogText, "r")
            for line in file:
                if re.search(pattern, line):
                    spl = line.strip().split(':')
                    if sys.platform == "win32":
                        self.virusfile.append(spl[0]+":"+spl[1])
                        self.infection.append(spl[2].replace('FOUND', '').strip())
                    else:
                        self.virusfile.append(spl[0])
                        self.infection.append(spl[1].replace('FOUND', '').strip())
            file.close()

            self.FillTreeView()

    def FillTreeView(self):
        model = self.ui.trvLogDetails.model()
        # Empty model
        if model.hasChildren():
            model.removeRows(0, model.rowCount())
        # Empty virusnames[]
        if len(self.virusnames) > 0:
            self.virusnames.clear()

        # if there are viruses
        if len(self.virusfile) > 0:
            self.virusfile.reverse() # reverse list
            self.infection.reverse() # reverse list
            for v in self.virusfile: # for every virus in list
                model.insertRow(0)   # insert a row to model
                model.setData(model.index(0, 0), v) # insert data to model
                if os.path.isfile(str(v)):
                    model.setData(model.index(0, 2), "FOUND")  # insert data to model
                else:
                    model.setData(model.index(0, 2), "DELETED")  # insert data to model

            # Get viruses names
            self.virusfile.reverse()
            for vir in self.virusfile:
                if sys.platform == "win32":
                    virus = vir.split('\\')
                else:
                    virus = vir.split('/')

                self.virusnames.append(virus[::-1][0])

            # # test if virus is in quarantine
            r=0 #reset counter
            for q in self.virusnames:
                if os.path.isfile(str(self.QuarantineDirectory + q)):
                    model.setData(model.index(r, 2), "QUARANTINED")  # insert data to model
                r = r + 1  # set counter

            # fill infections to model
            r=0 #reset counter
            for i in self.infection: # for every infection in list
                model.setData(model.index(r, 1), i) # insert data to model
                r=r+1 # set counter

class UpdateWindow(QDialog):
    def __init__(self, parent=None):
        super(UpdateWindow, self).__init__(parent)
        self.ui = Ui_UpdateWindow()
        self.ui.setupUi(self)
        self.setModal(True)
        self.ui.btnClose.clicked.connect(self.closeUpdateWindow)
        self.proc = QProcess(self)
        self.proc.finished.connect(self.OnUpdateProcFinished)
        self.freshclam = self.InitFreshClam()  # Get freshclam binary
        self.ui.txtUpdate.setEnabled(False)
        self.ui.btnClose.setIcon(QIcon(":/clamguard/images/close32.png"))

    def OnUpdateWindowShow(self):
        self.FreshClam()  # Get new virus list

    def OnUpdateProcFinished(self, exitCode, exitStatus):
        self.ui.txtUpdate.append("------------------------")
        self.ui.txtUpdate.append("Update is finished!")
        self.ui.txtUpdate.append("exitCode: " + str(exitCode))
        self.ui.txtUpdate.append("exitStatus: " + str(exitStatus))
        self.ui.txtUpdate.append("------------------------")


    def FreshClam(self):
        self.ui.txtUpdate.clear()  # Initializing txtScan
        self.ui.txtUpdate.append("Initializing freshclam...")
        self.ui.txtUpdate.append("freshclam program is: " + self.freshclam)
        self.proc.readyReadStandardOutput.connect(self.on_readyReadStandardOutput)

        # Get platform system
        if sys.platform == "win32":
            self.ui.txtUpdate.append("System is windows...")
            self.proc.start(self.freshclam)
        else:
            self.ui.txtUpdate.append("System is posix...")
            # self.proc.start("sudo " + self.freshclam)
            self.proc.start("pkexec " + self.freshclam)

        self.ui.txtUpdate.append("Get fresh signatures...")

    @QtCore.pyqtSlot()
    def on_readyReadStandardOutput(self):
        text = self.proc.readAllStandardOutput().data().decode()
        self.ui.txtUpdate.append(text.strip())

    def InitFreshClam(self):
        # Find freshclam
        try:
            r = os.popen("which freshclam").read()
            return r.rstrip()
        except Exception as e:
            print("ERROR: " + str(e))

    def closeUpdateWindow(self):
        # check if proc is running
        try:
            if self.proc.state() == QProcess.ProcessState(2): # If proc RUNNING
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Please wait update to finish before closing!')
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            else:
                # proc is not running, close window
                self.close()
        except Exception as e:
            self.ui.txtUpdate.append("ERROR: " + str(e))
            self.close()

class ScanWindow(QDialog):
    def __init__(self, parent=None):
        super(ScanWindow, self).__init__(parent)
        self.ui = Ui_ScanWindow()
        self.ui.setupUi(self)
        self.setModal(True)
        self.ui.btnClose.clicked.connect(self.closeScanWindow)
        self.scanPath = ""  # Init scan path
        self.LogPath = ""  # Init log path
        self.clamscan = self.InitClamScan()  # Init path for clamscan
        self.command = self.InitCommand()
        self.proc = QProcess(self)
        self.proc.finished.connect(self.OnScanProcFinished)
        self.ui.btnCancelScan.clicked.connect(self.KillScan)
        self.ui.txtScan.setEnabled(False)
        self.isCancel = False
        self.ui.btnClose.setIcon(QIcon(":/clamguard/images/close32.png"))
        self.ui.btnCancelScan.setIcon(QIcon(":/clamguard/images/exit32.png"))

    def OnScanProcFinished(self):
        try:
            if self.isCancel:
                self.ui.txtScan.append("Delete log file...")
                os.remove(self.LogPath)
                self.isCancel = False
        except Exception as e:
            self.ui.txtScan.append("ERROR: " + str(e))

    def KillScan(self):
        try:
            if self.proc.state() == QProcess.ProcessState(QProcess.Running): # If proc RUNNING
                self.ui.txtScan.append("Terminate scan...")
                self.proc.kill()
                self.ui.txtScan.append("Scan process is killed...")
                self.isCancel = True
            else:
                self.ui.txtScan.append("clamscan is not running...")
                self.isCancel = False
        except Exception as e:
            self.ui.txtScan.append("ERROR: " + str(e))

    @QtCore.pyqtSlot()
    def on_readyReadStandardOutput(self):
        text = self.proc.readAllStandardOutput().data().decode()
        self.ui.txtScan.append(text.strip())

    def MakeScan(self):
        self.ui.txtScan.clear()  # Initializing txtScan
        self.ui.txtScan.append("Initializing clamscan...")
        self.command = self.InitCommand()
        program = self.command[0]
        self.ui.txtScan.append("clamscan program is: " + program)
        args = self.command[1:]
        self.ui.txtScan.append("args are: " + str(args))
        self.proc.readyReadStandardOutput.connect(self.on_readyReadStandardOutput)  # 调用扫描功能
        self.proc.start(program, args)
        self.ui.txtScan.append("Scanning...")

    def GetTimestampLogName(self):
        dateTimeObj = datetime.now()
        f = str(dateTimeObj.year) + "_" + str(dateTimeObj.month) + "_" + str(dateTimeObj.day) + "_" + str(
            dateTimeObj.hour) + "_" + str(dateTimeObj.minute) + "_" + str(dateTimeObj.second)
        return f

    def InitCommand(self):
        # Format Log file name
        name = os.getcwd() + "/Logs/" + self.GetTimestampLogName() + str(".log")
        self.LogPath = name

        # Get platform system
        if sys.platform == "win32":
            # Format path
            self.scanPath = self.scanPath.replace("/", "\\")

        # Test if is dir or file
        if os.path.isdir(self.scanPath):
            cmd = [self.clamscan, "-i", "-v", "-l", name, "-r", self.scanPath]
            return cmd
        elif os.path.isfile(self.scanPath):
            cmd = [self.clamscan, "-i", "-v", "-l", name, self.scanPath]
            return cmd
        else:
            return []

    def InitClamScan(self):
        # Find clamscan
        try:
            r = os.popen("which clamscan").read()
            return r.rstrip()
        except Exception as e:
            print("ERROR: " + str(e))

    def IsScanPathDir(self):
        try:
            return os.path.isdir(self.scanPath)
        except Exception as e:
            print("ERROR: " + str(e))

    def IsScanPathFile(self):
        try:
            return os.path.isfile(self.scanPath)
        except Exception as e:
            print("ERROR: " + str(e))

    def closeScanWindow(self):
        # check if proc is running
        try:
            if self.proc.state() == QProcess.ProcessState(2): # If proc RUNNING
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Please terminate scan before closing!')
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            else:
                # proc is not running, close window
                self.close()
        except Exception as e:
            self.ui.txtScan.append("ERROR: " + str(e))
            self.close()

class DL_DetectWindow(QDialog):
    def __init__(self, parent=None):
        super(DL_DetectWindow, self).__init__(parent)
        self.ui = Ui_DeepLearningWindow()
        self.ui.setupUi(self)
        self.setModal(True)
        self.ui.btnClose.clicked.connect(self.closeDL_DetectWindow)
        self.scanPath = ""  # Init scan path
        self.DL_model = load_model("./DL/Checkpoints/ckp1.pth")  # 加载模型
        self.proc = QProcess(self)  # 准备检测进程
        self.ui.btnCancelScan.clicked.connect(self.KillDetect)
        self.ui.txtScan.setEnabled(False)
        self.isCancel = False
        self.ui.btnClose.setIcon(QIcon(":/clamguard/images/close32.png"))
        self.ui.btnCancelScan.setIcon(QIcon(":/clamguard/images/exit32.png"))

    def KillDetect(self):
        try:
            if self.proc.state() == QProcess.ProcessState(QProcess.Running): # If proc RUNNING
                self.ui.txtScan.append("Terminate detect...")
                self.proc.kill()
                self.ui.txtScan.append("Detect process is killed...")
                self.isCancel = True
            else:
                self.ui.txtScan.append("Detect is not running...")
                self.isCancel = False
        except Exception as e:
            self.ui.txtScan.append("ERROR: " + str(e))

    @QtCore.pyqtSlot() # 此行的作用是声明后接的函数是个槽函数
    def MakeDetect(self):   # 用训练好的模型进行检测
        self.ui.txtScan.clear()  # Initializing txtScan
        self.ui.txtScan.append("Model is loaded\nInitializing Detect...")
        if self.IsScanPathFile():
            # 读取文件并转为灰白图
            create_grayscale_image(self.scanPath, "./images/tmp_img.png")
            # 将灰白图大小重塑为128*128
            resize_image("tmp_img.png", "./images/", "./images/")
            # 调用模型进行预测
            result = predict("./images/tmp_img.png", self.DL_model)
            if result[0]==1:
                self.ui.txtScan.append(f"{self.scanPath} has detected malicious code. Warning!!!" )
            else :
                self.ui.txtScan.append(f"{self.scanPath} is OK!")
        elif self.IsScanPathDir():
            files = os.listdir(self.scanPath)
            for f in files:
                if os.path.isfile(os.path.join(self.scanPath,f)):
                    # 读取文件并转为灰白图
                    create_grayscale_image(os.path.join(self.scanPath,f), "./images/tmp_img.png")
                    # 将灰白图大小重塑为128*128
                    resize_image("tmp_img.png", "./images/", "./images/")
                    # 调用模型进行预测
                    result = predict("./images/tmp_img.png", self.DL_model)
                    if result[0]==1:
                        self.ui.txtScan.append(f"{os.path.join(self.scanPath,f)} has detected malicious code. Warning!!!")
                    else:
                        self.ui.txtScan.append(f"{os.path.join(self.scanPath,f)} is OK!")
        self.ui.txtScan.append("\nDetect finished！")



    def IsScanPathDir(self):
        try:
            return os.path.isdir(self.scanPath)
        except Exception as e:
            print("ERROR: " + str(e))

    def IsScanPathFile(self):
        try:
            return os.path.isfile(self.scanPath)
        except Exception as e:
            print("ERROR: " + str(e))

    def closeDL_DetectWindow(self):
        # check if proc is running
        try:
            if self.proc.state() == QProcess.ProcessState(2): # If proc RUNNING
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Please terminate detect before closing!')
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            else:
                # proc is not running, close window
                self.close()
        except Exception as e:
            self.ui.txtScan.append("ERROR: " + str(e))
            self.close()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Quarantine form
        #self.frmQuarantine = QuarantineWindow()
        #self.frmQuarantine.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        #self.ui.btnViewQuarantine.clicked.connect(self.showQuarantine)

        # DL form
        self.frmDL = DL_DetectWindow()
        self.frmDL.setWindowFlags(Qt.WindowMinimizeButtonHint)
        self.ui.btnDL_Detect.clicked.connect(self.showDL_DetectWindow)

        # Scan form
        self.frmScan = ScanWindow()
        self.frmScan.setWindowFlags(Qt.WindowMinimizeButtonHint)
        self.ui.btnScan.clicked.connect(self.showScanWindow)

        # UpdateWindow form
        self.frmUpdate = UpdateWindow()
        self.frmUpdate.setWindowFlags(Qt.WindowMinimizeButtonHint)
        self.ui.btnUpdate.clicked.connect(self.showUpdateWindow)

        # ViewLlogs form
        self.frmViewLogs = ViewLogsWindow()
        self.frmViewLogs.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.ui.btnViewLogs.clicked.connect(self.showViewLogsWindow)

        # Init trvSystem
        self.scanPath = ""  # Init scan path
        self.ui.trvSystem.clicked.connect(self.getScanPath)
        self.get_discs()

        # Set icons
        self.ui.btnScan.setIcon(QIcon(":/clamguard/images/scan32.png"))
        self.ui.btnUpdate.setIcon(QIcon(":/clamguard/images/update32.png"))
        self.ui.btnViewLogs.setIcon(QIcon(":/clamguard/images/log32.png"))
        self.ui.btnDL_Detect.setIcon(QIcon(":/clamguard/images/quarantine_virus32.png"))

    def getScanPath(self, index):
        # Get path from local dirs
        self.scanPath = self.ui.trvSystem.model.filePath(index)
        self.frmScan.scanPath = self.scanPath
        self.frmDL.scanPath = self.scanPath

    def get_discs(self):
        self.ui.trvSystem.model = QFileSystemModel()
        self.ui.trvSystem.model.setRootPath("")
        self.ui.trvSystem.setModel(self.ui.trvSystem.model)
        self.ui.trvSystem.setAnimated(False)
        self.ui.trvSystem.setIndentation(20)
        self.ui.trvSystem.sortByColumn(0, 0)
        self.ui.trvSystem.setSortingEnabled(True)
        self.ui.trvSystem.setColumnWidth(0, 200)

    def showDL_DetectWindow(self):
        if self.scanPath == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Please select a folder or a file to detect!')
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        self.frmDL.show()
        QApplication.processEvents() # 强制性地处理事件队列中的所有事件
        self.frmDL.MakeDetect()

    def showScanWindow(self):
        if self.scanPath == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Please select a folder or a file to scan!')
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        self.frmScan.InitCommand()
        self.frmScan.show()
        QApplication.processEvents()
        self.frmScan.MakeScan()

    def showUpdateWindow(self):
        self.frmUpdate.show()
        self.frmUpdate.OnUpdateWindowShow()

    def showViewLogsWindow(self):
        self.frmViewLogs.ListLogs()
#        self.frmViewLogs.GetLogDetails()
        self.frmViewLogs.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontShowIconsInMenus, False)
    w = MainWindow()
    #   Disable maximize window button
    w.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
    w.show()
    sys.exit(app.exec_())
