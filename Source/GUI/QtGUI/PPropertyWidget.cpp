#include "PPropertyWidget.h"
#include "Framework/Module.h"
#include "Framework/Node.h"
#include "Framework/Framework/SceneGraph.h"

#include "PVTKOpenGLWidget.h"
#include "PCustomWidgets.h"
#include "Nodes/QtNodeWidget.h"
#include "Nodes/QtModuleWidget.h"

#include "Common.h"

#include "vtkRenderer.h"
#include <vtkRenderWindow.h>

#include <QGroupBox>
#include <QLabel>
#include <QCheckBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QRegularExpression>
#include <QMessageBox>
#include <QDir>
#include <QThread>

namespace PhysIKA {
// add by HNU
QString* QTextFieldWidget::m_InputPathSaver = nullptr;
// disable for using lib
//QString* QTextFieldWidget::m_solverPathSaver = nullptr;
QString* QTextFieldWidget::m_workDirPathSaver = nullptr;
QString* QTextFieldWidget::m_previewSaver     = nullptr;

QTextFieldWidget::QTextFieldWidget(Field* field)
    : QGroupBox()
{
    m_field = field;

    this->setStyleSheet("border:none");
    QGridLayout* layout = new QGridLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    this->setLayout(layout);

    QLabel* name = new QLabel();
    name->setFixedSize(160, 18);
    name->setText(FormatFieldWidgetName(field->getObjectName()));

    m_inputCombo->setFixedWidth(360);
    m_inputCombo->insertItem(0, *QTextFieldWidget::getInputPathSaver());
    m_inputCombo->setCurrentIndex(0);

    m_btnManager[0] = new QPushButton;
    m_btnManager[0]->setObjectName("inputBtn");
    btnAddPicHelper(m_btnManager[0]);

    layout->addWidget(name, 1, 0);
    layout->addWidget(m_inputCombo, 1, 1);
    layout->addWidget(m_btnManager[0], 1, 2);

    connect(m_btnManager[0], &QPushButton::clicked, this, &QTextFieldWidget::btnClicked);
    //connect(m_inputCombo, &QComboBox::currentTextChanged, this, &QTextFieldWidget::emitLoadFileSignal);

    // inputFileUrl split with  "\\" by Signal&Slot
    connect(m_inputCombo, &QComboBox::currentTextChanged, [&](const QString str) {
        if (m_inputCombo->currentText() != "")
            emitLoadFileSignal(k2Obj(str));
    });

    //std::cout << name->text().toStdString();

    // solver Ui
    if (field->getDescription().compare("Solver") == 0)
    {
        initSolverUi(layout);
    }

    // output Ui
    if (field->getDescription().compare("outputView") == 0)
    {
        initOutputUi(layout);
    }
}

void QTextFieldWidget::initSolverUi(QGridLayout* layout)
{
    layout->setSpacing(3);

    // achieve by => m_previewLineedit->setText(*QTextFieldWidget::getParviewSaver());
    //updatePreview();

    // row 1
    QLabel* coreLabel = new QLabel("CPU/GPU");
    QLabel* numLabel  = new QLabel("Num");
    QLabel* IDLabel   = new QLabel("ID");
    coreLabel->setAlignment(Qt::AlignCenter);
    numLabel->setAlignment(Qt::AlignCenter);
    IDLabel->setAlignment(Qt::AlignCenter);
    coreLabel->setFixedWidth(50);
    numLabel->setFixedWidth(20);
    IDLabel->setFixedWidth(20);
    layout->addWidget(coreLabel, 0, 3);
    layout->addWidget(numLabel, 0, 4);
    layout->addWidget(IDLabel, 0, 5);

    // row 2
    // disable for using lib
    //QLabel* solverLabel = new QLabel("Solver");
    //m_solverCombo->setFixedWidth(360);
    //m_solverCombo->insertItem(0, *QTextFieldWidget::getSolverPathSaver());
    //m_solverCombo->setCurrentIndex(0);

    // disable for using lib
    //m_btnManager[1] = new QPushButton;
    //m_btnManager[1]->setObjectName("solverBtn");
    //btnAddPicHelper(m_btnManager[1]);

    QComboBox* cgCombo = new QComboBox;
    cgCombo->insertItem(0, "GPU");
    cgCombo->insertItem(1, "CPU");
    cgCombo->setFixedWidth(50);

    m_numLineedit->setFixedWidth(20);
    m_IDLineedit->setFixedWidth(20);

    // disable for using lib
    //layout->addWidget(solverLabel, 1, 0);
    //layout->addWidget(m_solverCombo, 1, 1);
    //layout->addWidget(m_btnManager[1], 1, 2);
    layout->addWidget(cgCombo, 1, 3);
    layout->addWidget(m_numLineedit, 1, 4);
    layout->addWidget(m_IDLineedit, 1, 5);
    //connect(m_btnManager[1], &QPushButton::clicked, this, &QTextFieldWidget::btnClicked);

    // row 3
    QLabel* workDirLabel    = new QLabel("WorkDir");
    QLabel* DTLabel         = new QLabel("DT");
    QLabel* OTLabel         = new QLabel("OT");
    QLabel* searchStepLabel = new QLabel("SS");

    m_workdirCombo->setFixedWidth(360);
    m_workdirCombo->insertItem(0, *QTextFieldWidget::getWorkDirPathSaver());
    m_workdirCombo->setCurrentIndex(0);

    m_btnManager[2] = new QPushButton;
    m_btnManager[2]->setObjectName("workdirBtn");
    btnAddPicHelper(m_btnManager[2]);

    DTLabel->setAlignment(Qt::AlignCenter);
    DTLabel->setAlignment(Qt::AlignCenter);
    searchStepLabel->setAlignment(Qt::AlignCenter);

    DTLabel->setFixedWidth(50);
    OTLabel->setFixedWidth(20);
    searchStepLabel->setFixedWidth(20);

    layout->addWidget(workDirLabel, 2, 0);
    layout->addWidget(m_workdirCombo, 2, 1);
    layout->addWidget(m_btnManager[2], 2, 2);
    layout->addWidget(DTLabel, 2, 3);
    layout->addWidget(OTLabel, 2, 4);
    layout->addWidget(searchStepLabel, 2, 5);

    connect(m_btnManager[2], &QPushButton::clicked, this, &QTextFieldWidget::btnClicked);

    // row 4
    //previewLineedit->setDisabled(true);

    QLabel* previewLabel = new QLabel("Preview");

    m_btnManager[3] = new QPushButton;
    m_btnManager[3]->setObjectName("startBtn");
    QPixmap pixmap = QPixmap(":/startSolver.png");
    QIcon   buttonIcon(pixmap);
    m_btnManager[3]->setIcon(buttonIcon);
    m_btnManager[3]->setIconSize(pixmap.rect().size());

    m_DTLineedit->setFixedWidth(50);
    m_OTLineedit->setFixedWidth(20);
    m_searchStepLineedit->setFixedWidth(20);

    m_previewLineedit->setText(*QTextFieldWidget::getParviewSaver());

    layout->addWidget(previewLabel, 3, 0);
    layout->addWidget(m_previewLineedit, 3, 1);
    layout->addWidget(m_btnManager[3], 3, 2);
    layout->addWidget(m_DTLineedit, 3, 3);
    layout->addWidget(m_OTLineedit, 3, 4);
    layout->addWidget(m_searchStepLineedit, 3, 5);

    connect(m_btnManager[3], &QPushButton::clicked, this, &QTextFieldWidget::btnClicked);

    // parameter changed and update preview
    connect(m_numLineedit, &QLineEdit::textChanged, this, &QTextFieldWidget::updatePreview);
    connect(m_IDLineedit, &QLineEdit::textChanged, this, &QTextFieldWidget::updatePreview);
    connect(m_DTLineedit, &QLineEdit::textChanged, this, &QTextFieldWidget::updatePreview);
    connect(m_OTLineedit, &QLineEdit::textChanged, this, &QTextFieldWidget::updatePreview);
    connect(m_searchStepLineedit, &QLineEdit::textChanged, this, &QTextFieldWidget::updatePreview);
}

void QTextFieldWidget::initOutputUi(QGridLayout* layout)
{
    QLabel* outputLabel = new QLabel("output Url");

    m_outputCombo->setFixedWidth(360);
    m_outputCombo->insertItem(0, *QTextFieldWidget::getWorkDirPathSaver());
    m_outputCombo->setCurrentIndex(0);

    m_btnManager[4] = new QPushButton;
    m_btnManager[4]->setObjectName("outputBtn");
    btnAddPicHelper(m_btnManager[4]);

    m_btnManager[5] = new QPushButton;
    m_btnManager[5]->setObjectName("showBtn");
    QPixmap pixmap = QPixmap(":/startSolver.png");
    QIcon   buttonIcon(pixmap);
    m_btnManager[5]->setIcon(buttonIcon);
    m_btnManager[5]->setIconSize(pixmap.rect().size());

    layout->addWidget(outputLabel, 2, 0);
    layout->addWidget(m_outputCombo, 2, 1);
    layout->addWidget(m_btnManager[4], 2, 2);
    layout->addWidget(m_btnManager[5], 2, 3);

    connect(m_btnManager[4], &QPushButton::clicked, this, &QTextFieldWidget::btnClicked);
    connect(m_btnManager[5], &QPushButton::clicked, this, &QTextFieldWidget::btnClicked);
}

void QTextFieldWidget::btnClicked()
{

    QString btnName = QObject::sender()->objectName();
    //std::cout << btnName.toStdString();

    if (btnName == "inputBtn")
    {
        QString fileName = QFileDialog::getOpenFileName(
            this,
            tr("open import file"),
            ".",
            tr("k(*.k);;All Files(*.*)"));
        if (fileName.isEmpty())
        {
            QMessageBox::warning(this, "Warning", "Failed to open import file");
            return;
        }
        else
        {
            *QTextFieldWidget::getInputPathSaver() = QDir::toNativeSeparators(fileName);
            m_inputCombo->setEditable(false);
            m_inputCombo->insertItem(0, QDir::toNativeSeparators(fileName));
            m_inputCombo->setCurrentIndex(0);
            updatePreview();
        }
    }
    else if (btnName == "solverBtn")
    {
        // disable for using lib
        //QString fileName = QFileDialog::getOpenFileName(
        //    this,
        //    tr("open import file"),
        //    ".",
        //    tr("executable(*.exe);;All Files(*.*)"));
        //if (fileName.isEmpty())
        //{
        //    QMessageBox::warning(this, "Warning", "Failed to choose an solver", QMessageBox::Ok);
        //    return;
        //}
        //else
        //{
        //    *QTextFieldWidget::getSolverPathSaver() = QDir::toNativeSeparators(fileName);
        //    m_solverCombo->setEditable(false);
        //    m_solverCombo->insertItem(0, QDir::toNativeSeparators(fileName));
        //    m_solverCombo->setCurrentIndex(0);
        //    updatePreview();
        //}
    }
    else if (btnName == "workdirBtn")
    {
        QString path = QFileDialog::getExistingDirectory(this,
                                                         tr("choose work dir"),
                                                         ".");
        if (path.isEmpty())
        {
            QMessageBox::warning(this,
                                 tr("Warning"),
                                 tr("Failed to Choose an Valid Path"),
                                 QMessageBox::Ok);
            return;
        }
        else
        {
            *QTextFieldWidget::getWorkDirPathSaver() = QDir::toNativeSeparators(path);
            m_workdirCombo->setEditable(false);
            m_workdirCombo->insertItem(0, QDir::toNativeSeparators(path));
            m_workdirCombo->setCurrentIndex(0);
            updatePreview();
        }
    }
    else if (btnName == "startBtn")
    {
        // establish objFile folder
        QDir objFileFolder(*getWorkDirPathSaver());
        if (!objFileFolder.exists("objFile"))
            objFileFolder.mkdir("objFile");

        //proCmd->startDetached("cmd.exe", QStringList() << "/k" << m_Preview);
        //proCmd->start("cmd.exe", QStringList() << "/c" << m_Preview);
        emit startCalculate(m_Preview);
    }
    else if (btnName == "outputBtn")
    {
        if (*getWorkDirPathSaver() == "")
        {
            QMessageBox::warning(this, "Warning", "open output file url after start a calculation mission.", QMessageBox::Ok);
            return;
        }

        QString outputFileUrl = QFileDialog::getOpenFileName(
            this,
            tr("open import file"),
            *getWorkDirPathSaver(),
            tr("vtk(*.vtk);;All Files(*.*)"));
        if (outputFileUrl.isEmpty())
        {
            QMessageBox::warning(this, "Warning", "Failed to choose an result file.", QMessageBox::Ok);
            return;
        }
        else
        {
            //qDebug() << outputFileUrl;
            //qDebug() << vtk2Obj(outputFileUrl);
            emitLoadFileSignal(vtk2Obj(outputFileUrl));
        }
    }
    else if (btnName == "showBtn")
    {
        //qDebug() << "showBtn clicked";
        //startAnimating(*getWorkDirPathSaver() + "\\objFile");
        startAnimating(*getWorkDirPathSaver());
    }
    else
    {
        QMessageBox::warning(this,
                             tr("Warning"),
                             tr("Unknown Button was clicked"),
                             QMessageBox::Ok);
    }
}
void QTextFieldWidget::btnAddPicHelper(QPushButton* btn)
{
    QPixmap pixmap = QPixmap(":/openFolder.png");
    QIcon   buttonIcon(pixmap);
    btn->setIcon(buttonIcon);
    btn->setIconSize(pixmap.rect().size());
}

QString QTextFieldWidget::updatePreview()
{
    // disable for using lib
    //QString solver = m_solverCombo->currentText();
    QString solver = " ";
    QString inp    = m_inputCombo->currentText();
    //工作路径
    QString WD = m_workdirCombo->currentText();

    //参与计算的GPU数
    QString NGPU;
    if (m_numLineedit->text() == "")
        NGPU = "1";
    else
        NGPU = m_numLineedit->text();

    //单GPU计算时指定采用GPU的ID
    QString IDG;
    if (m_IDLineedit->text() == "")
        IDG = "0";
    else
        IDG = m_IDLineedit->text();

    //指定时间步长paraview输出时间间隔指定搜索间隔
    QString DT = m_DTLineedit->text();

    //paraview输出时间间隔
    QString OT = m_OTLineedit->text();

    //指定搜索间隔
    QString CGN = m_searchStepLineedit->text();

    //根据 DT OT Search Step 三者状态确定输出内容
    if (DT == "")
    {
        if (OT == "")
            if (CGN == "")
            {  // 三者为空，均依赖输入文件
                m_Preview = solver + " INP=" + inp + " WD=" + WD + " NGPU=" + NGPU + " IDG=" + IDG;
            }
            else
            {  // CGN 不为空
                m_Preview = solver + " INP=" + inp + " WD=" + WD + " NGPU=" + NGPU + " IDG=" + IDG + " CGN=" + CGN;
            }
        else
        {  //OT 不为空
            if (CGN == "")
            {  // OT 不空 CGN 为空
                m_Preview = solver + " INP=" + inp + " WD=" + WD + " OT=" + OT + " NGPU=" + NGPU + " IDG=" + IDG;
            }
            else
            {  // CGN 不为空
                m_Preview = solver + " INP=" + inp + " WD=" + WD + " OT=" + OT + " NGPU=" + NGPU + " IDG=" + IDG + " CGN=" + CGN;
            }
        }
    }
    else
    {
        if (OT == "")
            if (CGN == "")
            {  // 三者为空，均依赖输入文件
                m_Preview = solver + " INP=" + inp + " WD=" + WD + " DT=" + DT + " NGPU=" + NGPU + " IDG=" + IDG;
            }
            else
            {  // CGN 不为空
                m_Preview = solver + " INP=" + inp + " WD=" + WD + " DT=" + DT + " NGPU=" + NGPU + " IDG=" + IDG + " CGN=" + CGN;
            }
        else
        {  //OT 不为空
            if (CGN == "")
            {  // OT 不空 CGN 为空
                m_Preview = solver + " INP=" + inp + " WD=" + WD + " DT=" + DT + " OT=" + OT + " NGPU=" + NGPU + " IDG=" + IDG;
            }
            else
            {  // CGN 不为空
                m_Preview = solver + " INP=" + inp + " WD=" + WD + " DT=" + DT + " OT=" + OT + " NGPU=" + NGPU + " IDG=" + IDG + " CGN=" + CGN;
            }
        }
    }

    //Preview = solver + " INP=" + inp + " WD=" + WD + " DT=" + DT + " OT=" + OT + " NGPU=" + NGPU + " IDG=" + IDG + " CGN=" + CGN;

    m_previewLineedit->setText(m_Preview);
    *QTextFieldWidget::getParviewSaver() = m_Preview;
    return m_Preview;
}

// k2Obj get url through Signal&Slot and split with "\\"
// vtk2Obj get url through QFileDialog::getOpenFileName and split with "/"
QString QTextFieldWidget::k2Obj(const QString inputUrl)
{
    if (inputUrl == "")
        return "";

    //qDebug() << inputUrl;
    QFile       inputFile(inputUrl);
    QStringList urlSplit = inputUrl.trimmed().split("\\");

    QFileInfo fileInfo(inputFile);
    //QString objectName = fileInfo.fileName();
    QString objectName = fileInfo.fileName();
    QString cropedName = objectName.section(".", 0, 0);
    //qDebug() << cropedName;

    //qDebug() << urlSplit;
    urlSplit.replace(urlSplit.size() - 1, "verticeFile.txt");
    //qDebug() << urlSplit.join("/");
    QFile verticeFile(urlSplit.join("/"));
    urlSplit.replace(urlSplit.size() - 1, "faceFile.txt");
    QFile faceFile(urlSplit.join("/"));

    QStringList tempList = urlSplit;
    tempList.replace(urlSplit.size() - 1, "");
    //qDebug() << tempList.join("/");
    QDir tempFileFolder(tempList.join("/"));
    //system("pause");
    if (!tempFileFolder.exists("temp"))
        tempFileFolder.mkdir("temp");
    urlSplit.replace(urlSplit.size() - 1, "temp");
    urlSplit.append(cropedName + ".obj");
    QFile outputFile(urlSplit.join("/"));

    QString forwardDeclear;

    unsigned long VerticesNum = 0;
    unsigned long FacesNum    = 0;

    if (!inputFile.open(QIODevice::ReadOnly | QIODevice::Text) || !verticeFile.open(QIODevice::WriteOnly | QIODevice::Text) || !faceFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << "open file failed or create temp file failed.";
        return "";
    }
    else
    {
        QTextStream in(&inputFile);

        while (in.readLine() != "*NODE")
            ;

        QString line = in.readLine();
        while (line.trimmed()[0] != "*")
        {
            QStringList strList = line.trimmed().split(" ");

            strList.replace(0, "v");
            strList.removeAll(QString(""));

            while (strList.size() > 4)
                strList.removeLast();

            //qDebug() << strList.join(" ");
            verticeFile.write(strList.join(" ").toUtf8());
            verticeFile.write("\n");

            line = in.readLine().toUtf8();

            ++VerticesNum;
        }
        // jump to data area
        verticeFile.close();

        while (line != "*ELEMENT_SHELL")
        {
            line = in.readLine();
            //    qDebug() << line;
        }

        line = in.readLine();
        //while (line != "*END" && line != "$$")
        while (!(line == "*END" || line == "$$"))
        {
            QStringList strList = line.trimmed().split(" ");
            strList.removeAll(QString(""));
            //strList.replace(0, "");
            //strList.replace(1, "f");
            //strList.replace(5, "");
            //strList.join("").trimmed();

            //    qDebug() << "f " + strList[2] + " " + strList[3] + " " + strList[4];
            faceFile.write(("f " + strList[2] + " " + strList[3] + " " + strList[4]).toUtf8());
            faceFile.write("\n");
            ++FacesNum;
            line = in.readLine();
        }
        inputFile.close();
        faceFile.close();
    }

    if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Text) || !verticeFile.open(QIODevice::ReadOnly | QIODevice::Text) || !faceFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "output file or tempfile open failed";
        return "";
    }
    else
    {
        forwardDeclear =
            "####\n\
#\n\
# OBJ File Generated by MXTeam\n\
#\n\
####\n\
# Object " + cropedName
            + ".obj\n\
#\n\
# Vertices: "
            + QString::number(VerticesNum) + "\n\
# Faces: " + QString::number(FacesNum)
            + "\n\
#\n\
####\n";
        //qDebug() << forwardDeclear;

        QTextStream out(&outputFile);
        QTextStream verticeStream(&verticeFile);
        QTextStream faceStream(&faceFile);
        // move pointer to begin
        verticeStream.seek(0);
        faceStream.seek(0);

        outputFile.write(forwardDeclear.toUtf8());

        while (!verticeStream.atEnd())
        {
            QString line = verticeStream.readLine();
            //qDebug() << line;
            outputFile.write(line.toUtf8());
            outputFile.write("\n");
        }
        verticeFile.close();

        outputFile.write(("# " + QString::number(VerticesNum) + ", 0 vertices normals\n").toUtf8());
        outputFile.write("\n");

        while (!faceStream.atEnd())
        {
            QString line = faceStream.readLine();
            //qDebug() << line;
            outputFile.write(line.toUtf8());
            outputFile.write("\n");
        }
        outputFile.write(("# " + QString::number(FacesNum) + " faces, 0 coords texture\n").toUtf8());
        outputFile.write("\n");
        outputFile.write("# End of File\n");
        outputFile.write("\n");

        faceFile.close();

        outputFile.close();
    }

    // delete intermedite files
    verticeFile.remove();
    faceFile.remove();
    // return convert file url
    return urlSplit.join("/");
}

QString QTextFieldWidget::vtk2Obj(const QString outputFileUrl)
{
    if (outputFileUrl == "")
        return "";

    QFile       outputFile(outputFileUrl);
    QStringList urlSplit = outputFileUrl.trimmed().split("/");

    QFileInfo fileInfo(outputFile);
    QString   cropedName = fileInfo.fileName().section(".", 0, 0);

    // add file number
    // result_xx
    // result_x
    // result
    if (cropedName.size() == 9)
        cropedName = cropedName.insert(7, "0");
    if (cropedName.size() == 8)
        cropedName = cropedName.insert(7, "00");
    if (cropedName.size() == 6)
        cropedName = cropedName.insert(6, "_000");

    // generate showFile Url and file
    urlSplit.removeLast();
    //    qDebug() << urlSplit.join("/");
    QDir dir(urlSplit.join("/"));

    if (!dir.exists("objFile"))
        dir.mkdir("objFile");

    urlSplit.append("objFile");
    urlSplit.append(cropedName + ".obj");
    QFile showFile(urlSplit.join("/"));
    //    qDebug() << urlSplit.join("/");

    QString       forwardDeclear;
    unsigned long VerticesNum = 0;
    unsigned long FacesNum    = 0;

    if (!(outputFile.open(QIODevice::ReadOnly | QIODevice::Text) && showFile.open(QIODevice::WriteOnly | QIODevice::Text)))
    {
        qDebug() << "open output file or open showFile failed.";
        return "";
    }
    else
    {
        QTextStream in(&outputFile);

        QString line = in.readLine();
        while (!line.trimmed().startsWith("POINTS"))
            line = in.readLine();
        QStringList strListTemp = line.simplified().split(" ");
        VerticesNum             = strListTemp[1].toLong();
        //qDebug() << VerticesNum;

        while (!line.trimmed().startsWith("CELLS"))
            line = in.readLine();
        strListTemp = line.simplified().split(" ");
        FacesNum    = strListTemp[1].toLong();
        //qDebug() << FacesNum;

        forwardDeclear =
            "####\n\
#\n\
# OBJ File Generated by MXTeam\n\
#\n\
####\n\
# Object " + cropedName
            + ".obj\n\
#\n\
# Vertices: "
            + QString::number(VerticesNum) + "\n\
# Faces: " + QString::number(FacesNum)
            + "\n\
#\n\
####\n";

        showFile.write(forwardDeclear.toUtf8());

        in.seek(0);

        while (!line.trimmed().startsWith("POINTS"))
            line = in.readLine();

        for (int i = 0; i < VerticesNum; i++)
        {
            line = in.readLine().simplified();
            //qDebug() << line;
            showFile.write(("v " + line).toUtf8());
            showFile.write("\n");
        }

        showFile.write(("# " + QString::number(VerticesNum) + ", 0 vertices normals\n").toUtf8());
        showFile.write("\n");

        while (!line.trimmed().startsWith("CELLS"))
            line = in.readLine();

        for (int i = 0; i < FacesNum; i++)
        {
            line = in.readLine().simplified();
            //qDebug() << line.simplified();
            QStringList temp = line.split(" ");
            temp.replace(0, "f");
            for (int i = 1; i < temp.size(); i++)
            {
                unsigned long tempNum = temp[i].toLong();
                temp.replace(i, QString::number(tempNum + 1));
            }
            //qDebug() << temp;
            showFile.write(temp.join(" ").toUtf8());
            showFile.write("\n");
        }

        showFile.write(("# " + QString::number(FacesNum) + " faces, 0 coords texture\n").toUtf8());
        showFile.write("\n");
        showFile.write("# End of File\n");
        showFile.write("\n");

        outputFile.close();
        showFile.close();
    }
    return urlSplit.join("/");
}

void QTextFieldWidget::multiVTK2Obj(QString showFileUrl)
{
    if (showFileUrl == "")
    {
        QMessageBox::warning(this, "Warning", "animating after starting a calculating mission.", QMessageBox::Ok);
        return;
    }

    // multi transvers
    // get vtk files and transvers to obj
    showFileUrl = QDir::fromNativeSeparators(showFileUrl);
    QDir outputDir(showFileUrl);
    if (!outputDir.exists())
        qDebug() << false;
    outputDir.setFilter(QDir::Files);
    outputDir.setSorting(QDir::Name);
    outputDir.setNameFilters(QString("*.vtk").split(";"));
    QStringList vtkList = outputDir.entryList();
    //qDebug() << vtkList;

    if (vtkList.size() == -1)
    {
        QMessageBox::warning(this, "Warning", "start animating after finishing a mission.", QMessageBox::Ok);
        return;
    }

    for (int i = 0; i < vtkList.size(); i++)
    {
        // multiThread
        //vtk2Obj(showFileUrl + "/" + vtkList[i]);
        QtConcurrent::run(vtk2Obj, showFileUrl + "/" + vtkList[i]);
    }

    qDebug() << "transver files finished.";
}

int QTextFieldWidget::startAnimating(QString showFileUrl)
{
    if (showFileUrl == "")
    {
        QMessageBox::warning(this, "Warning", "animating after starting a calculating mission.", QMessageBox::Ok);
        return -1;
    }

    QDir outputDir(showFileUrl);
    if (!outputDir.exists())
        qDebug() << false;
    outputDir.setFilter(QDir::Files);
    outputDir.setSorting(QDir::Name);
    outputDir.setNameFilters(QString("*.vtk").split(";"));
    QStringList vtkList = outputDir.entryList();

    if (vtkList.size() == 0)
    {
        QMessageBox::warning(this, "Warning", "animating after finished a calculating mission.", QMessageBox::Ok);
        return -1;
    }

    showFileUrl = QDir::fromNativeSeparators(showFileUrl);
    //qDebug() << showFileUrl;
    QDir objFileDir(showFileUrl + "/objFile");
    if (!objFileDir.exists())
        qDebug() << false;

    objFileDir.setFilter(QDir::Files);
    objFileDir.setSorting(QDir::Name);
    objFileDir.setNameFilters(QString("*.obj").split(";"));
    QStringList objList = objFileDir.entryList();
    //qDebug() << objList;

    // adjust data 20210327
    if (objList.size() < vtkList.size())
        multiVTK2Obj(showFileUrl);
    //qDebug() << objList.size() << "objSize()";
    //qDebug() << vtkList.size() << "vtkSize()";

    // adjust 20210621
    for (int i = 0; i < objList.size();)
    {
        //QtConcurrent::run([=]() {
        //    emit loadFileSignal(showFileUrl + "/objFile/" + objList[i]);
        //});
        //QThread::msleep(30000);
        //if (i + 9 > objList.size())
        //    emit loadFileSignal(showFileUrl + "/objFile/" + objList[objList.size() - 1]);
        //i = i + 9;

        emit loadFileSignal(showFileUrl + "/objFile/" + objList[i]);
        if (i + 9 >= objList.size())
            emit loadFileSignal(showFileUrl + "/objFile/" + objList[objList.size() - 1]);

        i = i + 9;
    }
    return 1;
}

/************************ add by HNU **************************/

QBoolFieldWidget::QBoolFieldWidget(Field* field)
    : QGroupBox()
{
    m_field           = field;
    VarField<bool>* f = TypeInfo::CastPointerDown<VarField<bool>>(m_field);
    if (f == nullptr)
    {
        return;
    }

    this->setStyleSheet("border:none");
    QGridLayout* layout = new QGridLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    this->setLayout(layout);

    QLabel* name = new QLabel();
    name->setFixedSize(160, 18);
    name->setText(FormatFieldWidgetName(field->getObjectName()));

    QCheckBox* checkbox = new QCheckBox();
    //checkbox->setFixedSize(40, 18);
    layout->addWidget(name, 0, 0);
    layout->addWidget(checkbox, 0, 1);

    connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(changeValue(int)));
    checkbox->setChecked(f->getValue());
}

void QBoolFieldWidget::changeValue(int status)
{
    VarField<bool>* f = TypeInfo::CastPointerDown<VarField<bool>>(m_field);
    if (f == nullptr)
    {
        return;
    }

    if (status == Qt::Checked)
    {
        f->setValue(true);
        f->update();
    }
    else if (status == Qt::PartiallyChecked)
    {
        //m_pLabel->setText("PartiallyChecked");
    }
    else
    {
        f->setValue(false);
        f->update();
    }

    emit fieldChanged();
}

QIntegerFieldWidget::QIntegerFieldWidget(Field* field)
    : QGroupBox()
{
    m_field          = field;
    VarField<int>* f = TypeInfo::CastPointerDown<VarField<int>>(m_field);
    if (f == nullptr)
    {
        return;
    }

    this->setStyleSheet("border:none");
    QGridLayout* layout = new QGridLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    this->setLayout(layout);

    QLabel* name = new QLabel();
    name->setFixedSize(160, 18);
    name->setText(FormatFieldWidgetName(field->getObjectName()));

    QSpinBox* spinner = new QSpinBox;
    spinner->setValue(f->getValue());

    layout->addWidget(name, 0, 0);
    layout->addWidget(spinner, 0, 1, Qt::AlignRight);

    this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
}

void QIntegerFieldWidget::changeValue(int value)
{
    VarField<int>* f = TypeInfo::CastPointerDown<VarField<int>>(m_field);
    if (f == nullptr)
    {
        return;
    }

    f->setValue(value);
    f->update();

    emit fieldChanged();
}

QRealFieldWidget::QRealFieldWidget(Field* field)
    : QGroupBox()
{
    m_field = field;

    this->setStyleSheet("border:none");
    QGridLayout* layout = new QGridLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    this->setLayout(layout);

    QLabel* name = new QLabel();
    name->setFixedSize(160, 18);
    name->setText(FormatFieldWidgetName(field->getObjectName()));

    QDoubleSlider* slider = new QDoubleSlider;
    //slider->setFixedSize(80,18);
    //slider->setRange(m_field->getMin(), m_field->getMax());
    slider->setRange(0, 1);

    QLabel* spc = new QLabel();
    spc->setFixedSize(10, 18);

    QDoubleSpinner* spinner = new QDoubleSpinner;
    spinner->setFixedSize(100, 18);
    //spinner->setRange(m_field->getMin(), m_field->getMax());
    spinner->setRange(0, 1);

    layout->addWidget(name, 0, 0);
    layout->addWidget(slider, 0, 1);
    layout->addWidget(spc, 0, 2);
    layout->addWidget(spinner, 0, 3, Qt::AlignRight);

    QObject::connect(slider, SIGNAL(valueChanged(double)), spinner, SLOT(setValue(double)));
    QObject::connect(spinner, SIGNAL(valueChanged(double)), slider, SLOT(setValue(double)));
    QObject::connect(spinner, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));

    std::string template_name = field->getTemplateName();
    if (template_name == std::string(typeid(float).name()))
    {
        VarField<float>* f = TypeInfo::CastPointerDown<VarField<float>>(m_field);
        slider->setValue(( double )f->getValue());
    }
    else if (template_name == std::string(typeid(double).name()))
    {
        VarField<double>* f = TypeInfo::CastPointerDown<VarField<double>>(m_field);
        slider->setValue(f->getValue());
    }

    FormatFieldWidgetName(field->getObjectName());
}

void QRealFieldWidget::changeValue(double value)
{
    std::string template_name = m_field->getTemplateName();

    if (template_name == std::string(typeid(float).name()))
    {
        VarField<float>* f = TypeInfo::CastPointerDown<VarField<float>>(m_field);
        f->setValue(( float )value);
        f->update();
    }
    else if (template_name == std::string(typeid(double).name()))
    {
        VarField<double>* f = TypeInfo::CastPointerDown<VarField<double>>(m_field);
        f->setValue(value);
        f->update();
    }

    emit fieldChanged();
}

QVector3FieldWidget::QVector3FieldWidget(Field* field)
    : QGroupBox()
{
    m_field = field;

    this->setStyleSheet("border:none");
    QGridLayout* layout = new QGridLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    this->setLayout(layout);

    QLabel* name = new QLabel();
    name->setFixedSize(160, 18);
    name->setText(FormatFieldWidgetName(field->getObjectName()));

    spinner1 = new QDoubleSpinner;
    spinner1->setRange(m_field->getMin(), m_field->getMax());

    spinner2 = new QDoubleSpinner;
    spinner2->setRange(m_field->getMin(), m_field->getMax());

    spinner3 = new QDoubleSpinner;
    spinner3->setRange(m_field->getMin(), m_field->getMax());

    layout->addWidget(name, 0, 0);
    layout->addWidget(spinner1, 0, 1);
    layout->addWidget(spinner2, 0, 2);
    layout->addWidget(spinner3, 0, 3);

    std::string template_name = m_field->getTemplateName();

    double v1 = 0;
    double v2 = 0;
    double v3 = 0;

    if (template_name == std::string(typeid(Vector3f).name()))
    {
        VarField<Vector3f>* f = TypeInfo::CastPointerDown<VarField<Vector3f>>(m_field);
        auto                v = f->getValue();
        v1                    = v[0];
        v2                    = v[1];
        v3                    = v[2];
    }
    else if (template_name == std::string(typeid(Vector3d).name()))
    {
        VarField<Vector3d>* f = TypeInfo::CastPointerDown<VarField<Vector3d>>(m_field);
        auto                v = f->getValue();

        v1 = v[0];
        v2 = v[1];
        v3 = v[2];
    }

    spinner1->setValue(v1);
    spinner2->setValue(v2);
    spinner3->setValue(v3);

    QObject::connect(spinner1, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));
    QObject::connect(spinner2, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));
    QObject::connect(spinner3, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));
}

void QVector3FieldWidget::changeValue(double value)
{
    double v1 = spinner1->value();
    double v2 = spinner2->value();
    double v3 = spinner3->value();

    std::string template_name = m_field->getTemplateName();

    if (template_name == std::string(typeid(Vector3f).name()))
    {
        VarField<Vector3f>* f = TypeInfo::CastPointerDown<VarField<Vector3f>>(m_field);
        f->setValue(Vector3f(( float )v1, ( float )v2, ( float )v3));
        f->update();
    }
    else if (template_name == std::string(typeid(Vector3d).name()))
    {
        VarField<Vector3d>* f = TypeInfo::CastPointerDown<VarField<Vector3d>>(m_field);
        f->setValue(Vector3d(v1, v2, v3));
        f->update();
    }

    emit fieldChanged();
}

//QWidget-->QVBoxLayout-->QScrollArea-->QWidget-->QGridLayout
PPropertyWidget::PPropertyWidget(QWidget* parent)
    : QWidget(parent)
    , m_main_layout()
{
    m_main_layout = new QVBoxLayout;
    m_scroll_area = new QScrollArea;

    m_main_layout->setContentsMargins(0, 0, 0, 0);
    m_main_layout->setSpacing(0);
    m_main_layout->addWidget(m_scroll_area);

    m_scroll_area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_scroll_area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_scroll_area->setWidgetResizable(true);

    m_scroll_layout = new QGridLayout;
    m_scroll_layout->setAlignment(Qt::AlignLeft | Qt::AlignTop);

    QWidget* m_scroll_widget = new QWidget;
    m_scroll_widget->setLayout(m_scroll_layout);

    m_scroll_area->setWidget(m_scroll_widget);

    setMinimumWidth(250);
    setLayout(m_main_layout);
}

PPropertyWidget::~PPropertyWidget()
{
    m_widgets.clear();
}

QSize PPropertyWidget::sizeHint() const
{
    return QSize(20, 20);
}

QWidget* PPropertyWidget::addWidget(QWidget* widget)
{
    m_scroll_layout->addWidget(widget);
    m_widgets.push_back(widget);

    return widget;
}

void PPropertyWidget::removeAllWidgets()
{
    //TODO: check whether m_widgets[i] should be explicitly deleted
    for (int i = 0; i < m_widgets.size(); i++)
    {
        m_scroll_layout->removeWidget(m_widgets[i]);
        delete m_widgets[i];
    }
    m_widgets.clear();
}

void PPropertyWidget::showProperty(Module* module)
{
    //        clear();

    updateContext(module);
}

void PPropertyWidget::showProperty(Node* node)
{
    //        clear();

    updateContext(node);
}

void PPropertyWidget::showBlockProperty(QtNodes::QtBlock& block)
{
    auto dataModel = block.nodeDataModel();

    auto node = dynamic_cast<QtNodes::QtNodeWidget*>(dataModel);
    if (node != nullptr)
    {
        this->showProperty(node->getNode().get());
    }
    else
    {
        auto module = dynamic_cast<QtNodes::QtModuleWidget*>(dataModel);
        if (module != nullptr)
        {
            this->showProperty(module->getModule());
        }
    }
}

void PPropertyWidget::updateDisplay()
{
    //        PVTKOpenGLWidget::getCurrentRenderer()->GetActors()->RemoveAllItems();
    SceneGraph::getInstance().draw();
    PVTKOpenGLWidget::getCurrentRenderer()->GetRenderWindow()->Render();
}

void PPropertyWidget::updateContext(Base* base)
{
    if (base == nullptr)
    {
        return;
    }

    this->removeAllWidgets();

    std::vector<Field*>& fields = base->getAllFields();

    for (Field* var : fields)
    {
        if (var != nullptr)
        {
            if (var->getClassName() == std::string("Variable"))
            {
                this->addScalarFieldWidget(var);
            }
        }
    }
}

void PPropertyWidget::addScalarFieldWidget(Field* field)
{
    std::string template_name = field->getTemplateName();
    if (template_name == std::string(typeid(bool).name()))
    {
        auto fw = new QBoolFieldWidget(field);
        this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(updateDisplay()));

        this->addWidget(fw);
    }
    else if (template_name == std::string(typeid(int).name()))
    {
        auto fw = new QIntegerFieldWidget(field);
        this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(updateDisplay()));

        this->addWidget(fw);
        //            this->addWidget(new QIntegerFieldWidget(new VarField<int>()));
    }
    else if (template_name == std::string(typeid(float).name()))
    {
        this->addWidget(new QRealFieldWidget(field));
    }
    else if (template_name == std::string(typeid(Vector3f).name()))
    {
        this->addWidget(new QVector3FieldWidget(field));
    }
    else if (template_name == std::string(typeid(std::string).name()))
    {
        //    QTextFieldWidget::m_textFieldInstance = new QTextFieldWidget(field);
        m_TextFieldWidget = new QTextFieldWidget(field);
        this->addWidget(m_TextFieldWidget);
        //this->addWidget(new QTextFieldWidget(field));
    }
}

void PPropertyWidget::addArrayFieldWidget(Field* field)
{
}

QTextFieldWidget* PPropertyWidget::getTextFieldWidget()
{
    return m_TextFieldWidget;
}

void PPropertyWidget::sendQTextFieldInitSignal()
{
    //qDebug() << "QTextFieldWidget init";
    emit QTextFieldInitSignal();
}
}  // namespace PhysIKA
