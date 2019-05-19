/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the demonstration applications of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/
#include "QLichtWidget.h"
#include "QtWindow.h"
#include "DockWidget.h"
#include "ToolBar.h"

#include <QAction>
#include <QLayout>
#include <QMenu>
#include <QMenuBar>
#include <QStatusBar>
#include <QTextEdit>
#include <QFile>
#include <QDataStream>
#include <QFileDialog>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QSignalMapper>
#include <QApplication>
#include <QPainter>
#include <QMouseEvent>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QDebug>
#include <QtWidgets/QOpenGLWidget>

// #include "Node/NodeData.hpp"
// #include "Node/FlowScene.hpp"
// #include "Node/FlowView.hpp"
// #include "Node/FlowViewStyle.hpp"
// #include "Node/ConnectionStyle.hpp"
// #include "Node/DataModelRegistry.hpp"

//#include "models.h"

Q_DECLARE_METATYPE(QDockWidget::DockWidgetFeatures)

QtWindow::QtWindow(QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
{
    setObjectName("MainWindow");
    setWindowTitle("Physika");

	setCentralView();
    setupToolBar();
    setupMenuBar();
    setupDockWidgets();

	statusBar()->showMessage(tr("Status Bar"));
}

void QtWindow::mainLoop()
{
}

void QtWindow::createWindow(int width, int height)
{

}

void QtWindow::newScene()
{
	QMessageBox::StandardButton reply;

	reply = QMessageBox::question(this, "Save", "Do you want to save your changes?",
		QMessageBox::Ok | QMessageBox::Cancel);
}

void QtWindow::setCentralView()
{
// 	QtNodes::FlowViewStyle::setStyle(
// 		R"(
// 		  {
// 			"FlowViewStyle": {
// 			  "BackgroundColor": [255, 255, 240],
// 			  "FineGridColor": [245, 245, 230],
// 			  "CoarseGridColor": [235, 235, 220]
// 			}
// 		  }
// 		  )");
// 
// 	QtNodes::NodeStyle::setNodeStyle(
// 		R"(
// 			{
// 				"NodeStyle": {
// 					"NormalBoundaryColor": "darkgray",
// 					"SelectedBoundaryColor": "deepskyblue",
// 					"GradientColor0": "mintcream",
// 					"GradientColor1": "mintcream",
// 					"GradientColor2": "mintcream",
// 					"GradientColor3": "mintcream",
// 					"ShadowColor": [200, 200, 200],
// 					"FontColor": [10, 10, 10],
// 					"FontColorFaded": [100, 100, 100],
// 					"ConnectionPointColor": "white",
// 					"PenWidth": 2.0,
// 					"HoveredPenWidth": 2.5,
// 					"ConnectionPointDiameter": 10.0,
// 					"Opacity": 1.0
// 				}
// 			}
// 			)");
// 
// 	QtNodes::ConnectionStyle::setConnectionStyle(
// 		R"(
// 		  {
// 			"ConnectionStyle": {
// 			  "ConstructionColor": "gray",
// 			  "NormalColor": "black",
// 			  "SelectedColor": "gray",
// 			  "SelectedHaloColor": "deepskyblue",
// 			  "HoveredColor": "deepskyblue",
// 
// 			  "LineWidth": 3.0,
// 			  "ConstructionLineWidth": 2.0,
// 			  "PointDiameter": 10.0,
// 
// 			  "UseDataDefinedColors": false
// 			}
// 		  }
// 		  )");

// 

// 	auto ret = std::make_shared<QtNodes::DataModelRegistry>();

// 	ret->registerModel<MyDataModel>(

	QTabWidget* tabWidget = new QTabWidget();
	tabWidget->setObjectName(QStringLiteral("tabWidget"));
	tabWidget->setGeometry(QRect(140, 60, 361, 241));
	QWidget* tabView = new QWidget();
	tabView->setObjectName(QStringLiteral("tabView"));
	tabWidget->addTab(tabView, QString());
	QWidget* tabEditor = new QWidget();
	tabEditor->setObjectName(QStringLiteral("tabEditor"));
	tabWidget->addTab(tabEditor, QString());

	tabWidget->setTabText(tabWidget->indexOf(tabView), QApplication::translate("MainWindow", "View", Q_NULLPTR));
	tabWidget->setTabText(tabWidget->indexOf(tabEditor), QApplication::translate("MainWindow", "Node Editor", Q_NULLPTR));

	QGridLayout * layout_1 = new QGridLayout(tabView);
	layout_1->setMargin(0);
	openGLWidget = new QLichtWidget(tabView);
	//QOpenGLWidget *openGLWidget = new QOpenGLWidget(tabView);
//	OpenglWidget* openGLWidget = new OpenglWidget(tabView);
//	openGLWidget = new QWidget(tabView);
	openGLWidget->setObjectName(QStringLiteral("openGLWidget"));
	openGLWidget->initIrrlicht();
	layout_1->addWidget(openGLWidget, 0, 0, 1, 1);

//	openGLWidget->initialize();
// 	app = std::shared_ptr<App>(new App(
// 		G3D::GApp::Settings()));

// 	openGLWidget->pushLoopBody(app.get());


	QGridLayout * layout_2 = new QGridLayout(tabEditor);
	layout_2->setMargin(0);
// 	QtNodes::FlowScene* scene = new QtNodes::FlowScene(ret);
// 	QtNodes::FlowView* view = new QtNodes::FlowView(scene);

// 	layout_2->addWidget(view, 0, 0, 1, 1);


	setCentralWidget(tabWidget);
}

void QtWindow::setupToolBar()
{
	ToolBar *tb = new ToolBar(tr("Tool Bar"), this);
	toolBars.append(tb);
	addToolBar(tb);
}

void QtWindow::setupMenuBar()
{
    QMenu *menu = menuBar()->addMenu(tr("&File"));

	menu->addAction(tr("New ..."), this, &QtWindow::newScene);
    menu->addAction(tr("Load ..."), this, &QtWindow::loadScene);
	menu->addAction(tr("Save ..."), this, &QtWindow::saveScene);

    menu->addSeparator();
    menu->addAction(tr("&Quit"), this, &QWidget::close);

    mainWindowMenu = menuBar()->addMenu(tr("&View"));
	mainWindowMenu->addAction(tr("FullScreen"), this, &QtWindow::fullScreen);

#ifdef Q_OS_OSX
    toolBarMenu->addSeparator();

    action = toolBarMenu->addAction(tr("Unified"));
    action->setCheckable(true);
    action->setChecked(unifiedTitleAndToolBarOnMac());
    connect(action, &QAction::toggled, this, &QMainWindow::setUnifiedTitleAndToolBarOnMac);
#endif

    windowMenu = menuBar()->addMenu(tr("&Window"));
	for (int i = 0; i < toolBars.count(); ++i)
		windowMenu->addMenu(toolBars.at(i)->toolbarMenu());

	aboutMenu = menuBar()->addMenu(tr("&Help"));
	aboutMenu->addAction(tr("Show Help ..."), this, &QtWindow::showHelp);
	aboutMenu->addAction(tr("About ..."), this, &QtWindow::showAbout);
}

void QtWindow::saveScene()
{
	return;
}

void QtWindow::fullScreen()
{
	return;
}

void QtWindow::showHelp()
{
	return;
}

void QtWindow::showAbout()
{
	QMessageBox::about(this, tr("PhysLab"), tr("PhysLab 1.0"));
	return;
}

void QtWindow::loadScene()
{
    return;
}

void QtWindow::setupDockWidgets()
{
    qRegisterMetaType<QDockWidget::DockWidgetFeatures>();

    windowMenu->addSeparator();

    static const struct Set {
        const char * name;
        uint flags;
        Qt::DockWidgetArea area;
    } sets [] = {
        { "White", 0, Qt::LeftDockWidgetArea },
        { "Blue", 0, Qt::BottomDockWidgetArea },
//        { "Yellow", 0, Qt::BottomDockWidgetArea }
    };
    const int setCount = sizeof(sets) / sizeof(Set);

    const QIcon qtIcon(QPixmap(":/res/qt.png"));
    for (int i = 0; i < setCount; ++i) {
        DockWidget *swatch = new DockWidget(tr(sets[i].name), this, Qt::WindowFlags(sets[i].flags));
        if (i % 2)
            swatch->setWindowIcon(qtIcon);
//         if (qstrcmp(sets[i].name, "Blue") == 0) {
//             BlueTitleBar *titlebar = new BlueTitleBar(swatch);
//             swatch->setTitleBarWidget(titlebar);
//             connect(swatch, &QDockWidget::topLevelChanged, titlebar, &BlueTitleBar::updateMask);
//             connect(swatch, &QDockWidget::featuresChanged, titlebar, &BlueTitleBar::updateMask, Qt::QueuedConnection);
//         }

        addDockWidget(sets[i].area, swatch);
        windowMenu->addMenu(swatch->colorSwatchMenu());
    }
}

void QtWindow::mousePressEvent(QMouseEvent *event)
{
// 	QLichtThread* m_thread = new QLichtThread(openGLWidget->winId());
// 	m_thread->start();
}
