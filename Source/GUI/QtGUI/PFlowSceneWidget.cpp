#include "PFlowSceneWidget.h"

//QT
#include <QGridLayout>
#include <QVBoxLayout>
#include <QMenuBar>

//Qt Nodes
#include "Nodes/FlowView.h"
#include "Nodes/DataModelRegistry.h"
#include "Nodes/AdditionModel.h"

namespace PhysIKA
{
	PFlowSceneWidget::PFlowSceneWidget(QWidget *parent) :
		QWidget(parent)
	{
		auto menuBar = new QMenuBar();
		auto newAction = menuBar->addAction("New..");
		auto saveAction = menuBar->addAction("Save..");
		auto loadAction = menuBar->addAction("Load..");
		auto clearAction = menuBar->addAction("Clear..");

		QVBoxLayout *l = new QVBoxLayout(this);

		l->addWidget(menuBar);
		scene = new ModuleFlowScene(this);
		l->addWidget(new QtNodes::FlowView(scene));
		l->setContentsMargins(0, 0, 0, 0);
		l->setSpacing(0);

		QObject::connect(newAction, &QAction::triggered,
			scene, &ModuleFlowScene::newNode);

		QObject::connect(saveAction, &QAction::triggered,
			scene, &ModuleFlowScene::save);

		QObject::connect(loadAction, &QAction::triggered,
			scene, &ModuleFlowScene::load);

		QObject::connect(clearAction, &QAction::triggered,
			scene, &ModuleFlowScene::clearScene);
	}

	PFlowSceneWidget::~PFlowSceneWidget()
	{
	}
}