#include "QtNodeFlowScene.h"

#include "QtBlock.h"
#include "QtFlowView.h"
#include "QtNodeWidget.h"

#include "Framework/Object.h"
#include "Framework/SceneGraph.h"

#include "DataModelRegistry.h"


namespace QtNodes
{

QtNodeFlowScene::QtNodeFlowScene(std::shared_ptr<DataModelRegistry> registry, QObject * parent)
  : QtFlowScene(registry, parent)
{
	connect(this, &QtFlowScene::nodeMoved, this, &QtNodeFlowScene::moveModulePosition);
}

QtNodeFlowScene::QtNodeFlowScene(QObject * parent)
	: QtFlowScene(parent)
{
	auto classMap = PhysIKA::Object::getClassMap();

	auto ret = std::make_shared<QtNodes::DataModelRegistry>();
	int id = 0;
	for (auto const c : *classMap)
	{
		id++;

		QString str = QString::fromStdString(c.first);
		auto obj = PhysIKA::Object::createObject(str.toStdString());
		auto module = dynamic_cast<PhysIKA::Node*>(obj);

		if (module != nullptr)
		{
			QtNodes::DataModelRegistry::RegistryItemCreator creator = [str, module]() {
				auto dat = std::make_unique<QtNodeWidget>(module);
				dat->setName(str);
				return dat; };

			QString category = "Default";// QString::fromStdString(module->getModuleType());
			ret->registerModel<QtNodeWidget>(category, creator);
		}
	}

	this->setRegistry(ret);
}


QtNodeFlowScene::~QtNodeFlowScene()
{

}


void QtNodeFlowScene::showSceneGraph(SceneGraph* node)
{
	
}

void QtNodeFlowScene::moveModulePosition(QtBlock& n, const QPointF& newLocation)
{

}

}
