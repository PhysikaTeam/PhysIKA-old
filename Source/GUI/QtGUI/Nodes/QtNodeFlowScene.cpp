#include "QtNodeFlowScene.h"

#include "QtBlock.h"
#include "QtFlowView.h"
#include "QtNodeWidget.h"

#include "Framework/Object.h"
#include "Framework/NodeIterator.h"
#include "Framework/NodePort.h"
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
		std::shared_ptr<Node> node(dynamic_cast<PhysIKA::Node*>(obj));

		if (node != nullptr)
		{
			QtNodes::DataModelRegistry::RegistryItemCreator creator = [str, node]() {
				auto dat = std::make_unique<QtNodeWidget>(std::move(node));
				dat->setName(str);
				return dat; };

			QString category = "Default";// QString::fromStdString(module->getModuleType());
			ret->registerModel<QtNodeWidget>(category, creator);
		}
	}

	this->setRegistry(ret);

	PhysIKA::SceneGraph& scn = PhysIKA::SceneGraph::getInstance();
	showSceneGraph(&scn);
}


QtNodeFlowScene::~QtNodeFlowScene()
{

}


void QtNodeFlowScene::showSceneGraph(SceneGraph* scn)
{
	std::map<std::string, QtBlock*> nodeMap;

	auto root = scn->getRootNode();

	auto addNodeWidget = [&](std::shared_ptr<Node> m) -> void
	{
		auto module_name = m->getName();

		auto type = std::make_unique<QtNodeWidget>(m);

		auto& node = this->createNode(std::move(type));

		nodeMap[module_name] = &node;

		QPointF posView(m->bx(), m->by());

		node.nodeGraphicsObject().setPos(posView);

		this->nodePlaced(node);
	};

	SceneGraph::Iterator it_end(nullptr);

	for (auto it = scn->begin(); it != it_end; it++)
	{
		addNodeWidget(it.get());
	}

	auto createNodeConnections = [&](std::shared_ptr<Node> nd) -> void
	{
		auto in_name = nd->getName();
		
		if (nodeMap.find(in_name) != nodeMap.end())
		{
			auto in_block = nodeMap[nd->getName()];

			auto ports = nd->getAllNodePorts();

			for (int i = 0; i < ports.size(); i++)
			{
				PhysIKA::NodePortType pType = ports[i]->getPortType();
				if (PhysIKA::Single == pType)
				{
					auto node = ports[i]->getNodes()[0];
					if (node != nullptr)
					{
						auto in_block = nodeMap[node->getName()];
						createConnection(*in_block, 0, *in_block, i);
					}
				}
				else if (PhysIKA::Multiple == pType)
				{
					auto nodes = ports[i]->getNodes();
					for (int j = 0; j < nodes.size(); j++)
					{
						if (nodes[j] != nullptr)
						{
							auto out_name = nodes[j]->getName();
							if (nodeMap.find(out_name) != nodeMap.end())
							{
								auto out_block = nodeMap[nodes[j]->getName()];
								createConnection(*in_block, i, *out_block, 0);
							}
							
						}
					}
				}
			}
		}
	};

	for (auto it = scn->begin(); it != it_end; it++)
	{
		createNodeConnections(it.get());
	}

}

void QtNodeFlowScene::moveModulePosition(QtBlock& n, const QPointF& newLocation)
{

}

}
