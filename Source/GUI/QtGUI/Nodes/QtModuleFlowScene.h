#pragma once

#include "QtFlowScene.h"

#include "Framework/Node.h"

namespace QtNodes
{

using PhysIKA::Node;

/// Scene holds connections and nodes.
class NODE_EDITOR_PUBLIC QtModuleFlowScene
  : public QtFlowScene
{
	Q_OBJECT
public:

	QtModuleFlowScene(std::shared_ptr<DataModelRegistry> registry,
			QObject * parent = Q_NULLPTR);

	QtModuleFlowScene(QObject * parent = Q_NULLPTR);


	~QtModuleFlowScene();



public Q_SLOTS:
	void showNodeFlow(Node* node);
	void moveModulePosition(QtBlock& n, const QPointF& newLocation);


private:
	std::weak_ptr<PhysIKA::Node> m_node;
};

}
