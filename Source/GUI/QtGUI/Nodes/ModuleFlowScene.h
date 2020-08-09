#pragma once

#include <QtCore/QUuid>
#include <QtWidgets/QGraphicsScene>

#include <unordered_map>
#include <tuple>
#include <functional>

#include "QUuidStdHash.h"
#include "Export.h"
#include "DataModelRegistry.h"
#include "TypeConverter.h"
#include "memory.h"

#include "Framework/Node.h"

namespace QtNodes
{

class NodeDataModel;
class FlowItemInterface;
class QtNode;
class NodeGraphicsObject;
class Connection;
class ConnectionGraphicsObject;
class NodeStyle;

using PhysIKA::Node;

/// Scene holds connections and nodes.
class NODE_EDITOR_PUBLIC ModuleFlowScene
  : public QGraphicsScene
{
  Q_OBJECT
public:

  ModuleFlowScene(std::shared_ptr<DataModelRegistry> registry,
            QObject * parent = Q_NULLPTR);

  ModuleFlowScene(QObject * parent = Q_NULLPTR);

  ~ModuleFlowScene();

public:

  std::shared_ptr<Connection>
  createConnection(PortType connectedPort,
                   QtNode& node,
                   PortIndex portIndex);

  std::shared_ptr<Connection>
  createConnection(QtNode& nodeIn,
                   PortIndex portIndexIn,
                   QtNode& nodeOut,
                   PortIndex portIndexOut,
                   TypeConverter const & converter = TypeConverter{});

  std::shared_ptr<Connection> restoreConnection(QJsonObject const &connectionJson);

  void deleteConnection(Connection& connection);

  QtNode&createNode(std::unique_ptr<NodeDataModel> && dataModel);

  QtNode&restoreNode(QJsonObject const& nodeJson);

  void removeNode(QtNode& node);

  DataModelRegistry&registry() const;

  void setRegistry(std::shared_ptr<DataModelRegistry> registry);

  void iterateOverNodes(std::function<void(QtNode*)> const & visitor);

  void iterateOverNodeData(std::function<void(NodeDataModel*)> const & visitor);

  void iterateOverNodeDataDependentOrder(std::function<void(NodeDataModel*)> const & visitor);

  QPointF getNodePosition(QtNode const& node) const;

  void setNodePosition(QtNode& node, QPointF const& pos) const;

  QSizeF getNodeSize(QtNode const& node) const;

public:

  std::unordered_map<QUuid, std::unique_ptr<QtNode> > const & nodes() const;

  std::unordered_map<QUuid, std::shared_ptr<Connection> > const & connections() const;

  std::vector<QtNode*> allNodes() const;

  std::vector<QtNode*> selectedNodes() const;

public:

  void clearScene();

  void newNode();

  void save() const;

  void load();

  QByteArray saveToMemory() const;

  void loadFromMemory(const QByteArray& data);

Q_SIGNALS:

	/**
	* @brief Node has been created but not on the scene yet.
	* @see nodePlaced()
	*/
	void nodeCreated(QtNode &n);

	/**
	* @brief Node has been added to the scene.
	* @details Connect to this signal if need a correct position of node.
	* @see nodeCreated()
	*/
	void nodePlaced(QtNode &n);

	void nodeDeleted(QtNode &n);

	void connectionCreated(Connection const &c);
	void connectionDeleted(Connection const &c);

	void nodeMoved(QtNode& n, const QPointF& newLocation);

	void nodeDoubleClicked(QtNode& n);

	void connectionHovered(Connection& c, QPoint screenPos);

	void nodeHovered(QtNode& n, QPoint screenPos);

	void connectionHoverLeft(Connection& c);

	void nodeHoverLeft(QtNode& n);

	void nodeContextMenu(QtNode& n, const QPointF& pos);

 public Q_SLOTS:
	  void showNodeFlow(Node* node);
	  void moveModulePosition(QtNode& n, const QPointF& newLocation);

private:

  using SharedConnection = std::shared_ptr<Connection>;
  using UniqueNode       = std::unique_ptr<QtNode>;

  std::unordered_map<QUuid, SharedConnection> _connections;
  std::unordered_map<QUuid, UniqueNode>       _nodes;
  std::shared_ptr<DataModelRegistry>          _registry;

  std::weak_ptr<PhysIKA::Node> m_node;

private Q_SLOTS:

  void setupConnectionSignals(Connection const& c);

  void sendConnectionCreatedToNodes(Connection const& c);
  void sendConnectionDeletedToNodes(Connection const& c);


};

QtNode*
locateNodeAt(QPointF scenePoint, ModuleFlowScene &scene,
             QTransform const & viewTransform);
}
