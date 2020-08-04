#include "ModuleFlowScene.h"

#include <stdexcept>
#include <utility>

#include <QtWidgets/QGraphicsSceneMoveEvent>
#include <QtWidgets/QFileDialog>
#include <QtCore/QByteArray>
#include <QtCore/QBuffer>
#include <QtCore/QDataStream>
#include <QtCore/QFile>

#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <QtCore/QtGlobal>
#include <QtCore/QDebug>

#include "QtNode.h"
#include "NodeGraphicsObject.h"

#include "NodeGraphicsObject.h"
#include "ConnectionGraphicsObject.h"

#include "Connection.h"

#include "Framework/Object.h"
#include "Framework/SceneGraph.h"
#include "ModuleWidget.h"

#include "GUI/QtGUI/PVTKSurfaceMeshRender.h"
#include "GUI/QtGUI/PVTKPointSetRender.h"

#include "Framework/Framework/SceneGraph.h"
#include "TestParticleElasticBody.h"



#include "FlowView.h"
#include "DataModelRegistry.h"

using QtNodes::ModuleFlowScene;
using QtNodes::QtNode;
using QtNodes::NodeGraphicsObject;
using QtNodes::Connection;
using QtNodes::DataModelRegistry;
using QtNodes::NodeDataModel;
using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::TypeConverter;


ModuleFlowScene::ModuleFlowScene(std::shared_ptr<DataModelRegistry> registry,
          QObject * parent)
  : QGraphicsScene(parent)
  , _registry(std::move(registry))
{
  setItemIndexMethod(QGraphicsScene::NoIndex);

  // This connection should come first
  connect(this, &ModuleFlowScene::connectionCreated, this, &ModuleFlowScene::setupConnectionSignals);
  connect(this, &ModuleFlowScene::connectionCreated, this, &ModuleFlowScene::sendConnectionCreatedToNodes);
  connect(this, &ModuleFlowScene::connectionDeleted, this, &ModuleFlowScene::sendConnectionDeletedToNodes);
}

ModuleFlowScene::ModuleFlowScene(QObject * parent)
	: ModuleFlowScene(std::make_shared<DataModelRegistry>(), parent)
{
	auto classMap = PhysIKA::Object::getClassMap();

	auto ret = std::make_shared<QtNodes::DataModelRegistry>();
	int id = 0;
	for (auto const c : *classMap)
	{
		id++;

		QString str = QString::fromStdString(c.first);
		PhysIKA::Object* obj = PhysIKA::Object::createObject(str.toStdString());
		PhysIKA::Module* module = dynamic_cast<PhysIKA::Module*>(obj);

		if (module != nullptr)
		{
			QtNodes::DataModelRegistry::RegistryItemCreator creator = [str, module]() {
				auto dat = std::make_unique<ModuleWidget>(module);
				dat->setName(str);
			return dat; };

			QString category = QString::fromStdString(module->getModuleType());
			ret->registerModel<ModuleWidget>(category, creator);
		}
	}

	_registry = std::move(ret);
}


ModuleFlowScene::~ModuleFlowScene()
{
  clearScene();
}


//------------------------------------------------------------------------------

std::shared_ptr<Connection>
ModuleFlowScene::createConnection(PortType connectedPort,
                 QtNode& node,
                 PortIndex portIndex)
{
  auto connection = std::make_shared<Connection>(connectedPort, node, portIndex);

  auto cgo = detail::make_unique<ConnectionGraphicsObject>(*this, *connection);

  // after this function connection points are set to node port
  connection->setGraphicsObject(std::move(cgo));

  _connections[connection->id()] = connection;

  // Note: this connection isn't truly created yet. It's only partially created.
  // Thus, don't send the connectionCreated(...) signal.

  connect(connection.get(),
          &Connection::connectionCompleted,
          this,
          [this](Connection const& c) {
            connectionCreated(c);
          });

  return connection;
}


std::shared_ptr<Connection>
ModuleFlowScene::createConnection(QtNode& nodeIn,
									 PortIndex portIndexIn,
									 QtNode& nodeOut,
									 PortIndex portIndexOut,
									 TypeConverter const &converter)
{
  auto connection =
    std::make_shared<Connection>(nodeIn,
                                 portIndexIn,
                                 nodeOut,
                                 portIndexOut,
                                 converter);

  auto cgo = detail::make_unique<ConnectionGraphicsObject>(*this, *connection);

  nodeIn.nodeState().setConnection(PortType::In, portIndexIn, *connection);
  nodeOut.nodeState().setConnection(PortType::Out, portIndexOut, *connection);

  // after this function connection points are set to node port
  connection->setGraphicsObject(std::move(cgo));

  // trigger data propagation
  nodeOut.onDataUpdated(portIndexOut);

  _connections[connection->id()] = connection;

  connectionCreated(*connection);

  return connection;
}


std::shared_ptr<Connection>
ModuleFlowScene::
restoreConnection(QJsonObject const &connectionJson)
{
  QUuid nodeInId  = QUuid(connectionJson["in_id"].toString());
  QUuid nodeOutId = QUuid(connectionJson["out_id"].toString());

  PortIndex portIndexIn  = connectionJson["in_index"].toInt();
  PortIndex portIndexOut = connectionJson["out_index"].toInt();

  auto nodeIn  = _nodes[nodeInId].get();
  auto nodeOut = _nodes[nodeOutId].get();

  auto getConverter = [&]()
  {
    QJsonValue converterVal = connectionJson["converter"];

    if (!converterVal.isUndefined())
    {
      QJsonObject converterJson = converterVal.toObject();

      NodeDataType inType { converterJson["in"].toObject()["id"].toString(),
                            converterJson["in"].toObject()["name"].toString() };

      NodeDataType outType { converterJson["out"].toObject()["id"].toString(),
                             converterJson["out"].toObject()["name"].toString() };

      auto converter  =
        registry().getTypeConverter(outType, inType);

      if (converter)
        return converter;
    }

    return TypeConverter{};
  };

  std::shared_ptr<Connection> connection =
    createConnection(*nodeIn, portIndexIn,
                     *nodeOut, portIndexOut,
                     getConverter());

  // Note: the connectionCreated(...) signal has already been sent
  // by createConnection(...)

  return connection;
}


void
ModuleFlowScene::
deleteConnection(Connection& connection)
{
  auto it = _connections.find(connection.id());
  if (it != _connections.end()) {
    connection.removeFromNodes();
    _connections.erase(it);
  }
}


QtNode&
ModuleFlowScene::
createNode(std::unique_ptr<NodeDataModel> && dataModel)
{
  auto node = detail::make_unique<QtNode>(std::move(dataModel));
  auto ngo  = detail::make_unique<NodeGraphicsObject>(*this, *node);

  node->setGraphicsObject(std::move(ngo));

  auto nodePtr = node.get();
  _nodes[node->id()] = std::move(node);

  nodeCreated(*nodePtr);
  return *nodePtr;
}


QtNode&
ModuleFlowScene::
restoreNode(QJsonObject const& nodeJson)
{
  QString modelName = nodeJson["model"].toObject()["name"].toString();

  auto dataModel = registry().create(modelName);

  if (!dataModel)
    throw std::logic_error(std::string("No registered model with name ") +
                           modelName.toLocal8Bit().data());

  auto node = detail::make_unique<QtNode>(std::move(dataModel));
  auto ngo  = detail::make_unique<NodeGraphicsObject>(*this, *node);
  node->setGraphicsObject(std::move(ngo));

  node->restore(nodeJson);

  auto nodePtr = node.get();
  _nodes[node->id()] = std::move(node);

  nodePlaced(*nodePtr);
  nodeCreated(*nodePtr);
  return *nodePtr;
}


void
ModuleFlowScene::
removeNode(QtNode& node)
{
  // call signal
  nodeDeleted(node);

  for(auto portType: {PortType::In,PortType::Out})
  {
    auto nodeState = node.nodeState();
    auto const & nodeEntries = nodeState.getEntries(portType);

    for (auto &connections : nodeEntries)
    {
      for (auto const &pair : connections)
        deleteConnection(*pair.second);
    }
  }

  _nodes.erase(node.id());
}


DataModelRegistry&
ModuleFlowScene::
registry() const
{
  return *_registry;
}


void
ModuleFlowScene::
setRegistry(std::shared_ptr<DataModelRegistry> registry)
{
  _registry = std::move(registry);
}


void
ModuleFlowScene::
iterateOverNodes(std::function<void(QtNode*)> const & visitor)
{
  for (const auto& _node : _nodes)
  {
    visitor(_node.second.get());
  }
}


void
ModuleFlowScene::
iterateOverNodeData(std::function<void(NodeDataModel*)> const & visitor)
{
  for (const auto& _node : _nodes)
  {
    visitor(_node.second->nodeDataModel());
  }
}


void
ModuleFlowScene::
iterateOverNodeDataDependentOrder(std::function<void(NodeDataModel*)> const & visitor)
{
  std::set<QUuid> visitedNodesSet;

  //A leaf node is a node with no input ports, or all possible input ports empty
  auto isNodeLeaf =
    [](QtNode const &node, NodeDataModel const &model)
    {
      for (unsigned int i = 0; i < model.nPorts(PortType::In); ++i)
      {
        auto connections = node.nodeState().connections(PortType::In, i);
        if (!connections.empty())
        {
          return false;
        }
      }

      return true;
    };

  //Iterate over "leaf" nodes
  for (auto const &_node : _nodes)
  {
    auto const &node = _node.second;
    auto model       = node->nodeDataModel();

    if (isNodeLeaf(*node, *model))
    {
      visitor(model);
      visitedNodesSet.insert(node->id());
    }
  }

  auto areNodeInputsVisitedBefore =
    [&](QtNode const &node, NodeDataModel const &model)
    {
      for (size_t i = 0; i < model.nPorts(PortType::In); ++i)
      {
        auto connections = node.nodeState().connections(PortType::In, i);

        for (auto& conn : connections)
        {
          if (visitedNodesSet.find(conn.second->getNode(PortType::Out)->id()) == visitedNodesSet.end())
          {
            return false;
          }
        }
      }

      return true;
    };

  //Iterate over dependent nodes
  while (_nodes.size() != visitedNodesSet.size())
  {
    for (auto const &_node : _nodes)
    {
      auto const &node = _node.second;
      if (visitedNodesSet.find(node->id()) != visitedNodesSet.end())
        continue;

      auto model = node->nodeDataModel();

      if (areNodeInputsVisitedBefore(*node, *model))
      {
        visitor(model);
        visitedNodesSet.insert(node->id());
      }
    }
  }
}


QPointF
ModuleFlowScene::
getNodePosition(const QtNode& node) const
{
  return node.nodeGraphicsObject().pos();
}


void
ModuleFlowScene::
setNodePosition(QtNode& node, const QPointF& pos) const
{
  node.nodeGraphicsObject().setPos(pos);
  node.nodeGraphicsObject().moveConnections();
}


QSizeF
ModuleFlowScene::
getNodeSize(const QtNode& node) const
{
  return QSizeF(node.nodeGeometry().width(), node.nodeGeometry().height());
}


std::unordered_map<QUuid, std::unique_ptr<QtNode> > const &
ModuleFlowScene::
nodes() const
{
  return _nodes;
}


std::unordered_map<QUuid, std::shared_ptr<Connection> > const &
ModuleFlowScene::
connections() const
{
  return _connections;
}


std::vector<QtNode*>
ModuleFlowScene::
allNodes() const
{
  std::vector<QtNode*> nodes;

  std::transform(_nodes.begin(),
                 _nodes.end(),
                 std::back_inserter(nodes),
                 [](std::pair<QUuid const, std::unique_ptr<QtNode>> const & p) { return p.second.get(); });

  return nodes;
}


std::vector<QtNode*>
ModuleFlowScene::
selectedNodes() const
{
  QList<QGraphicsItem*> graphicsItems = selectedItems();

  std::vector<QtNode*> ret;
  ret.reserve(graphicsItems.size());

  for (QGraphicsItem* item : graphicsItems)
  {
    auto ngo = qgraphicsitem_cast<NodeGraphicsObject*>(item);

    if (ngo != nullptr)
    {
      ret.push_back(&ngo->node());
    }
  }

  return ret;
}


//------------------------------------------------------------------------------

void
ModuleFlowScene::
clearScene()
{
  //Manual node cleanup. Simply clearing the holding datastructures doesn't work, the code crashes when
  // there are both nodes and connections in the scene. (The data propagation internal logic tries to propagate
  // data through already freed connections.)
  while (_connections.size() > 0)
  {
    deleteConnection( *_connections.begin()->second );
  }

  while (_nodes.size() > 0)
  {
    removeNode( *_nodes.begin()->second );
  }
}


void
ModuleFlowScene::
save() const
{
  QString fileName =
    QFileDialog::getSaveFileName(nullptr,
                                 tr("Open Flow Scene"),
                                 QDir::homePath(),
                                 tr("Flow Scene Files (*.flow)"));

  if (!fileName.isEmpty())
  {
    if (!fileName.endsWith("flow", Qt::CaseInsensitive))
      fileName += ".flow";

    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly))
    {
      file.write(saveToMemory());
    }
  }
}


void
ModuleFlowScene::
load()
{
  clearScene();

  //-------------

  QString fileName =
    QFileDialog::getOpenFileName(nullptr,
                                 tr("Open Flow Scene"),
                                 QDir::homePath(),
                                 tr("Flow Scene Files (*.flow)"));

  if (!QFileInfo::exists(fileName))
    return;

  QFile file(fileName);

  if (!file.open(QIODevice::ReadOnly))
    return;

  QByteArray wholeFile = file.readAll();

  loadFromMemory(wholeFile);
}


QByteArray
ModuleFlowScene::
saveToMemory() const
{
  QJsonObject sceneJson;

  QJsonArray nodesJsonArray;

  for (auto const & pair : _nodes)
  {
    auto const &node = pair.second;

    nodesJsonArray.append(node->save());
  }

  sceneJson["nodes"] = nodesJsonArray;

  QJsonArray connectionJsonArray;
  for (auto const & pair : _connections)
  {
    auto const &connection = pair.second;

    QJsonObject connectionJson = connection->save();

    if (!connectionJson.isEmpty())
      connectionJsonArray.append(connectionJson);
  }

  sceneJson["connections"] = connectionJsonArray;

  QJsonDocument document(sceneJson);

  return document.toJson();
}


void
ModuleFlowScene::
loadFromMemory(const QByteArray& data)
{
  QJsonObject const jsonDocument = QJsonDocument::fromJson(data).object();

  QJsonArray nodesJsonArray = jsonDocument["nodes"].toArray();

  for (QJsonValueRef node : nodesJsonArray)
  {
    restoreNode(node.toObject());
  }

  QJsonArray connectionJsonArray = jsonDocument["connections"].toArray();

  for (QJsonValueRef connection : connectionJsonArray)
  {
    restoreConnection(connection.toObject());
  }
}


void
ModuleFlowScene::
setupConnectionSignals(Connection const& c)
{
  connect(&c,
          &Connection::connectionMadeIncomplete,
          this,
          &ModuleFlowScene::connectionDeleted,
          Qt::UniqueConnection);
}


void
ModuleFlowScene::
sendConnectionCreatedToNodes(Connection const& c)
{
  QtNode* from = c.getNode(PortType::Out);
  QtNode* to   = c.getNode(PortType::In);

  Q_ASSERT(from != nullptr);
  Q_ASSERT(to != nullptr);

  from->nodeDataModel()->outputConnectionCreated(c);
  to->nodeDataModel()->inputConnectionCreated(c);
}


void
ModuleFlowScene::
sendConnectionDeletedToNodes(Connection const& c)
{
  QtNode* from = c.getNode(PortType::Out);
  QtNode* to   = c.getNode(PortType::In);

  Q_ASSERT(from != nullptr);
  Q_ASSERT(to != nullptr);

  from->nodeDataModel()->outputConnectionDeleted(c);
  to->nodeDataModel()->inputConnectionDeleted(c);
}


//------------------------------------------------------------------------------
namespace QtNodes
{

QtNode*
locateNodeAt(QPointF scenePoint, ModuleFlowScene &scene,
             QTransform const & viewTransform)
{
  // items under cursor
  QList<QGraphicsItem*> items =
    scene.items(scenePoint,
                Qt::IntersectsItemShape,
                Qt::DescendingOrder,
                viewTransform);

  //// items convertable to NodeGraphicsObject
  std::vector<QGraphicsItem*> filteredItems;

  std::copy_if(items.begin(),
               items.end(),
               std::back_inserter(filteredItems),
               [] (QGraphicsItem * item)
    {
      return (dynamic_cast<NodeGraphicsObject*>(item) != nullptr);
    });

  QtNode* resultNode = nullptr;

  if (!filteredItems.empty())
  {
    QGraphicsItem* graphicsItem = filteredItems.front();
    auto ngo = dynamic_cast<NodeGraphicsObject*>(graphicsItem);

    resultNode = &ngo->node();
  }

  return resultNode;
}


void ModuleFlowScene::newNode()
{
 	PhysIKA::SceneGraph& scene = PhysIKA::SceneGraph::getInstance();
	auto root = scene.getRootNode();

	root->removeAllChildren();

	std::shared_ptr<PhysIKA::TestParticleElasticBody<PhysIKA::DataType3f>> bunny = std::make_shared<PhysIKA::TestParticleElasticBody<PhysIKA::DataType3f>>();
	root->addChild(bunny);
	//	bunny->getRenderModule()->setColor(Vector3f(0, 1, 1));
	bunny->setMass(1.0);
	bunny->loadParticles("../../Media/bunny/bunny_points.obj");
	bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
	bunny->translate(PhysIKA::Vector3f(0.5, 0.2, 0.5));
	bunny->setVisible(true);

// 	auto renderer = std::make_shared<PhysIKA::PVTKSurfaceMeshRender>();
// 	renderer->setName("VTK Mesh Renderer");
// 	bunny->getSurfaceNode()->addVisualModule(renderer);

	auto pRenderer = std::make_shared<PhysIKA::PVTKPointSetRender>();
	pRenderer->setName("VTK Point Renderer");
	bunny->addVisualModule(pRenderer);

	scene.invalid();
	scene.initialize();

	auto mlist = bunny->getModuleList();

	auto c = bunny->getAnimationPipeline()->entry();

	std::map<std::string, QtNode*> moduleMap;

	int mSize = bunny->getAnimationPipeline()->size();

	for (; c != bunny->getAnimationPipeline()->finished(); c++)
	{
		auto module_name = c->getName();

		auto type = std::make_unique<ModuleWidget>(c.get());

		auto& node = this->createNode(std::move(type));

		moduleMap[module_name] = &node;

//		QPoint pos = event->pos();

		QPointF posView;

		node.nodeGraphicsObject().setPos(posView);

		this->nodePlaced(node);
	}

	c = bunny->getAnimationPipeline()->entry();
	for (; c != bunny->getAnimationPipeline()->finished(); c++)
	{
		auto out_node = moduleMap[c->getName()];

		auto fields = c->getOutputFields();

		for (int i = 0; i < fields.size(); i++)
		{
			auto sink_fields = fields[i]->getSinkFields();
			for (int j = 0; j < sink_fields.size(); j++)
			{
				auto in_module = dynamic_cast<Module*>(sink_fields[j]->getParent());
				if (in_module != nullptr)
				{
					auto in_fields = in_module->getInputFields();

					int in_port = -1;
					for (int t = 0; t < in_fields.size(); t++)
					{
						if (sink_fields[j] == in_fields[t])
						{
							in_port = t;
							break;
						}
					}

					if (in_port != -1)
					{
						auto in_node = moduleMap[in_module->getName()];

						createConnection(*in_node, in_port, *out_node, i);
					}
				}
			}
		}
	}
}

}
