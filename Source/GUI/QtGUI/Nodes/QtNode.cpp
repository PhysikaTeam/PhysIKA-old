#include "QtNode.h"

#include <QtCore/QObject>

#include <utility>
#include <iostream>

#include "ModuleFlowScene.h"

#include "NodeGraphicsObject.h"
#include "NodeDataModel.h"

#include "ConnectionGraphicsObject.h"
#include "ConnectionState.h"

using QtNodes::QtNode;
using QtNodes::NodeGeometry;
using QtNodes::NodeState;
using QtNodes::NodeData;
using QtNodes::NodeDataType;
using QtNodes::NodeDataModel;
using QtNodes::NodeGraphicsObject;
using QtNodes::PortIndex;
using QtNodes::PortType;

QtNode::
QtNode(std::unique_ptr<NodeDataModel> && dataModel)
  : _uid(QUuid::createUuid())
  , _nodeDataModel(std::move(dataModel))
  , _nodeState(_nodeDataModel)
  , _nodeGeometry(_nodeDataModel)
  , _nodeGraphicsObject(nullptr)
{
  _nodeGeometry.recalculateSize();

  // propagate data: model => node
  connect(_nodeDataModel.get(), &NodeDataModel::dataUpdated,
          this, &QtNode::onDataUpdated);

  connect(_nodeDataModel.get(), &NodeDataModel::embeddedWidgetSizeUpdated,
          this, &QtNode::onNodeSizeUpdated );
}


QtNode::
~QtNode() = default;

QJsonObject
QtNode::
save() const
{
  QJsonObject nodeJson;

  nodeJson["id"] = _uid.toString();

  nodeJson["model"] = _nodeDataModel->save();

  QJsonObject obj;
  obj["x"] = _nodeGraphicsObject->pos().x();
  obj["y"] = _nodeGraphicsObject->pos().y();
  nodeJson["position"] = obj;

  return nodeJson;
}


void
QtNode::
restore(QJsonObject const& json)
{
  _uid = QUuid(json["id"].toString());

  QJsonObject positionJson = json["position"].toObject();
  QPointF     point(positionJson["x"].toDouble(),
                    positionJson["y"].toDouble());
  _nodeGraphicsObject->setPos(point);

  _nodeDataModel->restore(json["model"].toObject());
}


QUuid
QtNode::
id() const
{
  return _uid;
}


void
QtNode::
reactToPossibleConnection(PortType reactingPortType,
                          NodeDataType const &reactingDataType,
                          QPointF const &scenePoint)
{
  QTransform const t = _nodeGraphicsObject->sceneTransform();

  QPointF p = t.inverted().map(scenePoint);

  _nodeGeometry.setDraggingPosition(p);

  _nodeGraphicsObject->update();

  _nodeState.setReaction(NodeState::REACTING,
                         reactingPortType,
                         reactingDataType);
}


void
QtNode::
resetReactionToConnection()
{
  _nodeState.setReaction(NodeState::NOT_REACTING);
  _nodeGraphicsObject->update();
}


NodeGraphicsObject const &
QtNode::
nodeGraphicsObject() const
{
  return *_nodeGraphicsObject.get();
}


NodeGraphicsObject &
QtNode::
nodeGraphicsObject()
{
  return *_nodeGraphicsObject.get();
}


void
QtNode::
setGraphicsObject(std::unique_ptr<NodeGraphicsObject>&& graphics)
{
  _nodeGraphicsObject = std::move(graphics);

  _nodeGeometry.recalculateSize();
}


NodeGeometry&
QtNode::
nodeGeometry()
{
  return _nodeGeometry;
}


NodeGeometry const&
QtNode::
nodeGeometry() const
{
  return _nodeGeometry;
}


NodeState const &
QtNode::
nodeState() const
{
  return _nodeState;
}


NodeState &
QtNode::
nodeState()
{
  return _nodeState;
}


NodeDataModel*
QtNode::
nodeDataModel() const
{
  return _nodeDataModel.get();
}


void
QtNode::
propagateData(std::shared_ptr<NodeData> nodeData,
              PortIndex inPortIndex) const
{
  _nodeDataModel->setInData(std::move(nodeData), inPortIndex);

  //Recalculate the nodes visuals. A data change can result in the node taking more space than before, so this forces a recalculate+repaint on the affected node
  _nodeGraphicsObject->setGeometryChanged();
  _nodeGeometry.recalculateSize();
  _nodeGraphicsObject->update();
  _nodeGraphicsObject->moveConnections();
}


void
QtNode::
onDataUpdated(PortIndex index)
{
  auto nodeData = _nodeDataModel->outData(index);

  auto connections =
    _nodeState.connections(PortType::Out, index);

  for (auto const & c : connections)
    c.second->propagateData(nodeData);
}

void
QtNode::
onNodeSizeUpdated()
{
    if( nodeDataModel()->embeddedWidget() )
    {
        nodeDataModel()->embeddedWidget()->adjustSize();
    }
    nodeGeometry().recalculateSize();
    for(PortType type: {PortType::In, PortType::Out})
    {
        for(auto& conn_set : nodeState().getEntries(type))
        {
            for(auto& pair: conn_set)
            {
                Connection* conn = pair.second;
                conn->getConnectionGraphicsObject().move();
            }
        }
    }
}
