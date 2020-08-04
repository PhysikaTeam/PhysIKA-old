#pragma once

#include <QtCore/QObject>
#include <QtCore/QUuid>
#include <QtCore/QVariant>

#include "PortType.h"
#include "NodeData.h"

#include "Serializable.h"
#include "ConnectionState.h"
#include "ConnectionGeometry.h"
#include "TypeConverter.h"
#include "QUuidStdHash.h"
#include "Export.h"
#include "memory.h"

class QPointF;

namespace QtNodes
{

class QtNode;
class NodeData;
class ConnectionGraphicsObject;

///
class NODE_EDITOR_PUBLIC Connection
  : public QObject
  , public Serializable
{

  Q_OBJECT

public:

  /// New Connection is attached to the port of the given Node.
  /// The port has parameters (portType, portIndex).
  /// The opposite connection end will require anothre port.
  Connection(PortType portType,
             QtNode& node,
             PortIndex portIndex);

  Connection(QtNode& nodeIn,
             PortIndex portIndexIn,
             QtNode& nodeOut,
             PortIndex portIndexOut,
             TypeConverter converter =
               TypeConverter{});

  Connection(const Connection&) = delete;
  Connection operator=(const Connection&) = delete;

  ~Connection();

public:

  QJsonObject
  save() const override;

public:

  QUuid
  id() const;

  /// Remembers the end being dragged.
  /// Invalidates Node address.
  /// Grabs mouse.
  void
  setRequiredPort(PortType portType);
  PortType
  requiredPort() const;

  void
  setGraphicsObject(std::unique_ptr<ConnectionGraphicsObject>&& graphics);

  /// Assigns a node to the required port.
  /// It is assumed that there is a required port, no extra checks
  void
  setNodeToPort(QtNode& node,
                PortType portType,
                PortIndex portIndex);

  void
  removeFromNodes() const;

public:

  ConnectionGraphicsObject&
  getConnectionGraphicsObject() const;

  ConnectionState const &
  connectionState() const;
  ConnectionState&
  connectionState();

  ConnectionGeometry&
  connectionGeometry();

  ConnectionGeometry const&
  connectionGeometry() const;

  QtNode*
  getNode(PortType portType) const;

  QtNode*&
  getNode(PortType portType);

  PortIndex
  getPortIndex(PortType portType) const;

  void
  clearNode(PortType portType);

  NodeDataType
  dataType(PortType portType) const;

  void
  setTypeConverter(TypeConverter converter);

  bool
  complete() const;

public: // data propagation

  void
  propagateData(std::shared_ptr<NodeData> nodeData) const;

  void
  propagateEmptyData() const;

Q_SIGNALS:

  void
  connectionCompleted(Connection const&) const;

  void
  connectionMadeIncomplete(Connection const&) const;

private:

  QUuid _uid;

private:

  QtNode* _outNode = nullptr;
  QtNode* _inNode  = nullptr;

  PortIndex _outPortIndex;
  PortIndex _inPortIndex;

private:

  ConnectionState    _connectionState;
  ConnectionGeometry _connectionGeometry;

  std::unique_ptr<ConnectionGraphicsObject>_connectionGraphicsObject;

  TypeConverter _converter;

Q_SIGNALS:

  void
  updated(Connection& conn) const;
};
}
