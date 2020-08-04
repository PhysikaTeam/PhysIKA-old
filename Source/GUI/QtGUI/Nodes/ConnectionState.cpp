#include "ConnectionState.h"

#include <iostream>

#include <QtCore/QPointF>

#include "ModuleFlowScene.h"
#include "QtNode.h"

using QtNodes::ConnectionState;
using QtNodes::QtNode;

ConnectionState::
~ConnectionState()
{
  resetLastHoveredNode();
}


void
ConnectionState::
interactWithNode(QtNode* node)
{
  if (node)
  {
    _lastHoveredNode = node;
  }
  else
  {
    resetLastHoveredNode();
  }
}


void
ConnectionState::
setLastHoveredNode(QtNode* node)
{
  _lastHoveredNode = node;
}


void
ConnectionState::
resetLastHoveredNode()
{
  if (_lastHoveredNode)
    _lastHoveredNode->resetReactionToConnection();

  _lastHoveredNode = nullptr;
}
