#pragma once

#include <QPainter>

#include "NodeGeometry.h"
#include "NodeDataModel.h"
#include "Export.h"

namespace QtNodes {

/// Class to allow for custom painting
class NODE_EDITOR_PUBLIC NodePainterDelegate
{

public:

  virtual
  ~NodePainterDelegate() = default;

  virtual void
  paint(QPainter* painter,
        NodeGeometry const& geom,
        NodeDataModel const * model) = 0;
};
}
