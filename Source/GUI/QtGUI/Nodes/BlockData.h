#pragma once

#include <QtCore/QString>

#include "Export.h"

namespace QtNodes
{

struct BlockDataType
{
  QString id;
  QString name;
};

/// Class represents data transferred between nodes.
/// @param type is used for comparing the types
/// The actual data is stored in subtypes
class NODE_EDITOR_PUBLIC BlockData
{
public:

  virtual ~BlockData() = default;

  virtual bool sameType(BlockData const &nodeData) const
  {
    return (this->type().id == nodeData.type().id);
  }

  /// Type for inner use
  virtual BlockDataType type() const = 0;
};
}
