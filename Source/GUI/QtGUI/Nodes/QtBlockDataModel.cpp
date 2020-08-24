#include "QtBlockDataModel.h"

#include "StyleCollection.h"

using QtNodes::QtBlockDataModel;
using QtNodes::BlockStyle;

QtBlockDataModel::
QtBlockDataModel()
  : _nodeStyle(StyleCollection::nodeStyle())
{
  // Derived classes can initialize specific style here
}


QJsonObject
QtBlockDataModel::
save() const
{
  QJsonObject modelJson;

  modelJson["name"] = name();

  return modelJson;
}


BlockStyle const&
QtBlockDataModel::
nodeStyle() const
{
  return _nodeStyle;
}


void
QtBlockDataModel::
setNodeStyle(BlockStyle const& style)
{
  _nodeStyle = style;
}
