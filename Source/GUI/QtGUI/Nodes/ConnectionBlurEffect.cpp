#include "ConnectionBlurEffect.h"

#include "ConnectionGraphicsObject.h"
#include "ConnectionPainter.h"

using QtNodes::ConnectionBlurEffect;
using QtNodes::ConnectionGraphicsObject;

ConnectionBlurEffect::
ConnectionBlurEffect(ConnectionGraphicsObject*)
{
  //
}


void
ConnectionBlurEffect::
draw(QPainter* painter)
{
  QGraphicsBlurEffect::draw(painter);

  //ConnectionPainter::paint(painter,
  //_object->connectionGeometry(),
  //_object->connectionState());

  //_item->paint(painter, nullptr, nullptr);
}
