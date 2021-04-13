#include "BlockStyle.h"

#include <iostream>

#include <QtCore/QFile>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonValueRef>
#include <QtCore/QJsonArray>

#include <QDebug>

#include "StyleCollection.h"

using QtNodes::BlockStyle;

inline void initResources() { Q_INIT_RESOURCE(resources); }

BlockStyle::
BlockStyle()
{
  // Explicit resources inialization for preventing the static initialization
  // order fiasco: https://isocpp.org/wiki/faq/ctors#static-init-order
  initResources();

  // This configuration is stored inside the compiled unit and is loaded statically
  loadJsonFile(":DefaultStyle.json");
}


BlockStyle::
BlockStyle(QString jsonText)
{
  loadJsonText(jsonText);
}


void
BlockStyle::
setNodeStyle(QString jsonText)
{
  BlockStyle style(jsonText);


  StyleCollection::setNodeStyle(style);
}


#ifdef STYLE_DEBUG
  #define NODE_STYLE_CHECK_UNDEFINED_VALUE(v, variable) { \
      if (v.type() == QJsonValue::Undefined || \
          v.type() == QJsonValue::Null) \
        qWarning() << "Undefined value for parameter:" << #variable; \
  }
#else
  #define NODE_STYLE_CHECK_UNDEFINED_VALUE(v, variable)
#endif

#define NODE_STYLE_READ_COLOR(values, variable)  { \
    auto valueRef = values[#variable]; \
    NODE_STYLE_CHECK_UNDEFINED_VALUE(valueRef, variable) \
    if (valueRef.isArray()) { \
      auto colorArray = valueRef.toArray(); \
      std::vector<int> rgb; rgb.reserve(3); \
      for (auto it = colorArray.begin(); it != colorArray.end(); ++it) { \
        rgb.push_back((*it).toInt()); \
      } \
      variable = QColor(rgb[0], rgb[1], rgb[2]); \
    } else { \
      variable = QColor(valueRef.toString()); \
    } \
}

#define NODE_STYLE_READ_FLOAT(values, variable)  { \
    auto valueRef = values[#variable]; \
    NODE_STYLE_CHECK_UNDEFINED_VALUE(valueRef, variable) \
    variable = valueRef.toDouble(); \
}

void
BlockStyle::
loadJsonFile(QString styleFile)
{
  QFile file(styleFile);

  if (!file.open(QIODevice::ReadOnly))
  {
    qWarning() << "Couldn't open file " << styleFile;

    return;
  }

  loadJsonFromByteArray(file.readAll());
}


void
BlockStyle::
loadJsonText(QString jsonText)
{
  loadJsonFromByteArray(jsonText.toUtf8());
}


void
BlockStyle::
loadJsonFromByteArray(QByteArray const &byteArray)
{
  QJsonDocument json(QJsonDocument::fromJson(byteArray));

  QJsonObject topLevelObject = json.object();

  QJsonValueRef nodeStyleValues = topLevelObject["NodeStyle"];

  QJsonObject obj = nodeStyleValues.toObject();

  NODE_STYLE_READ_COLOR(obj, NormalBoundaryColor);
  NODE_STYLE_READ_COLOR(obj, SelectedBoundaryColor);
  NODE_STYLE_READ_COLOR(obj, GradientColor0);
  NODE_STYLE_READ_COLOR(obj, GradientColor1);
  NODE_STYLE_READ_COLOR(obj, GradientColor2);
  NODE_STYLE_READ_COLOR(obj, GradientColor3);
  NODE_STYLE_READ_COLOR(obj, ShadowColor);
  NODE_STYLE_READ_COLOR(obj, FontColor);
  NODE_STYLE_READ_COLOR(obj, FontColorFaded);
  NODE_STYLE_READ_COLOR(obj, ConnectionPointColor);
  NODE_STYLE_READ_COLOR(obj, FilledConnectionPointColor);
  NODE_STYLE_READ_COLOR(obj, WarningColor);
  NODE_STYLE_READ_COLOR(obj, ErrorColor);

  NODE_STYLE_READ_FLOAT(obj, PenWidth);
  NODE_STYLE_READ_FLOAT(obj, HoveredPenWidth);
  NODE_STYLE_READ_FLOAT(obj, ConnectionPointDiameter);

  NODE_STYLE_READ_FLOAT(obj, Opacity);
}
