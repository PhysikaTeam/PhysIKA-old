#pragma once

#include <QMainWindow>

#include "Nodes/QtNodeWidget.h"

namespace PhysIKA {
class PNodeEditor : public QMainWindow
{
    Q_OBJECT
public:
    PNodeEditor(QtNodes::QtNodeWidget* node_widget);

private:
};
}  // namespace PhysIKA
