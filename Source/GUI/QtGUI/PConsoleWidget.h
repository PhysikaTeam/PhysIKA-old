#ifndef QCONSOLEWIDGET_H
#define QCONSOLEWIDGET_H

#include <QWidget>

namespace PhysIKA {

class PConsoleWidget : public QWidget
{
    Q_OBJECT
public:
    explicit PConsoleWidget(QWidget* parent = nullptr);

signals:

public slots:
};

}  // namespace PhysIKA

#endif  // QCONSOLEWIDGET_H
