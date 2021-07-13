#ifndef QSCENEGRAPHWIDGET_H
#define QSCENEGRAPHWIDGET_H

#include <QTreeWidget>

namespace PhysIKA {
class Node;

class PSceneGraphWidgetItem : public QTreeWidgetItem
{
public:
    explicit PSceneGraphWidgetItem(Node* node, QTreeWidget* treeview);
    explicit PSceneGraphWidgetItem(Node* node, QTreeWidgetItem* parent);

    inline Node* getNode()
    {
        return m_node;
    }

private:
    Node* m_node;
};

class PSceneGraphWidget : public QTreeWidget
{
    Q_OBJECT

public:
    PSceneGraphWidget(QWidget* parent = nullptr);

protected:
Q_SIGNALS:
    void notifyNodeSelected(Node* node);

    void notifyNodeDoubleClicked(Node* node);

public slots:
    void updateTree();
    void nodeClicked(QTreeWidgetItem* item, int index);
    void nodeDoubleClicked(QTreeWidgetItem* item, int index);
    void popMenu(const QPoint& pos);

    void nodeSelected(QTreeWidgetItem* item, int column);
};
}  // namespace PhysIKA
#endif  // QSCENEGRAPHWIDGET_H
