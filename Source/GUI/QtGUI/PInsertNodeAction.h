#ifndef PINSERTTREENODEACTION_H
#define PINSERTTREENODEACTION_H

#include <stack>
#include "Framework/Action/Action.h"

#include <QTreeWidget>

namespace PhysIKA {
class Node;
class PSceneGraphWidgetItem;

class PInsertTreeNodeAction : public Action
{
public:
    PInsertTreeNodeAction(QTreeWidget* widget);
    virtual ~PInsertTreeNodeAction(){};

public:
    void start(Node* node) override;
    void end(Node* node) override;

private:
    QTreeWidget* m_treeWidget;

    std::stack<PSceneGraphWidgetItem*> treeItemStack;
};
}  // namespace PhysIKA

#endif  // QTREEWIDGETNODEITEM_H
