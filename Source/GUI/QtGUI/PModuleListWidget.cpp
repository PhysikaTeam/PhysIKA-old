#include "PModuleListWidget.h"
#include "Framework/Node.h"
#include "Framework/Module.h"

namespace PhysIKA {
PModuleListItem::PModuleListItem(Module* module, QListWidget* listview)
    : QListWidgetItem(listview), m_module(module)
{
    if (module != nullptr)
    {
        setText(m_module->getName().c_str());
    }
}

PModuleListWidget::PModuleListWidget(QWidget* parent)
    : QListWidget(parent)
{
    setContextMenuPolicy(Qt::CustomContextMenu);

    connect(this, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(moduleSelected(QListWidgetItem*)));
}

void PModuleListWidget::updateModule(Node* node)
{
    clear();

    if (node == nullptr)
    {
        return;
    }

    auto& mlist = node->getModuleList();

    for (std::shared_ptr<Module> m : mlist)
    {
        addItem(new PModuleListItem(m.get(), this));
    }
}

void PModuleListWidget::moduleSelected(QListWidgetItem* item)
{
    PModuleListItem* nodeItem = dynamic_cast<PModuleListItem*>(item);
    if (nodeItem != nullptr)
        emit notifyModuleSelected(nodeItem->getModule());
}

}  // namespace PhysIKA
