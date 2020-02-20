#include "PModuleListWidget.h"
#include "Framework/Node.h"
#include "Framework/Module.h"

namespace PhysIKA
{	
	PModuleListItem::PModuleListItem(Module* module, QListWidget *listview) :
		QListWidgetItem(listview),
		m_module(module)
	{
		if (module != nullptr)
		{
			setText(m_module->getName().c_str());
		}
	}


	PModuleListWidget::PModuleListWidget()
	{

	}



	void PModuleListWidget::updateModule(Node* node)
	{
		clear();

		if (node == nullptr)
		{
			return;
		}

		auto& mlist = node->getModuleList();

		for each (std::shared_ptr<Module> m in mlist)
		{
			addItem(new PModuleListItem(m.get(), this)); 
			
		}
	}

}

