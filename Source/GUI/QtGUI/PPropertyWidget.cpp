#include "PPropertyWidget.h"
#include "Framework/Module.h"
#include "Framework/Node.h"

namespace PhysIKA
{

	PPropertyWidget::PPropertyWidget(QWidget *parent) :
		QListWidget(parent)
	{
		setMinimumWidth(200);
	}

// 	void PPropertyWidget::clear()
// 	{
// 
// 	}

	void PPropertyWidget::showProperty(Module* module)
	{
		clear();

		updateContext(module);
	}

	void PPropertyWidget::showProperty(Node* node)
	{
		clear();

		updateContext(node);
	}

	void PPropertyWidget::updateContext(Base* base)
	{
		if (base == nullptr)
		{
			return;
		}

		std::vector<Field*>& fields = base->getAllFields();

		for each (Field* var in fields)
		{
			if (var != nullptr)
			{
				addItem(new QListWidgetItem(var->getObjectName().c_str(), this));
			}
		}
	}

}
