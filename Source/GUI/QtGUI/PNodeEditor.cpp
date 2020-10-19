#include "PNodeEditor.h"

#include <QHBoxLayout>

#include "PPropertyWidget.h"
#include "PModuleFlowWidget.h"

namespace PhysIKA
{
	PNodeEditor::PNodeEditor(QtNodeWidget* node_widget)
		: QWidget(nullptr)
	{
		QHBoxLayout* mainLayout = new QHBoxLayout(this);
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setSpacing(0);


		PPropertyWidget* propertyWidget = new PPropertyWidget();
		mainLayout->addWidget(propertyWidget, 0);

		PModuleFlowWidget* moduleFlowView = new PModuleFlowWidget();
		mainLayout->addWidget(moduleFlowView, 1);
	}

}
