#include "ModuleWidget.h"

#include "DecimalData.hpp"
#include "Framework/Module.h"


ModuleWidget::ModuleWidget(PhysIKA::Module* base)
{
	m_module = base;
}

unsigned int
ModuleWidget::nPorts(PortType portType) const
{
	unsigned int result;

	if (portType == PortType::In)
	{
		if (m_module == nullptr)
		{
			result = 0;
		}
		else
			result = m_module->getAllFields().size();
	}
	else
	{
		if (m_module == nullptr)
		{
			result = 0;
		}
		else
		result = m_module->getAllFields().size();
	}

	return result;
}


NodeDataType ModuleWidget::dataType(PortType portType, PortIndex portIndex) const
{
	PhysIKA::Field* f = m_module->getAllFields()[portIndex];

	std::string name = f->getClassName();

	return NodeDataType{ name.c_str(), name.c_str() };
}


std::shared_ptr<NodeData>
ModuleWidget::outData(PortIndex)
{
	return std::static_pointer_cast<NodeData>(_result);
}


void
ModuleWidget::setInData(std::shared_ptr<NodeData> data, PortIndex portIndex)
{
	auto numberData =
		std::dynamic_pointer_cast<DecimalData>(data);

	if (portIndex == 0)
	{
		_number1 = numberData;
	}
	else
	{
		_number2 = numberData;
	}

	compute();
}


NodeValidationState
ModuleWidget::validationState() const
{
	return modelValidationState;
}


QString
ModuleWidget::validationMessage() const
{
	return modelValidationError;
}

