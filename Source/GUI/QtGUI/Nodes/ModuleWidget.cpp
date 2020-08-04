#include "ModuleWidget.h"

#include "FieldData.h"
#include "DecimalData.h"
#include "Framework/Module.h"


ModuleWidget::ModuleWidget(Module* base)
{
	m_module = base;

	if (m_module != nullptr)
	{
		//initialize out ports
		int output_num = getOutputFields().size();
		output_fields.resize(output_num);
		auto outputs = getOutputFields();
		for (int i = 0; i < outputs.size(); i++)
		{
			output_fields[i] = std::make_shared<FieldData>(outputs[i]);
		}

		//initialize in ports
		int input_num = getInputFields().size();
		input_fields.resize(input_num);
// 		auto inputs = getInputFields();
// 		for (int i = 0; i < outputs.size(); i++)
// 		{
// 			input_fields[i] = std::make_shared<FieldData>(inputs[i]);
// 		}
	}
}

unsigned int
ModuleWidget::nPorts(PortType portType) const
{
	unsigned int result;

	if (portType == PortType::In)
	{
		result = input_fields.size();
	}
	else
	{
		result = output_fields.size();
	}

	return result;
}


NodeDataType ModuleWidget::dataType(PortType portType, PortIndex portIndex) const
{
	PhysIKA::Field* f = this->getField(portType, portIndex);

	std::string name = f->getClassName();

	return NodeDataType{ name.c_str(), name.c_str() };
}


std::shared_ptr<NodeData>
ModuleWidget::outData(PortIndex port)
{
	return std::static_pointer_cast<NodeData>(output_fields[port]);
}


QString ModuleWidget::caption() const
{
	return QString::fromStdString(m_module->getClassInfo()->getClassName());
//	return m_name;
}

QString ModuleWidget::name() const
{
	return QString::fromStdString(m_module->getClassInfo()->getClassName());
}

bool ModuleWidget::portCaptionVisible(PortType portType, PortIndex portIndex) const
{
	Q_UNUSED(portType); Q_UNUSED(portIndex);
	return true;
}

QString ModuleWidget::portCaption(PortType portType, PortIndex portIndex) const
{
	PhysIKA::Field* f = this->getField(portType, portIndex);
	std::string name = f->getObjectName();

	return QString::fromStdString(name);
}

void ModuleWidget::setInData(std::shared_ptr<NodeData> data, PortIndex portIndex)
{
	auto field_port = std::dynamic_pointer_cast<FieldData>(data);

	input_fields[portIndex] = field_port;

	if (field_port != nullptr)
	{
		auto in_fields = getInputFields();
		field_port->getField()->connectPtr(in_fields[portIndex]);
	}
	
	updateModule();
}


NodeValidationState ModuleWidget::validationState() const
{
	return modelValidationState;
}

QString ModuleWidget::validationMessage() const
{
	return modelValidationError;
}

void ModuleWidget::updateModule()
{
	bool hasAllInputs = true;

	for(int i = 0; i < input_fields.size(); i++)
	{

		auto p = input_fields[i].lock();

		hasAllInputs &= (p != nullptr);
	}

	if (hasAllInputs)
	{
		modelValidationState = NodeValidationState::Valid;
		modelValidationError = QString();
	}
	else
	{
		modelValidationState = NodeValidationState::Warning;
		modelValidationError = QStringLiteral("Missing or incorrect inputs");
	}

	for (int i = 0; i < output_fields.size(); i++)
	{
		Q_EMIT dataUpdated(i);
	}
}

Field* ModuleWidget::getField(PortType portType, PortIndex portIndex) const
{
	return portType == PortType::In ? m_module->getInputFields()[portIndex] : m_module->getOutputFields()[portIndex];
}

std::vector<Field*>& ModuleWidget::getOutputFields()
{
	return m_module->getOutputFields();
}

std::vector<Field*>& ModuleWidget::getInputFields()
{
	return m_module->getInputFields();
}

