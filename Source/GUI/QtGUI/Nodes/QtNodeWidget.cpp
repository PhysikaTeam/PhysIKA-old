#include "QtNodeWidget.h"

#include "Framework/Node.h"
#include "Framework/NodePort.h"

#include "NodeData.h"

QtNodeWidget::QtNodeWidget(Node* base)
{
	m_node = base;

	if (m_node != nullptr)
	{
		//initialize in node ports
		int input_num = m_node->getAllNodePorts().size();

		input_nodes.resize(input_num);
		auto inputs = m_node->getAllNodePorts();
		for (int i = 0; i < inputs.size(); i++)
		{
			input_nodes[i] = std::make_shared<NodeData>(inputs[i]);
		}

		//initialize out node ports
		output_nodes.resize(1);
		output_nodes[0]  = std::make_shared<NodeData>(nullptr);
	}

}

unsigned int
QtNodeWidget::nPorts(PortType portType) const
{
	unsigned int result;

	if (portType == PortType::In)
	{
		result = m_node->getAllNodePorts().size();
	}
	else
	{
		result = 1;
	}

	return result;
}


BlockDataType QtNodeWidget::dataType(PortType portType, PortIndex portIndex) const
{
	return BlockDataType{ "", "" };
}


std::shared_ptr<BlockData>
QtNodeWidget::outData(PortIndex port)
{
	return std::static_pointer_cast<BlockData>(output_nodes[0]);;
}


QString QtNodeWidget::caption() const
{
	return QString::fromStdString(m_node->getClassInfo()->getClassName());
}

QString QtNodeWidget::name() const
{
	return QString::fromStdString(m_node->getClassInfo()->getClassName());
}

bool QtNodeWidget::portCaptionVisible(PortType portType, PortIndex portIndex) const
{
	Q_UNUSED(portType); Q_UNUSED(portIndex);
	return true;
}

QString QtNodeWidget::portCaption(PortType portType, PortIndex portIndex) const
{
	if (portType == PortType::In)
	{
		return QString::fromStdString(m_node->getAllNodePorts()[portIndex]->getPortName());
	}
	else
	{
		return QString::fromStdString(m_node->getClassInfo()->getClassName());
	}
}

void QtNodeWidget::setInData(std::shared_ptr<BlockData> data, PortIndex portIndex)
{
	
	updateModule();
}


ValidationState QtNodeWidget::validationState() const
{
	return modelValidationState;
}

Node* QtNodeWidget::getNode()
{
	return m_node;
}

QString QtNodeWidget::validationMessage() const
{
	return modelValidationError;
}

void QtNodeWidget::updateModule()
{
}


