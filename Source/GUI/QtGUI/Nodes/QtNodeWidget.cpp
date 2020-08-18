#include "QtNodeWidget.h"

#include "Framework/Node.h"


QtNodeWidget::QtNodeWidget(Node* base)
{
	m_module = base;
}

unsigned int
QtNodeWidget::nPorts(PortType portType) const
{
	unsigned int result;

	if (portType == PortType::In)
	{
		result = 0;
	}
	else
	{
		result = 0;
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
	return nullptr;
}


QString QtNodeWidget::caption() const
{
	return QString::fromStdString(m_module->getClassInfo()->getClassName());
}

QString QtNodeWidget::name() const
{
	return QString::fromStdString(m_module->getClassInfo()->getClassName());
}

bool QtNodeWidget::portCaptionVisible(PortType portType, PortIndex portIndex) const
{
	Q_UNUSED(portType); Q_UNUSED(portIndex);
	return true;
}

QString QtNodeWidget::portCaption(PortType portType, PortIndex portIndex) const
{

	return QString::fromStdString("");
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
	return m_module;
}

QString QtNodeWidget::validationMessage() const
{
	return modelValidationError;
}

void QtNodeWidget::updateModule()
{
}


