#pragma once

#include <QtCore/QObject>
#include <QtCore/QJsonObject>
#include <QtWidgets/QLabel>

#include "QtBlockDataModel.h"
#include "Framework/Node.h"

#include <iostream>

class DecimalData;

using PhysIKA::Node;

using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::BlockData;
using QtNodes::BlockDataType;
using QtNodes::QtBlockDataModel;
using QtNodes::ValidationState;

/// The model dictates the number of inputs and outputs for the Node.
/// In this example it has no logic.
class QtNodeWidget : public QtBlockDataModel
{
	Q_OBJECT

public:
	QtNodeWidget(Node* base = nullptr);

	virtual	~QtNodeWidget() {}

public:

	QString caption() const override;

	QString name() const override;
	void setName(QString name) { m_name = name; }
	
	QString	portCaption(PortType portType, PortIndex portIndex) const override;

	QString	validationMessage() const override;


	unsigned int nPorts(PortType portType) const override;

	
	bool portCaptionVisible(PortType portType, PortIndex portIndex) const override;

	std::shared_ptr<BlockData> outData(PortIndex port) override;

	void setInData(std::shared_ptr<BlockData> data, PortIndex portIndex) override;

	BlockDataType dataType(PortType portType, PortIndex portIndex) const override;


	QWidget* embeddedWidget() override { return nullptr; }

	ValidationState validationState() const override;

	Node* getNode();

protected:
	virtual void updateModule();

protected:
	QString m_name;

	Node* m_module = nullptr;

	ValidationState modelValidationState = ValidationState::Warning;
	QString modelValidationError = QString("Missing or incorrect inputs");

private:
};

