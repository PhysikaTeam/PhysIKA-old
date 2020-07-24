#pragma once

#include <QtCore/QObject>
#include <QtCore/QJsonObject>
#include <QtWidgets/QLabel>

#include "NodeDataModel.hpp"
#include "Framework/Module.h"

#include <iostream>

class DecimalData;

using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::NodeData;
using QtNodes::NodeDataType;
using QtNodes::NodeDataModel;
using QtNodes::NodeValidationState;

/// The model dictates the number of inputs and outputs for the Node.
/// In this example it has no logic.
class ModuleWidget : public NodeDataModel
{
	Q_OBJECT

public:
	ModuleWidget(PhysIKA::Module* base = nullptr);

	virtual	~ModuleWidget() {}

public:

	void setName(QString name) { _name = name; }

	unsigned int nPorts(PortType portType) const override;

	NodeDataType dataType(PortType portType, PortIndex portIndex) const override;

	std::shared_ptr<NodeData> outData(PortIndex port) override;

	QString caption() const override
	{
		return _name;
	}

	QString name() const override
	{
		return _name;
	}

	void setInData(std::shared_ptr<NodeData> data, PortIndex portIndex) override;

	QWidget* embeddedWidget() override { return nullptr; }

	NodeValidationState validationState() const override;

	QString	validationMessage() const override;

protected:

	virtual void compute() {};

protected:

	std::weak_ptr<DecimalData> _number1;
	std::weak_ptr<DecimalData> _number2;

	std::shared_ptr<DecimalData> _result;

	QString _name;

	PhysIKA::Module* m_module = nullptr;

	NodeValidationState modelValidationState = NodeValidationState::Warning;
	QString modelValidationError = QString("Missing or incorrect inputs");
};

