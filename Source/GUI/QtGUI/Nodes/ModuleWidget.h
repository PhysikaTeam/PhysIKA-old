#pragma once

#include <QtCore/QObject>
#include <QtCore/QJsonObject>
#include <QtWidgets/QLabel>

#include "NodeDataModel.h"
#include "Framework/Module.h"

#include <iostream>

class DecimalData;
class FieldData;

using PhysIKA::Module;
using PhysIKA::Field;

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
	ModuleWidget(Module* base = nullptr);

	virtual	~ModuleWidget() {}

public:

	QString caption() const override;

	QString name() const override;
	void setName(QString name) { m_name = name; }
	
	QString	portCaption(PortType portType, PortIndex portIndex) const override;

	QString	validationMessage() const override;


	unsigned int nPorts(PortType portType) const override;

	
	bool portCaptionVisible(PortType portType, PortIndex portIndex) const override;

	std::shared_ptr<NodeData> outData(PortIndex port) override;

	void setInData(std::shared_ptr<NodeData> data, PortIndex portIndex) override;

	NodeDataType dataType(PortType portType, PortIndex portIndex) const override;


	QWidget* embeddedWidget() override { return nullptr; }

	NodeValidationState validationState() const override;

	Module* getModule();

protected:
	virtual void updateModule();

protected:

	using OutFieldPtr = std::vector<std::shared_ptr<FieldData>>;
	using InFieldPtr = std::vector<std::weak_ptr<FieldData>>;

	InFieldPtr input_fields;
	OutFieldPtr output_fields;
	
	QString m_name;

	Module* m_module = nullptr;

	NodeValidationState modelValidationState = NodeValidationState::Warning;
	QString modelValidationError = QString("Missing or incorrect inputs");

private:

	Field* getField(PortType portType, PortIndex portIndex) const;

	std::vector<Field*>& getOutputFields();
	std::vector<Field*>& getInputFields();
};

