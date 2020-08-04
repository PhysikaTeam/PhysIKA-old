#pragma once

#include "NodeDataModel.h"
#include "Framework/Field.h"

using QtNodes::NodeDataType;
using QtNodes::NodeData;

using PhysIKA::Field;

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class FieldData : public NodeData
{
public:

	FieldData()
	{}

	FieldData(Field* f)
		: field(f)
	{}

	NodeDataType type() const override
	{
		return NodeDataType{ "decimal",
							 "Decimal" };
	}

	Field* getField() { return field; }

	bool isEmpty() { return field == nullptr; }

private:

	Field* field = nullptr;
};
