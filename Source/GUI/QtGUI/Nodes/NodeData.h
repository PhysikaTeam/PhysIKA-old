#pragma once

#include "QtBlockDataModel.h"
#include "Framework/Node.h"

using QtNodes::BlockDataType;
using QtNodes::BlockData;

using PhysIKA::Node;

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class NodeData : public BlockData
{
public:

	NodeData()
	{}

	NodeData(Node* n)
		: node(n)
	{}

	BlockDataType type() const override
	{
		return BlockDataType{ "node",
							 "Node" };
	}

	Node* getNode() { return node; }

	bool isEmpty() { return node == nullptr; }

private:

	Node* node = nullptr;
};
