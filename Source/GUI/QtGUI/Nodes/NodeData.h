#pragma once

#include "QtBlockDataModel.h"
#include "Framework/NodePort.h"

using QtNodes::BlockDataType;
using QtNodes::BlockData;

using PhysIKA::NodePort;

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class NodeData : public BlockData
{
public:

	NodeData()
	{}

	NodeData(NodePort* n)
		: node_port(n)
	{}

	BlockDataType type() const override
	{
		return BlockDataType{ "nodeport",
							 "NodePort" };
	}

	NodePort* getNode() { return node_port; }

	bool isEmpty() { return node_port == nullptr; }

private:

	NodePort* node_port = nullptr;
};
