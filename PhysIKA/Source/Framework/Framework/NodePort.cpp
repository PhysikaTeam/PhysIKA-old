#include "NodePort.h"
#include "Node.h"

namespace PhysIKA
{

	NodePort::NodePort(std::string name, std::string description, Node* parent /*= nullptr*/)
		: m_name(name)
		, m_description(description)
		, m_portType(NodePortType::Unknown)
		, m_parent(parent)
	{
		parent->addNodePort(this);
	}

	NodePortType NodePort::getPortType()
	{
		return m_portType;
	}

	void NodePort::setPortType(NodePortType portType)
	{
		m_portType = portType;
	}

	void NodePort::clear()
	{
		m_nodes.clear();
	}

	bool NodePort::addNodeToParent(std::shared_ptr<Node> node)
	{
		if (!m_parent->hasChild(node))
		{
			m_parent->addChild(node);
			return true;
		}

		return false;
	}

	bool NodePort::removeNodeFromParent(std::shared_ptr<Node> node)
	{
		if (m_parent->hasChild(node))
		{
			m_parent->removeChild(node);
			return true;
		}

		return false;
	}

}

