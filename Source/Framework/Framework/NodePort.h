#pragma once
#include <string>
#include <vector>
#include <memory>

namespace PhysIKA {

	class Node;


	enum NodePortType
	{
		Single,
		Multiple,
		Unknown
	};

/*!
*	\class	NodePort
*	\brief	Input ports for Node.
*/
class NodePort
{
public:
	NodePort(std::string name, std::string description, Node* parent = nullptr);
	virtual ~NodePort() {};

	virtual std::string getPortName() { return m_name; };

	NodePortType getPortType();

	void setPortType(NodePortType portType);

	virtual std::vector<std::weak_ptr<Node>>& getNodes() { return m_nodes; }

	virtual void addNode(std::weak_ptr<Node> node) = 0;

protected:
	std::vector<std::weak_ptr<Node>> m_nodes;

private:

	std::string m_name;
	std::string m_description;
	NodePortType m_portType;
};


template<typename T>
class SingleNodePort : NodePort
{
public:
	SingleNodePort(std::string name, std::string description, Node* parent = nullptr)
		: NodePort(name, description, parent)
	{
		this->setPortType(NodePortType::Single);
		this->getNodes().resize(1);
	};
	~SingleNodePort() {};

	void addNode(std::weak_ptr<Node> node) override { m_nodes[0] = node; }
};


template<typename T>
class MultipleNodePort : NodePort
{
public:
	MultipleNodePort(std::string name, std::string description, Node* parent = nullptr)
		: NodePort(name, description, parent) 
	{
		this->setPortType(NodePortType::Multiple);
	};

	~MultipleNodePort() {};

	void addNode(std::weak_ptr<Node> node) override { 
		auto d_node = std::dynamic_pointer_cast<T>(node.lock());
		if (d_node != nullptr)
		{
			m_derived_nodes.push_back(d_node);
		}
	}

	std::vector<std::weak_ptr<Node>>& getNodes() override
	{
		m_nodes.resize(m_derived_nodes.size());
		for (int i = 0; i < m_nodes.size(); i++)
		{
			m_nodes[i] = std::dynamic_pointer_cast<Node>(m_derived_nodes[i].lock());
		}
		return m_nodes;
	}

	inline std::vector<std::weak_ptr<T>>& getDerivedNodes()
	{
		return m_derived_nodes;
	}
private:
	std::vector<std::weak_ptr<T>> m_derived_nodes;
};


}