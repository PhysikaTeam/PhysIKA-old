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

	std::vector<Node*>& getNodes() { return m_nodes; }

	virtual void addNode(Node* node) = 0;

protected:
	std::vector<Node*> m_nodes;

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

	void addNode(Node* node) override { m_nodes[0] = node; }
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

	void addNode(Node* node) override { m_nodes.push_back(node); }
};


}