#include "NodeIterator.h"
#include "Node.h"

namespace PhysIKA
{

NodeIterator::NodeIterator()
{
	node_current = nullptr;
}


NodeIterator::NodeIterator(Node* node)
{
	node_current = node;

	auto children = node_current->getChildren();
	for each (auto c in children)
	{
		node_stack.push(c.get());
	}
}


NodeIterator::~NodeIterator()
{

}

NodeIterator& NodeIterator::operator++()
{
	if (node_stack.empty())
		node_current = nullptr;
	else
	{
		node_current = node_stack.top();
		node_stack.pop();

		auto children = node_current->getChildren();
		for each (auto c in children)
		{
			node_stack.push(c.get());
		}
	}

	return *this;
}


NodeIterator& NodeIterator::operator++(int)
{
	return operator++();
}

Node* NodeIterator::operator->() const
{
	return node_current;
}

Node* NodeIterator::get() const
{
	return node_current;
}

bool NodeIterator::operator!=(const NodeIterator &iterator) const
{
	return node_current != iterator.get();
}

bool NodeIterator::operator==(const NodeIterator &iterator) const
{
	return node_current == iterator.get();
}


}