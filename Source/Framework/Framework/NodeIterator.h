#pragma once
#include <stack>

namespace PhysIKA
{

	class Node;

	class NodeIterator
	{
	public:
		NodeIterator();
		NodeIterator(Node* node);

		~NodeIterator();
		
		bool operator== (const NodeIterator &iterator) const;
		bool operator!= (const NodeIterator &iterator) const;

		NodeIterator& operator++();
		NodeIterator& operator++(int);

		Node* operator->() const;

		Node* get() const;

	protected:
		Node* node_current;

		std::stack<Node*> node_stack;

		friend class Node;
	};

}

