#pragma once
#include <stack>
#include <memory>

namespace PhysIKA {

class Node;

class NodeIterator
{
public:
    NodeIterator();
    NodeIterator(std::shared_ptr<Node> node);

    ~NodeIterator();

    bool operator==(const NodeIterator& iterator) const;
    bool operator!=(const NodeIterator& iterator) const;

    NodeIterator& operator++();
    NodeIterator& operator++(int);

    std::shared_ptr<Node> operator->() const;

    std::shared_ptr<Node> get() const;

protected:
    std::shared_ptr<Node> node_current;

    std::stack<std::shared_ptr<Node>> node_stack;

    friend class Node;
};

}  // namespace PhysIKA
