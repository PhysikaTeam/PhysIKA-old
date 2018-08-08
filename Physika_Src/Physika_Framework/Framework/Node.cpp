#include "Node.h"
#include "Action/Action.h"
#include "Framework/DeviceContext.h"
#include "Framework/Module.h"
#include "Framework/ControllerAnimation.h"
#include "Framework/ControllerRender.h"

namespace Physika
{
Node::Node(std::string name)
{
	setName(name);

	m_active = this->allocHostVariable<bool>("active", "this is a variable!");	m_active->setValue(true);
	m_visible = this->allocHostVariable<bool>("visible", "this is a variable!"); m_visible->setValue(true);
	m_time = this->allocHostVariable<float>("time", "this is a variable!"); m_time->setValue(0.0f);
		
	m_dt = 0.001f;

	m_parent = NULL;
	m_context = NULL;
}

Node::~Node()
{
}

void Node::setName(std::string name)
{
	if (m_node_name == nullptr)
	{
		m_node_name = this->allocHostVariable<std::string>("node_name", "Node name");
	}
	m_node_name->setValue(name);
}

std::string Node::getName()
{
	return m_node_name->getValue();
}


Node* Node::getChild(std::string name)
{
	for (ListPtr<Node>::iterator it = m_children.begin(); it != m_children.end(); ++it)
	{
		if ((*it)->getName() == name)
			return it->get();
	}
	return NULL;
}

Node* Node::getParent()
{
	return m_parent;
}

Node* Node::getRoot()
{
	Node* root = this;
	while (root->getParent() != NULL)
	{
		root = root->getParent();
	}
	return root;
}

bool Node::isActive()
{
	return m_active->getValue();
}

void Node::setActive(bool active)
{
	m_active->setValue(active);
}

bool Node::isVisible()
{
	return m_visible->getValue();
}

void Node::setVisible(bool visible)
{
	m_visible->setValue(visible);
}

float Node::getTime()
{
	return m_time->getValue();
}

float Node::getDt()
{
	return m_dt;
}

void Node::removeChild(std::shared_ptr<Node> child)
{
	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); )
	{
		if (*iter == child)
		{
			m_children.erase(iter++);
		}
		else
		{
			++iter;
		}
	}
}

std::shared_ptr<DeviceContext> Node::getContext()
{
	if (m_context == nullptr)
	{
		m_context = TypeInfo::New<DeviceContext>();
	}
	return m_context;
}

bool Node::addModule(std::string name, Module* module)
{
	if (getContext() == nullptr || module == NULL)
	{
		std::cout << "Context or module does not exist!" << std::endl;
		return false;
	}

	std::map<std::string, Module*>::iterator found = m_modules.find(name);
	if (found != m_modules.end())
	{
		std::cout << "Module name already exists!" << std::endl;
		return false;
	}
	else
	{
		m_modules[name] = module;
		m_module_list.push_back(module);

		module->insertToNode(this);
	}

	return true;
}

bool Node::addModule(Module* module)
{
	std::list<Module*>::iterator found = std::find(m_module_list.begin(), m_module_list.end(), module);
	if (found == m_module_list.end())
	{
		m_module_list.push_back(module);
		module->setParent(this);
		return true;
	}

	return false;
}

bool Node::deleteModule(Module* module)
{
	return false;
}

void Node::doTraverse(Action* act)
{
	if (!this->isActive())	return;

	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); iter++)
	{
		(*iter)->traverse(act);
	}

	act->Process(this);
}

void Node::traverse(Action* act)
{
	doTraverse(act);
}

void Node::setAsCurrentContext()
{
	getContext()->enable();
}

Module* Node::getModule(std::string name)
{
	std::map<std::string, Module*>::iterator result = m_modules.find(name);
	if (result == m_modules.end())
	{
		return NULL;
	}

	return result->second;
}

bool Node::execute(std::string name)
{
	Module* module = getModule(name);
	if (module == NULL)
	{
		return false;
	}

	return module->execute();
}

}