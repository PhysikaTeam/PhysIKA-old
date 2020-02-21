#include "Module.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA
{

Module::Module(std::string name)
	: m_node(nullptr)
	, m_initialized(false)
{
//	attachField(&m_module_name, "module_name", "Module name", false);

//	m_module_name.setValue(name);
	m_module_name = name;
}

Module::~Module(void)
{

}

bool Module::initialize()
{
	if (m_initialized)
	{
		return true;
	}
	m_initialized = initializeImpl();

	return m_initialized;
}

void Module::setName(std::string name)
{
	//m_module_name.setValue(name);
	m_module_name = name;
}

void Module::setParent(Node* node)
{
	m_node = node;
}

std::string Module::getName()
{
	return m_module_name;
}

bool Module::isInitialized()
{
	return m_initialized;
}

bool Module::initializeImpl()
{
	if (m_node == nullptr)
	{
		Log::sendMessage(Log::Warning, "Parent is not set");
		return false;
	}

	return true;
}

}