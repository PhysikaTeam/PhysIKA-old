#include "Module.h"
#include "Physika_Framework/Framework/Node.h"

namespace Physika
{

Module::Module()
	: m_node(NULL)
	, m_initialized(false)
{

}

Module::~Module(void)
{
	m_arguments.clear();
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

bool Module::isArgumentComplete()
{
	std::list<BaseSlot*>::iterator iter = m_arguments.begin();
	for (; iter != m_arguments.end(); iter++)
	{
		if ((*iter)->isInitialized())
		{
			std::cout << "Argument: " << (*iter)->getName() << " is not initialized! " << std::endl;
		}
		return false;
	}
	return true;
}

bool Module::isInitialized()
{
	return m_initialized;
}

void Module::initArgument(BaseSlot* arg, std::string name, std::string desc)
{
	arg->setName(name);
	arg->setDescription(desc);
	m_arguments.push_back(arg);
}

}