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

bool Module::isInitialized()
{
	return m_initialized;
}

}