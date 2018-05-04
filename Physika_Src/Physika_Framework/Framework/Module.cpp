#include "Module.h"

namespace Physika
{
IMPLEMENT_CLASS(Module)

Module::Module()
{

}

Module::~Module(void)
{
	m_ins.clear();
	m_outs.clear();
}

}