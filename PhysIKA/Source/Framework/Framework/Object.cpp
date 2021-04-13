#include <map>
#include "Object.h"

namespace PhysIKA
{
static std::map< std::string, ClassInfo*> *classInfoMap = NULL;

IMPLEMENT_CLASS(Object)
	
bool Object::registerClass(ClassInfo* ci)
{
	if (!classInfoMap) {
		classInfoMap = new std::map< std::string, ClassInfo*>();
	}
	if (ci) {
		if (classInfoMap->find(ci->m_className) == classInfoMap->end()) {
			classInfoMap->insert(std::map< std::string, ClassInfo*>::value_type(ci->m_className, ci));
		}
	}
	return true;
}
Object* Object::createObject(std::string name)
{
	std::map< std::string, ClassInfo*>::const_iterator iter = classInfoMap->find(name);
	if (classInfoMap->end() != iter) {
		return iter->second->createObject();
	}
	return NULL;
}

std::map< std::string, ClassInfo*>* Object::getClassMap()
{
	return classInfoMap;
}

bool Register(ClassInfo* ci)
{
	return Object::registerClass(ci);
}
}
