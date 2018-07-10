#include "SceneLoaderXML.h"
#include <algorithm>
#include "Physika_Dependency/tinyxml/tinyxml2.h"


namespace Physika
{
	Physika::Node* SceneLoaderXML::load(const std::string filename)
	{
		return NULL;
	}

	bool SceneLoaderXML::canLoadFileByExtension(const std::string extension)
	{
		std::string str = extension;
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);
		return (extension == "xml");
	}

}