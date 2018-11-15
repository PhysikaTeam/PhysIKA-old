#pragma once
#include "Framework/Node.h"
#include "SceneLoaderFactory.h"
#include "Physika_Dependency/tinyxml/tinyxml2.h"
using namespace tinyxml2;

namespace Physika {

	class SceneLoaderXML : public SceneLoader
	{
	public:
		std::shared_ptr<Node> load(const std::string filename) override;

	private:
		std::shared_ptr<Node> processNode(XMLElement* nodeXML);
		std::shared_ptr<Module> processModule(XMLElement* moduleXML);
		bool addModule(std::shared_ptr<Node> node, std::shared_ptr<Module> module);

		std::vector<std::string> split(std::string str, std::string pattern);

		virtual bool canLoadFileByExtension(const std::string extension);
	};
}
