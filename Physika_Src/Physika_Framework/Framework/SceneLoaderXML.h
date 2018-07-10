#pragma once
#include "Framework/Node.h"
#include "SceneLoaderFactory.h"

namespace Physika {

	class SceneLoaderXML : public SceneLoader
	{
	public:
		virtual Node* load(const std::string filename);

		virtual bool canLoadFileByExtension(const std::string extension);
	};
}
