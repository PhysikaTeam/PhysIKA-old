#pragma once
#include "Framework/Framework/Node.h"

namespace PhysIKA {

	class SceneLoader
	{
	public:
		virtual std::shared_ptr<Node> load(const std::string filename) { return nullptr; }

		virtual bool canLoadFileByName(const std::string filename) {
			std::string str = filename;
			std::string::size_type pos = str.find_last_of('.');
			if (pos == std::string::npos)
				return false; // no extension

			return canLoadFileByExtension(str.substr(pos + 1));
		}

		virtual bool canLoadFileByExtension(const std::string extension) { return false; }
	};

	class SceneLoaderFactory
	{
	public:
		typedef std::vector<SceneLoader*> SceneLoaderList;

		/// Get the ObjectFactory singleton instance
		static SceneLoaderFactory& getInstance();

	public:
		/// Get an entry given a file extension
		SceneLoader* getEntryByFileExtension(std::string extension);

		/// Get an entry given a file name
		SceneLoader* getEntryByFileName(std::string filename);

		/// Add a scene loader
		SceneLoader* addEntry(SceneLoader *loader);

		/// Get the list of loaders
		SceneLoaderList* getEntryList() { return &m_loaders; }

	private:
		SceneLoaderFactory();

		/// Main class registry
		SceneLoaderList m_loaders;
	};


}
