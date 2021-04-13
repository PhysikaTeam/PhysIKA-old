#pragma once
#include "Framework/Framework/Module.h"

namespace PhysIKA
{

	class ModuleIterator
	{
	public:
		ModuleIterator();

		~ModuleIterator();

		ModuleIterator(const ModuleIterator &iterator);

		ModuleIterator& operator= (const ModuleIterator &iterator);

		bool operator== (const ModuleIterator &iterator) const;

		bool operator!= (const ModuleIterator &iterator) const;

		ModuleIterator& operator++ ();
		ModuleIterator& operator++ (int);

		std::shared_ptr<Module> operator *();

		Module* operator->();

		Module* get();

	protected:

		std::weak_ptr<Module> module;

		friend class ControllerModule;
	};

	class ControllerModule : public Module
	{
		DECLARE_CLASS(ControllerModule)
	public:
		typedef ModuleIterator Iterator;

		ControllerModule();
		virtual ~ControllerModule();

		Iterator entry();
		Iterator finished();

		unsigned int size();

		void push_back(std::weak_ptr<Module> m);

	private:
		std::weak_ptr<Module> start_module;
		std::weak_ptr<Module> current_module;
		std::weak_ptr<Module> end_module;

		unsigned int num = 0;
	};
}

