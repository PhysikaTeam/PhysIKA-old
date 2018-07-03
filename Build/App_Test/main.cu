
#include <stdio.h>
#include "Physika_Core/Vectors/vector_fixed.h"
#include "Framework/Base.h"
#include "Framework/Field.h"
#include "Framework/Module.h"
#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Dynamics/ParticleSystem/ParticleSystem.h"
#include <string>

int main()
{
	std::string* str = new std::string;
	*str = "zzz";

	Physika::Base b;
	std::shared_ptr< HostVariable<std::string> > var = b.allocHostVariable<std::string>("name", "description");
	var->setValue(std::string("aa"));


	std::shared_ptr<SceneGraph> scene = SceneGraph::getInstance();

	std::shared_ptr<ParticleSystem<DataType3f>> psystem =
		scene->createNewScene<ParticleSystem<DataType3f>>("root");

	psystem->initialize();

// 	Physika::Base b;
// 	std::shared_ptr<Physika::Field> f(new Physika::Field);
//	b.addField(NULL);
//	b.removeField(f);

    return 0;
}