#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

//#include "Dynamics/Sand/swe/SandSimulator.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"

#include "demoParticleSand.h"
#include "demoSandRigid.h"
#include <iostream>
using namespace std;
using namespace PhysIKA;



int main()
{
	//DemoSSESand demo;
	//demo.createScene();
	//demo.run();





	//DemoParticleSand* demo = DemoParticleSand::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoParticleSandRigid_Sphere* demo = DemoParticleSandRigid_Sphere::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoParticleSandSlop* demo = DemoParticleSandSlop::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoParticleSandSlide* demo = DemoParticleSandSlide::getInstance();
	//demo->createScene();
	//demo->run();


	//DemoParticleSandSlide2* demo = DemoParticleSandSlide2::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoParticleSandLand* demo = DemoParticleSandLand::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoParticleSandLand2* demo = DemoParticleSandLand2::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoParticleSandMultiRigid* demo = DemoParticleSandMultiRigid::getInstance();
	//demo->createScene();
	//demo->run();


	//DemoParticleSandPile* demo = DemoParticleSandPile::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoParticleAvalanche* demo = DemoParticleAvalanche::getInstance();
	//demo->createScene();
	//demo->run();


	DemoParticleRiver* demo = DemoParticleRiver::getInstance();
	demo->createScene();
	demo->run();


	// **********************

	//DemoHeightFieldSand* demo = DemoHeightFieldSand::getInstance();
	//demo->createScene();
	//demo->run();


	//DemoHeightFieldSandRigid_Sphere* demo = DemoHeightFieldSandRigid_Sphere::getInstance();
	//demo->createScene();
	//demo->run();


	//DemoHeightFieldSandLandRigid* demo = DemoHeightFieldSandLandRigid::getInstance();
	//demo->createScene();
	//demo->run();


	//DemoHeightFieldSandSlide* demo = DemoHeightFieldSandSlide::getInstance();
	//demo->createScene();
	//demo->run();


	//DemoHeightFieldSandLandMultiRigid* demo = DemoHeightFieldSandLandMultiRigid::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoHeightFieldSandLandMultiRigid2* demo = DemoHeightFieldSandLandMultiRigid2::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoHeightFieldSandLandMultiRigidTest* demo = DemoHeightFieldSandLandMultiRigidTest::getInstance();
	//demo->createScene();
	//demo->run();

	//DemoHeightFieldSandValley* demo = DemoHeightFieldSandValley::getInstance();
	//demo->createScene();
	//demo->run();

	return 0;
}


