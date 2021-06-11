#include "rigid_body_demo.h"


#include "Dynamics/RigidBody/RigidUtil.h"
#include "demoCar.h"
#include "demoPBD.h"





int main()
{
	std::cout << "Input to choose a simulation scene: " << std::endl;
	std::cout << "   0:  Normal 4 wheels car." << std::endl;
	std::cout << "   1:  Tank car." << std::endl;


	int examp = 0;
	std::cin >> examp;

	switch (examp)
	{
	case 1:
	{
		// Tank car.
		DemoTankCar* demo1 = DemoTankCar::getInstance();
		demo1->build(true);
		break;
	}
	default:
	{
		// Normal 4 wheel car.
		DemoCar2* demo0 = DemoCar2::getInstance();
		demo0->build(false);
		break;
	}
	}

	//testMatInverse();

	//demoLoadFile();

	//demo_PrismaticJoint();
	//demo_PlanarJoint();

	//demo_middleAxis();

	//demo_SphericalJoint();

	//demo_MultiRigid<1>();


	//DemoPBDPositionConstraint demo(100, true);
	//demo.run();

	//DemoPBDRotationConstraint demo(10, true);
	//demo.run();

	//DemoPBDCommonRigid demo(3, true);
	//demo.run();

	//DemoCar2* demo = DemoCar2::getInstance();
	////root->addChild(demo);
	//demo->build();

	//DemoPBDCar* demo = DemoPBDCar::getInstance();
	//demo->build(false);


	



	//DemoSlope* demo = DemoSlope::getInstance();
	//demo->build(true);

	//DemoPBDSingleHFCollide* demo = DemoPBDSingleHFCollide::getInstance();
	//demo->createScene();
	//demo->run();
	

	//DemoCollisionTest* demo = DemoCollisionTest::getInstance();
	//demo->build(false);

	//DemoPendulumTest* demo = DemoPendulumTest::getInstance();
	//demo->build(false);

	//DemoContactTest* demo = DemoContactTest::getInstance();
	//demo->build(true);


	/*GLApp window;
	window.createWindow(1024, 768);
	window.mainLoop();*/



	//system("pause");
	return 0;

}

