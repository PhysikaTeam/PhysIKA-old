1. the project is built through cmake.
2. cuda library is required and if cudalib cannot be linked, link directory must be specified explicitly.
3. binary is in the bin/Release under the build directory and you must run it in the bin directory due to the relative model path.
4. four new app, i.e. App_CollisionHybridOne, App_CollisionHybridTwo, App_MassSpring and App_FiniteElement is added. The first two show the collision and the last two show the simulation.
