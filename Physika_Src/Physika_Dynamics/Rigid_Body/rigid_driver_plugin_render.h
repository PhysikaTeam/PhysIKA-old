/*
 * @file rigid_driver_plugin_render.h 
 * @Render plugin of rigid body driver. This is a singleton class
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_RENDER_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_RENDER_H_

#include<vector>

namespace Physika{

template <typename Scalar,int Dim> class RigidDriverPlugin;
class GlutWindow;
class RenderBase;

template <typename Scalar,int Dim>
class RigidDriverPluginRender: public RigidDriverPlugin<Scalar, Dim>
{
public:
	RigidDriverPluginRender();
	~RigidDriverPluginRender();

	//functions called in driver
	void onRun();
	void onAdvanceFrame();
	void onInitialize();
	void onAdvanceStep(Scalar dt);
	void onAddRigidBody(RigidBody<Scalar, Dim>* rigid_body);

	//basic function
	void setDriver(RigidBodyDriver<Scalar, Dim>* driver);
	void setWindow(GlutWindow* window);

	//singleton function
	void active();//active this instance

protected:
	GlutWindow* window_;
	std::vector<RenderBase*> render_queue_;

	//singleton
	static RigidDriverPluginRender<Scalar, Dim>* active_render_;//current active instance
	static void idle();//idle function which is set to window_. Called each frame by window and call simulation functions in driver
};


} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_RENDER_H_