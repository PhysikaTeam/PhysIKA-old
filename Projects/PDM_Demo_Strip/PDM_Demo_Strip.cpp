/*
 * @file PDM_Demo_Strip.cpp 
 * @brief PDM Demo: Strip.
 * @author WeiChen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <set>

#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_Render/Point_Render/point_render.h"

#include "Physika_Dynamics/PDM/PDM_particle.h"
#include "Physika_Dynamics/PDM/PDM_state.h"

#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_render.h"
#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_boundary.h"
#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_output_mesh.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_state_viscoplasticity_verlet.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_state_viscoplasticity_semi_implicit.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Fracture_Methods/PDM_fracture_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_grid.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_space_hash_3d.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_Force.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_Sphere.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Boundary_Condition_Methods/PDM_boundary_condition_method.h"

using namespace std;
using namespace Physika;

int main()
{
    //mesh and boudary vertex file name
    string mesh_file_name;

    //material property
    double delta_ratio;
    double density;
    double k;
    double u;

    //fracture
    double alpha;
    double criticle_s;

    //visco & plasticity
    double Sb_limit;
    double relax_time;
    double lambda;
    double Sp_limit;
    double yield_critical_val;
    double Rcp;
    double Ecp_limit;

    //collision detection
    bool enable_collision;
    double Kc;
    unsigned int hash_table_size;
    bool use_edge_intersection;

    //boundary condition
    bool enable_boundary_condition;
    int cancel_boundary_condition_time_step;
    bool render_specified_particle;
    double fixed_ref_pos;
    double boundary_ref_pos;
    bool use_force;
    Vector<double, 3> vel;
    Vector<double, 3> force;

    //topology control
    bool enable_topology_control;
    double crack_smooth_level;
    bool enable_smooth_crack_vertex;

    bool enable_adjust_vertex;
    bool enable_rotate_vertex;

    bool enable_rigid_constrain;
    double critical_ele_quality;
    double max_rotate_degree;


    //global parameters
    double dt;
    double gravity;
    double Kd;
    double vel_decay_ratio;
    bool enable_plastic_statistics;
    bool skip_isolate_ele;
    bool capture_screen;
    bool creat_window;
    unsigned int wait_time_per_step;
    unsigned int time_step_per_frame;

    ///////////////////////////////////////////////////////////////////////////////
    freopen("parameters.txt","r",stdin);

    string parameter_name;
    cin>>parameter_name>>mesh_file_name;                cout<<parameter_name<<mesh_file_name<<endl;

    cin>>parameter_name>>delta_ratio;                   cout<<parameter_name<<delta_ratio<<endl;
    cin>>parameter_name>>density;                       cout<<parameter_name<<density<<endl;
    cin>>parameter_name>>k;                             cout<<parameter_name<<k<<endl;
    cin>>parameter_name>>u;                             cout<<parameter_name<<u<<endl;

    cin>>parameter_name>>alpha;                         cout<<parameter_name<<alpha<<endl;
    cin>>parameter_name>>criticle_s;                    cout<<parameter_name<<criticle_s<<endl;

    cin>>parameter_name>>Sb_limit;                      cout<<parameter_name<<Sb_limit<<endl;
    cin>>parameter_name>>relax_time;                    cout<<parameter_name<<relax_time<<endl;
    cin>>parameter_name>>lambda;                        cout<<parameter_name<<lambda<<endl;
    cin>>parameter_name>>Sp_limit;                      cout<<parameter_name<<Sp_limit<<endl;
    cin>>parameter_name>>yield_critical_val;            cout<<parameter_name<<yield_critical_val<<endl;
    cin>>parameter_name>>Rcp;                           cout<<parameter_name<<Rcp<<endl;
    cin>>parameter_name>>Ecp_limit;                     cout<<parameter_name<<Ecp_limit<<endl;

    cin>>parameter_name>>enable_collision;              cout<<parameter_name<<enable_collision<<endl;
    cin>>parameter_name>>Kc;                            cout<<parameter_name<<Kc<<endl;
    cin>>parameter_name>>hash_table_size;               cout<<parameter_name<<hash_table_size<<endl;
    cin>>parameter_name>>use_edge_intersection;         cout<<parameter_name<<use_edge_intersection<<endl;

    cin>>parameter_name>>enable_boundary_condition;               cout<<parameter_name<<enable_boundary_condition<<endl;
    cin>>parameter_name>>cancel_boundary_condition_time_step;     cout<<parameter_name<<cancel_boundary_condition_time_step<<endl;
    cin>>parameter_name>>render_specified_particle;               cout<<parameter_name<<render_specified_particle<<endl;

    cin>>parameter_name>>fixed_ref_pos;                           cout<<parameter_name<<fixed_ref_pos<<endl;
    cin>>parameter_name>>boundary_ref_pos;                        cout<<parameter_name<<boundary_ref_pos<<endl;
    cin>>parameter_name>>use_force;                               cout<<parameter_name<<use_force<<endl;
    cin>>parameter_name>>vel[0]>>vel[1]>>vel[2];                  cout<<parameter_name<<vel<<endl;
    cin>>parameter_name>>force[0]>>force[1]>>force[2];            cout<<parameter_name<<force<<endl;

    cin>>parameter_name>>enable_topology_control;       cout<<parameter_name<<enable_topology_control<<endl;
    cin>>parameter_name>>crack_smooth_level;            cout<<parameter_name<<crack_smooth_level<<endl;
    cin>>parameter_name>>enable_smooth_crack_vertex;    cout<<parameter_name<<enable_smooth_crack_vertex<<endl;
    cin>>parameter_name>>enable_adjust_vertex;          cout<<parameter_name<<enable_adjust_vertex<<endl;
    cin>>parameter_name>>enable_rotate_vertex;          cout<<parameter_name<<enable_rotate_vertex<<endl;
    cin>>parameter_name>>enable_rigid_constrain;        cout<<parameter_name<<enable_rigid_constrain<<endl;
    cin>>parameter_name>>critical_ele_quality;          cout<<parameter_name<<critical_ele_quality<<endl;
    cin>>parameter_name>>max_rotate_degree;             cout<<parameter_name<<max_rotate_degree<<endl;

    cin>>parameter_name>>dt;                            cout<<parameter_name<<dt<<endl;
    cin>>parameter_name>>gravity;                       cout<<parameter_name<<gravity<<endl;
    cin>>parameter_name>>Kd;                            cout<<parameter_name<<Kd<<endl;
    cin>>parameter_name>>vel_decay_ratio;               cout<<parameter_name<<vel_decay_ratio<<endl;
    cin>>parameter_name>>enable_plastic_statistics;     cout<<parameter_name<<enable_plastic_statistics<<endl;
    cin>>parameter_name>>skip_isolate_ele;              cout<<parameter_name<<skip_isolate_ele<<endl;
    cin>>parameter_name>>capture_screen;                cout<<parameter_name<<capture_screen<<endl;
    cin>>parameter_name>>creat_window;                  cout<<parameter_name<<creat_window<<endl;
    cin>>parameter_name>>wait_time_per_step;            cout<<parameter_name<<wait_time_per_step<<endl;
    cin>>parameter_name>>time_step_per_frame;           cout<<parameter_name<<time_step_per_frame<<endl;

    fclose(stdin);
    freopen("CON","r",stdin);

    system("pause");

    //////////////////////////////////////////////////////////////////////////////

    PDMState<double, 3> pdm_state;

    pdm_state.setHomogeneousBulkModulus(k);
    pdm_state.setHomogeneousShearModulus(u*k);

    //pdm_state.setParticlesViaMesh(mesh_file_name, delta_ratio, false);
    pdm_state.autoSetParticlesViaMesh(mesh_file_name, delta_ratio);
    pdm_state.setMassViaHomogeneousDensity(density);

    // step_method
    //PDMStepMethodStateViscoPlasticityImplicit<double,3> step_method;
    //PDMStepMethodStateViscoPlasticityVerlet<double, 3> step_method;
    PDMStepMethodStateViscoPlasticitySemiImplicit<double, 3> step_method;
    pdm_state.setStepMethod(&step_method);

    //statistics
    if(enable_plastic_statistics) step_method.enablePlasticStatistics();

    //velocity decay
    step_method.setKd(Kd);
    step_method.setVelDecayRatio(vel_decay_ratio);

    //Visco & plasticity
    pdm_state.setHomogeneousEbStretchLimit(Sb_limit);
    step_method.setRelaxTime(relax_time);
    step_method.setLambda(lambda);
    pdm_state.setHomogeneousEpStretchLimit(Sp_limit);
    step_method.setHomogeneousYieldCriticalVal(yield_critical_val);
    step_method.setRcp(Rcp);
    step_method.setEcpLimit(Ecp_limit);

    //fracture
    PDMFractureMethodBase<double,3> fracture_method;  
    step_method.setFractureMethod(&fracture_method);
    step_method.enableFracture();

    fracture_method.setAlpha(alpha);
    fracture_method.setHomogeneousCriticalStretch(criticle_s);

    //collision
    PDMCollisionMethodSpaceHash<double, 3> collision_method;
    step_method.setCollisionMethod(&collision_method);
    if (enable_collision) step_method.enableCollision();

    collision_method.setKc(Kc);
    collision_method.setHashTableSize(hash_table_size);
    collision_method.autoSetGridCellSize();
    cout<<"grid cell size: "<<collision_method.gridCellSize()<<endl;
    if (use_edge_intersection) collision_method.enableEdgeIntersect();

    //boundary 
    PDMBoundaryConditionMethod<double, 3> boundary_method;
    step_method.setBoundaryConditionMethod(&boundary_method);
    if (enable_boundary_condition) step_method.enableBoundaryCondition();
    boundary_method.setCancelBoundaryConditionTimeStep(cancel_boundary_condition_time_step);

    vector<unsigned int> fixed_particles;
    vector<unsigned int> boundary_particles;

    for (unsigned int par_id = 0; par_id < pdm_state.numSimParticles(); par_id++)
    {
        Vector<double, 3> pos = pdm_state.particleRestPosition(par_id);
        if (pos[0] < fixed_ref_pos) fixed_particles.push_back(par_id);
        if (pos[0] > boundary_ref_pos) boundary_particles.push_back(par_id);
    }

    for (unsigned int i = 0; i < fixed_particles.size(); i++)
        boundary_method.addFixParticle(fixed_particles[i]);

    if (use_force == false)
    {
        for (unsigned int i = 0; i < boundary_particles.size(); i++)
            boundary_method.addSetParticleVel(boundary_particles[i], vel);
    }
    else
    {
        for (unsigned int i = 0; i < boundary_particles.size(); i++)
        {
            boundary_method.addParticleExtraForce(boundary_particles[i], force);
            fracture_method.setCriticalStretch(boundary_particles[i], 1.0);
        }
    }


    //topology control
    PDMTopologyControlMethod<double,3> topology_control_method;
    step_method.setTopologyControlMethod(&topology_control_method);
    if (enable_topology_control) step_method.enableTopologyControl();

    topology_control_method.setCrackSmoothLevel(crack_smooth_level);
    if (enable_smooth_crack_vertex) topology_control_method.enableSmoothCrackVertexPos();

    if (enable_adjust_vertex)       topology_control_method.enableAdjustMeshVertexPos();

    topology_control_method.setRotVelDecayRatio(0.01);
    if (enable_rotate_vertex)       topology_control_method.enableRotateMeshVertex();

    topology_control_method.setCriticalEleQuality(critical_ele_quality);
    topology_control_method.setMaxRigidRotDegree(max_rotate_degree);
    if (enable_rigid_constrain) topology_control_method.enableRigidConstrain();

    //render plugin
    PDMPluginRender<double, 3> render_plugin;
    GlutWindow glut_window;
    render_plugin.setWindow(&glut_window);
    if (capture_screen) render_plugin.enableCaptureScreen();
    if (creat_window) pdm_state.addPlugin(&render_plugin);

    render_plugin.setPointSize(2);
    render_plugin.enableRenderMesh();

    if (creat_window && render_specified_particle)
    {
        for (unsigned int i = 0; i < fixed_particles.size(); i++)
            render_plugin.addIdxToSpecifiedParVec(fixed_particles[i]);
        for (unsigned int i = 0; i < boundary_particles.size(); i++)
            render_plugin.addIdxToSpecifiedParVec(boundary_particles[i]);
    }

    render_plugin.enableRenderSpecifiedParticle();

    //mesh output plugin
    PDMPluginOutputMesh<double, 3> output_mesh_plugin;
    if (enable_topology_control) pdm_state.addPlugin(&output_mesh_plugin);
    if (skip_isolate_ele) output_mesh_plugin.enableSkipIsolateEle();

    //global parameters
    pdm_state.setGravity(gravity);
    pdm_state.setMaxDt(dt);
    pdm_state.setWaitTimePerStep(wait_time_per_step);
    pdm_state.setStartFrame(0);
    pdm_state.setEndFrame(1000000);
    pdm_state.setFrameRate(1.0/(time_step_per_frame*dt));

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    render_scene_config.setCameraNearClip(0.01);
    render_scene_config.setCameraFarClip(1000.0);
    render_scene_config.setCameraPosition(Vector<double,3>(0, 0, 20));
    render_scene_config.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
    render_scene_config.setCameraFOV(45);
    
    //glut window
    glut_window.enableEventMode();
    if (creat_window) glut_window.createWindow();

    //run simulation
    if (!creat_window) pdm_state.forwardSimulation();
    pdm_state.run();

    return 0;
}