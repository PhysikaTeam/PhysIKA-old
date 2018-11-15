/*
* @file render_demo.cpp
* @brief render demo of Physika.
* @author WeiChen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/
#include <iostream>
#include <fstream>
#include <GL/freeglut.h>

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Transform/transform_3d.h"

#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

#include "Physika_Render/Lights/point_light.h"
#include "Physika_Render/Lights/spot_light.h"
#include "Physika_Render/Lights/flash_light.h"
#include "Physika_Render/Lights/flex_spot_light.h"

#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"

#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render_util.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_solid_render_task.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_point_render_task.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_wireframe_render_task.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_point_vector_render_task.h"

#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_render_util.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_point_render_task.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_point_vector_render_task.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_wireframe_render_task.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_solid_render_task.h"

#include "Physika_Render/Fluid_Render/fluid_render_util.h"
#include "Physika_Render/Fluid_Render/fluid_point_render_task.h"
#include "Physika_Render/Fluid_Render/fluid_render_task.h"

#include "Physika_Render/ColorBar_Render/color_bar_render_task.h"
#include "Physika_Render/Plane_Render/plane_render_task.h"

#include "Physika_Render/Screen_Based_Render_Manager/screen_based_render_manager.h"
#include "Physika_Render/ColorBar/ColorMap/color_map.h"

using namespace std;
using namespace Physika;


//-------------------------------------------------------------------------------------
SurfaceMesh<double> ball_mesh;
SurfaceMesh<double> teapot_mesh;
SurfaceMesh<double> antique_mesh;
SurfaceMesh<double> flower_mesh;

shared_ptr<SpotLight> spot_light_1;
shared_ptr<SpotLight> spot_light_2;
shared_ptr<SpotLight> spot_light_3;
shared_ptr<SpotLight> spot_light_4;
shared_ptr<SpotLight> spot_light_5;

shared_ptr<FlashLight> flash_light;

//-------------------------------------------------------------------------------------

//==========================================================================================================================================================================

void addTeaPotMeshRenderTasks()
{
    cout << "loading mesh......" << endl;
    ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/teapot.obj", &teapot_mesh);
    cout << "teapot surface mesh loaded successfully!\n";

    //-----------------------------------------------------------------------------------------------------------------------------------------
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    ColorMap<float> color_map;

    //-----------------------------------------------------------------------------------------------------------------------------------------
    auto teapot_render_util = make_shared<SurfaceMeshRenderUtil<double>>(&teapot_mesh);

    vector<Color4f> vert_col_vec;
    for (int i = 0; i < teapot_mesh.numVertices(); ++i)
        vert_col_vec.push_back(color_map.colorViaRatio(float(i) / teapot_mesh.numVertices()));

    //---------------------------------------------------------------------------------------------------------
    auto teapot_solid_render_task = make_shared<SurfaceMeshSolidRenderTask<double>>(teapot_render_util);

    Transform<float, 3> teapot_solid_transform;
    teapot_solid_transform.setScale({ 2,2,2 });
    teapot_solid_transform.setTranslation({ -58, 2, 0 });

    teapot_solid_render_task->setCustomVertexColors(vert_col_vec);
    teapot_solid_render_task->enableUseCustomColor();
    //teapot_solid_render_task->disableUseLight();
    teapot_solid_render_task->setTransform(teapot_solid_transform);

    render_scene_config.pushBackRenderTask(teapot_solid_render_task);

    //---------------------------------------------------------------------------------------------------------
    auto teapot_wireframe_render_task = make_shared<SurfaceMeshWireframeRenderTask<double>>(teapot_render_util);

    Transform<float, 3> teapot_wireframe_transform;
    teapot_wireframe_transform.setScale({ 2,2,2 });
    teapot_wireframe_transform.setTranslation({ -42, 2, 0 });
    teapot_wireframe_render_task->setTransform(teapot_wireframe_transform);

    teapot_wireframe_render_task->setCustomVertexColors(vert_col_vec);
    teapot_wireframe_render_task->enableUseCustomColor();

    render_scene_config.pushBackRenderTask(teapot_wireframe_render_task);

    //---------------------------------------------------------------------------------------------------------
    auto teapot_point_vector_render_task = make_shared<SurfaceMeshPointVectorRenderTask<double>>(teapot_render_util);

    teapot_point_vector_render_task->setTransform(teapot_solid_transform);

    teapot_point_vector_render_task->setPointVectorsAsNormalVector();
    teapot_point_vector_render_task->enableUsePointVectorColor();
    teapot_point_vector_render_task->setScaleFactor(0.5);

    render_scene_config.pushBackRenderTask(teapot_point_vector_render_task);

    //---------------------------------------------------------------------------------------------------------
    auto teapot_solid_no_light_render_task = make_shared<SurfaceMeshSolidRenderTask<double>>(teapot_render_util);

    Transform<float, 3> teapot_solid_no_light_transform;
    teapot_solid_no_light_transform.setScale({ 2,2,2 });
    teapot_solid_no_light_transform.setTranslation({ -50, 2, -8 });
    teapot_solid_no_light_render_task->setTransform(teapot_solid_no_light_transform);

    teapot_solid_no_light_render_task->setCustomVertexColors(vert_col_vec);
    teapot_solid_no_light_render_task->enableUseCustomColor();
    teapot_solid_no_light_render_task->disableUseLight();

    render_scene_config.pushBackRenderTask(teapot_solid_no_light_render_task);

    //---------------------------------------------------------------------------------------------------------
    auto teapot_point_render_task = make_shared<SurfaceMeshPointRenderTask<double>>(teapot_render_util);

    Transform<float, 3> teapot_point_transform;
    teapot_point_transform.setScale({ 2,2,2 });
    teapot_point_transform.setTranslation({ -50, 2, 8 });
    teapot_point_render_task->setTransform(teapot_point_transform);

    teapot_point_render_task->setPointColors(vert_col_vec);
    teapot_point_render_task->setPointScaleForPointSprite(100.0);

    render_scene_config.pushBackRenderTask(teapot_point_render_task);



    //---------------------------------------------------------------------------------------------------------
    auto teapot_no_color_solid_render_task = make_shared<SurfaceMeshSolidRenderTask<double>>(teapot_render_util);

    Transform<float, 3> teapot_no_color_solid_transform;
    teapot_no_color_solid_transform.setScale({ 3,3,3 });
    teapot_no_color_solid_transform.setTranslation({ -8, 3, 0 });
    teapot_no_color_solid_render_task->setTransform(teapot_no_color_solid_transform);

    render_scene_config.pushBackRenderTask(teapot_no_color_solid_render_task);

    //---------------------------------------------------------------------------------------------------------
    auto teapot_black_wireframe_render_task = make_shared<SurfaceMeshWireframeRenderTask<double>>(teapot_render_util);

    Transform<float, 3> teapot_black_wireframe_transform;
    teapot_black_wireframe_transform.setScale({ 3,3,3 });
    teapot_black_wireframe_transform.setTranslation({ -8, 3, 0 });
    teapot_black_wireframe_render_task->setTransform(teapot_black_wireframe_transform);

    teapot_black_wireframe_render_task->setUniformColor(Color4f::Black());
    teapot_black_wireframe_render_task->enableUseCustomColor();

    render_scene_config.pushBackRenderTask(teapot_black_wireframe_render_task);

}

//==========================================================================================================================================================================

void addBallAndFlowerAndAntiqueMeshRenderTasks()
{
    cout << "loading mesh......" << endl;
    ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/ball_high.obj", &ball_mesh);
    ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/SunFlower.obj", &flower_mesh);
    ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/scene_dense_mesh_refine_texture.obj", &antique_mesh);
    cout << "load ball&flower&antique mesh successfully!\n";

    //-----------------------------------------------------------------------------------------------------------------------------------------
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    auto ball_render_util = make_shared<SurfaceMeshRenderUtil<double>>(&ball_mesh);
    auto ball_solid_render_task = make_shared<SurfaceMeshSolidRenderTask<double>>(ball_render_util);

    Transform<float, 3> ball_solid_transform;
    ball_solid_transform.setScale({ 0.05f, 0.05f, 0.05f });
    ball_solid_transform.setTranslation({ 50, 2, -5 });
    ball_solid_render_task->setTransform(ball_solid_transform);

    render_scene_config.pushBackRenderTask(ball_solid_render_task);

    //-----------------------------------------------------------------------------------------------------------------------------------------
    auto flower_render_util = make_shared<SurfaceMeshRenderUtil<double>>(&flower_mesh);
    auto flower_solid_render_task = make_shared<SurfaceMeshSolidRenderTask<double>>(flower_render_util);

    Transform<float, 3> flower_solid_transform;
    flower_solid_transform.setScale({ 0.5f, 0.5f, 0.5f });
    flower_solid_transform.setTranslation({ 60, 5, 5 });
    flower_solid_render_task->setTransform(flower_solid_transform);

    render_scene_config.pushBackRenderTask(flower_solid_render_task);

    //-----------------------------------------------------------------------------------------------------------------------------------------

    auto antique_render_util = make_shared<SurfaceMeshRenderUtil<double>>(&antique_mesh);
    auto antique_solid_render_task = make_shared<SurfaceMeshSolidRenderTask<double>>(antique_render_util);

    Transform<float, 3> antique_solid_transform;
    antique_solid_transform.setScale({ 5.0f, 5.0f, 5.0f });
    antique_solid_transform.setTranslation({ 45, 5, 0});
    antique_solid_transform.setRotation({ 0.0, 0.0, 1.0 }, PI / 2.0);
    antique_solid_render_task->setTransform(antique_solid_transform);
    

    render_scene_config.pushBackRenderTask(antique_solid_render_task);

}

//==========================================================================================================================================================================

void addVolMeshRenderTasks()
{
    VolumetricMesh<double, 3> * tire_mesh = nullptr;
    VolumetricMesh<double, 3> * armadillo_mesh = nullptr;
    VolumetricMesh<double, 3> * flower_mesh = nullptr;

    cout << "loading mesh ......" << endl;
    tire_mesh = VolumetricMeshIO<double, 3>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Vol_Mesh/tire.smesh");
    armadillo_mesh = VolumetricMeshIO<double, 3>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Vol_Mesh/armadillo-coarse.smesh");
    flower_mesh = VolumetricMeshIO<double, 3>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Vol_Mesh/flower.smesh");
    cout << "load mesh successfully" << endl;

    //---------------------------------------------------------------------------------------------------------------------
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    auto tire_render_util = make_shared<VolumetricMeshRenderUtil<double, 3>>(tire_mesh);
    auto armadillo_render_util = make_shared<VolumetricMeshRenderUtil<double, 3>>(armadillo_mesh);
    auto flower_render_util = make_shared<VolumetricMeshRenderUtil<double, 3>>(flower_mesh);
    //---------------------------------------------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------------------------------------------
    auto tire_solid_render_task = make_shared<VolumetricMeshSolidRenderTask<double, 3>>(tire_render_util);

    Transform<float, 3> tire_solid_transform;
    tire_solid_transform.setTranslation({ 5, 10, -8 });
    tire_solid_render_task->setTransform(tire_solid_transform);

    tire_solid_render_task->setUnifromColor(Color4f::Yellow());

    render_scene_config.pushBackRenderTask(tire_solid_render_task);

    //---------------------------------------------------------------------------------------------------------------------
    auto armadillo_solid_render_task = make_shared<VolumetricMeshSolidRenderTask<double, 3>>(armadillo_render_util);

    Transform<float, 3> armadillo_solid_transform;
    armadillo_solid_transform.setScale({ 2.0f, 2.0f, 2.0f });
    armadillo_solid_transform.setTranslation({ 5, 10, 8 });
    armadillo_solid_render_task->setTransform(armadillo_solid_transform);

    render_scene_config.pushBackRenderTask(armadillo_solid_render_task);

    //---------------------------------------------------------------------------------------------------------------------
    auto flower_solid_render_task = make_shared<VolumetricMeshSolidRenderTask<double, 3>>(flower_render_util);

    Transform<float, 3> flower_solid_transform;
    flower_solid_transform.setScale({ 0.5f, 0.5f, 0.5f });
    flower_solid_transform.setTranslation({ 12, 8, 0 });
    flower_solid_render_task->setTransform(flower_solid_transform);

    ColorMap<float> color_map;
    vector<Color4f> ele_col_vec;
    for (unsigned int i = 0; i < flower_render_util->eleNum(); ++i)
        ele_col_vec.push_back(color_map.colorViaRatio(float(i) / flower_render_util->eleNum()));
    flower_solid_render_task->setElementColors(ele_col_vec);

    render_scene_config.pushBackRenderTask(flower_solid_render_task);
}

//==========================================================================================================================================================================

void addPlaneRenderTasks()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    auto plane_render_task = make_shared<PlaneRenderTask>();
    plane_render_task->addPlane({ 0.0f, 1.0f, 0.0f, 0.0f });
    render_scene_config.pushBackRenderTask(plane_render_task);
}

//==========================================================================================================================================================================

void addColorBarTasks()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    ColorBar<float> color_bar;
    auto color_bar_render_task = make_shared<ColorBarRenderTask<float>>(color_bar);
    render_scene_config.pushBackRenderTask(color_bar_render_task);
}

//==========================================================================================================================================================================

GLfloat * readVec4File(const string & file_name, int & num)
{
    fstream file(file_name, ios::in);
    if (file.good() == false)
    {
        cerr << "can't open file: " << file_name << endl;
        exit(EXIT_FAILURE);
    }

    file >> num;
    cout << "file_name: " << file_name << endl;
    cout << "num: " << num << endl;

    GLfloat * data = new GLfloat[4 * num];
    for (int i = 0; i < num; ++i)
        file >> data[4 * i] >> data[4 * i + 1] >> data[4 * i + 2] >> data[4 * i + 3];

    return data;
}

GLfloat * readDensityFile(const string & file_name, int & num)
{
    fstream file(file_name, ios::in);
    if (file.good() == false)
    {
        cerr << "can't open file: " << file_name << endl;
        exit(EXIT_FAILURE);
    }

    file >> num;
    cout << "file_name: " << file_name << endl;
    cout << "num: " << num << endl;

    GLfloat * data = new GLfloat[num];
    for (int i = 0; i < num; ++i)
        file >> data[i];

    return data;
}

GLuint * readIndicesFile(const string & file_name, int & num)
{
    fstream file(file_name, ios::in);
    if (file.good() == false)
    {
        cerr << "can't open file: " << file_name << endl;
        exit(EXIT_FAILURE);
    }

    file >> num;
    cout << "file_name: " << file_name << endl;
    cout << "num: " << num << endl;

    GLuint * data = new GLuint[num];
    for (int i = 0; i < num; ++i)
        file >> data[i];

    return data;
}

void addFluidRenderTasks()
{
    int fluid_particle_num;
    int id = 0;

    cout << "read data ..." << endl;
    string path = "Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Screen_Based_Render_Manager/";

    GLfloat * pos = readVec4File(path + "data/pos_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_1 = readVec4File(path + "data/anisotropy_1_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_2 = readVec4File(path + "data/anisotropy_2_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_3 = readVec4File(path + "data/anisotropy_3_" + to_string(id) + ".txt", fluid_particle_num);

    GLfloat * density = readDensityFile(path + "data/density_" + to_string(id) + ".txt", fluid_particle_num);
    GLuint  * indices = readIndicesFile(path + "data/indice_" + to_string(id) + ".txt", fluid_particle_num);

    cout << "fluid particle num: " << fluid_particle_num << endl;
    cout << "read data successfully!" << endl;

    //-----------------------------------------------------------------------------------------------------------------------------------------
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    auto fluid_render_util = make_shared<FluidRenderUtil>(fluid_particle_num, 0);
    fluid_render_util->updateFluidParticleBuffer(pos, density, anisotropy_1, anisotropy_2, anisotropy_3, indices, fluid_particle_num);

    //-----------------------------------------------------------------------------------------------------------------------------------------
    auto fluid_point_render_task = make_shared<FluidPointRenderTask>(fluid_render_util);

    Transform<float, 3> fluid_point_transform;
    fluid_point_transform.setScale({ 4.0f, 4.0f, 4.0f });
    fluid_point_transform.setTranslation({ -5, 0, 45 });
    fluid_point_render_task->setTransform(fluid_point_transform);

    fluid_point_render_task->setRadius(0.04f);

    render_scene_config.pushBackRenderTask(fluid_point_render_task);

    //-----------------------------------------------------------------------------------------------------------------------------------------
    auto fluid_render_task = make_shared<FluidRenderTask>(fluid_render_util);

    Transform<float, 3> fluid_transform;
    fluid_transform.setScale({ 4.0f, 4.0f, 4.0f });
    fluid_transform.setTranslation({ -5, 0, 55 });
    fluid_render_task->setTransform(fluid_transform);

    fluid_render_task->setFluidRadius(0.04f);

    render_scene_config.pushBackRenderTask(fluid_render_task);
}

void addLights()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    //ambient light
    spot_light_1 = make_shared<SpotLight>();
    spot_light_1->setPos({ 0, 50, 0 });
    spot_light_1->setSpotDirection({ 0, -1, 0 });
    spot_light_1->setSpotCutoff(0.0);
    spot_light_1->setSpotOuterCutoff(20);
    spot_light_1->setAmbient(Color4f::Gray());
    spot_light_1->setDiffuse(Color4f::Black());
    spot_light_1->setSpecular(Color4f::Black());
    render_scene_config.pushBackLight(spot_light_1);

    
    spot_light_2 = make_shared<SpotLight>();
    spot_light_2->setPos({ 0, 50, 0 });
    spot_light_2->setSpotDirection({ 0, -1, 0 });
    spot_light_2->setSpotCutoff(0.0);
    spot_light_2->setSpotOuterCutoff(20);
    render_scene_config.pushBackLight(spot_light_2);

    spot_light_3 = make_shared<SpotLight>();
    spot_light_3->setPos({ 50, 50, 0 });
    spot_light_3->setSpotDirection({ 0, -1, 0 });
    spot_light_3->setSpotCutoff(0.0);
    spot_light_3->setSpotOuterCutoff(20);
    render_scene_config.pushBackLight(spot_light_3);

    spot_light_4 = make_shared<SpotLight>();
    spot_light_4->setPos({ -50, 50, 0 });
    spot_light_4->setSpotDirection({ 0, -1, 0 });
    spot_light_4->setSpotCutoff(0.0);
    spot_light_4->setSpotOuterCutoff(20);
    render_scene_config.pushBackLight(spot_light_4);

    spot_light_5 = make_shared<SpotLight>();
    spot_light_5->setPos({ 0, 50, 50 });
    spot_light_5->setSpotDirection({ 0, -1, 0 });
    spot_light_5->setSpotCutoff(0.0);
    spot_light_5->setSpotOuterCutoff(20);
    render_scene_config.pushBackLight(spot_light_5);
    

    flash_light = make_shared<FlashLight>();
    flash_light->disableLighting();
    render_scene_config.pushBackLight(flash_light);

}

void initFunction()
{
    addTeaPotMeshRenderTasks();
    
    addBallAndFlowerAndAntiqueMeshRenderTasks();
    
    addVolMeshRenderTasks();

    addPlaneRenderTasks();

    addFluidRenderTasks();

    addColorBarTasks();

    addLights();
    
    //-----------------------------------------------------------------------------------------------------------------------------------------
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glClearDepth(1.0);
    glClearColor(0.5, 0.5, 0.5, 1.0);

    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);

    
}

void displayFunction()
{
    //Note: it's important to enable polygon offset for combination with wireframe and solid render task.
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0, 1.0);

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    ScreenBasedRenderManager & render_manager = render_scene_config.screenBasedRenderManager();

    render_manager.render();

    GlutWindow *cur_window = (GlutWindow*)glutGetWindowData();
    cur_window->displayFrameRate();
    glutPostRedisplay();
    glutSwapBuffers();
}

void keyboardFunction(unsigned char key, int x, int y)
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    ScreenBasedRenderManager & render_manager = render_scene_config.screenBasedRenderManager();

    switch (key)
    {
    case '1':
    {
        if (spot_light_1->isEnableLighting())
            spot_light_1->disableLighting();
        else
            spot_light_1->enableLighting();
        break;
    }

    case '2':
    {
        if (spot_light_2->isEnableLighting())
            spot_light_2->disableLighting();
        else
            spot_light_2->enableLighting();
        break;
    }

    case '3':
    {
        if (spot_light_3->isEnableLighting())
            spot_light_3->disableLighting();
        else
            spot_light_3->enableLighting();
        break;
    }

    case '4':
    {
        if (spot_light_4->isEnableLighting())
            spot_light_4->disableLighting();
        else
            spot_light_4->enableLighting();
        break;
    }

    case '5':
    {
        if (spot_light_5->isEnableLighting())
            spot_light_5->disableLighting();
        else
            spot_light_5->enableLighting();
        break;
    }

    case 'f':
    {
        if (flash_light->isEnableLighting())
            flash_light->disableLighting();
        else
            flash_light->enableLighting();
        break;
    }

    case  's':
    {
        if (render_manager.isUseShadowmap())
            render_manager.disableUseShadowmap();
        else
            render_manager.enableUseShadowmap();
        break;
            
    }

    case  'g':
    {
        if (render_manager.isUseGammaCorrection())
            render_manager.disableUseGammaCorrection();
        else
            render_manager.enableUseGammaCorrection();
        break;

    }

    default:
        break;
    }
}

int main()
{
    GlutWindow glut_window("screen_based_render_manager_test", 1280, 720);

    cout << "Window name: " << glut_window.name() << "\n";
    cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    render_scene_config.setCameraPosition(Vector<double, 3>(0, 0, 200));
    render_scene_config.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
    render_scene_config.setCameraNearClip(0.001);
    render_scene_config.setCameraFarClip(1.0e4);

    glut_window.setDisplayFunction(displayFunction);
    glut_window.setKeyboardFunction(keyboardFunction);
    glut_window.setInitFunction(initFunction);
    glut_window.createWindow();
    return 0;
}