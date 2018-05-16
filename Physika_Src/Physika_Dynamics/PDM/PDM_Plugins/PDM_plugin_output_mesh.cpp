/*
 * @file PDM_plugin_output_mesh.cpp
 * @brief output particle information for mesh generating
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <fstream>
#include <sstream>
#include <map>

#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_output_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tri3d_mesh.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMPluginOutputMesh<Scalar, Dim>::PDMPluginOutputMesh()
    :skip_isolate_ele_(false), save_intermediate_state_time_step_(-1), impact_method_(NULL)
{

}

template <typename Scalar, int Dim>
PDMPluginOutputMesh<Scalar, Dim>::~PDMPluginOutputMesh()
{

}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::onBeginFrame(unsigned int frame)
{

}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::onEndFrame(unsigned int frame)
{
    std::stringstream sstream;
    sstream<<frame;
    std::string frame_str;
    sstream>>frame_str;
    std::string file_name("mesh/frame_"+frame_str+".obj");

    PDMBase<Scalar,Dim> * pdm_base = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(pdm_base);

    // save surface mesh
    VolumetricMesh<Scalar, Dim> * mesh = pdm_base->mesh();
    if (mesh->elementType() == VolumetricMeshInternal::TET)
    {
        this->saveBoundaryMesh(static_cast<TetMesh<Scalar> *>(mesh), file_name);
    }
    else if(mesh->elementType() == VolumetricMeshInternal::TRI3D)
    {
        this->saveBoundaryMesh(static_cast<Tri3DMesh<Scalar> *>(mesh), file_name);
    }
    else
    {
        std::cerr<<"error: element type is wrong!\n";
        std::exit(EXIT_FAILURE);
    }

    //save impact pos
    if (this->impact_method_ != NULL)
    {
        this->saveImpactPos();
    }
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::onBeginTimeStep(Scalar time, Scalar dt)
{

}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::onEndTimeStep(Scalar time, Scalar dt)
{
    PDMBase<Scalar,Dim> * pdm_base = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(pdm_base);

    //need further consideration
    /////////////////////////////////////////////////////////////////////////////////////////
    std::fstream file("save_intermediate_state_time_step.txt", std::ios::in);
    if (file.fail() == false)
    {
        file>>this->save_intermediate_state_time_step_;
        std::cout<<"update save intermediate state time step: "<<this->save_intermediate_state_time_step_<<std::endl;
    }
    file.close();
    ////////////////////////////////////////////////////////////////////////////////////////

    //save itermediate state
    if (pdm_base->timeStepId() == this->save_intermediate_state_time_step_)
    {
        this->saveParticlePos();
        this->saveParticleVel();
        this->saveVolumetricMesh();
    }
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::enableSkipIsolateEle()
{
    this->skip_isolate_ele_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::disableSkipIsolateEle()
{
    this->skip_isolate_ele_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::setSaveIntermediateStateTimeStep(int save_intermediate_state_time_step)
{
    this->save_intermediate_state_time_step_ = save_intermediate_state_time_step;
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::setImpactMethod(PDMImpactMethodBase<Scalar, Dim> * impact_method)
{
    this->impact_method_ = impact_method;

    //need further consideration
    std::fstream file("impact_pos_data.txt", std::ios::out|std::ios::trunc);
    file.close();
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::saveBoundaryMesh(TetMesh<Scalar> * mesh, const std::string & file_name)
{
    std::vector<unsigned int> boundary_vertices;
    mesh->boundaryVertices(boundary_vertices);
    std::cout<<"boundary_vertices size: "<<boundary_vertices.size()<<std::endl;

    SurfaceMesh<Scalar> surf_mesh;
    std::map<unsigned int, unsigned int> vert_id_map;
    for (unsigned int vert_idx=0; vert_idx<boundary_vertices.size(); vert_idx++)
    {
        surf_mesh.addVertexPosition(mesh->vertPos(boundary_vertices[vert_idx]));
        vert_id_map[boundary_vertices[vert_idx]] = vert_idx;
    }

    SurfaceMeshInternal::FaceGroup<Scalar> group;
    surf_mesh.addGroup(group);
    SurfaceMeshInternal::FaceGroup<Scalar> & group_ref = surf_mesh.group(surf_mesh.numGroups()-1);

    std::vector<unsigned int> face(3);
    std::vector<std::vector<unsigned int> > boundary_faces;
    for(unsigned int ele_idx = 0; ele_idx < mesh->eleNum(); ele_idx++)
    {
        if (mesh->isBoundaryElement(ele_idx) == false) 
            continue;

        unsigned int vert_idx0 = mesh->eleVertIndex(ele_idx,0);
        unsigned int vert_idx1 = mesh->eleVertIndex(ele_idx,1);
        unsigned int vert_idx2 = mesh->eleVertIndex(ele_idx,2);
        unsigned int vert_idx3 = mesh->eleVertIndex(ele_idx,3);

        /*
        PDMBase<Scalar,Dim> * pdm_base = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
        const PDMParticle<Scalar, Dim> & particle = pdm_base->particle(ele_idx);

        if (this->skip_isolate_ele_ == true && particle.validFamilySize() == 0)
            continue;

        if (particle.validFamilySize() == 0) 
        {
            std::vector<Vector<Scalar, Dim> > ele_vert_pos;
            mesh->eleVertPos(ele_idx, ele_vert_pos);

            Vector<Scalar, Dim> x1_minus_x0 = ele_vert_pos[1] - ele_vert_pos[0];
            Vector<Scalar, Dim> x2_minus_x0 = ele_vert_pos[2] - ele_vert_pos[0];
            Vector<Scalar, Dim> x3_minus_x0 = ele_vert_pos[3] - ele_vert_pos[0];

            //cross vector used for 2D-case compile
            Vector<Scalar, Dim> cross_vector(x1_minus_x0.cross(x2_minus_x0));
            Scalar signed_volume = 1.0/6.0*x3_minus_x0.dot(cross_vector);

            if (signed_volume < 0.0) std::swap(vert_idx0, vert_idx1);
        }
        */
            
        //face 021
        face[0] = vert_idx0; face[1] = vert_idx2; face[2] = vert_idx1;
        if (mesh->isBoundaryFace(face)) boundary_faces.push_back(face);
        //face 013
        face[0] = vert_idx0; face[1] = vert_idx1; face[2] = vert_idx3;
        if (mesh->isBoundaryFace(face)) boundary_faces.push_back(face);
        //face 032
        face[0] = vert_idx0; face[1] = vert_idx3; face[2] = vert_idx2;
        if (mesh->isBoundaryFace(face)) boundary_faces.push_back(face);
        //face 123
        face[0] = vert_idx1; face[1] = vert_idx2; face[2] = vert_idx3;
        if (mesh->isBoundaryFace(face)) boundary_faces.push_back(face);
    }

    for (unsigned int face_idx = 0; face_idx<boundary_faces.size(); face_idx++)
    {
        SurfaceMeshInternal::Face<Scalar> face;
        group_ref.addFace(face);

        Physika::SurfaceMeshInternal::Face<Scalar> & face_ref = group_ref.face(group_ref.numFaces()-1);
        Physika::BoundaryMeshInternal::Vertex<Scalar> vertex_0;
        Physika::BoundaryMeshInternal::Vertex<Scalar> vertex_1;
        Physika::BoundaryMeshInternal::Vertex<Scalar> vertex_2;

        const std::vector<unsigned int> & face_vec = boundary_faces[face_idx];
        unsigned int vert_0_id;
        unsigned int vert_1_id;
        unsigned int vert_2_id;
        std::map<unsigned int, unsigned int>::iterator iter_0 = vert_id_map.find(face_vec[0]);
        std::map<unsigned int, unsigned int>::iterator iter_1 = vert_id_map.find(face_vec[1]);
        std::map<unsigned int, unsigned int>::iterator iter_2 = vert_id_map.find(face_vec[2]);

        if (iter_0 == vert_id_map.end()||iter_1 == vert_id_map.end()||iter_2 == vert_id_map.end())
        {
            std::cerr<<"error: can't find the vertex index in surface mesh!\n";
            std::exit(EXIT_FAILURE);
        }

        vert_0_id = iter_0->second;
        vert_1_id = iter_1->second;
        vert_2_id = iter_2->second;
      
        vertex_0.setPositionIndex(vert_0_id);
        vertex_1.setPositionIndex(vert_1_id);
        vertex_2.setPositionIndex(vert_2_id);

        face_ref.addVertex(vertex_0);
        face_ref.addVertex(vertex_1);
        face_ref.addVertex(vertex_2);
    }

    surf_mesh.computeAllVertexNormals(SurfaceMesh<Scalar>::AVERAGE_FACE_NORMAL);
    ObjMeshIO<Scalar>::save(file_name, &surf_mesh);
}


template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::saveBoundaryMesh(const Tri3DMesh<Scalar> * mesh, const std::string & file_name)
{
    std::cout<<"boundary_vertices size: "<<mesh->vertNum()<<std::endl;
    SurfaceMesh<Scalar> surf_mesh;

    //add vertices
    for (unsigned int vert_idx = 0; vert_idx < mesh->vertNum(); vert_idx++)
    {
        surf_mesh.addVertexPosition(mesh->vertPos(vert_idx));
    }

    //add group
    SurfaceMeshInternal::FaceGroup<Scalar> group;
    surf_mesh.addGroup(group);
    SurfaceMeshInternal::FaceGroup<Scalar> & group_ref = surf_mesh.group(surf_mesh.numGroups()-1);

    //add faces
    for(unsigned int ele_idx = 0; ele_idx < mesh->eleNum(); ele_idx++)
    {
        unsigned int vert_idx0 = mesh->eleVertIndex(ele_idx,0);
        unsigned int vert_idx1 = mesh->eleVertIndex(ele_idx,1);
        unsigned int vert_idx2 = mesh->eleVertIndex(ele_idx,2);

        SurfaceMeshInternal::Face<Scalar> face;
        group_ref.addFace(face);

        Physika::SurfaceMeshInternal::Face<Scalar> & face_ref = group_ref.face(group_ref.numFaces()-1);
        Physika::BoundaryMeshInternal::Vertex<Scalar> vertex_0;
        Physika::BoundaryMeshInternal::Vertex<Scalar> vertex_1;
        Physika::BoundaryMeshInternal::Vertex<Scalar> vertex_2;

        vertex_0.setPositionIndex(vert_idx0);
        vertex_1.setPositionIndex(vert_idx1);
        vertex_2.setPositionIndex(vert_idx2);

        face_ref.addVertex(vertex_0);
        face_ref.addVertex(vertex_1);
        face_ref.addVertex(vertex_2);
    }

    surf_mesh.computeAllVertexNormals(SurfaceMesh<Scalar>::AVERAGE_FACE_NORMAL);
    ObjMeshIO<Scalar>::save(file_name, &surf_mesh);

}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::saveParticlePos()
{
    PDMBase<Scalar,Dim> * pdm_base = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(pdm_base);

    std::fstream file("./VolumetricMesh/intermediate_state_particle_pos.txt", std::ios::out|std::ios::trunc);

    if (file.fail() == true)
    {
        std::cerr<<"error: can't open ./VolumetricMesh/intermediate_state_particle_pos.txt!\n";
        std::exit(EXIT_FAILURE);
    }

    for(unsigned int par_id = 0; par_id < pdm_base->numSimParticles(); par_id++)
    {
        const Vector<Scalar, Dim> & pos = pdm_base->particleCurrentPosition(par_id);
        file<<pos[0]<<" "<<pos[1]<<" "<<pos[2]<<std::endl;
    }
    file.close();
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::saveParticleVel()
{
    PDMBase<Scalar,Dim> * pdm_base = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(pdm_base);

    std::fstream file("./VolumetricMesh/intermediate_state_particle_pos.txt", std::ios::out|std::ios::trunc);

    if (file.fail() == true)
    {
        std::cerr<<"error: can't open ./VolumetricMesh/intermediate_state_particle_pos.txt!\n";
        std::exit(EXIT_FAILURE);
    }

    for(unsigned int par_id = 0; par_id < pdm_base->numSimParticles(); par_id++)
    {
        const Vector<Scalar, Dim> & vel = pdm_base->particleVelocity(par_id);
        file<<vel[0]<<" "<<vel[1]<<" "<<vel[2]<<std::endl;
    }
    file.close();
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::saveVolumetricMesh()
{
    PDMBase<Scalar,Dim> * pdm_base = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(pdm_base);

    VolumetricMeshIO<Scalar, Dim>::save("./VolumetricMesh/intermediate_state_volumetric_mesh.smesh", pdm_base->mesh(), VolumetricMeshIOInternal::SINGLE_FILE);
}

template <typename Scalar, int Dim>
void PDMPluginOutputMesh<Scalar, Dim>::saveImpactPos()
{
    std::fstream file("impact_pos_data.txt", std::ios::out|std::ios::app);
    const Vector<Scalar, Dim> & impact_pos = this->impact_method_->impactPos();
    file<<impact_pos[0]<<" "<<impact_pos[1]<<" "<<impact_pos[2]<<std::endl;
    file.close();
}

// explicit instantiations
template class PDMPluginOutputMesh<double, 3>;
template class PDMPluginOutputMesh<float, 3>;



}