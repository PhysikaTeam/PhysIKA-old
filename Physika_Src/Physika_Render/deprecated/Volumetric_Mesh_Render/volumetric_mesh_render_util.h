/*
 * @file volumetric_mesh_render_util.h 
 * @Basic class VolumetricMeshRenderUtil
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#pragma once

#include <memory>
#include "Physika_Core/Utilities/dimension_trait.h"

namespace Physika{

template <typename Scalar, int Dim>
class VolumetricMesh;

template <typename Scalar, int Dim>
class VolumetricMeshRenderUtilBase;

class PointRenderUtil;

//Note: VolumetricMeshRenderUtil do not inherit from VolumetricMeshRenderUtilBase.
//      Here we delegate the task of VolumetricMeshRenderUtil to the derived class of VolumetricMeshRenderUtilBase,
//      which means that we use combination. 

template <typename Scalar, int Dim>
class VolumetricMeshRenderUtil
{
public:
    VolumetricMeshRenderUtil(VolumetricMesh<Scalar, Dim> * mesh, bool auto_compute_normal = true);


    const VolumetricMesh<Scalar, Dim> * mesh() const;
    void setMesh(VolumetricMesh<Scalar, Dim> * mesh, bool auto_compute_normal = true);

    //design for render tasks
    std::shared_ptr<VolumetricMeshRenderUtilBase<Scalar, Dim>> getInnerVolumetricMeshRenderUtilBase();
    std::shared_ptr<PointRenderUtil> getInnerPointRenderUtil();

    unsigned int pointNum() const;
    unsigned int eleNum() const;


    /*
    
    You are not gonna need it!

    void bindPointVAO();
    void unbindPointVAO();

    void bindLineVAO();
    void unbindLineVAO();

    void bindMeshVAO();
    void unbindMeshVAO();
    */

private:
    void setRenderUtil(VolumetricMesh<Scalar, Dim> * mesh, bool auto_compute_normal);
    void setRenderUtil(VolumetricMesh<Scalar, 2> * mesh, bool auto_compute_normal, DimensionTrait<2>);
    void setRenderUtil(VolumetricMesh<Scalar, 3> * mesh, bool auto_compute_normal, DimensionTrait<3>);

private:
    std::shared_ptr<VolumetricMeshRenderUtilBase<Scalar, Dim>> render_util_;
};

}