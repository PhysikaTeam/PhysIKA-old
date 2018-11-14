/*
 * @file volumetric_mesh_render_util_base.h 
 * @Basic class VolumetricMeshRenderUtilBase
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

namespace Physika{

template <typename Scalar, int Dim>
class VolumetricMesh;

class PointRenderUtil;

//Note: Consider all VolumetricMeshs have the same PointRenderUtil, we thus define VolumetricMeshRenderUtilBase to encapsulate it.
//      However, we may should use combination rather than inheritance to implement this. 
//      The other reason  we choose inheritance is the design of VolumetricMeshRenderUtil where we just need a base class. 
   
template <typename Scalar, int Dim>
class VolumetricMeshRenderUtilBase
{
public:
    VolumetricMeshRenderUtilBase(VolumetricMesh<Scalar, Dim> * mesh);
    virtual ~VolumetricMeshRenderUtilBase() = 0;

    //disable copy
    VolumetricMeshRenderUtilBase(const VolumetricMeshRenderUtilBase &) = delete;
    VolumetricMeshRenderUtilBase & operator = (const VolumetricMeshRenderUtilBase &) = delete;

    virtual const VolumetricMesh<Scalar, Dim> * mesh() const;
    virtual void setMesh(VolumetricMesh<Scalar, Dim> * mesh, bool auto_compute_normal = true);

    unsigned int pointNum() const;
    virtual unsigned int eleNum() const = 0;  //for different type of volumetric mesh

    std::shared_ptr<PointRenderUtil> getInnerPointRenderUtil();

    void bindPointVAO();
    void unbindPointVAO();

private:
    void initPointRenderUtil();

private:
    VolumetricMesh<Scalar, Dim> * mesh_ = nullptr;
    std::shared_ptr<PointRenderUtil> point_render_util_;
};

}//end of namespace Physika