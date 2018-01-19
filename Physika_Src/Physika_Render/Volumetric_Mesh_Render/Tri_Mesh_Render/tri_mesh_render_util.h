/*
 * @file tri_mesh_render_util.h 
 * @Basic class TriMeshRenderUtil
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

#include "Physika_Render/Volumetric_Mesh_Render/Volumetric_Mesh_Render_Util_Base/volumetric_mesh_render_util_base.h"

namespace Physika{

template <typename Scalar>
class TriMesh;

class TriangleRenderUtil;

template <typename Scalar>
class TriMeshRenderUtil: public VolumetricMeshRenderUtilBase<Scalar, 2>
{
public:
    explicit TriMeshRenderUtil(TriMesh<Scalar> * mesh, bool auto_compute_normal = true);
    ~TriMeshRenderUtil() = default;

    //disable copy
    TriMeshRenderUtil(const TriMeshRenderUtil &) = delete;
    TriMeshRenderUtil & operator = (const TriMeshRenderUtil &) = delete;

    const VolumetricMesh<Scalar, 2> * mesh() const override;
    void setMesh(VolumetricMesh<Scalar, 2> * mesh, bool auto_compute_normal = true) override;

    unsigned int triangleNum() const;
    unsigned int eleNum() const override;
    std::shared_ptr<TriangleRenderUtil> getInnerTriangleRenderUtil();

    void bindTriMeshVAO();
    void unbindTriMeshVAO();

private:
    void initTriangleRenderUtil(bool auto_compute_normal);

private:
    TriMesh<Scalar> * mesh_ = nullptr;
    
    std::shared_ptr<TriangleRenderUtil> tri_render_util_;
};

}//end of namespace Physika