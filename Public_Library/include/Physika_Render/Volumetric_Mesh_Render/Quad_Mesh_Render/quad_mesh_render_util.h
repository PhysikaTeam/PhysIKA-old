/*
 * @file quad_mesh_render_util.h 
 * @Basic class QuadMeshRenderUtil
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
class QuadMesh;

class QuadRenderUtil;

template <typename Scalar>
class QuadMeshRenderUtil: public VolumetricMeshRenderUtilBase<Scalar, 2>
{
public:
    explicit QuadMeshRenderUtil(QuadMesh<Scalar> * mesh, bool auto_compute_normal = true);
    ~QuadMeshRenderUtil() = default;

    //disable copy
    QuadMeshRenderUtil(const QuadMeshRenderUtil &) = delete;
    QuadMeshRenderUtil & operator = (const QuadMeshRenderUtil &) = delete;

    const VolumetricMesh<Scalar, 2> * mesh() const override;
    void setMesh(VolumetricMesh<Scalar, 2> * mesh, bool auto_compute_normal = true) override;

    unsigned int quadNum() const;
    unsigned int eleNum() const override;
    std::shared_ptr<QuadRenderUtil> getInnerQuadRenderUtil();

    void bindQuadMeshLineVAO();
    void unbindQuadMeshLineVAO();

    void bindQuadMeshVAO();
    void unbindQuadMeshVAO();

private:
    void initQuadRenderUtil(bool auto_compute_normal);

private:
    QuadMesh<Scalar> * mesh_ = nullptr;
    
    std::shared_ptr<QuadRenderUtil> quad_render_util_;
};

}//end of namespace Physika