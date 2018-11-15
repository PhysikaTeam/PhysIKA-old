/*
 * @file tetrahedron_solid_render_task.h 
 * @Basic solid render task of tetrahedron
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

#include "Physika_Render/Triangle_Render/triangle_solid_render_task.h"


namespace Physika{

class TetrahedronRenderUtil;

class TetrahedronSolidRenderTask: public TriangleSolidRenderTask
{
public:
    explicit TetrahedronSolidRenderTask(std::shared_ptr<TetrahedronRenderUtil> render_util);
    ~TetrahedronSolidRenderTask();

    //disable copy
    TetrahedronSolidRenderTask(const TetrahedronSolidRenderTask &) = delete;
    TetrahedronSolidRenderTask & operator = (const TetrahedronSolidRenderTask &) = delete;

    void setTetColors(const std::vector<Color4f> & colors);
    void setElementColors(const std::vector<Color4f> & colors) override;

private:
    std::shared_ptr<TetrahedronRenderUtil> render_util_;
};

}//end of namespace Physika    