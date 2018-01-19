/*
 * @file render_task_manager.h 
 * @Basic render task manager
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

#include <list>
#include <memory>

namespace Physika{

class RenderTaskBase;

class RenderTaskManager
{
public:
    RenderTaskManager() = default;
    ~RenderTaskManager() = default;

    unsigned int numRenderTasks() const;

    void insertBack(std::shared_ptr<RenderTaskBase> render_task);
    void insertFront(std::shared_ptr<RenderTaskBase> render_task);
    void insertAtIndex(unsigned int index, std::shared_ptr<RenderTaskBase> render_task);

    void removeBack();
    void removeFront();
    void removeAtIndex(unsigned int index);
    void removeAll();

    std::shared_ptr<const RenderTaskBase> taskAtIndex(unsigned int index) const;
    std::shared_ptr<RenderTaskBase> taskAtIndex(unsigned int index);
    int taskIndex(const std::shared_ptr<const RenderTaskBase> & task) const;

    void renderAllTasks();
    void renderTaskAtIndex(unsigned int index);
private:
    std::list<std::shared_ptr<RenderTaskBase> > render_task_list_;
};
    
}//end of namespace Physika