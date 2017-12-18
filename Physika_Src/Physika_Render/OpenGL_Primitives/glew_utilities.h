/*
 * @file glew_utilities.h 
 * @Brief openGL glew utilities
 * @author: Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

/* Note: since glew.h must be included before gl.h to pass compiler,
 *       you need include this file (glew_utilities.h) before opengl_primitives.h
 *       
 *       This file is used to add utility function that use glew library
 */

#pragma  once

#include <iostream>
#include <GL/glew.h>

inline GLenum openGlCheckCurFramebufferStatus(GLenum target = GL_FRAMEBUFFER)
{
    GLenum status = glCheckFramebufferStatus(target);
    switch (status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        break;

    case GL_FRAMEBUFFER_UNDEFINED:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_UNDEFINED " << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT " << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT " << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER" << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER" << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_UNSUPPORTED:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_UNSUPPORTED" << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE" << std::endl;
        break;
    }

    default:
    {
        std::cerr << "framebuffer error: unknown" << std::endl;
        break;
    }

    }

    return status;
}