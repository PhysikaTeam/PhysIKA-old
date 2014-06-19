/*
 * @file opengl_headers.h 
 * @Brief This header points to right directories of opengl headers for different platforms.
 *        Include this header instead of manually include specific files, unless you take
 *        cross-platform into considerations.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_OPENGL_HEADERS_H_
#define PHYSIKA_CORE_UTILITIES_OPENGL_HEADERS_H_

#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#endif

#endif //PHYSIKA_CORE_UTILITIES_OPENGL_HEADERS_H_
