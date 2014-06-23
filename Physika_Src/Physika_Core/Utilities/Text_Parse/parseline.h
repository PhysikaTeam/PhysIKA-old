/*
 * @file parseline.h 
 * @brief Some universal functions when preprocessing an input line.
 * @author LiYou Xu
 * @acknowledge Jernej Barbic, author of VegaFEM
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_TEXT_PARSE_PARSELINE_H_
#define PHYSIKA_CORE_UTILITIES_TEXT_PARSE_PARSELINE_H_

#include<string>

namespace Physika{

namespace TextParse{

//remove abundant whitespaces
std::string removeWhitespaces(const std::string &line_, unsigned int numRetainedSpaces = 1);

} //end of namespace TextParse

} //end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_TEXT_PARSE_PARSELINE_H_
