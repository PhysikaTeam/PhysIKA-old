/*
 * @file file_path_utilities.h 
 * @brief Some universal functions when processing files' path.
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

#ifndef PHYSIKA_CORE_UTILITIES_FILE_PATH_UTILITIES_FILE_PATH_UTILITIES_H_
#define PHYSIKA_CORE_UTILITIES_FILE_PATH_UTILITIES_FILE_PATH_UTILITIES_H_

#include<string>

namespace Physika{

namespace FilePathUtilities{

std::string dirName(const std::string &path);   //abstract father_directory of a file path
std::string filenameInPath(const std::string &path);   //abstract filename in a path of a file

} //end of namespace File_Path_Utilities

} //end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_FILE_PATH_UTILITIES_FILE_PATH_UTILITIES_H_
