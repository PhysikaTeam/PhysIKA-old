/*
 * @file parseline.cpp
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


#include "Physika_Core/Utilities/File_Utilities/parse_line.h"
using std::string;

namespace Physika{

namespace FileUtilities{

string removeWhitespaces(const string &line_, unsigned int numRetainedSpaces)
{
    string::size_type pos;
    string line = line_;
    string whitespace(" "), retained_whitespaces(" ");
    for(unsigned int i=0; i<numRetainedSpaces; ++i)retained_whitespaces+= whitespace;
    while(line[0] == ' ')line = line.substr(1);
    while((pos=line.find(retained_whitespaces)) != string::npos)line.erase(pos,1);
    return line;
}

} //end of namespace FileUtilities

} //end of namespace Physika

