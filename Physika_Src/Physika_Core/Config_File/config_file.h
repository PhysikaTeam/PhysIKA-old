/*
 * @file config_file.h 
 * @A tool to parse predefined parameters. eg. Read some of simulation fix parameters: gravity, timestep, etc..;
 * @author Sheng Yang
 * @Suggestion: The name of a config file should be "***.config"
 * @The rule and format of the config file as follows: 
 --------------------------------------
 * @Rule:
 * 1. a '*' stands for a parameter's name 
 * 2. '#' stands for a note or you can use it as a hidden value for the nearest upper parameter. 
 * 3. no special character stands for a value of the nearest upper parameter .
 --------------------------------------
 * @Format:
    *parameter_0_name
    parameter_0_value
    *parameter_1_name
    #parameter_1_note
    #parameter_1_note
    #parameter_1_note
    parameter_1_value
    #parameter_1_note
    *................
    #................
    *................
    #................
 --------------------------------------
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHSYIKA_CORE_CONFIG_FILE_CONFIG_FILE_H_
#define PHSYIKA_CORE_CONFIG_FILE_CONFIG_FILE_H_

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace Physika{

class ConfigFile
{

    enum OptionType
    {
        Option_Int,
        Option_Bool,
        Option_Float,
        Option_Double,
        Option_String,
    };


public:

    ConfigFile();
    ~ConfigFile();

    int addOption(string option_name, int* dest_location);
    int addOption(string option_name, bool* dest_location);
    int addOption(string option_name, float* dest_location);
    int addOption(string option_name, double* dest_location);
    int addOption(string option_name, string* dest_location);

    template <class T>
    int addOptionOptional(string option_name, T* dest_location, T default_value);

    int parseFile(string file_name);

    void printOptions(); //print all options alread read in memory.
    
protected:

    vector<string> option_names_;
    vector<int> option_types_;
    vector<void*> dest_locations_;
    vector<bool> option_set_;

    template <class T>
    int addOptionOperation(string option_name, T* dest_location);


    int findOption(string option_name);// find a option in the option_names_, if not find return -1,else return the index;
};



}//end of namespace Physika

#endif // PHSYIKA_CORE_CONFIG_FILE_CONFIG_FILE_H_