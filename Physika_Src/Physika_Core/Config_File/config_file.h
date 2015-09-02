/*
 * @file config_file.h
 * @brief A tool to parse predefined parameters from config files.
 * @author Sheng Yang, Fei Zhu
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_CONFIG_FILE_CONFIG_FILE_H_
#define PHYSIKA_CORE_CONFIG_FILE_CONFIG_FILE_H_

#include <vector>
#include <string>

namespace Physika{

/*
* @Suggestion: The name of a config file should be "***.config"
* @The rule and format of the config file as follows:
* --------------------------------------
* @Format of config file:
* 1. '*': name of parameters
* 2. '#': comment
* 3. regular characters: parameter value
* --------------------------------------
* @Example:
*   *parameter_0_name
*   parameter_0_value
*   *parameter_1_name
*   #parameter_1_note
*   #parameter_1_note
*   #parameter_1_note
*   parameter_1_value
*   #parameter_1_note
*   *................
*   #................
*   *................
*   #................
* --------------------------------------
*/

class ConfigFile
{
enum OptionType
{
    Option_Int,
    Option_Unsigned_Int,
    Option_Bool,
    Option_Float,
    Option_Double,
    Option_String,
};
public:
    ConfigFile();
    ~ConfigFile();

    bool addOption(const std::string &option_name, int* dest_location);
	bool addOption(const std::string &option_name, unsigned int* dest_location);
    bool addOption(const std::string &option_name, bool* dest_location);
    bool addOption(const std::string &option_name, float* dest_location);
    bool addOption(const std::string &option_name, double* dest_location);
    bool addOption(const std::string &option_name, std::string* dest_location);

    template <class T>
    bool addOptionOptional(const std::string &option_name, T* dest_location, T default_value);

    //parse file after options are added
    bool parseFile(const std::string &file_name);

    void printOptions() const; //print all options already read in memory
    bool saveOptions(const std::string &file_name) const;
protected:
    template <class T>
    bool addOptionOperation(const std::string &option_name, T* dest_location);
    //return -1 if cannot find
    int findOption(const std::string &option_name);
protected:
    std::vector<std::string> option_names_;
    std::vector<int> option_types_;
    std::vector<void*> dest_locations_;
    std::vector<bool> option_set_;
};

}//end of namespace Physika

#endif // PHYSIKA_CORE_CONFIG_FILE_CONFIG_FILE_H_
