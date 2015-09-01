/*
 * @file config_file.cpp
 * @A tool to parse predefined parameters from config files.
 * @author Sheng Yang, Fei Zhu
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "Physika_Core/Config_File/config_file.h"

namespace Physika{

ConfigFile::ConfigFile()
{
}

ConfigFile::~ConfigFile()
{
}

bool ConfigFile::addOption(const std::string &option_name, int* dest_location)
{
    if(!addOptionOperation(option_name, dest_location))
        return false;
    option_types_.push_back(Option_Int);
    return true;
}

bool ConfigFile::addOption(const std::string &option_name, unsigned int* dest_location)
{
    if(!addOptionOperation(option_name, dest_location))
        return false;
    option_types_.push_back(Option_Unsigned_Int);
    return true;
}

bool ConfigFile::addOption(const std::string &option_name, bool* dest_location)
{
    if(!addOptionOperation(option_name, dest_location))
        return false;
    option_types_.push_back(Option_Bool);
    return true;
}

bool ConfigFile::addOption(const std::string &option_name, float* dest_location)
{
    if(!addOptionOperation(option_name, dest_location))
        return false;
    option_types_.push_back(Option_Float);
    return true;
}

bool ConfigFile::addOption(const std::string &option_name, double* dest_location)
{
    if(!addOptionOperation(option_name, dest_location))
        return false;
    option_types_.push_back(Option_Double);
    return true;
}

bool ConfigFile::addOption(const std::string &option_name, std::string* dest_location)
{
    if(!addOptionOperation(option_name, dest_location))
        return false;
    option_types_.push_back(Option_String);
    return true;
}

bool ConfigFile::parseFile(const std::string &file_name)
{
    std::ifstream inputstream(file_name.c_str());
    if(!inputstream)
    {
        std::cerr<<"Couldn't open config file :"<<file_name<<std::endl;
        return false;
    }
    const unsigned int maxlen = 1000;
    char line[maxlen];
    unsigned int count = 0;
    while (inputstream)
    {
        count++;
        inputstream.getline(line, maxlen);
        std::string str(line);
        if(str[0] == '#' || str[0] == '\0')
            continue;
        if(str[0] != '*')
        {
            std::cerr<<"Error: invalid line "<<count<<", "<<str<<std::endl;
            inputstream.close();
            return false;
        }
        std::string option_name(str.begin()+1, str.end());
        int index = findOption(option_name);
        if(index == -1)
        {
            std::cerr<<"Warning: unknown option on line "<<count<<", "<<option_name<<std::endl;
            continue;
        }
        if(!inputstream)
        {
            std::cerr<<"Error: EOF reached without specifying option value."<<std::endl;
            return false;
        }
        //found a valid option_name, now set a value;
        std::string data_entry("");
        while(inputstream)
        {
            count++;
            inputstream.getline(line, maxlen);
            std::string data(line);
            if(data[0] == '#' || data[0] == '\0')
                continue;
            data_entry = data;
            break;
        }
        if(data_entry == "")
        {
            std::cerr<<"Error: EOF reached without specifying option value."<<std::endl;
            std::cerr<<"Error: Not specifying for: "<<option_names_[index]<<std::endl;
            return false;
        }

        if(option_types_[index] == Option_String)
        {
           *(std::string*)dest_locations_[index] = data_entry;
        }
        else if(option_types_[index] == Option_Bool)
        {
            if(data_entry == "0" || data_entry == "1" || data_entry == "true" || data_entry == "false")
            {
                if(data_entry == "0" || data_entry == "false")
                    *(bool*) dest_locations_[index] = false;
                else
                    *(bool*) dest_locations_[index] = true;
            }
            else
            {
                std::cerr<<"Error: invalid boolean specification: line "<<count<<" "<<data_entry<<std::endl;
                inputstream.close();
                return false;
            }
        }
        else if(option_types_[index] == Option_Unsigned_Int)
        {
            for (unsigned int i = 1; i < data_entry.length(); i++)
            {
                if(data_entry[i] < '0' || data_entry[i] > '9')
                {
                    std::cerr<<"Error: invalid int specification: line "<<count<<" "<<data_entry<<std::endl;
                    inputstream.close();
                    return false;
                }
            }
            std::stringstream stream;
            stream << data_entry;
            int data;
            stream >> data;
            *(unsigned int*) dest_locations_[index] = data;
        }
		else if(option_types_[index] == Option_Int)
        {
            for (unsigned int i = 0; i < data_entry.length(); i++)
            {
                if(data_entry[i] < '0' || data_entry[i] > '9')
                {
					if(i == 0 && (data_entry[i] == '+' || data_entry[i] == '-'))
						continue;
                    std::cerr<<"Error: invalid int specification: line "<<count<<" "<<data_entry<<std::endl;
                    inputstream.close();
                    return false;
                }
            }
            std::stringstream stream;
            stream << data_entry;
            int data;
            stream >> data;
            *(int*) dest_locations_[index] = data;
        }
        else
        {
            std::string data_tmp("");
            if(data_entry[data_entry.length()-1] == 'f' || data_entry[data_entry.length()-1] == 'd')
               data_tmp.assign(data_entry.begin(), data_entry.end()-1);
            else
               data_tmp = data_entry;

			bool dot_flag = false;
            if(data_tmp != "")
            {
                for (unsigned int i = 0; i < data_tmp.length()-1; i++)
                {
                    if(data_tmp[i] < '0' || data_tmp[i] > '9')
                    {
                        if(data_tmp[i] == '.' && dot_flag == false)
                        {
							dot_flag = true;
							continue;
                        }
                        if(i == 0 && (data_tmp[i] == '+' || data_tmp[i] == '-'))
                            continue;
                        std::cerr<<"Error: invalid float/double specification: line "<<count<<" "<<data_entry<<std::endl;
                        inputstream.close();
                        return false;
                    }
                }
            }
            std::stringstream stream;
            stream << data_tmp;
            if(option_types_[index] == Option_Float)
            {
                float data;
                stream >> data;
                *(float*) dest_locations_[index] = data;
            }
            else
            {
                double data;
                stream >> data;
                *(double*) dest_locations_[index] = data;
            }
        }
        option_set_[index] = true;
    }
    inputstream.close();
    for (unsigned int i = 0; i < option_names_.size(); i++)
    {
        if(!option_set_[i])
        {
            std::cerr<<"Error: option "<<option_names_[i]<<" didn't have an entry in the config file!"<<std::endl;
            return false;
        }
    }
    return true;
}

template <class T>
bool ConfigFile::addOptionOptional(const std::string &option_name, T* dest_location, T default_value)
{
    if(!addOption(option_name, dest_location))
        return false;
    *dest_location = default_value;
    option_set_[option_set_.size() - 1] = true;
    return true;
}

int ConfigFile::findOption(const std::string &option_name)
{
    std::vector<std::string>::iterator iter = std::find(option_names_.begin(),option_names_.end(),option_name);
    if(iter == option_names_.end())
        return -1;
    else
        return iter - option_names_.begin();
}

void ConfigFile::printOptions()
{
    for (unsigned int i = 0; i < option_names_.size(); i++)
    {
        switch (option_types_[i])
        {
        case Option_Int:
            std::cout<<"Option name: "<<option_names_[i].c_str()<<", value: "<<*(int*)(dest_locations_[i])<<std::endl;
            break;
		case Option_Unsigned_Int:
            std::cout<<"Option name: "<<option_names_[i].c_str()<<", value: "<<*(unsigned int*)(dest_locations_[i])<<std::endl;
            break;
        case Option_Bool:
            std::cout<<"Option name: "<<option_names_[i].c_str()<<", value: "<<*(bool*)(dest_locations_[i])<<std::endl;
            break;
        case Option_Float:
            std::cout<<"Option name: "<<option_names_[i].c_str()<<", value: "<<*(float*)(dest_locations_[i])<<std::endl;
            break;
        case Option_Double:
            std::cout<<"Option name: "<<option_names_[i].c_str()<<", value: "<<*(double*)(dest_locations_[i])<<std::endl;
            break;
        case Option_String:
            std::cout<<"Option name: "<<option_names_[i].c_str()<<", value: "<<*(std::string*)(dest_locations_[i])<<std::endl;
            break;
        default:
            std::cerr<<"Warning: invalid type requested !"<<std::endl;
            break;
        }
    }

}

template <class T>
bool ConfigFile::addOptionOperation(const std::string &option_name, T* dest_location)
{
    if(findOption(option_name) != -1)
    {
        std::cerr<<"Warning: option "<<option_name<<" already exists. Ignoring request tp re-add it."<<std::endl;
        return false;
    }
    option_names_.push_back(option_name);
    dest_locations_.push_back((void*)dest_location);
    option_set_.push_back(false);
    return true;
}

template bool ConfigFile::addOptionOptional<int> (const std::string &option_name, int* dest_location, int default_value);
template bool ConfigFile::addOptionOptional<unsigned int> (const std::string &option_name, unsigned int* dest_location, unsigned int default_value);
template bool ConfigFile::addOptionOptional<float> (const std::string &option_name, float* dest_location, float default_value);
template bool ConfigFile::addOptionOptional<double> (const std::string &option_name, double* dest_location, double default_value);
template bool ConfigFile::addOptionOptional<bool> (const std::string &option_name, bool* dest_location, bool default_value);
template bool ConfigFile::addOptionOptional<std::string> (const std::string &option_name, std::string* dest_location, std::string default_value);

}//end of namespace Physika
