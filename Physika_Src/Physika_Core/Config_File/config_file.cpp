/*
 * @file config_file.cpp
 * @A tool to parse predefined parameters. eg. Read some of simulation fix parameters: gravity, timestep, etc..;
 * @author Sheng Yang
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstdlib>
#include <fstream>
#include <sstream>
#include "Physika_Core/Config_File/config_file.h"

using std::cout;
using std::endl;
using std::fstream;
using std::ifstream;
using std::stringstream;
using std::cerr;
using std::string;
using std::vector;

namespace Physika{

ConfigFile::ConfigFile()
{

}

ConfigFile::~ConfigFile()
{

}

int ConfigFile::findOption(string option_name)
{
    for (unsigned int i = 0; i < option_names_.size(); i++)
    {
        if(option_names_[i] == option_name)
            return i;
    }
    return -1;
}

void ConfigFile::printOptions()
{
    for (unsigned int i = 0; i < option_names_.size(); i++)
    {
        switch (option_types_[i])
        {
            case Option_Int:
                cout<<"Option name: "<<option_names_[i].c_str()<<", it's value: "<<*(int*)(dest_locations_[i])<<endl;
                break;
			case Option_Unsigned_Int:
                cout<<"Option name: "<<option_names_[i].c_str()<<", it's value: "<<*(int*)(dest_locations_[i])<<endl;
                break;
            case Option_Bool:
                cout<<"Option name: "<<option_names_[i].c_str()<<", it's value: "<<*(bool*)(dest_locations_[i])<<endl;
                break;
            case Option_Float:
                cout<<"Option name: "<<option_names_[i].c_str()<<", it's value: "<<*(float*)(dest_locations_[i])<<endl;
                break;
            case Option_Double:
                cout<<"Option name: "<<option_names_[i].c_str()<<", it's value: "<<*(double*)(dest_locations_[i])<<endl;
                break;
            case Option_String:
                cout<<"Option name: "<<option_names_[i].c_str()<<", it's value: "<<*(string*)(dest_locations_[i])<<endl;
                break;
            default:
                cerr<<"Error: invalid type requested !"<<endl;
                std::exit(EXIT_FAILURE);
        }
    }

}


int ConfigFile::addOption(string optionName, int* dest_location)
{
    int code = -1;
    if((code == addOptionOperation(optionName, dest_location)) != 0)
        return code;

    option_types_.push_back(Option_Int);

    return 0;
}

int ConfigFile::addOption(std::string option_name, unsigned int* dest_location)
{
	int code = -1;
    if((code == addOptionOperation(option_name, dest_location)) != 0)
        return code;

    option_types_.push_back(Option_Unsigned_Int);

    return 0;
}


int ConfigFile::addOption(string option_name, bool* dest_location)
{
    int code = -1;
    if((code == addOptionOperation(option_name, dest_location)) != 0)
        return code;

    option_types_.push_back(Option_Bool);

    return 0;
}

int ConfigFile::addOption(string option_name, float* dest_location)
{
    int code = -1;
    if((code == addOptionOperation(option_name, dest_location)) != 0)
        return code;

    option_types_.push_back(Option_Float);

    return 0;
}

int ConfigFile::addOption(string option_name, double* dest_location)
{
    int code = -1;
    if((code == addOptionOperation(option_name, dest_location)) != 0)
        return code;

    option_types_.push_back(Option_Double);

    return 0;
}

int ConfigFile::addOption(string option_name, string* dest_location)
{
    int code = -1;
    if((code == addOptionOperation(option_name, dest_location)) != 0)
        return code;

    option_types_.push_back(Option_String);

    return 0;
}

bool ConfigFile::parseFile(string file_name)
{
    ifstream inputstream(file_name.c_str());

    if(!inputstream)
    {
        cout<<"Couldn't open config file :"<<file_name<<endl;
        return false;
    }

    const int maxlen = 1000;
    char line[maxlen];
    unsigned int count = 0;

    while (inputstream)
    {
        count++;

        inputstream.getline(line, maxlen);
        string str(line);

        if(str[0] == '#' || str[0] == '\0')
            continue;

        if(str[0] != '*')
        {
            cout<<"Error: invalid line "<<count<<", "<<str<<endl;
            inputstream.close();
            return false;
        }

        string option_name(str.begin()+1, str.end());
        int index = findOption(option_name);

        if(index == -1)
        {
            cout<<"Warning: unknown option on line "<<count<<", "<<option_name<<endl;
            continue;
        }
        
        if(!inputstream)
        {
            cerr<<"Error: EOF reached without specifying option value."<<endl;
            return false;
           // std::exit(EXIT_FAILURE);
        }

        //find a invalid option_name, now set a value;
        string data_entry = "";
        while(inputstream)
        {
            count++;
            inputstream.getline(line, maxlen);
            string data(line);
            if(data[0] == '#' || data[0] == '\0')
                continue;
            data_entry = data;
            break;
        }
        if(data_entry == "")
        {
            cerr<<"Error: EOF reached without specifying option value."<<endl;
            cerr<<"Error: Not specifying for: "<<option_names_[index]<<endl;
            return false;
           // std::exit(EXIT_FAILURE);
        }

        if(option_types_[index] == Option_String)
        {
           *(string*)dest_locations_[index] = data_entry;
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
                cerr<<"Error: invalid boolean specification: line "<<count<<" "<<data_entry<<endl;
                inputstream.close();
                return false;
            //    std::exit(EXIT_FAILURE);
            }
        }
        else if(option_types_[index] == Option_Unsigned_Int)
        {
            for (unsigned int i = 1; i < data_entry.length(); i++)
            {
                if(data_entry[i] < '0' || data_entry[i] > '9')
                {
                    cout<<"Error: invalid int specification: line "<<count<<" "<<data_entry<<endl;
                    inputstream.close();
                    return false;
                 //   return 1;
                }
            }
            stringstream stream;
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
                    cout<<"Error: invalid int specification: line "<<count<<" "<<data_entry<<endl;
                    inputstream.close();
                    return false;
                 //   return 1;
                }
            }
            stringstream stream;
            stream << data_entry;
            int data;
            stream >> data;
            *(int*) dest_locations_[index] = data;
        }
        else
        {
            string data_tmp = "";
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

                        cerr<<"Error: invalid float/double specification: line "<<count<<" "<<data_entry<<endl;
                        inputstream.close();
                        return false;
                        //  std::exit(EXIT_FAILURE);
                    }
                }
            }

            stringstream stream;
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
            cerr<<"Error: option "<<option_names_[i]<<" didn't have an entry in the config file!"<<endl;
            return false;
           // std::exit(EXIT_FAILURE);
        }
    }

    return 0;
}



template <class T>
int ConfigFile::addOptionOptional(std::string option_name, T* dest_location, T default_value)
{
    int code = addOption(option_name, dest_location);
    *dest_location = default_value;
    option_set_[option_set_.size() - 1] = true;
    return code;
}

template <class T>
int ConfigFile::addOptionOperation(std::string option_name, T* dest_location)
{
    if(findOption(option_name) != -1)
    {
        std::cout<<"Warning: option "<<option_name<<" already exists. Ignoring request tp re-add it."<<std::endl;
        return false;
    }

    option_names_.push_back(option_name);
    dest_locations_.push_back((void*)dest_location);
    option_set_.push_back(false);
    return true;
}

template int ConfigFile::addOptionOptional<int> (std::string option_name, int* dest_location, int default_value);
template int ConfigFile::addOptionOptional<unsigned int> (std::string option_name, unsigned int* dest_location, unsigned int default_value);
template int ConfigFile::addOptionOptional<float> (std::string option_name, float* dest_location, float default_value);
template int ConfigFile::addOptionOptional<double> (std::string option_name, double* dest_location, double default_value);
template int ConfigFile::addOptionOptional<bool> (std::string option_name, bool* dest_location, bool default_value);
template int ConfigFile::addOptionOptional<std::string> (std::string option_name, std::string* dest_location, std::string default_value);

}//end of namespace Physika

