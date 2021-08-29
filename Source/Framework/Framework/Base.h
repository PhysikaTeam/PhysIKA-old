/**
 * @file Base.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief Base class that is used to control all fields
 * @version 0.1
 * @date 2019-06-12
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include <iostream>
#include "Framework/Framework/Field.h"
#include "Framework/Framework/Object.h"

namespace PhysIKA {
/**
*  \brief Base class for modules
*
*  This class contains all functionality shared by every module in PhysIKA.
*  It defines how to retrieve information about an class (name, type, data fields).
*
*/

typedef std::string FieldID;

class Base : public Object
{
public:
    typedef std::vector<Field*>       FieldVector;
    typedef std::map<FieldID, Field*> FieldMap;

    Base()
        : Object(){};
    ~Base() override{};

    /**
     * @brief Add a field to Base
     * FieldID will be set to the name of Field by default
     */
    bool addField(Field* data);
    /**
     * @brief Add a field to Base
     * 
     * @param Field name
     * @param Field pointer
     */
    bool addField(FieldID name, Field* data);
    bool addFieldAlias(FieldID name, Field* data);
    bool addFieldAlias(FieldID name, Field* data, FieldMap& fieldAlias);

    /**
     * @brief Find a field by its pointer
     * 
     * @param data Field pointer
     */
    bool findField(Field* data);
    /**
     * @brief Find a field by its name
     * 
     * @param name Field name
     */
    bool findFieldAlias(const FieldID name);
    /**
     * @brief Find a field in fieldAlias by its name
     * This function is typically called by other functions
     * 
     * @param name Field name
     * @param fieldAlias All fields the searching is taken on
     */
    bool findFieldAlias(const FieldID name, FieldMap& fieldAlias);

    /**
     * @brief Remove a field by its pointer
     * 
     */
    bool removeField(Field* data);
    /**
     * @brief Remove a field by its name
     * 
     */
    bool removeFieldAlias(const FieldID name);
    bool removeFieldAlias(const FieldID name, FieldMap& fieldAlias);

    /**
     * @brief Return a field by its name
     * 
     */
    Field* getField(const FieldID name);

    std::vector<Field*>& getAllFields();

    /**
     * @brief Attach a field to Base
     * 
     * @param field Field pointer
     * @param name Field name
     * @param desc Field description
     * @param autoDestroy The field will be destroyed by Base if true, otherwise, the field should be explicitly destroyed by its creator.
     * 
     * @return Return false if the name conflicts with exists fields' names
     */
    virtual bool attachField(Field* field, std::string name, std::string desc, bool autoDestroy = true);

    template <typename T>
    T* getField(FieldID name)
    {
        FieldMap::iterator iter = m_fieldAlias.find(name);
        if (iter != m_fieldAlias.end())
        {
            return dynamic_cast<T*>(iter->second);
        }
        return nullptr;
    }

    /**
     * @brief Check the completeness of all required fields
     */
    bool isAllFieldsReady();

    std::vector<FieldID> getFieldAlias(Field* data);
    int                  getFieldAliasCount(Field* data);

    inline void setBlockCoord(float x, float y)
    {
        block_x = x;
        block_y = y;
    }

    inline float bx()
    {
        return block_x;
    }
    inline float by()
    {
        return block_y;
    }

private:
    float block_x = 0.0f;
    float block_y = 0.0f;

    FieldVector m_field;
    FieldMap    m_fieldAlias;
};

}  // namespace PhysIKA
