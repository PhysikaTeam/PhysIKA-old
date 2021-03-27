import React, { useState, useEffect } from 'react';
import { Button, Modal, Form, InputNumber, Input, Row, Col, Select, Switch, Upload, Checkbox } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

const { Option } = Select;
const CheckboxGroup = Checkbox.Group;

//TreeNodeAttrModal组件中Form的样式
const formItemLayout = {
    labelCol: { span: 4 },
    wrapperCol: { span: 20 },
};

//upload附带body内容
const uploadBodyContent = {};
//存储上传文件内容
let fileContent;

//使用Hook实现的树结点属性显示Modal
const TreeNodeAttrModal = ({ treeNodeAttr, treeNodeText, visible, hideModal, changeData }) => {
    const [form] = Form.useForm();
    //使用treeNodeText为form赋初值
    const formInitialValues = {};
    //存储Select控件的Options
    const selectOptions = [];
    //存储Checkbox控件的Options
    const checkboxOptions = [];

    const [okDisabled, setOkDisabled] = useState(false);

    useEffect(() => {
        if (form && visible) {
            setFormInitialValues();
            form.resetFields();
        }
    }, [visible]);

    useEffect(() => {
        if (visible && treeNodeAttr.class == 'File') {
            setUploadBodyContent();
        }
    }, [visible]);

    function setUploadBodyContent() {
        uploadBodyContent.userID = window.localStorage.userID;
        uploadBodyContent.uploadDate = Date.now();
    }

    //捕获upload事件对象
    function normFile(e) {
        console.log('Upload event:', e);
        if (e.fileList.length > 1) {
            e.fileList.shift();
        }
        if (e.fileList[0].status !== 'done') {
            setOkDisabled(true);
        }
        //如果文件上传成功，则可以保存
        if (e.fileList[0].status === 'done') {
            setOkDisabled(false);
        }
        return e && e.fileList;
    };

    function isDisabled() {
        return (treeNodeAttr.disabled === 'true');
    }

    function setSelectOptions() {
        treeNodeAttr.enum.split(' ').forEach(item => {
            selectOptions.push(item);
        });
        //注意大小括号：Array.map(item=>(不需要return))；Array.map(item=>{需要return}) 
        return selectOptions.map((item, index) => (
            <Option value={index} key={index}>{item}</Option>
        ));
    }

    function setCheckboxOptions() {
        treeNodeAttr.check.split(' ').forEach(item => {
            checkboxOptions.push(item);
        });
        return checkboxOptions;
    }

    //设置Form的初始化值
    function setFormInitialValues() {
        formInitialValues.name = treeNodeAttr.name;
        formInitialValues.class = treeNodeAttr.class;
        if (treeNodeText !== undefined) {
            switch (treeNodeAttr.class) {
                case 'Real':
                    formInitialValues.real = treeNodeText;
                    break;
                case 'Unsigned':
                    formInitialValues.unsigned = treeNodeText;
                    break;
                case 'Vector2u':
                    formInitialValues.v2u_X = treeNodeText[0];
                    formInitialValues.v2u_Y = treeNodeText[1];
                    break;
                case 'Vector2f':
                    formInitialValues.v2f_X = treeNodeText[0];
                    formInitialValues.v2f_Y = treeNodeText[1];
                    break;
                case 'Vector3f':
                    formInitialValues.v3f_X = treeNodeText[0];
                    formInitialValues.v3f_Y = treeNodeText[1];
                    formInitialValues.v3f_Z = treeNodeText[2];
                    break;
                case 'Enum':
                    formInitialValues.enum_value = selectOptions[treeNodeText];
                    break;
                case 'Bool':
                    formInitialValues.checked = treeNodeText;
                    break;
                case 'File':
                    formInitialValues.upload = (treeNodeText === 'null') ? [] : treeNodeText;
                    break;
                case 'String':
                    formInitialValues.string = treeNodeText;
                    break;
                case 'Check':
                    const checkedList = [];
                    treeNodeText.split(';').forEach(item => {
                        checkedList.push(item);
                    });
                    console.log(checkedList);
                    formInitialValues.checkbox = checkedList;
            }
        }
    }

    //返回树结点修改后的数据
    function returnTreeNodeData(value) {
        const obj = {
            _attributes: treeNodeAttr,
            //_text: ''
        };

        if (treeNodeText !== undefined) {
            switch (treeNodeAttr.class) {
                case 'Real':
                    obj._text = value.real;
                    break;
                case 'Unsigned':
                    obj._text = value.unsigned;
                    break;
                case 'Vector2u':
                    obj._text = [value.v2u_X, value.v2u_Y];
                    break;
                case 'Vector2f':
                    obj._text = [value.v2f_X, value.v2f_Y];
                    break;
                case 'Vector3f':
                    obj._text = [value.v3f_X, value.v3f_Y, value.v3f_Z];
                    break;
                case 'Enum':
                    if (typeof value.enum_value === 'number') {
                        obj._text = value.enum_value;
                    }
                    else {
                        selectOptions.forEach((item, index) => {
                            if (item === value.enum_value)
                                obj._text = index;
                        });
                    }
                    break;
                case 'Bool':
                    obj._text = value.checked;
                    break;
                case 'File':
                    if (!value.upload[0].uploadDate) {
                        value.upload[0].uploadDate = uploadBodyContent.uploadDate;
                    }
                    obj._text = value.upload;
                    obj.fileContent = fileContent;
                    break;
                case 'String':
                    obj._text = value.string;
                    break;
                case 'Check':
                    for (let i = 0; i < value.checkbox.length; ++i) {
                        if (i == 0) {
                            obj._text = value.checkbox[0];
                        }
                        else {
                            obj._text += ';' + value.checkbox[i];
                        }
                    }
                    break;
            }
        }

        changeData(obj);
    }

    return (
        <Modal
            title={"结点属性"}
            visible={visible}
            onOk={() => {
                form.validateFields()
                    .then(value => {
                        //console.log("validateFields", value);
                        returnTreeNodeData(value);
                    })
                    .catch(info => {
                        console.log('Validate Failed:', info);
                    });
            }}
            onCancel={hideModal}
            okText="保存"
            cancelText="取消"
            okButtonProps={{ disabled: okDisabled }}
        >
            <Form
                {...formItemLayout}
                form={form}
                name="nodeAttrModal"
                initialValues={formInitialValues}
            >
                <Form.Item name="name" label="Name" >
                    <Input disabled={true} />
                </Form.Item>
                <Form.Item name="class" label="Class" >
                    <Input disabled={true} />
                </Form.Item>
                {
                    (treeNodeAttr.class === 'Real') &&
                    <Form.Item name="real" label="Value"
                        rules={[{ required: true, message: 'Value cannot be empty!' }]}
                    >
                        <InputNumber step={0.001} disabled={isDisabled()} />
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'Unsigned') &&
                    <Form.Item name='unsigned' label="Value"
                        rules={[{ required: true, message: 'Value cannot be empty!' }]}
                    >
                        <InputNumber formatter={value => `${value}`.replace(/[^\d]+/g, '')} parser={value => value.replace(/[^\d]+/g, '')} disabled={isDisabled()} />
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'Vector2u') &&
                    <Form.Item label="Value">
                        <Row>
                            <Col span={8}>
                                <Form.Item name="v2u_X" label="X"
                                    rules={[{ required: true, message: 'X cannot be empty!' }]}
                                >
                                    <InputNumber formatter={value => `${value}`.replace(/[^\d]+/g, '')} parser={value => value.replace(/[^\d]+/g, '')} disabled={isDisabled()} />
                                </Form.Item>
                            </Col>
                            <Col span={8}>
                                <Form.Item name="v2u_Y" label="Y"
                                    rules={[{ required: true, message: 'Y cannot be empty!' }]}
                                >
                                    <InputNumber formatter={value => `${value}`.replace(/[^\d]+/g, '')} parser={value => value.replace(/[^\d]+/g, '')} disabled={isDisabled()} />
                                </Form.Item>
                            </Col>
                        </Row>
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'Vector2f') &&
                    <Form.Item label="Value">
                        <Row>
                            <Col span={8}>
                                <Form.Item name="v2f_X" label="X"
                                    rules={[{ required: true, message: 'X cannot be empty!' }]}
                                >
                                    <InputNumber step={0.001} disabled={isDisabled()} />
                                </Form.Item>
                            </Col>
                            <Col span={8}>
                                <Form.Item name="v2f_Y" label="Y"
                                    rules={[{ required: true, message: 'Y cannot be empty!' }]}
                                >
                                    <InputNumber step={0.001} disabled={isDisabled()} />
                                </Form.Item>
                            </Col>
                        </Row>
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'Vector3f') &&
                    <Form.Item label="Value">
                        <Row>
                            <Col span={8}>
                                <Form.Item name="v3f_X" label="X"
                                    rules={[{ required: true, message: 'X cannot be empty!' }]}
                                >
                                    <InputNumber step={0.001} disabled={isDisabled()} />
                                </Form.Item>
                            </Col>
                            <Col span={8}>
                                <Form.Item name="v3f_Y" label="Y"
                                    rules={[{ required: true, message: 'Y cannot be empty!' }]}
                                >
                                    <InputNumber step={0.001} disabled={isDisabled()} />
                                </Form.Item>
                            </Col>
                            <Col span={8}>
                                <Form.Item name="v3f_Z" label="Z"
                                    rules={[{ required: true, message: 'Z cannot be empty!' }]}
                                >
                                    <InputNumber step={0.001} disabled={isDisabled()} />
                                </Form.Item>
                            </Col>
                        </Row>
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'Enum') &&
                    <Form.Item name="enum_value" label="Type"
                        rules={[{ required: true }]}
                    >
                        <Select
                            disabled={isDisabled()}
                        >
                            {setSelectOptions()}
                        </Select>
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'Bool') &&
                    <Form.Item name="checked" label="Switch"
                        valuePropName="checked"
                    >
                        <Switch
                            checkedChildren="Y"
                            unCheckedChildren="N"
                            disabled={isDisabled()}
                        >
                            {treeNodeAttr.name}
                        </Switch>
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'File') &&
                    <Form.Item name="upload" label="Upload"
                        valuePropName="fileList"
                        getValueFromEvent={normFile}
                        rules={[{ required: true, message: 'Please upload the corresponding file!' }]}
                    >
                        <Upload action="/uploadFile" listType="picture" showUploadList={{ showRemoveIcon: false }}
                            accept={treeNodeAttr.accept} data={uploadBodyContent}
                            beforeUpload={file => {
                                //获取文件内容
                                const reader = new FileReader();
                                reader.onload = e => {
                                    fileContent = reader.result;
                                };
                                reader.readAsText(file);
                                return true;
                            }}
                        >
                            <Button icon={<UploadOutlined />} disabled={isDisabled()}>Click to upload</Button>
                        </Upload>
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'String') &&
                    <Form.Item name="string" label="Value" >
                        <Input rules={[{ required: true, message: 'Value cannot be empty!' }]} disabled={isDisabled()} />
                    </Form.Item>
                }
                {
                    (treeNodeAttr.class === 'Check') &&
                    <Form.Item name="checkbox" label="Options">
                        <CheckboxGroup options={setCheckboxOptions()}></CheckboxGroup>
                    </Form.Item>
                }
            </Form>
        </Modal>
    );
}

export {
    TreeNodeAttrModal as PhysikaTreeNodeAttrModal
};
