import React, { useState, useEffect } from 'react';
import { Modal, Form, Input, Button, Row, Col } from 'antd';

//Form的样式
const formItemLayout = {
    labelCol: {
        span: 6,
    },
    wrapperCol: {
        span: 14,
    },
};

const tailLayout = {
    wrapperCol: {
        offset: 6,
        span: 14,
    },
};

const LoginModal = ({ visible, hideModal, changeUserStatus }) => {
    const [form] = Form.useForm();
    const [register, setRegister] = useState(false);
    const [RLoading, setRLoading] = useState(false);
    const [LLoading, setLLoading] = useState(false);

    const registerSuccess = (values) => {
        setRLoading(true);
        setTimeout(() => {
            //注册
            console.log('注册成功', values);
            window.localStorage.clear();
            window.localStorage.userID = values.RUsername;
            //token
            setRLoading(false);
            changeUserStatus(true);
        }, 5000);
        
    }

    const registerFailed = (errorInfo) => {
        console.log('注册失败', errorInfo);
    }

    const loginSuccess = (values) => {
        //登录
        console.log('Success:', values);
        window.localStorage.clear();
        //token
        window.localStorage.userID = values.LUsername;
        changeUserStatus(true);
    };

    const loginFailed = (errorInfo) => {
        console.log('Failed:', errorInfo);
    };

    return (
        <Modal
            title={register ? "用户注册" : "用户登录"}
            visible={visible}
            //closable={false}
            onCancel={hideModal}
            destroyOnClose={true}
            footer={null}
        >
            {
                register
                    ? <Form
                        {...formItemLayout}
                        form={form}
                        name="Register"
                        preserve={false}
                        onFinish={registerSuccess}
                        onFinishFailed={registerFailed}
                    >
                        <Form.Item
                            label="用户名"
                            name="RUsername"
                            rules={[
                                { required: true, message: 'Please input your username!' }
                            ]}
                        >
                            <Input />
                        </Form.Item>

                        <Form.Item
                            label="密码"
                            name="RPassword"
                            rules={[
                                { required: true, message: 'Please input your password!' },
                            ]}
                        >
                            <Input.Password />
                        </Form.Item>

                        <Form.Item
                            label="确认密码"
                            name="confirm"
                            dependencies={['RPassword']}
                            hasFeedback
                            rules={[
                                { required: true, message: 'Please input your password!' },
                                ({ getFieldValue }) => ({
                                    validator(_, value) {
                                        if (!value || getFieldValue('RPassword') === value) {
                                            return Promise.resolve();
                                        }
                                        return Promise.reject('The two passwords that you entered do not match!');
                                    },
                                }),
                            ]}
                        >
                            <Input.Password />
                        </Form.Item>

                        <Form.Item {...tailLayout}>
                            <Row>
                                <Col span={8}>
                                    <Button type="primary" htmlType="submit" loading={RLoading}>注册</Button>
                                </Col>
                                <Col span={8}>
                                    <Button type="link" onClick={() => setRegister(false)}>返回登录</Button>
                                </Col>
                            </Row>
                        </Form.Item>
                    </Form>
                    : <Form
                        {...formItemLayout}
                        form={form}
                        name="Login"
                        preserve={false}
                        onFinish={loginSuccess}
                        onFinishFailed={loginFailed}
                    >
                        <Form.Item
                            label="用户名"
                            name="LUsername"
                            rules={[
                                { required: true, message: 'Please input your username!' }
                            ]}
                        >
                            <Input />
                        </Form.Item>

                        <Form.Item
                            label="密码"
                            name="LPassword"
                            rules={[
                                { required: true, message: 'Please input your password!' },
                            ]}
                        >
                            <Input.Password />
                        </Form.Item>

                        <Form.Item {...tailLayout}>
                            <Row>
                                <Col span={8}>
                                    <Button type="primary" htmlType="submit" loading={LLoading}>登录</Button>
                                </Col>
                                <Col span={8}>
                                    <Button type="link" onClick={() => setRegister(true)}>注册新账号</Button>
                                </Col>
                            </Row>
                        </Form.Item>
                    </Form>
            }
        </Modal>
    );
}

export {
    LoginModal as LoginModal
};