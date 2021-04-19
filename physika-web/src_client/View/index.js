import React, { useState, useEffect } from 'react';
import { Layout, Menu, Dropdown, Button, Row, Col, Avatar } from 'antd';
import { DownOutlined, UserOutlined } from '@ant-design/icons';
import '../../static/css/antdesign.css';
import { PhysikaCloudNature } from './CloudNature';
import { PhysikaCloudSatellite } from './CloudSatellite';
import { PhysikaCloudWRF } from './CloudWRF';
import { LoginModal } from './LoginModal';
import { MySPH } from './SPH';
import { PhysikaCloth } from './Cloth';
import {ParticleEvolution} from './ParticleEvolution';

const { Header, Content, Sider } = Layout;

//屏蔽全局浏览器右键菜单
document.oncontextmenu = function () {
    return false;
}

function PhysikaWeb() {
    const [simType, setSimType] = useState(-1);
    const [userStatus, setUserStatus] = useState(false);
    const [visible, setVisible] = useState(false);

    const simTypeMenu = (
        <Menu onClick={auth}>
            <Menu.Item key='1'>自然云图像建模</Menu.Item>
            <Menu.Item key='2'>卫星云图像建模</Menu.Item>
            <Menu.Item key='3'>WRF数据建模</Menu.Item>
            <Menu.Item key='4'>SPH流体仿真</Menu.Item>
            <Menu.Item key='5'>布料仿真</Menu.Item>
            <Menu.Item key='6'>基于PBF的形状演化</Menu.Item>
        </Menu>
    )

    const userMenu = (
        window.localStorage.userIDSPH
            ? <Menu onClick={userAction}>
                <Menu.Item key="0">注销</Menu.Item>
            </Menu>
            : <Menu onClick={userAction}>
                <Menu.Item key="0">登录</Menu.Item>
            </Menu>
    )

    //用户选择仿真类型前需要进行认证
    function auth(e) {
        setSimType(e.key);
        if (!window.localStorage.userID) {
            setVisible(true);
        }
        else {
            //token验证
            setUserStatus(true);
        }
    }

    //登录/注册成功
    function changeUserStatus(status) {
        setUserStatus(status);
        setVisible(false);
    }

    //用户相关操作
    function userAction(e) {
        if (window.localStorage.userID) {
            switch (e.key) {
                case '0':
                    window.localStorage.clear();
                    setUserStatus(false);
                    //用于解决：用户没有注销直接关闭网页之后再次打开网页点击注销时无法触发更新
                    setSimType(-2);
                    break;
                default:
            }
        }
        else {
            setVisible(true);
        }
    }

    return (

        <Layout>
            <Header className="header" style={{ backgroundColor: "lavender" }}>
                <Row>
                    <Col span={4} style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
                        <div className="logo" style={{ textAlign: "center", fontSize: "20px", background: "inherit", width: "200px" }}>
                            云建模/物理仿真平台
                        </div>
                    </Col>
                    <Col span={16}></Col>
                    <Col span={3} style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
                        <Dropdown overlay={simTypeMenu} placement="bottomCenter">
                            <Button>选择仿真类型<DownOutlined /></Button>
                        </Dropdown>
                    </Col>
                    <Col span={1} style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
                        <Dropdown overlay={userMenu} placement="bottomCenter" arrow>
                            {
                                window.localStorage.userID
                                    ? <Avatar size={40}>{window.localStorage.userID.substring(0, 1)}</Avatar>
                                    : <Avatar size={40} icon={<UserOutlined />} />
                            }
                        </Dropdown>
                    </Col>
                </Row>
            </Header>
            <Layout style={{ height: "93vh" }}>
                <Sider width={270} className="site-layout-background" style={{ overflow: 'scroll' }}>
                    {
                        userStatus && (simType === "1") &&
                        <PhysikaCloudNature></PhysikaCloudNature>
                    }
                    {
                        userStatus && (simType === "2") &&
                        <PhysikaCloudSatellite></PhysikaCloudSatellite>
                    }
                    {
                        userStatus && (simType === "3") &&
                        <PhysikaCloudWRF></PhysikaCloudWRF>
                    }
                    {
                        userStatus && (simType === "4") &&
                        <MySPH></MySPH>
                    }
                    {
                        userStatus && (simType === "5") &&
                        <PhysikaCloth></PhysikaCloth>
                    }
                    {
                        userStatus && (simType === "6") &&
                        <ParticleEvolution></ParticleEvolution>
                    }
                </Sider>
                <Layout style={{ padding: '24px 24px 24px' }}>
                    <Content
                        className="site-layout-background"
                        style={{
                            padding: 0,
                            margin: 0,
                            minHeight: 280,
                        }}
                    >
                        <div id="geoViewer" style={{ height: "100%", width: "100%", position: "relative" }}></div>
                    </Content>
                </Layout>
            </Layout>
            <div>
                <LoginModal
                    visible={visible}
                    hideModal={() => setVisible(false)}
                    changeUserStatus={(status) => changeUserStatus(status)}
                ></LoginModal>
            </div>
        </Layout >

    );
}

export default PhysikaWeb;
