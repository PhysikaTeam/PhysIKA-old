import React from 'react';
import { Tree, Button, Divider, Descriptions, Collapse} from 'antd';
const { TreeNode } = Tree;
const { Panel } = Collapse;
//antd样式
import 'antd/dist/antd.css';
//渲染窗口
import vtkFullScreenRenderWindow from 'vtk.js/Sources/Rendering/Misc/FullScreenRenderWindow';
import vtkVolumeController from '../Widget/VolumeController'
import vtkFPSMonitor from '../Widget/FPSMonitor'

import { physikaLoadConfig } from '../../IO/LoadConfig'
import { physikaUploadConfig } from '../../IO/UploadConfig'
import { PhysikaTreeNodeAttrModal } from '../TreeNodeAttrModal'
import { physikaInitVti } from '../../IO/InitVti'
import { getOrientationMarkerWidget } from '../Widget/OrientationMarkerWidget'
import { parseSimulationResult, checkUploadConfig } from '../../Common'

import WebworkerPromise from 'webworker-promise';
import WSWorker from '../../Worker/ws.worker';

const simType = "CloudSatellite";

//load:重新加载初始化文件，并清空界面；upload：只会清空界面。
class CloudSatellite extends React.Component {
    constructor(props) {
        super(props);
        this.state = {

            data: [],
            treeNodeAttr: {},
            treeNodeText: "",
            treeNodeKey: -1,

            description: [],

            isTreeNodeAttrModalShow: false,
            uploadDisabled: true,
            simLoading: false,
        };
    }

    componentDidMount() {
        //---------初始化渲染窗口
        this.fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
            background: [0.0, 0.0, 0.0],
            rootContainer: geoViewer,
            //关键操作！！！能把canvas大小限制在div里了！
            containerStyle: { height: 'inherit', width: 'inherit' }
        });
        this.renderer = this.fullScreenRenderer.getRenderer();
        this.renderWindow = this.fullScreenRenderer.getRenderWindow();
        //添加旋转控制控件
        this.orientationMarkerWidget = getOrientationMarkerWidget(this.renderWindow);
        this.fileName = '';
        this.frameSum = 0;
        this.curScene = {};
        //记录本次upload的时间
        this.uploadDate = null;
        //worker创建及WebSocket初始化
        this.wsWorker = new WebworkerPromise(new WSWorker());
        this.wsWorker.postMessage({ init: true });
    }

    componentWillUnmount() {
        console.log('子组件将卸载');
        let renderWindowDOM = document.getElementById("geoViewer");
        renderWindowDOM.innerHTML = ``;
        //关闭WebSocket
        this.wsWorker.postMessage({ close: true });
        this.wsWorker.terminate();
        //是否需要？
        if(this.FPSWidget){
            this.FPSWidget.delete();
        }
    }

    clean = () => {
        this.renderer.removeActor(this.curScene.actor);
        this.curScene = {};
        let geoViewer = document.getElementById("geoViewer");
        if (document.getElementById("volumeController")) {
            geoViewer.removeChild(document.getElementById("volumeController"));
        }
        this.renderer.resetCamera();
        this.renderWindow.render();

        this.setState({
            description: [],
        });
    }

    load = () => {
        physikaLoadConfig(simType)
            .then(res => {
                console.log("成功获取初始化配置");
                this.setState({
                    data: res,
                    uploadDisabled: false
                });
                this.clean();
            })
            .catch(err => {
                console.log("Error loading: ", err);
            })
    }

    //递归渲染每个树节点（这里必须用map遍历！因为需要返回数组）
    renderTreeNodes = (data) => data.map((item, index) => {
        item.title = (
            <div>
                {
                    item._text
                        ? <Button type="text" size="small" onClick={() => this.showTreeNodeAttrModal(item)}>{item._attributes.name}</Button>
                        : <span className="ant-rate-text">{item._attributes.name}</span>
                }
            </div>
        );

        if (item.children) {
            return (
                <TreeNode title={item.title} key={item.key} >
                    {this.renderTreeNodes(item.children)}
                </TreeNode>
            );
        }

        return <TreeNode {...item} />;
    });

    showTreeNodeAttrModal = (item) => {
        this.setState({
            isTreeNodeAttrModalShow: true,
            treeNodeAttr: item._attributes,
            treeNodeKey: item.key,
            treeNodeText: item._text
        });
    }

    hideTreeNodeAttrModal = () => {
        this.setState({
            isTreeNodeAttrModalShow: false
        });
    }

    //接收TreeNodeAttrModal返回的结点数据并更新树
    changeData = (obj) => {
        //注意：这里直接改变this.state.data本身不会触发渲染，
        //真正触发渲染的是hideTreeNodeAttrModal()函数的setState！
        //官方并不建议直接修改this.state中的值，因为这样不会触发渲染，
        //但是React的setState本身并不能处理nested object的更新。
        //若该函数不再包含hideTreeNodeAttrModal()函数，则需要另想办法更新this.state.data！
        let eachKey = this.state.treeNodeKey.split('-');
        let count = 0;
        const findTreeNodeKey = (node) => {
            if (count === eachKey.length - 1) {
                //找到treeNodeKey对应树结点，更新数据
                if (obj.hasOwnProperty('_text')) {
                    console.log("obj ", obj);
                    node[eachKey[count]]._text = obj._text;
                }
                //若以后需修改_attributes属性，则在此添加代码
                return;
            }
            findTreeNodeKey(node[eachKey[count++]].children);
        };
        findTreeNodeKey(this.state.data);
        this.hideTreeNodeAttrModal();
    }

    updateScene = (newScene) => {
        //移除旧场景actor
        this.renderer.removeActor(this.curScene.actor);
        this.curScene = newScene;
        //添加新场景actor
        this.renderer.addActor(this.curScene.actor);
        this.renderer.resetCamera();
        this.renderWindow.render();
    }

    initVolumeController = () => {
        //动态删除添加volume这个div
        let geoViewer = document.getElementById("geoViewer");
        let volumeControllerContainer = document.createElement("div");
        volumeControllerContainer.id = "volumeController";
        geoViewer.append(volumeControllerContainer);

        this.controllerWidget = vtkVolumeController.newInstance({
            size: [400, 150],
            rescaleColorMap: true,
        });
        this.controllerWidget.setContainer(volumeControllerContainer);
        this.controllerWidget.setupContent(this.renderWindow, this.curScene.actor, true);
    }

    initFPS = () => {
        let FPSContainer = document.getElementById("fps");
        if (FPSContainer.children.length === 0) {
            this.FPSWidget = vtkFPSMonitor.newInstance({infoVisibility: false});
            this.FPSWidget.setContainer(FPSContainer);
            this.FPSWidget.setRenderWindow(this.renderWindow);
            this.FPSWidget.setOrientation('vertical');
        }
    }

    //现在upload不更新data！
    upload = () => {
        if (!checkUploadConfig(this.state.data)) {
            return;
        }
        this.clean();
        this.uploadDate = Date.now();
        this.setState({
            uploadDisabled: true,
            simLoading: true,
        }, () => {
            const extraInfo = {
                userID: window.localStorage.userID,
                uploadDate: this.uploadDate,
                simType: simType,
            }
            physikaUploadConfig(this.state.data, extraInfo)
                .then(res => {
                    console.log("成功上传配置并获取到仿真结果配置");
                    const resultInfo = parseSimulationResult(res);
                    this.fileName = resultInfo.fileName;
                    this.frameSum = resultInfo.frameSum;
                    this.setState({
                        description: resultInfo.description,
                    });
                    if (this.frameSum > 0) {
                        return this.wsWorker.postMessage({
                            data: {
                                userID: window.localStorage.userID,
                                uploadDate: this.uploadDate,
                                fileName: this.fileName + '.vti'
                            }
                        });
                    }
                    else {
                        return Promise.reject('模拟帧数不大于0！');
                    }
                })
                .then(res => {
                    return physikaInitVti(res, 'zip');
                })
                .then(res => {
                    this.updateScene(res);
                    const dimensions = this.curScene.source.getDimensions();
                    this.state.description.push({
                        name: "体素数",
                        content: dimensions[0] * dimensions[1] * dimensions[2]
                    });
                    this.initVolumeController();
                    this.orientationMarkerWidget.setEnabled(true);
                    this.initFPS();
                    this.setState({
                        uploadDisabled: false,
                        simLoading:false,
                    });
                })
                .catch(err => {
                    console.log("Error uploading: ", err);
                })
        });
    }

    renderDescriptions = () => this.state.description.map((item, index) => {
        return <Descriptions.Item label={item.name} key={index}>{item.content}</Descriptions.Item>
    })

    render() {
        console.log("tree:", this.state.data);
        return (
            <div>
                <Divider>卫星云图像建模</Divider>
                <Collapse defaultActiveKey={['1']}>
                    <Panel header="仿真初始化" key="1">
                        <Button type="primary" size={'small'} block onClick={this.load}>加载场景</Button>
                        <Tree style={{ overflowX: 'auto', width: '200px' }}>
                            {this.renderTreeNodes(this.state.data)}
                        </Tree>
                        <br />
                        <Button type="primary" size={'small'} block onClick={this.upload} disabled={this.state.uploadDisabled}
                            loading={this.state.simLoading}>开始仿真</Button>
                    </Panel>
                    <Panel header="仿真结果信息" key="2">
                        <Descriptions column={1} layout={'horizontal'}>
                            {this.renderDescriptions()}
                        </Descriptions>
                    </Panel>
                    {/* forceRender为true，即折叠面板未打开时也渲染其中组件；若为false，则未打开面板前无法获得其中组件 */}
                    <Panel header="绘制信息" key="3" forceRender="true">
                        <div id="fps"></div>
                    </Panel>
                    {/* <Panel header="仿真展示控制" key="3"></Panel> */}
                </Collapse>
                <div>
                    <PhysikaTreeNodeAttrModal
                        treeNodeAttr={this.state.treeNodeAttr}
                        treeNodeText={this.state.treeNodeText}
                        visible={this.state.isTreeNodeAttrModalShow}
                        hideModal={this.hideTreeNodeAttrModal}
                        changeData={(obj) => this.changeData(obj)}
                    ></PhysikaTreeNodeAttrModal>
                </div>
            </div>
        )
    }
}

export {
    CloudSatellite as PhysikaCloudSatellite
}
