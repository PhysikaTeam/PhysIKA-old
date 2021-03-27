import React from 'react';
import { Tree, Button, Slider, Divider, Descriptions, Collapse, Row, Col, InputNumber } from 'antd';
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

import db from '../../db';

const simType = 0;

//load:重新加载初始化文件，并清空界面；upload：只会清空界面。
class CloudEuler extends React.Component {
    constructor(props) {
        super(props);
        this.state = {

            data: [],
            treeNodeAttr: {},
            treeNodeText: "",
            treeNodeKey: -1,

            //结果展示信息
            description: [],
            inputFrameIndex: 0,

            isTreeNodeAttrModalShow: false,
            uploadDisabled: true,
            animation: false,
            isSliderShow: false,
        };
    }

    componentDidMount() {
        //---------初始化渲染窗口
        this.fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
            //background: [1.0, 1.0, 1.0],
            background: [0.75, 0.76, 0.79],
            rootContainer: geoViewer,
            //关键操作！！！能把canvas大小限制在div里了！
            containerStyle: { height: 'inherit', width: 'inherit' }
        });
        this.renderer = this.fullScreenRenderer.getRenderer();
        this.renderWindow = this.fullScreenRenderer.getRenderWindow();
        //添加旋转控制控件
        this.orientationMarkerWidget = getOrientationMarkerWidget(this.renderWindow);
        //文件名
        this.fileName = '';
        //总帧数
        this.frameSum = 0;
        //当前帧序号
        this.curFrameIndex = 0;
        //curScene={source, mapper, actor}
        this.curScene = {};
        //frameStateArray保存每一帧仿真模型的当前状态：
        // 0（未获取）、1（正在获取）、2（已获取，未存入indexedDB）、3（在indexedDB中）、4（在内存对象中（无法获知js对象在内存中的大小，没法设定内存对象大小。。。））
        this.frameStateArray = [];
        //fetchFrameQueue保存未获取帧的获取序列
        this.fetchFrameQueue = [];
        //加载到内存中的帧
        //this.frameSeq = [];
        //控制给worker发送信息的锁
        this.workerLock = false;
        //获取模型操作的定时器
        this.fetchModelTimer = null;
        //记录本次upload的时间
        this.uploadDate = null;
        //用于标记是否在workerLock为true的情况下触发了load方法
        this.loadTag = 0;
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
        //关闭定时器
        if (this.fetchModelTimer !== null) {
            clearInterval(this.fetchModelTimer);
        }
        //是否需要？
        if(this.FPSWidget){
            this.FPSWidget.delete();
        }
    }

    clean = () => {
        this.renderer.removeActor(this.curScene.actor);
        this.curFrameIndex = 0;
        this.curScene = {};
        this.frameStateArray = [];
        this.fetchFrameQueue = [];
        if (this.fetchModelTimer !== null) {
            clearInterval(this.fetchModelTimer);
        }
        db.table('model').where('uploadDate').below(Date.now()).delete()
            .then(() => { console.log('删除旧数据成功!') })
            .catch(err => { console.log('删除旧数据出错! ' + err) });
        let geoViewer = document.getElementById("geoViewer");
        if (document.getElementById("volumeController")) {
            geoViewer.removeChild(document.getElementById("volumeController"));
        }
        this.renderer.resetCamera();
        this.renderWindow.render();

        this.setState({
            description: [],
            animation: false,
            isSliderShow: false,
        });
    }

    load = () => {
        if (this.workerLock) {
            this.loadTag = 1;
            return;
        }
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
            this.FPSWidget.setOrientation('vertical');
            this.FPSWidget.setRenderWindow(this.renderWindow);
        }
    }

    //现在upload不更新data！
    upload = () => {
        if (this.workerLock) {
            this.loadTag = 2;
            return;
        }
        if (!checkUploadConfig(this.state.data)) {
            return;
        }
        this.clean();
        //存储提交日期用于区分新旧数据，并删除旧数据
        //this.uploadDate = Date.now();
        //测试就将uploadDate调为0；
        this.uploadDate = 0;
        this.setState({
            uploadDisabled: true,
        }, () => {
            //设置后端存储使用的额外信息
            const extraInfo = {
                userID: window.localStorage.userID,
                uploadDate: this.uploadDate,
                simType: simType,
            }
            //第一个参数data，第二个参数仿真类型
            physikaUploadConfig(this.state.data, extraInfo)
                .then(res => {
                    console.log("成功上传配置并获取到仿真结果配置");
                    const resultInfo = parseSimulationResult(res);
                    this.fileName = resultInfo.fileName;
                    this.frameSum = resultInfo.frameSum;
                    this.setState({
                        description: resultInfo.description,
                        animation: resultInfo.animation,
                    });
                    //强制加载第0帧，然后再显示其他内容！
                    if (this.frameSum > 0) {
                        for (let i = 0; i < this.frameSum; i++) {
                            //根据帧总数初始化this.frameStateArray
                            this.frameStateArray.push(0);
                            //初始化获取帧序列
                            this.fetchFrameQueue.push(i);
                        }
                        this.fetchFrameQueue.shift();
                        console.log(this.frameStateArray, this.fetchFrameQueue, this.frameSum);
                        return this.wsWorker.postMessage({
                            data: {
                                userID: window.localStorage.userID,
                                uploadDate: this.uploadDate,
                                fileName: this.fileName + '_0.vti'
                            }
                        });
                    }
                    else {
                        return Promise.reject('模拟帧数不大于0！');
                    }
                })
                .then(res => {
                    //第一帧获取时在开启获取定时器之前，故不需要锁
                    //开启获取定时器
                    this.fetchModelTimer = setInterval(this.checkFrameQueue, 1000);
                    //存入indexedDB
                    //注意：先执行完下一个then中的updateScene等操作才会执行写数据（writeModel中的异步操作放在当前微任务队列最后）
                    this.writeModel(0, res);
                    //注意后缀！
                    return physikaInitVti(res, 'zip');
                })
                .then(res => {
                    this.frameStateArray[0] = 2;
                    this.updateScene(res);
                    //初始化体素显示控制控件
                    this.initVolumeController();
                    //显示方向标记部件
                    this.orientationMarkerWidget.setEnabled(true);
                    //初始化fps控件
                    this.initFPS();
                    this.setState({
                        inputFrameIndex: 0,
                        uploadDisabled: false,
                        isSliderShow: true,
                    });
                })
                .catch(err => {
                    console.log("Error uploading: ", err);
                })
        });
    }

    checkFrameQueue = () => {
        //如果获取帧队列不为空 且 worker未上锁
        if (this.fetchFrameQueue.length !== 0 && !this.workerLock) {
            //开启worker锁
            this.workerLock = true;
            this.fetchModel(this.fetchFrameQueue.shift());
        }
        if (this.fetchFrameQueue.length === 0) {
            clearInterval(this.fetchModelTimer);
            console.log("获取帧队列为空，清除定时器", this.fetchFrameQueue);
        }
    }

    fetchModel = (frameIndex) => {
        //设定当前帧状态为获取中
        this.frameStateArray[frameIndex] = 1;
        this.wsWorker.postMessage({
            data: {
                userID: window.localStorage.userID,
                uploadDate: this.uploadDate,
                fileName: this.fileName + '_' + frameIndex + '.vti'
            }
        })
            .then(res => {
                //关闭worker锁
                this.workerLock = false;
                if (this.loadTag === 0) {
                    //设定当前帧状态为已获取但未存入indexedDB
                    this.frameStateArray[frameIndex] = 2;
                    console.log('获取到第', frameIndex, '帧，', this.frameStateArray, this.fetchFrameQueue);
                    //将模型写入indexedDB
                    this.writeModel(frameIndex, res);
                    return physikaInitVti(res, 'zip');
                }
                else {
                    if (this.loadTag === 1) {
                        this.loadTag = 0;
                        this.load();
                    }
                    if (this.loadTag === 2) {
                        this.loadTag = 0;
                        this.upload();
                    }
                    return Promise.reject('忽略该帧！')
                }
            })
            .then(res => {
                if (frameIndex === this.curFrameIndex) {
                    this.updateScene(res);
                    this.controllerWidget.changeActor(this.curScene.actor);
                }
            })
            .catch(err => {
                console.log("Error fetchModel: ", err);
            });
    }

    writeModel = (frameIndex, arrayBuffer) => {
        db.table('model').add({
            userID: window.localStorage.userID, uploadDate: this.uploadDate, frameIndex: frameIndex, arrayBuffer: arrayBuffer
        }).then(id => {
            this.frameStateArray[frameIndex] = 3;
            console.log(id, '成功存入第' + frameIndex + '帧！', this.frameStateArray);
        }).catch(err => {
            console.log(err);
        })
    }

    readModel = (uploadDate, frameIndex) => {
        //then后面if判断
        db.table('model').get({
            uploadDate: uploadDate, frameIndex: frameIndex
        }).then(model => {
            return physikaInitVti(model.arrayBuffer, 'zip');
        }).then(res => {
            if (uploadDate === this.uploadDate && frameIndex === this.curFrameIndex) {
                this.updateScene(res);
                this.controllerWidget.changeActor(this.curScene.actor);
            }
            else {
                //快速改变已获取帧数可以看到如下显示
                console.log('Old model ' + frameIndex + ' is no longer uesd!')
            }
        }).catch(err => {
            console.log('Error readModel: ', err);
        });
    }

    changeInput = (value) => {
        console.log('changeInput: ', value);
        this.setState({ inputFrameIndex: value }, () => {
            this.curFrameIndex = value;
            switch (this.frameStateArray[value]) {
                case 0:
                    //未获取
                    console.log("-------", this.fetchFrameQueue);
                    //考虑到value在数组中的位置，前方数组可能比后面大，
                    //执行过多的push会导致效率太低，考虑将数组变为队列应该可以改善效率
                    const pos = this.fetchFrameQueue.indexOf(value);
                    const frontArray = this.fetchFrameQueue.splice(0, pos);
                    for (const item of frontArray) {
                        this.fetchFrameQueue.push(item);
                    }
                    console.log("-------", this.fetchFrameQueue);
                    break;
                case 1:
                    //获取中,不管！
                    break;
                case 2:
                    //已获取，但未存入indexedDB
                    let uploadDate = this.uploadDate;
                    setTimeout(() => {
                        if (value === this.curFrameIndex) {
                            //这里可能有问题！
                            this.readModel(uploadDate, value);
                            console.log('尝试从indexedDB中读第' + value + '帧！');
                        }
                    }, 1000);
                    break;
                case 3:
                    this.readModel(this.uploadDate, value);
                    break;
                default:
                    break;
            }
        })

    }

    renderDescriptions = () => this.state.description.map((item, index) => {
        return <Descriptions.Item label={item.name} key={index}>{item.content}</Descriptions.Item>
    })

    render() {
        console.log("tree:", this.state.data);
        return (
            <div>
                <Divider>云欧拉仿真</Divider>
                <Collapse defaultActiveKey={['1']}>
                    <Panel header="仿真初始化" key="1">
                        <Button type="primary" size={'small'} block onClick={this.load}>加载场景</Button>
                        <Tree style={{ overflowX: 'auto', width: '200px' }}>
                            {this.renderTreeNodes(this.state.data)}
                        </Tree>
                        <br />
                        <Button type="primary" size={'small'} block onClick={this.upload} disabled={this.state.uploadDisabled}>开始仿真</Button>
                    </Panel>
                    <Panel header="仿真结果信息" key="2">
                        <Descriptions column={1} layout={'horizontal'}>
                            {this.renderDescriptions()}
                        </Descriptions>
                    </Panel>
                    <Panel header="多帧展示控制" key="3">
                        {
                            (this.state.isSliderShow) &&
                            <div>
                                <Slider min={0} max={this.frameSum - 1} value={this.state.inputFrameIndex} onChange={this.changeInput} />
                                <Row>
                                    <Col span={13} style={{ alignItems: 'center', display: 'flex' }}>
                                        <span className="ant-rate-text">当前帧序号：</span>
                                    </Col>
                                    <Col span={3}>
                                        <InputNumber min={0} max={this.frameSum - 1} value={this.state.inputFrameIndex} onChange={this.changeInput} />
                                    </Col>
                                </Row>
                            </div>
                        }
                    </Panel>
                    {/* forceRender为true，即折叠面板未打开时也渲染其中组件；若为false，则未打开面板前无法获得其中组件 */}
                    <Panel header="绘制信息" key="4" forceRender="true">
                        <div id="fps"></div>
                    </Panel>
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
    CloudEuler as PhysikaCloudEuler
}
