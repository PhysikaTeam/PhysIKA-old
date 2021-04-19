import React from 'react';
import { Tree, Button, Slider, Divider, Descriptions, Collapse, Row, Col, InputNumber, Select, Input } from 'antd';
const { TreeNode } = Tree;
const { Panel } = Collapse;
const { Option } = Select;
//antd样式
import 'antd/dist/antd.css';
//渲染窗口
import vtkFullScreenRenderWindow from 'vtk.js/Sources/Rendering/Misc/FullScreenRenderWindow';
import vtkFPSMonitor from '../Widget/FPSMonitor';

import { physikaLoadConfig } from '../../IO/LoadConfig';
import { physikaUploadConfig } from '../../IO/UploadConfig';
import { PhysikaTreeNodeAttrModal } from '../TreeNodeAttrModal';
import { getOrientationMarkerWidget } from '../Widget/OrientationMarkerWidget';
import { parseSimulationResult, checkUploadConfig, addNewNode, deleteNode } from '../../Common';
import { physikaInitObj } from '../../IO/InitObj';

import WebworkerPromise from 'webworker-promise';
import WSWorker from '../../Worker/ws.worker';

import db from '../../db';

const simType = "Cloth";
const filetype = 'zip';

const representationOptions = ['V', 'W', 'S'];

class Cloth extends React.Component {
    constructor(props) {
        super(props);
        this.state = {

            data: [],
            treeNodeAttr: {},
            treeNodeText: "",
            treeNodeKey: -1,

            simLoading: false,
            //结果展示信息
            description: [],
            //当前帧索引
            curFrameIndex: 0,
            //已获取帧数
            maxFrameIndex: 0,

            isTreeNodeAttrModalShow: false,
            uploadDisabled: true,
            isSliderShow: false,
            //控制播放动画按钮
            isPlay: false,
            //控制显示场景切换
            isShowResult: false
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
        //下一个要获取的帧的索引
        this.nextFetchFrameIndex = 0;
        //总帧数
        this.frameSum = 0;
        //模拟结果组成的场景
        //curScene=[{source, mapper, actor},{source, mapper, actor}...]
        this.curScene = [];
        //frameStateArray保存每一帧仿真模型的当前状态：0（已发获取请求）、1（已获取，未存入indexedDB）、2（在indexedDB中）
        this.frameStateArray = [];
        //记录本次upload的时间
        this.uploadDate = null;
        //worker创建及WebSocket初始化
        this.wsWorker = null;
        //RigidBodies所组成的场景
        this.cScene = [];

    }

    componentWillUnmount() {
        console.log('子组件将卸载');
        let renderWindowDOM = document.getElementById("geoViewer");
        renderWindowDOM.innerHTML = ``;
        this.curScene = [];
        this.frameStateArray = [];
        this.cScene = [];
        //关闭WebSocket
        if (this.wsWorker) {
            this.wsWorker.postMessage({ close: true });
            this.wsWorker.terminate();
        }
        db.table('model').where('[uploadDate+frameIndex]').between(
            [0, 0], [Date.now(), Number.MAX_SAFE_INTEGER]
        ).delete()
            .then(() => { console.log('删除旧数据成功!') })
            .catch(err => { console.log('删除旧数据出错! ' + err) });
        //是否需要？
        if (this.FPSWidget) {
            this.FPSWidget.delete();
        }
    }

    clean = (tag) => {
        if (tag === 0) {
            Object.keys(this.cScene).forEach(key => {
                this.renderer.removeActor(this.cScene[key].actor);
                this.cScene[key].source.delete();
            });
            this.cScene = [];
        }
        Object.keys(this.curScene).forEach(key => {
            this.renderer.removeActor(this.curScene[key].actor);
            this.curScene[key].source.delete();
            this.curScene = [];
        });
        let geoViewer = document.getElementById("geoViewer");
        if (document.getElementById("volumeController")) {
            geoViewer.removeChild(document.getElementById("volumeController"));
        }
        this.renderer.resetCamera();
        this.renderWindow.render();

        this.setState({
            curFrameIndex: 0,
            maxFrameIndex: 0,
            description: [],
            isSliderShow: false,
            isPlay: false,
        });
        this.nextFetchFrameIndex = 0;
        this.frameSum = 0;
        this.frameStateArray = [];
    }

    load = () => {
        if (this.wsWorker) {
            this.wsWorker.terminate();
            console.log("wsworker", this.wsWorker);
        }
        this.clean(0);
        physikaLoadConfig(simType)
            .then(res => {
                console.log("成功获取初始化配置");
                this.setState({
                    data: res,
                    uploadDisabled: false
                });
            })
            .catch(err => {
                console.log("Error loading: ", err);
            })
    }

    addNewNode = (item) => {
        const resObj = addNewNode(this.state.data, item);
        this.setState({
            data: resObj.tree
        }, () => {
            //
        });
    }

    deleteNode = (item) => {
        const resObj = deleteNode(this.state.data, item);
        this.setState({
            data: resObj.tree
        }, () => {
            const deletedNode = resObj.deletedNode[0];
            const index = Number(deletedNode.key.split('-').pop());
            if (this.cScene[index]) {
                switch (deletedNode.tag.split('_')[0]) {
                    case 'Cloth':
                        this.renderer.removeActor(this.cScene[index].actor);
                        this.cScene[index].source.delete();
                        this.cScene.splice(index, 1);
                        break;
                    default:
                        break;
                }
                this.renderer.resetCamera();
                this.renderWindow.render();
            }
        });
    }

    changeRepresentation = (e, key) => {
        const eachKey = key.split('-');
        //简单粗暴但不通用
        switch (this.state.data[eachKey[0]].children[eachKey[1]].tag) {
            case 'Cloths':
                if (this.cScene[eachKey[2]]) {
                    this.cScene[eachKey[2]].actor.getProperty().setRepresentation(e);
                }
                break;
            default:
                break;
        }
        this.renderer.resetCamera();
        this.renderWindow.render();
    }

    //递归渲染每个树节点（这里必须用map遍历！因为需要返回数组）
    renderTreeNodes = (data) => data.map((item, index) => {
        item.title = (
            <div>
                {
                    //不要随便用item._text这种形式做判断，会自动把数值转为bool做判断
                    item.hasOwnProperty('_text') ? <Button type="text" size="small" onClick={() => this.showTreeNodeAttrModal(item)}>{item._attributes.name}</Button>
                        : (item.tag === 'AddNode') ? <Button type="text" size="small" onClick={() => this.addNewNode(item)}>{item._attributes.name}</Button>
                            : item.deletable
                                ?
                                <div>
                                    <Input placeholder={item._attributes.name} bordered={false} size={'small'} style={{ width: '65px' }}></Input>
                                    <Select defaultValue={'S'} onChange={(e) => this.changeRepresentation(e, item.key)} bordered={false} size={'small'}>
                                        {this.renderRepresentationOptions()}
                                    </Select>
                                    <Button type="text" size="small" onClick={() => this.deleteNode(item)}>  -</Button>
                                </div>
                                :
                                <div>
                                    <span className="ant-rate-text">{item._attributes.name}</span>
                                    {
                                        (item.tag === 'Cloth_0') &&
                                        <Select defaultValue={'S'} onChange={(e) => this.changeRepresentation(e, item.key)} bordered={false} size={'small'}>
                                            {this.renderRepresentationOptions()}
                                        </Select>
                                    }
                                </div>
                }
            </div >
        );

        if (item.children) {
            return (
                <TreeNode title={item.title} key={item.key}>
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

    editRigidBody = (tag, item, index, attributes) => {
        switch (tag) {
            case 'geometryFile':
                physikaInitObj(item.fileContent, 'obj')
                    .then(res => {
                        this.renderer.addActor(res[0].actor);
                        if (this.state.isShowResult) {
                            res[0].actor.setVisibility(false);
                        }
                        if (this.cScene[index]) {
                            //必须先从renderer中移除该对象，否则会报错
                            this.renderer.removeActor(this.cScene[index].actor);
                            this.cScene[index].source.delete();
                        }
                        //只有在第一次加载actor时初始化
                        if (this.cScene.length === 0) {
                            //显示方向标记部件
                            this.orientationMarkerWidget.setEnabled(true);
                            //初始化fps控件
                            this.initFPS();
                        }
                        this.cScene[index] = res[0];
                        attributes.forEach(i => {
                            if (i.tag === 'geometryFile')
                                return;
                            this.editRigidBody(i.tag, i, index, attributes);
                        });
                        //因为没有别的属性，所以场景刷新只能放在then中
                        this.renderer.resetCamera();
                        this.renderWindow.render();
                    })
                    .catch(err => {
                        console.log('Error in obj importing: ', err);
                    })
                break;
            default:
                break;
        }
        this.renderer.resetCamera();
        this.renderWindow.render();
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
                    node[eachKey[count]]._text = obj._text;
                }
                //若以后需修改_attributes属性，则在此添加代码
                return;
            }
            findTreeNodeKey(node[eachKey[count++]].children);
        };
        findTreeNodeKey(this.state.data);

        //编辑初始化模型
        let node = this.state.data;
        for (let i = 0; i < count - 1; ++i) {
            node = node[eachKey[i]].children;
        }
        const fatherNode = node[eachKey[count - 1]];
        switch (fatherNode.tag.split('_')[0]) {
            case 'Cloth':
                this.editRigidBody(fatherNode.children[eachKey[count]].tag, obj, eachKey[count - 1], fatherNode.children);
                break;
            default:
                break;
        }

        this.hideTreeNodeAttrModal();
    }

    //只用于更新模拟结果模型
    updateScene = (newScene) => {
        //移除旧场景actor
        this.curScene.forEach(item => {
            Object.keys(item).forEach(key => {
                this.renderer.removeActor(item[key].actor);
                item[key].source.delete();
            });
        });
        this.curScene = newScene;
        //console.log(this.curScene);
        //添加新场景actor
        this.curScene.forEach(item => {
            Object.keys(item).forEach(key => {
                this.renderer.addActor(item[key].actor);
                if (!this.state.isShowResult) {
                    item[key].actor.setVisibility(false);
                }
            });
        });
        this.renderer.resetCamera();
        this.renderWindow.render();
    }

    initFPS = () => {
        let FPSContainer = document.getElementById("fps");
        if (FPSContainer.children.length === 0) {
            this.FPSWidget = vtkFPSMonitor.newInstance({ infoVisibility: false });
            this.FPSWidget.setContainer(FPSContainer);
            this.FPSWidget.setOrientation('vertical');
            this.FPSWidget.setRenderWindow(this.renderWindow);
        }
    }

    //现在upload不更新data！
    upload = () => {
        if (!checkUploadConfig(this.state.data)) {
            return;
        }
        if (this.wsWorker) {
            this.wsWorker.terminate();
            console.log("wsworker", this.wsWorker);
        }
        this.wsWorker = new WebworkerPromise(new WSWorker());
        this.wsWorker.postMessage({ init: true });
        this.clean(1);
        //存储提交日期用于区分新旧数据，并删除旧数据
        this.uploadDate = Date.now();
        this.setState({
            uploadDisabled: true,
            simLoading: true
        }, () => {
            //设置后端存储使用的额外信息
            const extraInfo = {
                userID: window.localStorage.userID,
                uploadDate: this.uploadDate,
                simType: simType,
            }

            //开始预取缓存
            setTimeout(this.fetchModel(0, false), 0);

            //第一个参数data，第二个参数仿真类型
            physikaUploadConfig(this.state.data, extraInfo)
                .then(res => {
                    console.log("成功上传配置并获取到仿真结果配置");
                    const resultInfo = parseSimulationResult(res);
                    this.frameSum = resultInfo.frameSum;
                    this.setState({
                        description: resultInfo.description,
                        uploadDisabled: false,
                        simLoading: false
                    });
                })
                .catch(err => {
                    console.log("Error uploading: ", err);
                    this.setState({
                        uploadDisabled: false,
                        simLoading: false
                    });
                })
        });
    }

    fetchModel = (frameIndex, isEnd) => {
        this.frameStateArray[frameIndex] = 0;
        this.wsWorker.postMessage({
            data: {
                userID: window.localStorage.userID,
                uploadDate: this.uploadDate,
                usePrefetch: true,
                simType: simType,
                frameIndex: frameIndex,
                isEnd: isEnd
            }
        })
            .then(arrayBuffer => {
                if (arrayBuffer.byteLength === 0) {
                    //在预取最后一帧时若发生超时，则重新取一次
                    if (frameIndex == this.frameSum - 1) {
                        this.fetchModel(this.nextFetchFrameIndex, true);
                    }
                    throw ("Prefetch timed out!");
                }
                else {
                    //设定当前帧状态为已获取但未存入indexedDB
                    this.frameStateArray[frameIndex] = 1;
                    console.log('获取到第' + frameIndex + '帧，', this.frameStateArray);
                    //将数据写入indexDB
                    return db.table('model').add({
                        userID: window.localStorage.userID, uploadDate: this.uploadDate, frameIndex: frameIndex, arrayBuffer: arrayBuffer
                    });
                }
            })
            .then(id => {
                this.frameStateArray[frameIndex] = 2;
                console.log(id, '成功存入第' + frameIndex + '帧！', this.frameStateArray);
                ++this.nextFetchFrameIndex;
                this.setState({
                    maxFrameIndex: this.nextFetchFrameIndex - 1
                });
                //判断是否为最后一帧
                if (this.nextFetchFrameIndex == this.frameSum - 1) {
                    this.fetchModel(this.nextFetchFrameIndex, true);
                }
                else if (this.nextFetchFrameIndex != this.frameSum) {
                    this.fetchModel(this.nextFetchFrameIndex, false);
                }
                //第0帧特殊处理
                if (frameIndex == 0) {
                    return db.table('model').where('[uploadDate+frameIndex]').equals([this.uploadDate, frameIndex]).toArray();
                }
                else {
                    //不是第0帧直接终止Promise链
                    return Promise.reject(0);
                }
            })
            .then(model => {
                //注意后缀！
                return physikaInitObj(model[0].arrayBuffer, filetype);
            })
            .then(res => {
                this.updateScene([res]);
                this.setState({
                    isSliderShow: true
                });
            })
            .catch(state => {
                if (state == 0) {
                    console.log("FrameIndex is not 0.");
                }
                else {
                    console.log("Error fetchModel: ", state);
                }
            });
    }

    readModel = (uploadDate, frameIndex) => {
        db.table('model').where('[uploadDate+frameIndex]').equals([uploadDate, frameIndex]).toArray()
            .then(model => {
                return physikaInitObj(model[0].arrayBuffer, filetype);
            })
            .then(res => {
                if (frameIndex === this.state.curFrameIndex && uploadDate === this.uploadDate) {
                    this.updateScene([res]);
                }
                else {
                    //快速改变已获取帧数可以看到如下显示
                    console.log('Old model ' + frameIndex + ' is no longer uesd!');
                }
            })
            .catch(err => {
                console.log('Error readModel: ', err);
            });
    }

    changeInput = (value) => {
        console.log('changeInput: ', value);
        this.setState({ curFrameIndex: value }, () => {
            switch (this.frameStateArray[value]) {
                case 0:
                    //
                    break;
                case 1:
                    setTimeout(this.changeInput(value), 1000);
                    break;
                case 2:
                    this.readModel(this.uploadDate, value);
                    break;
                default:
                    break;
            }
        });
    }

    playClick = () => {
        //若当前帧为所能获取的最大帧数，则忽略点击操作
        if (this.state.curFrameIndex < this.state.maxFrameIndex) {
            if (!this.state.isPlay) {
                this.setState({
                    isPlay: true
                }, () => {
                    const playNextFrame = () => {
                        db.table('model').where('[uploadDate+frameIndex]').equals([this.uploadDate, this.state.curFrameIndex + 1]).toArray()
                            .then(model => {
                                return physikaInitObj(model[0].arrayBuffer, filetype);
                            })
                            .then(res => {
                                this.updateScene([res]);
                                if (this.state.curFrameIndex == this.state.maxFrameIndex - 1 || !this.state.isPlay) {
                                    this.setState({
                                        curFrameIndex: ++this.state.curFrameIndex,
                                        isPlay: false
                                    });
                                }
                                else {
                                    this.setState({
                                        curFrameIndex: ++this.state.curFrameIndex
                                    }, () => {
                                        setTimeout(playNextFrame, 10);
                                    })
                                }
                            })
                            .catch(err => {
                                console.log('Error readModel: ', err);
                            });
                    }
                    playNextFrame();
                });
            }
            else {
                this.setState({ isPlay: false });
            }
        }
    }

    switchScene = () => {
        this.setState({
            isShowResult: !this.state.isShowResult
        }, () => {
            console.log(this.curScene);
            if (this.state.isShowResult) {
                this.cScene.forEach(item => {
                    item.actor.setVisibility(false);
                });
                this.curScene.forEach(item => {
                    Object.keys(item).forEach(key => {
                        item[key].actor.setVisibility(true);
                    });
                });
            }
            else {
                this.cScene.forEach(item => {
                    item.actor.setVisibility(true);
                });
                this.curScene.forEach(item => {
                    Object.keys(item).forEach(key => {
                        item[key].actor.setVisibility(false);
                    });
                });
            }
            this.renderer.resetCamera();
            this.renderWindow.render();
        });
    }

    renderRepresentationOptions = () => representationOptions.map((item, index) => {
        return <Option value={index} key={index}>{item}</Option>
    })

    renderDescriptions = () => this.state.description.map((item, index) => {
        return <Descriptions.Item label={item.name} key={index}>{item.content}</Descriptions.Item>
    })

    render() {
        console.log("tree:", this.state.data);
        return (
            <div>
                <Divider>布料仿真</Divider>
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
                    <Panel header="多帧展示控制" key="3">
                        {
                            (this.state.isSliderShow) &&
                            <div>
                                <Slider min={0} max={this.state.maxFrameIndex} value={this.state.curFrameIndex} onChange={this.changeInput} disabled={this.state.isPlay} />
                                <Row>
                                    <Col span={13} style={{ alignItems: 'center', display: 'flex' }}>
                                        <span className="ant-rate-text">当前帧序号：</span>
                                    </Col>
                                    <Col span={3}>
                                        <InputNumber min={0} max={this.state.maxFrameIndex} value={this.state.curFrameIndex} onChange={this.changeInput} disabled={this.state.isPlay} />
                                    </Col>
                                </Row>
                                <Divider />
                                <Row>
                                    <Col span={13} style={{ alignItems: 'center', display: 'flex' }}>
                                        <span className="ant-rate-text">逐帧播放：</span>
                                    </Col>
                                    <Col span={3}>
                                        <Button onClick={this.playClick}>{this.state.isPlay ? 'Stop' : 'Play'}</Button>
                                    </Col>
                                </Row>
                            </div>
                        }
                    </Panel>
                    {/* forceRender为true，即折叠面板未打开时也渲染其中组件；若为false，则未打开面板前无法获得其中组件 */}
                    <Panel header="绘制信息" key="4" forceRender="true">
                        <Row>
                            <Col span={13} style={{ alignItems: 'center', display: 'flex' }}>
                                <span className="ant-rate-text">场景切换：</span>
                            </Col>
                            <Col span={3}>
                                <Button onClick={this.switchScene}>{this.state.isShowResult ? '初始化场景' : '模拟结果场景'}</Button>
                            </Col>
                        </Row>
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
    Cloth as PhysikaCloth
}