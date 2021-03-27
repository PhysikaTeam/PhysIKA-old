//深拷贝
function deepCopy(obj) {
    var newobj = obj.constructor === Array ? [] : {};
    if (typeof obj !== 'object') {
        return;
    }
    for (var i in obj) {
        //title成员中含有的dom标签在深拷贝时有很多问题！！！！
        if (i !== 'title') {
            newobj[i] = (typeof obj[i] === 'object' && obj[i] !== null) ? deepCopy(obj[i]) : obj[i];
        }
    }
    return newobj;
}

//判断输入对象是否为Object
const isObject = (object) => {
    return Object.prototype.toString.call(object) === '[object Object]';
}

//添加拾取的面片对应的树节点
function addPickedCell(cellId, node) {
    //添加pick对象
    let hasPickObj = false;
    if (!node.hasOwnProperty('children')) {
        node.children = [];
    }
    let pickObjIndex = node.children.length;
    node.children.forEach((item, index) => {
        if (item.tag == 'Pick') {
            hasPickObj = true;
            pickObjIndex = index;
        }
    });
    if (!hasPickObj) {
        let pickOBj = {
            children: [],
            key: `${node.key}-${node.children.length}`,
            tag: 'Pick',
            _attributes: {
                class: 'Pick',
                name: '面片拾取'
            }
        };
        node.children.push(pickOBj);
    }
    //添加面对象
    const pickObj = node.children[pickObjIndex];
    let hasCellObj = false;
    let cellObjIndex = pickObj.children.length;
    pickObj.children.forEach((item, index) => {
        if (item._attributes.name == `cell-${cellId}`) {
            hasCellObj = true;
            cellObjIndex = index;
        }
    });
    if (!hasCellObj) {
        let cellObj = {
            children: [],
            key: `${pickObj.key}-${pickObj.children.length}`,
            tag: 'Cell',
            _attributes: {
                class: 'Cell',
                name: `cell-${cellId}`
            }
        };
        pickObj.children.push(cellObj);
    }
    const cellObj = pickObj.children[cellObjIndex];
    if (cellObj.children.length === 0) {
        let fieldeObj = {
            key: `${cellObj.key}-0`,
            tag: 'Field',
            _text: '0.0 0.0 0.0',
            _attributes: {
                class: 'Vector3f',
                name: '施加力'
            }
        };
        cellObj.children.push(fieldeObj);
    }
    return cellObj.children[0];
}

function parseSimulationResult(data) {
    let simRunObj;
    for (let i = 0; i < data[0].children.length; i++) {
        if (data[0].children[i].tag === 'SimulationRun') {
            simRunObj = data[0].children[i];
            data[0].children.splice(i, 1);
        }
    }
    const resultInfo = {
        fileName: '',
        frameSum: 0,
        animation: false,
        description: []
    }
    for (const item of simRunObj.children) {
        if (item.tag === 'FileName') {
            resultInfo.fileName = item._text;
        }
        if (item.tag === 'FrameSum') {
            resultInfo.frameSum = item._text;
        }
        if (item.tag === 'Animation') {
            resultInfo.animation = (item._text === 'true');
        }
        resultInfo.description.push(
            {
                name: item._attributes.name,
                content: item._text
            }
        );
    }
    return resultInfo;
}

function checkUploadConfig(data) {
    let passTag = true;
    let errorTag = false;
    const check = (children) => {
        for (const item of children) {
            if (item.children) {
                const path = check(item.children);
                if (errorTag) {
                    return item._attributes.name + '->' + path;
                }
                continue;
            }
            if (item._attributes.class === 'File' && item._text === 'null') {
                errorTag = true;
                return item._attributes.name + '不能为空！';
            }
            if (item._text === undefined && item.tag !== 'AddNode') {
                errorTag = true;
                return item._attributes.name + '不能为空！';
            }
        }
    }
    let errorPath = check(data);
    if (errorTag) {
        alert('警告：' + errorPath);
        passTag = false;
    }
    return passTag;
}

//---------------删除添加树结点（目前只支持操作结点的子结点没有孩子）-----------------------
//newNode为新增加的结点，addNodeFinalKey为点击增加新结点的结点的最后一个key值
//流体块初始化对象
function buildFluidBlockObj() {
    const obj = {};
    obj.tag = "FluidBlock_0";
    obj._attributes = { name: "默认流体块" };
    obj.children = [];
    obj.children.push({
        tag: "denseMode",
        _attributes: {
            class: "Enum",
            disabled: "true",
            enum: "常规采样 大密集采样 密集采样",
            name: "采样方式"
        },
        _text: 0
    });
    obj.children.push({
        tag: "start",
        _attributes: {
            class: "Vector3f",
            disabled: "false",
            name: "起始坐标"
        },
        _text: [-0.5, 0, -0.5]
    });
    obj.children.push({
        tag: "end",
        _attributes: {
            class: "Vector3f",
            disabled: "false",
            name: "终止坐标"
        },
        _text: [0.5, 1, 0.5]
    });
    obj.children.push({
        tag: "translation",
        _attributes: {
            class: "Vector3f",
            disabled: "false",
            name: "平移向量"
        },
        _text: [-1.45, 0, 0]
    });
    obj.children.push({
        tag: "scale",
        _attributes: {
            class: "Vector3f",
            disabled: "true",
            name: "规模"
        },
        _text: [1, 1, 1]
    });
    obj.children.push({
        tag: "initialVelocity",
        _attributes: {
            class: "Vector3f",
            disabled: "false",
            name: "初始速度"
        },
        _text: [0, 0, 0]
    });
    return obj;
}

function initNewNode(newNode, fatherKey, sonKey) {
    newNode._attributes.name = '新结点'
    const tagArray = newNode.tag.split('_');
    newNode.tag = tagArray[0] + '_' + sonKey;
    newNode.key = fatherKey + '-' + sonKey;
    if (newNode.children) {
        newNode.children.forEach((item, index) => {
            if (item._attributes.class === 'File') {
                item._text = 'null';
            }
            if (item.tag === 'isWall') {
                item._text = false;
            }
            if (item.tag === 'mapInvert') {
                item._text = false;
            }
            item.key = fatherKey + '-' + sonKey + '-' + index;
        })
    }
    newNode.deletable = true;
}

function addNewNode(tree, item) {
    let newNode;
    let eachKey = item.key.split('-');
    let count = 0;
    const findTreeNodeKey = (node) => {
        if (count === eachKey.length - 1) {
            node.pop();
            let sonKey = '';
            for (let i = item.key.length - 1; i > 0; --i) {
                if (item.key[i] !== '-') {
                    sonKey = item.key[i] + sonKey;
                }
                else {
                    sonKey = Number(sonKey);
                    const fatherKey = item.key.substring(0, i);
                    //根据父节点tag选择新结点属性
                    let tmpNode = tree;
                    for (let i = 0; i < count - 1; ++i) {
                        tmpNode = tmpNode[eachKey[i]].children;
                    }
                    switch (tmpNode[eachKey[count - 1]].tag) {
                        case 'RigidBodies':
                            newNode = deepCopy(node[0]);
                            break;
                        case 'FluidBlocks':
                            newNode = buildFluidBlockObj();
                            break;
                        case 'Cloths':
                            newNode = deepCopy(node[0]);
                            break;
                        default:
                            break;
                    }
                    initNewNode(newNode, fatherKey, sonKey);
                    node.push(newNode);
                    item.key = fatherKey + '-' + (sonKey + 1);
                    node.push(item);
                    break;
                }
            }
            return;
        }
        findTreeNodeKey(node[eachKey[count++]].children);
    };
    findTreeNodeKey(tree);
    return { tree, newNode };
}

function changeNodeKeyAfterDelete(node, sonKey, sonKeyIndex) {
    for (let i = sonKey; i < node.length; ++i) {
        const nodeKeyArray = node[i].key.split('-');
        let newKey = nodeKeyArray[0];
        for (let keyIndex = 1; keyIndex < nodeKeyArray.length; ++keyIndex) {
            if (keyIndex === sonKeyIndex) {
                newKey += '-' + (nodeKeyArray[keyIndex] - 1);
            }
            else {
                newKey += '-' + nodeKeyArray[keyIndex];
            }

        }
        node[i].key = newKey;
        if (node[i].children) {
            const tagArray = node[i].tag.split('_');
            node[i].tag = tagArray[0] + '_' + sonKey;
            changeNodeKeyAfterDelete(node[i].children, 0, sonKeyIndex);
        }
    }
}

function deleteNode(tree, item) {
    let deletedNode;
    let eachKey = item.key.split('-');
    let count = 0;
    const findTreeNodeKey = (node) => {
        if (count === eachKey.length - 1) {
            deletedNode = node.splice(eachKey[count], 1);
            const sonKey = eachKey[count];
            const sonKeyIndex = count;
            changeNodeKeyAfterDelete(node, sonKey, sonKeyIndex);
            return;
        }
        findTreeNodeKey(node[eachKey[count++]].children);
    };
    findTreeNodeKey(tree);
    return { tree, deletedNode };
}
//----------------------------------------------------------------------------------

export { deepCopy, isObject, parseSimulationResult, addPickedCell, checkUploadConfig, addNewNode, deleteNode };