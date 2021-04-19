# physika_web

# 配置安装
nodejs版本要求14.16.0

把项目clone到本地目录
```shell script
git clone https://github.com/cenyc/Physika-web.git
```
进入clone文件目录
```shell script
cd Physika-web
```
安装所需到node_modules
```shell script
npm install
```
使用webpack打包项目
```shell script
npm run build
```
运行项目
```shell script
node .\src_server\websocket.js
```
在浏览器访问http://localhost:8888 即可

src_server目录下的pathconfig.json文件用于配置数据存放路径和python接口路径：
"userPath"：用于存放用户相关文件目录；
每个仿真方法路径配置都由一个json对象保存，如下示例：
"CloudEuler":{
    "simType":"CloudEuler",
    "initConfigFileName":"./data/init_config_file/cloud_euler.xml",
    "callPythonFileName":""
}
"CloudEuler"：代表仿真类型名，需和对应仿真方法前端界面代码中的simType相同；
对象中存储有"simType"、"initConfigFileName"、"callPythonFileName"，
其中"simType"为仿真类型名，"initConfigFileName"为初始化配置文件的路径，"callPythonFileName"为调用对应仿真方法接口文件的路径。

src_server目录下的websocket.js文件中的queryCountMax参数为预取缓存机制中轮询的次数，每次间隔1秒钟，超出设定的最大次数后会产生预取超时的响应，可根据电脑性能和仿真时间进行调整。

测试用例在 ./data/test_case/ 中。
