from xml_parsing import XmlParsing
from Cloud_Gen_Based_Euler import CloudEulerGen

if __name__ == '__main__':
    xml_parsed = XmlParsing('euler.xml')
    arg = []
    arg.append(xml_parsed.simType) # 模拟对象
    arg.append(xml_parsed.scalar) # 模拟空间大小(不包含边界)
    arg.append(0) # 发射源起始x坐标
    arg.append(30) # 发射源结束x坐标
    arg.append(0) # 发射源起始y坐标
    arg.append(30) # 发射源结束y坐标
    arg.append(xml_parsed.noise) # 噪声种类

    epsilon = xml_parsed.vortex # 涡旋抑制系数

    cloud = CloudEulerGen(arg, epsilon)
    cloud.CloudGenFrameRun()

    print("只是告诉你跑完了...")