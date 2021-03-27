import xml.etree.ElementTree as ET

class XmlParsing:
    def __init__(self, filename):
        tree = ET.parse(filename)
        scene = tree.getroot()

        domain = scene[0]
        self.simType = int(domain[0].text) # 模拟对象
        self.noise = int(domain[1].text) # 源噪声
        self.vortex = float(domain[2].text) # epsilon
        self.scalar = int(domain[3].text) # 模拟空间

        self.gravity = list(map(float, scene[1].text.split())) # 重力
        if(scene[2].attrib['type'] == '0(自适应步长)'):
            self.timeStep = float(0) # 模拟步长（自适应步长）
        else:
            self.timeStep = float(scene[2].text) # 模拟步长（给定步长）
        self.path = scene[3].text # 路径
        self.frame = int(scene[4].text) # 模拟帧数




if __name__ == '__main__':
    xml_parsed = XmlParsing('euler.xml')
    print('模拟对象(simType)：\t', xml_parsed.simType)
    print('源噪声(noise)：    \t', xml_parsed.noise)
    print('epsilon(vortex)： \t', xml_parsed.vortex)
    print('模拟空间(scalar)： \t', xml_parsed.scalar)
    print('重力(gravity)：    \t', xml_parsed.gravity)
    print('模拟步长(timeStep)：\t', xml_parsed.timeStep)
    print('路径(path)：       \t', xml_parsed.path)
    print('模拟帧数(frame)：  \t', xml_parsed.frame)