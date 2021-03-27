import xml.etree.ElementTree as ET

# 给输出xml文件增加缩进和换行
def indent(elem, level=0):
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class XmlParsing:
    # 读取并解析xml文件
    def __init__(self, in_file_path):
        with open(in_file_path, 'r', encoding='UTF-8') as xml_file:
            self.in_file_path = in_file_path
            tree = ET.parse(xml_file)
            scene = tree.getroot()
            # in 'Cloth' tag 
            cloths = scene.find('Cloths')
            cloth_0=cloths.find('Cloth_0')
            for elem in cloth_0.iter():
                tag = elem.tag
                if(tag == 'geometryFile'):
                    self.geometryFile = elem.text

    
    # 输出结果xml文件
    def write_result_xml(self, out_xml_file_path, extra_info):
        scene = ET.Element('Scene')
        scene.set('class', '')
        scene.set('name', '场景')

        simulation_run = ET.SubElement(scene, 'SimulationRun')
        simulation_run.set('name', '模拟运行结果')
        # add 'FrameSum' tag
        frame_sum = ET.SubElement(simulation_run, 'FrameSum')
        frame_sum.set('name', '帧总数')
        frame_sum.text = str(extra_info['frame_sum'])
        # add 'Animation' tag
        animation = ET.SubElement(simulation_run, 'Animation')
        animation.set('name', '是否支持动画')
        animation.text = extra_info['animation']

        indent(scene)
        tree = ET.ElementTree(scene)
        tree.write(out_xml_file_path, encoding="utf-8", xml_declaration=True)


if __name__ == '__main__':
    xml_parsed = XmlParsing('cloth.xml')
    print('几何文件(geometryFile)：\t\t', xml_parsed.geometryFile)

    extra_info = {}
    extra_info['frame_sum'] = 100
    extra_info['animation'] = 'true'
    out_xml_file_path = 'cloth_config_file.xml'
    xml_parsed.write_result_xml(out_xml_file_path, extra_info)