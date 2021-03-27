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
            # in 'WRF' tag 
            wrf = scene.find('WRF')
            for elem in wrf.iter():
                tag = elem.tag
                if(tag == 'NetCDF'):
                    self.netCDF = elem.text


    
    # 输出结果xml文件
    def write_result_xml(self, out_xml_file_path, extra_info):
        with open(self.in_file_path, 'r', encoding='UTF-8') as xml_file:
            tree = ET.parse(xml_file)
            scene = tree.getroot()
            simulation_run = ET.Element('SimulationRun')
            simulation_run.set('name', '模拟运行结果')
            # add 'FileName' tag
            file_name = ET.SubElement(simulation_run, 'FileName')
            file_name.set('name', '仿真文件名')
            file_name.text = extra_info['file_name']
            # add 'FrameSum' tag
            frame_sum = ET.SubElement(simulation_run, 'FrameSum')
            frame_sum.set('name', '帧总数')
            frame_sum.text = str(extra_info['frame_sum'])
            # add 'Animation' tag
            animation = ET.SubElement(simulation_run, 'Animation')
            animation.set('name', '是否支持动画')
            animation.text = extra_info['animation']

            scene.append(simulation_run)
            indent(scene)
            tree.write(out_xml_file_path, encoding="utf-8", xml_declaration=True)



# if __name__ == '__main__':
#     xml_parsed = XmlParsing('euler_i.xml')
#     print('起始位置(Start)：\t\t', xml_parsed.start_x, xml_parsed.start_y)
#     print('结束位置(End)：    \t\t', xml_parsed.end_x, xml_parsed.end_y)
#     print('涡旋抑制系数(Vortex)： \t\t', xml_parsed.vortex)
#     print('源噪声(Noise)： \t\t', xml_parsed.noise)
#     print('大小(Scale)：    \t\t', xml_parsed.scale)
#     print('仿真类型(SimulationMethod)：\t', xml_parsed.sim_type)
#     print('模拟帧数(Frame)：  \t\t', xml_parsed.frame)

#     xml_parsed.write_result_xml('euler_s_test.xml', './data/visualize_data/head-binary-zlib.vti')