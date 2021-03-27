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
            # in 'InitialState' tag 
            initialState = scene.find('InitialState')
            for elem in initialState.iter():
                tag = elem.tag
                if(tag == 'Particle'):
                    self.init_particle = elem.text
                if(tag == 'CDF'):
                    self.init_cdf = elem.text
            
            # in 'GoalState' tag
            goalState = scene.find('GoalState')
            for elem in goalState.iter():
                tag = elem.tag
                if(tag == 'Particle'):
                    self.goal_particle = elem.text
                if(tag == 'CDF'):
                    self.goal_cdf = elem.text

            # in 'Frame' tag
            frame = scene.find('Frame')
            self.frame = int(frame.text) 



    
    # 输出结果xml文件
    def write_result_xml(self, out_xml_file_path, extra_info):
        scene = ET.Element('Scene')
        scene.set('class', '')
        scene.set('name', '场景')

        simulation_run = ET.SubElement(scene, 'SimulationRun')
        simulation_run.set('name', '模拟运行结果')
        # add 'FrameSum' tag
        frameSum = ET.SubElement(simulation_run, 'FrameSum')
        frameSum.set('name', '帧总数')
        frameSum.text = str(extra_info['frame_sum'])
        # add 'Animation' tag
        animation = ET.SubElement(simulation_run, 'Animation')
        animation.set('name', '是否支持动画')
        animation.text = extra_info['animation']

        indent(scene)
        tree = ET.ElementTree(scene)
        tree.write(out_xml_file_path, encoding="utf-8", xml_declaration=True)



if __name__ == '__main__':
    xml_parsed = XmlParsing('particle_evolution.xml')
    print('初始状态粒子对象(init_particle)：\t\t', xml_parsed.init_particle)
    print('初始状态CDF(init_cdf)：    \t\t', xml_parsed.init_cdf)
    print('目标状态粒子对象(goal_particle)： \t\t', xml_parsed.goal_particle)
    print('目标状态CDF(goal_cdf)： \t\t', xml_parsed.goal_cdf)
    print('生成帧数(frame)：   \t\t', xml_parsed.frame)

    extra_info = {}
    extra_info['animation'] = 'true'
    out_xml_file_path = 'particle_evolution_config_file.xml'
    xml_parsed.write_result_xml(out_xml_file_path, extra_info)