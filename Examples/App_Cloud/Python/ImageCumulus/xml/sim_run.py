from sim_xml_parsing import XmlParsing
import Sim_Cloud
import sys
import os
# python sim_run.py 
# E:\zhangqf\project\Sim_Cloud\xml\2021\upload_config_file.xml 
# E:\zhangqf\project\Sim_Cloud\natural_img\ 
# E:\zhangqf\project\Sim_Cloud\xml\2021\  
# E:\zhangqf\project\Sim_Cloud\xml\2021\sim_data\
if __name__ == '__main__':
    assert(len(sys.argv) == 5), "sim_run.py should have 4 arguments."
    upload_config_file_path = sys.argv[1]
    upload_file_dir = sys.argv[2]
    upload_date_dir = sys.argv[3]
    sim_data_dir = sys.argv[4]

    xml_parsed = XmlParsing(upload_config_file_path)
    
    upload_img_path = upload_file_dir + '/' + xml_parsed.upload_name
    sim_data_filename = 'cloud'
    sim_data_path = sim_data_dir + '/' + sim_data_filename + '.obj'
    # run!
    Sim_Cloud.sim_cloud(upload_img_path, sim_data_path)

    extra_info = {}
    extra_info['file_name'] = sim_data_filename
    extra_info['frame_sum'] = 1
    extra_info['animation'] = 'false'
    extra_info['sun_color'] = Sim_Cloud.get_sun_color()
    extra_info['img_WH'] = Sim_Cloud.get_img_WH()
    extra_info['num_vertices'] = Sim_Cloud.get_num_vertices()
    extra_info['num_faces'] = Sim_Cloud.get_num_faces()

    out_xml_file_path = upload_date_dir + '/' + 'sim_config_file.xml'
    xml_parsed.write_result_xml(out_xml_file_path, extra_info)

    #将路径输出到标准输出
    sys.stdout.write(out_xml_file_path)
