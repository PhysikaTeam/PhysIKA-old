from cloth_xml_parsing import XmlParsing
from App_Cloth2 import AppCloth
import sys
import os
os.chdir("D:\\PhysIKA_merge_build_python\\lib\\Release")

if __name__ == '__main__':
    assert(len(sys.argv) == 5), "cloth_run.py should have 4 arguments."
    upload_config_file_path = sys.argv[1]
    upload_file_dir = sys.argv[2]
    upload_date_dir = sys.argv[3]
    sim_data_dir = sys.argv[4]

    xml_parsed = XmlParsing(upload_config_file_path)
    geometry_path = upload_file_dir + '/' + xml_parsed.geometryFile
    
    sim_data_dir = sim_data_dir + '/'
    # run!
    AppCloth(geometry_path, sim_data_dir)
    
    extra_info = {}
    extra_info['frame_sum'] = 500
    extra_info['animation'] = 'true'

    out_xml_file_path = upload_date_dir + '/' + 'cloth_config_file.xml'
    xml_parsed.write_result_xml(out_xml_file_path, extra_info)

    #将路径输出到标准输出
    sys.stdout.write(out_xml_file_path)
