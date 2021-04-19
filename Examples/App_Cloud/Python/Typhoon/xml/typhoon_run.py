from typhoon_xml_parsing import XmlParsing
import SatImageTyphoon
import sys
import os
# python typhoon_run.py 
# E:\zhangqf\project\Sim_Cloud\xml\2021\upload_config_file.xml 
# E:\zhangqf\project\Sim_Cloud\natural_img 
# E:\zhangqf\project\Sim_Cloud\xml\2021  
# E:\zhangqf\project\Sim_Cloud\xml\2021\sim_data
if __name__ == '__main__':
    assert(len(sys.argv) == 5), "sim_run.py should have 4 arguments."
    upload_config_file_path = sys.argv[1]
    upload_file_dir = sys.argv[2]
    upload_date_dir = sys.argv[3]
    sim_data_dir = sys.argv[4]

    xml_parsed = XmlParsing(upload_config_file_path)
    VIS_path = upload_file_dir + '/' + xml_parsed.VIS
    IR1_path = upload_file_dir + '/' + xml_parsed.IR1
    IR2_path = upload_file_dir + '/' + xml_parsed.IR2
    WV_path = upload_file_dir + '/' + xml_parsed.WV
    SWIR_path = upload_file_dir + '/' + xml_parsed.SWIR
    files = [VIS_path, IR1_path, IR2_path, WV_path, SWIR_path]
    height = xml_parsed.range[0]
    width = xml_parsed.range[1]

    sim_data_filename = 'typhoon'
    sim_data_dir = sim_data_dir + '/'
    # run!
    typhoon = SatImageTyphoon.SatDataCloud()
    typhoon.Run(files, sim_data_dir, sim_data_filename, height,width)

    extra_info = {}
    extra_info['file_name'] = sim_data_filename
    extra_info['frame_sum'] = 1
    extra_info['animation'] = 'false'

    out_xml_file_path = upload_date_dir + '/' + 'typhoon_config_file.xml'
    xml_parsed.write_result_xml(out_xml_file_path, extra_info)

    #将路径输出到标准输出
    sys.stdout.write(out_xml_file_path)
