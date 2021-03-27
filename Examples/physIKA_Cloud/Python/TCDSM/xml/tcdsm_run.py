from tcdsm_xml_parsing import XmlParsing
import tcdsmModeler
import sys
import os
# python tcdsm_run.py 
# E:\zhangqf\project\TCDSM\xml\2021\upload_config_file.xml 
# E:\zhangqf\project\TCDSM\data 
# E:\zhangqf\project\TCDSM\xml\2021  
# E:\zhangqf\project\TCDSM\xml\2021\sim_data
if __name__ == '__main__':
    assert(len(sys.argv) == 5), "sim_run.py should have 4 arguments."
    upload_config_file_path = sys.argv[1]
    upload_file_dir = sys.argv[2]
    upload_date_dir = sys.argv[3]
    sim_data_dir = sys.argv[4]

    xml_parsed = XmlParsing(upload_config_file_path)
    netCDF_file_path = upload_file_dir + '/' + xml_parsed.netCDF

    sim_data_filename = 'frame0'
    sim_data_dir = sim_data_dir + '/'

    # run!
    tcdsmModeler.execute(netCDF_file_path, sim_data_dir)
    
    extra_info = {}
    extra_info['file_name'] = sim_data_filename
    extra_info['frame_sum'] = 1
    extra_info['animation'] = 'false'

    out_xml_file_path = upload_date_dir + '/' + 'tcdsm_config_file.xml'
    xml_parsed.write_result_xml(out_xml_file_path, extra_info)
    #将路径输出到标准输出
    sys.stdout.write(out_xml_file_path)
