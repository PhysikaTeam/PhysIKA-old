from fluid_xml_parsing import XmlParsing
import FluidEvolution
import sys
import os

# fluidEvaluation(std::string& oriCdfName, 
#					std::string& oriShapeName, 
#					std::string& tarCdfName, 
#					std::string& tarShapeName,
#					std::string& rootPath,
#					int max_steps=100);
if __name__ == '__main__':
    assert(len(sys.argv) == 5), "fluid_run.py should have 4 arguments."
    upload_config_file_path = sys.argv[1]
    upload_file_dir = sys.argv[2]
    upload_date_dir = sys.argv[3]
    sim_data_dir = sys.argv[4]
    
    xml_parsed = XmlParsing(upload_config_file_path)
    oriShapeName = upload_file_dir + '/' + xml_parsed.init_particle
    oriCdfName = upload_file_dir + '/' + xml_parsed.init_cdf
    tarShapeName = upload_file_dir + '/' + xml_parsed.goal_particle
    tarCdfName = upload_file_dir + '/' + xml_parsed.goal_cdf
    max_steps = xml_parsed.frame

    rootPath = sim_data_dir + '/'

    # run!
    FluidEvolution.fluidEvaluation(oriCdfName, oriShapeName, tarCdfName, tarShapeName, rootPath, max_steps)
    
    extra_info = {}
    extra_info['frame_sum'] = xml_parsed.frame
    extra_info['animation'] = 'true'

    out_xml_file_path = upload_date_dir + '/' + 'particle_evolution_config_file.xml'
    xml_parsed.write_result_xml(out_xml_file_path, extra_info)
    #将路径输出到标准输出
    sys.stdout.write(out_xml_file_path)