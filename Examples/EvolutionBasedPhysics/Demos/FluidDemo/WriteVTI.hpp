#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

// base64 encoding
static const char* base64_chars = {
			 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			 "abcdefghijklmnopqrstuvwxyz"
			 "0123456789"
			 "+/"};

std::string base64_encode(unsigned char const* bytes_to_encode, size_t in_len) {

	size_t len_encoded = (in_len + 2) / 3 * 4;

	unsigned char trailing_char = '=';

	const char* base64_chars_ = base64_chars;

	std::string ret;
	ret.reserve(len_encoded);

	unsigned int pos = 0;

	while (pos < in_len) {
		ret.push_back(base64_chars_[(bytes_to_encode[pos + 0] & 0xfc) >> 2]);

		if (pos + 1 < in_len) {
			ret.push_back(base64_chars_[((bytes_to_encode[pos + 0] & 0x03) << 4) + ((bytes_to_encode[pos + 1] & 0xf0) >> 4)]);

			if (pos + 2 < in_len) {
				ret.push_back(base64_chars_[((bytes_to_encode[pos + 1] & 0x0f) << 2) + ((bytes_to_encode[pos + 2] & 0xc0) >> 6)]);
				ret.push_back(base64_chars_[bytes_to_encode[pos + 2] & 0x3f]);
			}
			else {
				ret.push_back(base64_chars_[(bytes_to_encode[pos + 1] & 0x0f) << 2]);
				ret.push_back(trailing_char);
			}
		}
		else {

			ret.push_back(base64_chars_[(bytes_to_encode[pos + 0] & 0x03) << 4]);
			ret.push_back(trailing_char);
			ret.push_back(trailing_char);
		}

		pos += 3;
	}

	return ret;
}


// 将密度数据保存为.vti文件(ascii形式)(length，width，height：数据场长宽高；data：密度数据；path：文件保存路径；ascii：是否使用ascii形式存储)
bool WriteVTI(int length, int width, int height, const std::vector<float>& data, const std::string& path) {
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    // 这里改用ascii还是binary
    bool ascii = true;
    if (file) {
        file << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"";
        // 判断大小端
        int a = 1;
        char* p = reinterpret_cast<char*>(&a);
        if (*p == 1) {
            file << "LittleEndian";
        }
        else {
            file << "BigEndian";
        }
        // 判断int类型大小
        if(sizeof(int) == 8) {
			file << "\" header_type=\"UInt64\">" << std::endl;
		}
		else {
			file << "\" header_type=\"UInt32\">" << std::endl;
		}

        file << "<ImageData WholeExtent=\"" << "0 " << length << " 0 " << width << " 0 " << height
            << "\" Origin=\"0 0 0\" Spacing=\"1.0 1.0 1.0\">" << std::endl;
        file << "<Piece Extent=\"" << "0 " << length << " 0 " << width << " 0 " << height << "\">" << std::endl;
        file << "<PointData Scalars=\"Scalars_\">" << std::endl;
        // 计算密度值范围
        float rangeMin = 1.0f;
        float rangeMax = 0.0f;
        for(float value:data) {
            if(value < rangeMin) {
                rangeMin = value;
            }
            if(value > rangeMax) {
                rangeMax = value;
            }
        }
        // 判断float类型大小
        if(sizeof(float) == 8) {
			file << "<DataArray type=\"Float64\" Name=\"Scalars_\" format=\"";
		}
		else {
			file << "<DataArray type=\"Float32\" Name=\"Scalars_\" format=\"";
		}
        // 存储形式（ascii or binary）
        if(ascii) {
            file << "ascii";
			file << "\" RangeMin=\"" << rangeMin << "\" RangeMax=\"" << rangeMax << "\">" << std::endl;
            for (float value : data) {
                file << value << " ";
            }
        }
        else {
            file << "binary";
            file << "\" RangeMin=\"" << rangeMin << "\" RangeMax=\"" << rangeMax << "\">" << std::endl;
			std::stringstream ss;
			int size = data.size();
			ss.write(reinterpret_cast<char*>(&size), sizeof(int));
			for (float value : data) {
				ss.write(reinterpret_cast<char*>(&value), sizeof(float));
			}
			const std::string& dataString = ss.str();
			file << base64_encode(reinterpret_cast<const unsigned char*>(dataString.data()), data.size()*sizeof(float) + sizeof(int));
        }
        
        file << "</DataArray>" << std::endl;
        file << "</PointData>" << std::endl;
        file << "<CellData>" << std::endl;
        file << "</CellData>" << std::endl;
        file << "</Piece>" << std::endl;
        file << "</ImageData>" << std::endl;
        file << "</VTKFile>" << std::endl;
        file.close();

    }
    else {
        printf("Fail to save vti file: %s!\n", path.c_str());
        return false;
    }
    return true;
}