#include "maps.hpp"


std::string get_key(const float* coor){
	std::stringstream ss;
	char char_coor[9];
	
	for(int i=0; i<3; i++){
		snprintf(char_coor, sizeof(char_coor), "%8.3f", coor[i]);
		ss << char_coor;
	}

	return ss.str();
	
}

void parse_key(std::string key, float* coor){
	
	char char_coor[9];
	
	for(int i=0; i<3; i++){
		for(int j=0; j<8; j++){
			char_coor[j] = key.c_str()[j+i*8];
		}
		char_coor[8] = '\0';

		coor[i] = (float)atof(char_coor);
	}	
}
