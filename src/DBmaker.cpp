#include <string>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "FastaFile.h"

using namespace std;

int main(int argc, char* argv[])
{
	if(argc != 3){
		printf("Error pls refer to usage\n");
		exit(-1);
	}
	string filePath(argv[1]);
	string outputPath(argv[2]);
	printf("Processing database %s\n", filePath.c_str());	
	printf("Target file basename %s\n", outputPath.c_str());	
	FastaFile DBMaker(filePath, outputPath);
	try{
	DBMaker.run();	
	}catch(const char* e){
		cout<<e<<endl;
		exit(-1);
	}
	return 0;
}
