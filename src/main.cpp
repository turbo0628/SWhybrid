#include "DataPool.h"
#include "WorkerPool.h"
#include "Param.h"

using namespace std;

int main(int argc, char* argv[]){
	
	Param* param = new Param(argc, argv);
	param->startTimer();
	//param->sysConf();
	DataPool loader(param->topNum, param->queryPath, param->dbPath);//TODO topNum may be larger than seq num
	WorkerPool pool(&loader, param);

	return 0;
}
