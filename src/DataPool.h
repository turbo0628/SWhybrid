#ifndef _DATA_POOL_H_
#define _DATA_POOL_H_

#include <vector>
#include <string>
#include <fstream>
#include <pthread.h>

#include "Defs.h"


class DataPool {
public:
	DataPool(int topNum, char* qPath, char* dbPath);
	~DataPool();

	void Read();
	void run();
	bool fillChunk(char* dst, batchInfo* seqBatchInfo, size_t& infoSize, size_t& aaSize, size_t bufferSize, bool cpuFlag, size_t& baseIndex);
	bool initRead();
	const char* getQuery();
	size_t getQueryLen();
	size_t getTotalResidue(){
		return totalResidue;
	}
	void getTitle(size_t index, std::string& outTitle);
	const char* getTitle(size_t index);

private:
	void setPath(const std::string& path);
	void openRead();
	void closeFiles();
	bool checkFiles();
	bool fileExist(const std::string& name);
	void getQuery(const std::string& queryPath);

	std::string dbFilePath;
	std::string seqPath;
	std::string titlePath;
	std::string mapPath;

	std::fstream titleFileStream;
	std::fstream seqFileStream;
	std::fstream mapFileStream;

private:
	int topNum;
	bool readFlag;
	size_t totalResidue;
	size_t seqIdx;
	bool fill(char* dst, batchInfo* seqBatchInfo, size_t& infoSize, size_t& aaSize,  size_t bufferSize, bool cpuFlag, size_t& baseIndex);

	std::string seqBatch; //for CUDA, SSE and MIC

	std::vector<unsigned> seqIndices; //table for index

	char* buffer;
	int bufIdx;
	int bufIdxCalled;//parameter passed to the calling thread
	static const char amino_acids_trans[256];

	size_t totalNum;
	std::string queryPath;
	std::string dbPath;
	std::string queryName;
	std::string queryResidue;

	size_t remainderPos;
	size_t remainderNum;
	size_t globalNumSeqs;
	size_t globalSeqLen;

	size_t curBufferSize;
	int 	callIdx[2];	
};

#endif /* _DATAPOOL_H_ */
