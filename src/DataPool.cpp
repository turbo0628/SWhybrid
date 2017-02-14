#include "DataPool.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstring>

#include <unistd.h>

using namespace std;

#define _DEBUG

static const size_t maxBufferSize = 1 << 28;
static const size_t minBufferSize = maxBufferSize / 8;
static const int querySeqAlignedLength = 8;

const int mapBatchWidth = 128;

pthread_mutex_t readerMutex;

int lock(){
	return pthread_mutex_lock(&readerMutex);
}

int unlock(){
	return pthread_mutex_unlock(&readerMutex);
}

int tmpCnt;

const char DataPool::amino_acids_trans[256] = { -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 9, 10, 11, 12, -1, 13,
		14, 15, 16, 17, -1, 18, 19, 20, 21, 22, -1, -1, -1, -1, -1, -1, 0, 1, 2,
		3, 4, 5, 6, 7, 8, -1, 9, 10, 11, 12, -1, 13, 14, 15, 16, 17, -1, 18, 19,
		20, 21, 22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

DataPool::DataPool(int topNum, char* qPath, char* dbPath) :
	topNum(topNum),
	readFlag(false),
	buffer(NULL),
	bufIdx(0),
	seqIdx(0),
	totalResidue(0),
	bufIdxCalled(0),
	queryPath(qPath),
	dbPath(dbPath),
	callIdx()
{
	callIdx[0] = 0;
	callIdx[1] = 0;
	pthread_mutex_init(&readerMutex, NULL);
	tmpCnt = 0;
	totalNum = 0;
	run();
}

DataPool::~DataPool() {
	pthread_mutex_destroy(&readerMutex);
	closeFiles();
}

void DataPool::run() {
	setPath(dbPath.c_str());
	Read();
}

void DataPool::setPath(const string& path) {
	bool makeFlag = 0;
	int seqpos;
	int mappos;
	int titlepos;

	seqpos = path.find(".seq");
	mappos = path.find(".map");
	titlepos = path.find(".title");

	if (seqpos > 0) {
		dbFilePath = path.substr(0, seqpos);
	} else if (mappos > 0) {
		dbFilePath = path.substr(0, mappos);
	} else if (titlepos > 0) {
		dbFilePath = path.substr(0, titlepos);
	} else {
		dbFilePath = path;
	}
	seqPath = dbFilePath + ".seq";
	mapPath = dbFilePath + ".map";
	titlePath = dbFilePath + ".title";
	if (makeFlag) {
		throw("need to make database\n");
	}
}


void DataPool::openRead() {
	titleFileStream.open(titlePath.c_str(), ios::in);
	seqFileStream.open(seqPath.c_str(), ios::in | ios::binary);
	mapFileStream.open(mapPath.c_str(), ios::in | ios::binary);
}

void DataPool::closeFiles() {
	if (titleFileStream.is_open())
		titleFileStream.close();
	if (seqFileStream.is_open())
		seqFileStream.close();
	if (mapFileStream.is_open())
		mapFileStream.close();
}

void DataPool::Read() {
	if (checkFiles()) {
		try {
			getQuery(queryPath);
			openRead();
			initRead();
			printf("Query sequence length %ld\n", getQueryLen());
		} catch (const char * e) {
			cerr << e << endl;
			exit(-1);
		}
	}
}

const char* DataPool::getQuery(){
	return queryResidue.data();	
}

size_t DataPool::getQueryLen()
{
	return queryResidue.size();
}

bool DataPool::initRead()
{
	char flag;
	//read file head
	seqFileStream.seekg(0);
	seqFileStream.read(reinterpret_cast<char*>(&remainderPos), sizeof(size_t));//the boundry of batches
	seqFileStream.read(reinterpret_cast<char*>(&remainderNum), sizeof(size_t));//how many of these ones
	printf("Sequence Database size %2.2lfGB\n", (double) remainderPos /(1 << 30));
	//printf("%x %d\n", remainderPos, remainderNum);
	if (seqFileStream >> flag && flag == '$') {
		seqFileStream.read(reinterpret_cast<char*>(&globalNumSeqs),
				sizeof(size_t));
		seqFileStream.read(reinterpret_cast<char*>(&globalSeqLen),
				sizeof(size_t));
		//seqIdx = globalNumSeqs;
	}
	else{
		goto err;
	}
	return true;
err:
	throw("initialization sequence file failed, database file damaged\n");
	exit(0);
}

/*how about timing for every worker?*/
bool DataPool::fill(char* dstBuffer, batchInfo* seqBatchInfo, size_t& infoSize, size_t& aaSize,size_t bufferSize, bool cpuFlag, size_t& baseIndex)
{
		
#if 0
	if(bufferSize > maxBufferSize)
	{
		cerr<<"ERROR: buffer size of this work is too large"<<endl;
		exit(-1);
	}
#endif
	streampos pos = seqFileStream.tellg();
	//if(!cpuFlag && (remainderPos - pos <= maxBufferSize))
#if 0
	if(!cpuFlag && (remainderPos - pos <= maxBufferSize))
	{
		/*return for the last chunk*/;
		infoSize = 0;
		return false;
	}

	/*
	if(pos >= 1 << 27){
		infoSize = 0;
		return false;
	}*/
#endif
	char flag;
	baseIndex = seqIdx;
	seqIdx += globalNumSeqs;
	size_t seqLen = globalSeqLen;
	size_t numSeqs = globalNumSeqs;
	size_t readSize = seqLen * numSeqs;//remained
	char* bufp = dstBuffer;
	size_t numRead = seqLen * numSeqs;
	//batchInfo tmpInfo;
	batchInfo* infop = seqBatchInfo;
	infoSize = 0;
	
	//printf("buffer size %ldMB\n", bufferSize >> 20);
	if(remainderPos - pos <= 4 * maxBufferSize)
	{
		bufferSize = minBufferSize;	
	}
	
	//printf("[%2.2lf]\r", (double) seqFileStream.tellg() / remainderPos * 100);
	//fflush(stdout);

	if(dstBuffer == NULL)
		throw("reader buffer not allcated\n");	
	
	/*read one batch*/
	if((size_t) seqFileStream.tellp() < remainderPos){
		readFlag = true;
		while(1){
			/*read indices of each sequence*/
			/*check flags to make sure that the sequence length is correct.*/
			if (seqFileStream >> flag && flag == '#') {
			} else	{
				printf("inidices flag %c ", flag);
				goto err;
			}
			/*read sequence residues*/
			if (numRead <= bufferSize) {
				*(infop++) = batchInfo(numSeqs, seqLen);
				++infoSize;	
				seqFileStream.read(bufp, numRead * sizeof(char));
				bufp += numRead;
			} else {
				fprintf(stderr, "readSize %ld, bufferSize %ld\n", readSize, bufferSize);
				throw("error: readBufferSize too small\n");
			}

			/*read information of next batch*/
			if((size_t) seqFileStream.tellp() < remainderPos){
				if (seqFileStream >> flag && flag == '$') {
					seqFileStream.read(reinterpret_cast<char*>(&numSeqs),
							sizeof(size_t));
					seqFileStream.read(reinterpret_cast<char*>(&seqLen),
							sizeof(size_t));
					seqIdx += numSeqs;
				} else {
					printf("batch info flag %d, seqIdx %ld, numSeqs %ld, seqLen %ld\n", flag, seqIdx, numSeqs, seqLen);
					goto err;
				}
			}else{
				/*The last batch is read*/
				totalResidue += readSize;
				return true;
			}
			numRead = numSeqs * seqLen;
			readSize += numRead;

			if (readSize >= bufferSize) {
				globalSeqLen = seqLen;
				globalNumSeqs = numSeqs;
				seqIdx -= numSeqs;
				readSize -= numRead;
				totalResidue += readSize;
				aaSize = readSize;
				return true;
			}
		}//while
	}
	else{
		return false;
	}
err: throw("database reading failed, file may be damaged\n");
}

bool DataPool::fillChunk(char* dstBuffer, batchInfo* seqBatchInfo, size_t& infoSize, size_t& aaSize, size_t bufferSize, bool cpuFlag, size_t& baseIndex)
{
	lock();
	bool ret = fill(dstBuffer, seqBatchInfo, infoSize, aaSize, bufferSize, cpuFlag, baseIndex);
	unlock();
	return ret;
}


void DataPool::getTitle(size_t index, string& outTitle) {
	const int mapBatchWidth = 128;
		size_t batchIdx = index / mapBatchWidth;	
		size_t remIdx = index % mapBatchWidth;
		++batchIdx;//there is a seqNum at the head of map file
		++remIdx;
		size_t titlePos = -1;
		mapFileStream.seekg(batchIdx * sizeof(size_t));
		mapFileStream.read(reinterpret_cast<char*> (&titlePos), sizeof(size_t));
		string title("");	
		titleFileStream.seekg(titlePos);
		for(size_t j = 0; j != remIdx; ++j)		
			getline(titleFileStream, title);
		outTitle.append(title);
}


const char* DataPool::getTitle(size_t index){
		string out;
		getTitle(index, out);
		return out.c_str();
}

void DataPool::getQuery(const string& queryPath) {
	ifstream queryStream;
	queryStream.open(queryPath.c_str(), ios::in);
	char tmp;
	string line;
	do {
		queryStream >> tmp;
	} while (tmp != '>');

	queryName.push_back('>');
	getline(queryStream, line);
	queryName.append(line);
	while (queryStream.peek() != '<' && queryStream.peek() != EOF) {
		getline(queryStream, line);
		queryResidue.append(line);
	}

	
	int pad = (querySeqAlignedLength
			- queryResidue.size() % querySeqAlignedLength)
			% querySeqAlignedLength;

	for (int i = 0; i != pad; ++i) {
		queryResidue.push_back('@');
	}

	//translate characters into integers
	for (size_t i = 0; i != queryResidue.size(); ++i) {
		char tmpTrans;
		tmpTrans = amino_acids_trans[(int)queryResidue[i]];
		queryResidue[i] = tmpTrans;
	}
}

#if 0
// posix manner to check file
inline bool DataPool:fileExist(const std::string& name) {
	stat buffer;
	return (stat (name.c_str(), &buffer) == 0);
}
#else
bool DataPool::fileExist(const std::string& name) {
	ifstream f(name.c_str());
	if (f.good()) {
		f.close();
		return true;
	} else {
		f.close();
		return false;
	}
}
#endif

bool DataPool::checkFiles() {
	char c;
	if (fileExist(seqPath) && fileExist(mapPath) && fileExist(mapPath)) {
		//cout << "processed database file exists, Run!" << endl;
		return true;
	} else {
		cout << "cannot find the files, make database?(type in 'y' or 'n')";
		while (cin.get(c)) {
			if (c == 'y' || c == 'Y') {
				cout << "Making database...Please wait" << endl;
//				openWrite();
//				makeDatabase();
//				closeFiles();
				return true;
			} else if (c == 'n' || c == 'Y') {
				cout << "aborted, cannot find datafiles" << endl;
				break;
			}
		}

	}
	return false;
}
