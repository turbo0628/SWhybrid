/*
 * FastaFile.cpp
 *
 *  Created on: May 20, 2014
 *      Author: lan
 */

#include "FastaFile.h"
#include "Bucket.h"
#include <vector>
#include <string>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <sys/timeb.h>

using namespace std;

const int databaseSeqAligned = 4;
const int flushOutThreshold = 1 << 28; // 2^30 = 1GB, 2^28 = 256MB
const int readerBufferSize = 1 << 28;
//const int batchWidth = 16;
const size_t batchWidth = BATCH_WIDTH;
const int titleSearchGranularity = 50; //shall be adjustable

//timeb start, stop;
double totalTimeInVector = 0;

const char amino_acids_trans[256] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 9, 10, 11, 12, -1, 13, 14, 15, 16,
	17, -1, 18, 19, 20, 21, 22, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 9, 10,
	11, 12, -1, 13, 14, 15, 16, 17, -1, 18, 19, 20, 21, 22, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1 };


FastaFile::FastaFile(string filePath, string outputPath):
	staticBuckets(NULL),
	lenThreshold(3072),
	padding(4)
{
	totalNum = 0;
	dbFileStream.sync_with_stdio(false);
	titleFileStream.sync_with_stdio(false);
	seqFileStream.sync_with_stdio(false);

	setPath(filePath, outputPath);
}

void FastaFile::setPath(const string& baseFilePath, const string& outputFilePath) {
	dbFilePath = baseFilePath;

	titlePath = outputFilePath + ".title";
	seqPath = outputFilePath + ".seq";
	mapPath = outputFilePath + ".map";
}



FastaFile::~FastaFile() {
	closeFiles();
}

void FastaFile::openWrite() {
	dbFileStream.open(dbFilePath.c_str(), ios::in);
	titleFileStream.open(titlePath.c_str(), ios::trunc|ios::out);
	seqFileStream.open(seqPath.c_str(), ios::trunc|ios::out);
	mapFileStream.open(mapPath.c_str(), ios::trunc|ios::out);

	const size_t zero = 0;
	seqFileStream.write(reinterpret_cast<const char*>(&zero),sizeof(size_t));
	seqFileStream.write(reinterpret_cast<const char*>(&zero),sizeof(size_t));
	mapFileStream.write(reinterpret_cast<const char*>(&zero), sizeof(size_t));//total sequence number
	//mapFileStream.write(reinterpret_cast<const char*>(&titleSearchGranularity), sizeof(size_t));
}



void FastaFile::closeFiles() {
	printf("closing files\n");
	if (dbFileStream.is_open())
		dbFileStream.close();
	if (titleFileStream.is_open())
		titleFileStream.close();
	if (seqFileStream.is_open())
		seqFileStream.close();
	if (mapFileStream.is_open())
		mapFileStream.close();
	printf("file closed\n");
}



void FastaFile::run() {
	openWrite();
	makeDatabase();
}

size_t FastaFile::nextSeq(int pad) {
	int padLen;
	string tmpBuf;
	seqBuf.clear();
	titleBuf.clear();
	char next;

	if (dbFileStream.peek() == EOF) {
		return 0;
	} else {
		if (getline(dbFileStream, tmpBuf) == 0) {
			throw("bad format");
		}
		if (tmpBuf[0] == '>') {
			titleBuf.append(tmpBuf.begin(), tmpBuf.end()); 
			titleBuf.push_back('\n');
			do {
				tmpBuf.clear();
				getline(dbFileStream, tmpBuf);
				seqBuf.append(tmpBuf.begin(), tmpBuf.end() );
				next = dbFileStream.peek();
			} while (next != '>' && next != EOF);

			padLen = pad - seqBuf.size() % pad;
			padLen %= pad;
			for (int i = 0; i != padLen; ++i) {
				seqBuf.push_back('@');
			}
			for(int i = 0; i != seqBuf.size(); ++i){
				char ss = amino_acids_trans[seqBuf[i]];
				if(ss > 23)
					ss = 23;
				seqBuf[i] = ss;
			}
			return seqBuf.size();
		} else {
			throw("fasta format error in reading file");
		}
	}
}

inline double gettime(timeb start, timeb stop){
	return ((stop.time - start.time) * 1000 + (stop.millitm - start.millitm));
}

int FastaFile::vecContainLength(size_t len) {
	//ftime(&start);
	for (size_t i = 0; i != buckets.size(); ++i) {
		if (buckets[i].getLen() == len) {
		//	ftime(&stop);
			//totalTimeInVector += gettime();
			return i;
		}
	}
	//ftime(&stop);
	//totalTimeInVector += gettime();
	return -1;
}

void FastaFile::pushSeq(size_t seqLen, string& inSeq, string& inTitle){
	map<size_t, size_t>::iterator tableIt;
	vector<Bucket>::iterator 			bucketIt;
	tableIt = lookUpTable.find(seqLen);
	if(tableIt != lookUpTable.end())
	{
		size_t pos = tableIt->second;
		bucketIt = buckets.begin() + pos;
		bucketIt->appendSeq(seqBuf, titleBuf);
	}else{
		//didn't find the bucket, add a new one
		lookUpTable.insert(pair<size_t, size_t>(seqLen, buckets.size()));
		buckets.push_back(Bucket(seqLen, inSeq, inTitle));
	}
}

int FastaFile::makeDatabase() {
	int bucketIndex; //The number of containers are considered not to exceed int32, otherwise it would be horrible
	size_t seqIdx = 0;
	size_t seqLen = 0;
	size_t inMemoryResidueNum = 0;
	size_t flushIndex = 0;
	while (1) {
		seqLen = nextSeq(databaseSeqAligned);
		if (seqLen > 0) {
			pushSeq(seqLen, seqBuf, titleBuf);
			++seqIdx;
			inMemoryResidueNum += seqLen;
			if ((inMemoryResidueNum + titleBuf.size()) * sizeof(char)
					>= flushOutThreshold) {
				printf("flushing out chunk %ld of size %ldMB\n", flushIndex++, inMemoryResidueNum>>20);
				flushOutBatch();
				inMemoryResidueNum = 0;
			}
		} else
			break;
	}
	totalNum = seqIdx;
	printf("flushing out chunk %ld of size %ldMB\n", flushIndex++, inMemoryResidueNum>>20);
	flushOutBatch();
	flushOutRemain();
	mapFileStream.seekp(0);
	mapFileStream.write(reinterpret_cast<const char*>(&seqIdx), sizeof(size_t));
	printf("total sequence num is %ld %lx\n", totalNum, totalNum);
	return 0;
}

void FastaFile::writeBatch(size_t seqNum, size_t seqLen, string& seqLine, string& titleLine, vector<size_t>& vecMap){
	writeBatch(seqNum, seqLen, seqLine, titleLine, vecMap.data(), vecMap.size());
	seqLine.clear();
	titleLine.clear();
	vecMap.clear();
}

void FastaFile::writeBatch(size_t seqNum, size_t seqLen, std::string& seqLine, std::string& titleLine, size_t* vecMap, const size_t vecMapSize){
	seqFileStream.put('$');
	seqFileStream.write(reinterpret_cast<const char*>(&seqNum),sizeof(size_t));
	seqFileStream.write(reinterpret_cast<const char*>(&seqLen),sizeof(size_t));
	seqFileStream << '#';
	size_t base = titleFileStream.tellp();
	for(int i = 0; i != vecMapSize; ++i){
		vecMap[i] += base;
	}
	if(seqLine.size() > 0 && titleLine.size() > 0){
		seqFileStream << seqLine;
		titleFileStream << titleLine;
		mapFileStream.write(reinterpret_cast<const char*>(vecMap), sizeof(size_t)*vecMapSize);
	}else{
		throw("unexpected empty bucket");
	}
	//seqLine.clear();
	//titleLine.clear();
	//vecMap.clear();	
	seqFileStream.flush();
	mapFileStream.flush();
	titleFileStream.flush();
}

void FastaFile::flushOutBatch() {
	vector<Bucket>::iterator iter;
	string seqBackup;
	string seqLine;
	string titleLine;
	vector<size_t> vecMap;
	//double writeTime = 0;
	for (iter = buckets.begin(); iter != buckets.end(); ++iter) {
		size_t seqNum = iter->size();
		size_t seqLen = iter->getLen();
		seqNum = seqNum - seqNum % batchWidth;
		if(seqNum > 0){
			iter->flushOut(seqLine, titleLine, vecMap);
			writeBatch(seqNum, seqLen, seqLine, titleLine, vecMap);
		}
	} //for each bucket
}

/*
 * TODO: return pos of seq file and append into map table
 */

bool bucketLess(Bucket i, Bucket j)
{
	return i.getLen() < j.getLen();
}


void FastaFile::flushOutRemain() {
	vector<Bucket>::iterator iter;
	vector<Bucket>::iterator iter_i;
	vector<Bucket>::iterator iter_j;
	const size_t batchWidth_long = batchWidth;
	size_t remainPos = 0;
	size_t remainNum = 0;
	size_t remainRes = 0;

	/*sort buckets by length*/
	sort(buckets.begin(), buckets.end(), bucketLess);

	iter = buckets.begin();
	size_t num = 0;
	while(iter < buckets.end())
	{	
		num = 0;
		for(iter_j = iter; iter_j < buckets.end() && num < batchWidth; ++iter_j){
			while(iter_j->size() == 0 && iter_j != buckets.end()){
				++iter_j;
			}
			if(iter_j == buckets.end())
				break;	
			num += iter_j->size();
			if(num >= batchWidth){
				string seqLine;
				string titleLine;
				vector<size_t> vecMap;
				size_t len = iter_j->getLen();
				size_t rem = num % batchWidth;
				iter_j->flushOut(0, iter_j->size() - rem, seqLine, titleLine);
				num = rem;
				for(iter_i = iter; iter_i < iter_j; ++iter_i)
				{
					if(iter_i->size() == 0)
						continue;
					size_t cur_len = iter_i->getLen();
					size_t pads = len - cur_len;
					iter_i->flushOut(pads, iter_i->size(), seqLine, titleLine);
				}//if 

				/*write batch*/
				size_t zero = 0;
				vecMap.push_back(zero);
				writeBatch(batchWidth, len, seqLine, titleLine, vecMap);
			}
		}
		iter = iter_j;
	}//while

	cout<<"last batch num "<<num<<endl;

	//flush out the last batch
	if(num > 0)
	{
		string seqLine;
		string titleLine;
		vector<size_t> vecMap;
		vector<Bucket>::reverse_iterator riter;
		riter = buckets.rbegin();
		while(riter->size() == 0)
			++riter;
		size_t len = riter->getLen();
		int dummyNum = batchWidth - num;	
		int cnt = 0;
		while(cnt < num)
		{
			size_t curLen = riter->getLen();
			size_t pads = len - curLen;
			cnt += riter->size();
			riter->flushOut(pads, riter->size(), seqLine, titleLine);
			++riter;
		}
		//string dummySeq(len * dummyNum, '@');
		string dummySeq(len * dummyNum, 23);
		seqLine.append(dummySeq.begin(), dummySeq.end());
		size_t zero = 0;
		vecMap.push_back(zero);
		writeBatch(batchWidth, len, seqLine, titleLine, vecMap);
	}
	remainPos = seqFileStream.tellp();
	seqFileStream.seekp(SEEK_SET);
	seqFileStream.write(reinterpret_cast<const char*>(&remainPos),
			sizeof(size_t));
	seqFileStream.write(reinterpret_cast<const char*>(&remainNum),
			sizeof(size_t));
	//printf("pos %lx, num %lx, remain size is  %lfMB\n", remainPos, remainNum, remainRes / 1024.0 / 1024.0);
}
