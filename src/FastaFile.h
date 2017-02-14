/*
 * FastaFile.h
 *
 * Provide a method processing initial Fasta format and a dual-buffer reader for computing
 *
 *  Created on: May 20, 2014
 *      Author: lan
 */

#ifndef FASTAFILE_H_
#define FASTAFILE_H_


#include "Bucket.h"
#include "Defs.h"

#include <string>
#include <fstream>
#include <vector>
#include <map>

class FastaFile {
public:
	FastaFile(std::string filePath, std::string outputPath);
	~FastaFile();

	void setPath(const std::string& baseFilePath, const std::string& outputFilePath);
	void run();//only for development
private:

	//for basic file operations
	void openWrite();
	void closeFiles();

	std::string dbFilePath;
	std::string seqPath;
	std::string titlePath;
	std::string mapPath;

	std::ifstream dbFileStream;
	std::fstream titleFileStream;
	std::fstream seqFileStream;
	std::fstream mapFileStream;

	/*
	 * rem file:
	 * header: first index|sequence numbers|residue numbers
	 * sequence: length|sequence
	 */
	std::fstream remFileStream;

	/*
	 * batch file:
	 * header: sequence numbers|residue numbers|batch width
	 * sequence: length|batch sequence
	 */
	std::fstream batchFileStream;
private:
	//for pre-process database operations
	int makeDatabase();
	size_t nextSeq(int pad);
	int vecContainLength(size_t len);
	void pushSeq(size_t seqLen, std::string& inSeq, std::string& inTitle);
	void writeBatch(size_t seqNum, size_t seqLen, std::string& seqLine, std::string& titleLine, std::vector<size_t>& vecMap);
	void writeBatch(size_t seqNum, size_t seqLen, std::string& seqLine, std::string& titleLine, size_t* vecMap, size_t vecMapSize);
	void flushOutBatch();
	void flushOutRemain();

	size_t 				totalNum;
	std::string 			titleBuf;
	std::string 			seqBuf;
	std::vector<Bucket> 		buckets;
	std::map<size_t, size_t> 	lookUpTable;
	const size_t 			padding;
	unsigned char*			staticBuckets;
	int*				bucketIndices;
	const int 			lenThreshold;

};

#endif /* FASTAFILE_H_ */
