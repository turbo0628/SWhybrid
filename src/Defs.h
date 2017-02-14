/*
 * Defs.h
 *
 *  Created on: 2014-7-3
 *      Author: lan
 */

#ifndef DEFS_H_
#define DEFS_H_

#include <string>

const int alphabetSize = 32;
const int maxReaderBufferSize = 1 << 28;
const int BATCH_WIDTH = 128;//max batch width for dbmaker

struct Overflow{
	Overflow(int channel, int batchIdx);
	int batchIdx;
	int channel;
	std::string sequence;
};

struct batchInfo{
	batchInfo():numSeqs(0), seqLen(0){}
	batchInfo(unsigned numSeqs, unsigned seqLen):numSeqs(numSeqs), seqLen(seqLen){}
	unsigned numSeqs;
	unsigned seqLen;
};

#endif /* DEFS_H_ */
