/*
 * FastaFile.h
 *
 * Provide a method processing initial Fasta format and a dual-buffer reader for computing
 *
 *  Created on: May 20, 2014
 *      Author: lan
 */

#ifndef BUCKET_H_
#define BUCKET_H_

#include <string>
#include <fstream>
#include <vector>
#include "Defs.h"

class Bucket
{
	public:
		Bucket(size_t len);
		Bucket(size_t len, std::string& seq, std::string& title);
		~Bucket();
		void appendSeq(std::string& seq, std::string& title);
		void getTitle(size_t index, size_t& outPos, size_t& outLen);
		void flushOut(std::string& outSeq, std::string& outTitle, std::vector<size_t>& outMap);
		void flushOut(size_t flushNum, std::string& outSeq, std::string& outTitle, std::vector<size_t>& outMap);

		void flushOut(size_t flushNum, std::string& outSeq, std::string& outTitle);
		void flushOut(size_t pads, size_t flushNum, std::string& outSeq, std::string& outTitle);
		
		size_t size(){return sz;}
		size_t getLen(){return len;}

		void printBucket();
		
	private:
		std::string 				seq;
		std::string					title;
		std::vector<size_t> map;
		size_t			sz;
		size_t 			len;
};

#endif /* BUCKET_H_ */
