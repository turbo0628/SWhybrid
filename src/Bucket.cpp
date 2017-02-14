/*
 * Bucket.cpp
 *
 *  Created on: Dec 19, 2014
 *      Author: lan
 */

#include "Bucket.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

const size_t batchWidth = BATCH_WIDTH;

Bucket::Bucket(size_t len):sz(0), len(len){

}
Bucket::Bucket(size_t len, std::string& seq, std::string& title):sz(0), len(len)
{
	appendSeq(seq, title);
}
Bucket::~Bucket(){

}

void Bucket::appendSeq(string& inSeq, string& inTitle)
{
	if(sz % batchWidth == 0){
		map.push_back(title.size());
	}
	seq.append(inSeq.begin(), inSeq.end());	
	if(inTitle[inTitle.size() - 1] != '\n'){
		throw("illegal title\n");
	}
	title.append(inTitle.begin(), inTitle.end());
	++sz;
}

void Bucket::getTitle(size_t index, size_t& outPos, size_t& outLen)
{
	if(index >= this->sz){
		throw("Bucket: index out of range\n");
	}
	size_t pos = 0;
	//i is the index begins from zero
	for(int i = 0; i <= index; ++i)
	{
		outPos = pos;
		pos = this->title.find('\n', pos);
		++pos;
	}
	outLen = pos - outPos;
}

void Bucket::flushOut(string& outSeq, string& outTitle, vector<size_t>& outMap)
{
	//outSeq.append(seq.begin(), seq.end());
	//outTitle.append(title.begin(), title.end());
	//std::copy(map.begin(), map.end(), outMap.begin());
	//outMap.insert(outMap.end(), map.begin(), map.end());
	//map.clear();
	if(sz == 0)
		return;
	size_t rem = sz % batchWidth;
	size_t flushNum = sz - rem;
	flushOut(flushNum, outSeq, outTitle, outMap);
}

void Bucket::flushOut(size_t flushNum, string& outSeq, string& outTitle){
	if(flushNum > sz || flushNum <= 0)
	{
		printBucket();
		cout<<"size "<<sz<<" "<<flushNum<<endl;
		throw("Flush Out Error: Illegal flush number request in Bucket::flushOut\n");
	}

	//find ending position of the last title to be flushed.
#if 0
	size_t pos = 0;
	for(int i = 0; i != flushNum; ++i)
	{
		pos = title.find('\n', pos);
		++pos;
	}
#else
	size_t pos = title.size();
	size_t rem = sz - flushNum + 1;
	for(int i = 0; i != rem; ++i)
	{
		--pos;
		pos = title.rfind('\n', pos);
	}
	++pos;
#endif
	//cout<<"~~~~~~~~~~pos "<<pos<<" "<<title[pos + 1]<<"~~~~~~~~~~~~~"<<endl;
	outSeq.append(seq.begin(), seq.begin() + len * flushNum);
	outTitle.append(title.begin(), title.begin() + pos);
	sz -= flushNum;
	seq = string(seq.begin() + len * flushNum, seq.end());
	title = string(title.begin() + pos, title.end());	
}

void Bucket::flushOut(size_t flushNum, string& outSeq, string& outTitle, vector<size_t>& outMap)
{
	flushOut(flushNum, outSeq, outTitle);
	outMap.insert(outMap.end(), map.begin(), map.begin() + flushNum / batchWidth);
	//rebuilt map
	map.clear();
	size_t pos = 0;
	while(pos < title.size())
	{
		map.push_back(pos);
		for(int i = 0; i != batchWidth && pos < title.size(); ++i){
			pos = title.find('\n', pos);
			++pos;
		}
	}
}

void Bucket::flushOut(size_t pads, size_t flushNum, std::string& outSeq, std::string& outTitle){
	flushOut(flushNum, outSeq, outTitle);
	char dummy = 23;
	for(int i = 0; i != flushNum; ++i)
	{
		outSeq.insert( (i + 1) * (len + pads) - pads, pads, dummy);
	}	
}
void Bucket::printBucket()
{
	/*
	cout<<title;
	cout<<seq<<endl;
	copy(map.begin(), map.end(), ostream_iterator<size_t>(cout, ", "));
	cout<<endl;
	for(int i = 0; i != map.size(); ++i)
	{
		cout<<title[map[i]]<<endl ;
	}
	cout<<"size of the bucket "<<sz<<endl;*/
}
