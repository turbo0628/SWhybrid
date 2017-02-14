#ifndef RESULT_LIST_H_
#define RESULT_LIST_H_

#include <list>
#include <iostream>

template<class T>
class Result
{
	public:
		Result():score(0), idx(0){}
		Result(T score, size_t idx):score(score), idx(idx){}
		~Result(){}
		T score;
		size_t 	 idx;
};

template<class T>
class ResultList
{
	public:
		ResultList(size_t queueSize):
			_size(queueSize), 
			_bottom(0),
			_results()
		{
			_results.resize(_size);
		}


		~ResultList(){}

		std::list<Result<T> >& getResult(){
			return _results;
		}

		inline void push(T* inScores, size_t sz, size_t baseIdx){
			for(int i = 0; i != sz; ++i)
			{
				if(inScores[i] <= _bottom)
					continue;
				push(inScores[i], baseIdx + i);
			}
		}//push0

		void mergeList(ResultList<T>& newList){
			std::list<Result<T> > newListResult = newList.getResult();
			typename std::list<Result<T> >::iterator it;
			for(it = newListResult.begin(); it != newListResult.end(); ++it)
			{	
				push(it->score, it->idx);
			}
		}//push1

		void push(T inScore, size_t idx){
			if(inScore	<= _bottom)
				return;
			if(_results.size() == 0){
				_results.push_back(Result<T>(inScore, idx));
				return;
			}

			typename std::list<Result<T> >::iterator iter;
			for(iter = _results.begin(); iter != _results.end(); ++iter)
			{
				if(iter->score < inScore)
				{
					_results.insert(iter, Result<T>(inScore, idx));
					break;
				}
			}

			//if(_results.size() <= _size){
			//	_bottom = 0;
			//}else{
				_results.pop_back();
				_bottom = _results.rbegin()->score;
			//}
		}//_push

		void print(){
			typename std::list<Result<T> >::iterator it;
			for(it = _results.begin(); it != _results.end(); ++it)
			{
				printf("score %d id %ld\n", it->score, it->idx);
			}
		}
	private:
			size_t 							_size;
			T 								_bottom;
			std::list<Result<T> >           _results;
};
#endif
