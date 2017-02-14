#ifndef KNCWORKER_H_
#define KNCWORKER_H_
#include "Worker.h"

#include "RecalcWorker.h"


class KNCWorker : public Worker
{
	public:
		KNCWorker(DataPool* loader, Param* params, RecalcWorker* recalcWorker, int id);
		virtual ~KNCWorker();

		void who(){
			printf("KNC %d\n", id);}

		void setQuery(const char* query, const size_t qlen);

	private:
		void search();
		void alloc();
		void free();
		void asyncCopy();
		void showPerformance();

	private:
		int mapSize;

	private:

		int* score[2];

		int id;

		//for score profile
		int*  _g_profile;
		int** _g_qtable;

		int plainMatrix[32 * 32];

};

#endif /* MICWORKER_H_ */
