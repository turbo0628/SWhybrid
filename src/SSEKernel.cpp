/*for SSE2 implementations*/
#ifdef HAVE_SSSE3
#include <tmmintrin.h>
#else
#include <emmintrin.h>
#endif

#include <cstdio>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <sys/timeb.h>
#include <vector>
#include <unistd.h>
#include <omp.h>
#include <unistd.h>
#include <pthread.h>

#include "SSEKernel.h"

#define N16_CHANNELS 	16	/*16 parallel lanes*/
#define N8_CHANNELS		8	/*8 parallel lanes*/
#define CDEPTH 			4

typedef char BYTE;
typedef short WORD;

static pthread_mutex_t idxMutex;
static pthread_mutex_t cntMutex;

	inline void dprofile_fill7(BYTE * dprofile, BYTE * score_matrix,
			BYTE * dseq)
	{
		__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
		__m128i xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

		// 4 x 16 db symbols
		// ca 188 x 4 = 752 instructions

		for (int j = 0; j < CDEPTH; j++)
		{
			int d[N16_CHANNELS];
			for (int i = 0; i < N16_CHANNELS; i++)
				d[i] = dseq[j * N16_CHANNELS + i] << 5;

			xmm0 = _mm_loadl_epi64((__m128i *) (score_matrix + d[0]));
			xmm2 = _mm_loadl_epi64((__m128i *) (score_matrix + d[2]));
			xmm4 = _mm_loadl_epi64((__m128i *) (score_matrix + d[4]));
			xmm6 = _mm_loadl_epi64((__m128i *) (score_matrix + d[6]));
			xmm8 = _mm_loadl_epi64((__m128i *) (score_matrix + d[8]));
			xmm10 = _mm_loadl_epi64((__m128i *) (score_matrix + d[10]));
			xmm12 = _mm_loadl_epi64((__m128i *) (score_matrix + d[12]));
			xmm14 = _mm_loadl_epi64((__m128i *) (score_matrix + d[14]));

			xmm0 = _mm_unpacklo_epi8(xmm0, *(__m128i *) (score_matrix + d[1]));
			xmm2 = _mm_unpacklo_epi8(xmm2, *(__m128i *) (score_matrix + d[3]));
			xmm4 = _mm_unpacklo_epi8(xmm4, *(__m128i *) (score_matrix + d[5]));
			xmm6 = _mm_unpacklo_epi8(xmm6, *(__m128i *) (score_matrix + d[7]));
			xmm8 = _mm_unpacklo_epi8(xmm8, *(__m128i *) (score_matrix + d[9]));
			xmm10 = _mm_unpacklo_epi8(xmm10,
					*(__m128i *) (score_matrix + d[11]));
			xmm12 = _mm_unpacklo_epi8(xmm12,
					*(__m128i *) (score_matrix + d[13]));
			xmm14 = _mm_unpacklo_epi8(xmm14,
					*(__m128i *) (score_matrix + d[15]));

			xmm1 = xmm0;
			xmm0 = _mm_unpacklo_epi16(xmm0, xmm2);
			xmm1 = _mm_unpackhi_epi16(xmm1, xmm2);
			xmm5 = xmm4;
			xmm4 = _mm_unpacklo_epi16(xmm4, xmm6);
			xmm5 = _mm_unpackhi_epi16(xmm5, xmm6);
			xmm9 = xmm8;
			xmm8 = _mm_unpacklo_epi16(xmm8, xmm10);
			xmm9 = _mm_unpackhi_epi16(xmm9, xmm10);
			xmm13 = xmm12;
			xmm12 = _mm_unpacklo_epi16(xmm12, xmm14);
			xmm13 = _mm_unpackhi_epi16(xmm13, xmm14);

			xmm2 = xmm0;
			xmm0 = _mm_unpacklo_epi32(xmm0, xmm4);
			xmm2 = _mm_unpackhi_epi32(xmm2, xmm4);
			xmm6 = xmm1;
			xmm1 = _mm_unpacklo_epi32(xmm1, xmm5);
			xmm6 = _mm_unpackhi_epi32(xmm6, xmm5);
			xmm10 = xmm8;
			xmm8 = _mm_unpacklo_epi32(xmm8, xmm12);
			xmm10 = _mm_unpackhi_epi32(xmm10, xmm12);
			xmm14 = xmm9;
			xmm9 = _mm_unpacklo_epi32(xmm9, xmm13);
			xmm14 = _mm_unpackhi_epi32(xmm14, xmm13);

			xmm3 = xmm0;
			xmm0 = _mm_unpacklo_epi64(xmm0, xmm8);
			xmm3 = _mm_unpackhi_epi64(xmm3, xmm8);
			xmm7 = xmm2;
			xmm2 = _mm_unpacklo_epi64(xmm2, xmm10);
			xmm7 = _mm_unpackhi_epi64(xmm7, xmm10);
			xmm11 = xmm1;
			xmm1 = _mm_unpacklo_epi64(xmm1, xmm9);
			xmm11 = _mm_unpackhi_epi64(xmm11, xmm9);
			xmm15 = xmm6;
			xmm6 = _mm_unpacklo_epi64(xmm6, xmm14);
			xmm15 = _mm_unpackhi_epi64(xmm15, xmm14);

			_mm_store_si128((__m128i *) (dprofile + 16 * j + 0), xmm0);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 64), xmm3);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 128), xmm2);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 192), xmm7);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 256), xmm1);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 320), xmm11);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 384), xmm6);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 448), xmm15);

			xmm0 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[0]));
			xmm1 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[1]));
			xmm2 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[2]));
			xmm3 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[3]));
			xmm4 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[4]));
			xmm5 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[5]));
			xmm6 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[6]));
			xmm7 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[7]));
			xmm8 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[8]));
			xmm9 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[9]));
			xmm10 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[10]));
			xmm11 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[11]));
			xmm12 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[12]));
			xmm13 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[13]));
			xmm14 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[14]));
			xmm15 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[15]));

			xmm0 = _mm_unpacklo_epi8(xmm0, xmm1);
			xmm2 = _mm_unpacklo_epi8(xmm2, xmm3);
			xmm4 = _mm_unpacklo_epi8(xmm4, xmm5);
			xmm6 = _mm_unpacklo_epi8(xmm6, xmm7);
			xmm8 = _mm_unpacklo_epi8(xmm8, xmm9);
			xmm10 = _mm_unpacklo_epi8(xmm10, xmm11);
			xmm12 = _mm_unpacklo_epi8(xmm12, xmm13);
			xmm14 = _mm_unpacklo_epi8(xmm14, xmm15);

			xmm1 = xmm0;
			xmm0 = _mm_unpacklo_epi16(xmm0, xmm2);
			xmm1 = _mm_unpackhi_epi16(xmm1, xmm2);
			xmm5 = xmm4;
			xmm4 = _mm_unpacklo_epi16(xmm4, xmm6);
			xmm5 = _mm_unpackhi_epi16(xmm5, xmm6);
			xmm9 = xmm8;
			xmm8 = _mm_unpacklo_epi16(xmm8, xmm10);
			xmm9 = _mm_unpackhi_epi16(xmm9, xmm10);
			xmm13 = xmm12;
			xmm12 = _mm_unpacklo_epi16(xmm12, xmm14);
			xmm13 = _mm_unpackhi_epi16(xmm13, xmm14);

			xmm2 = xmm0;
			xmm0 = _mm_unpacklo_epi32(xmm0, xmm4);
			xmm2 = _mm_unpackhi_epi32(xmm2, xmm4);
			xmm6 = xmm1;
			xmm1 = _mm_unpacklo_epi32(xmm1, xmm5);
			xmm6 = _mm_unpackhi_epi32(xmm6, xmm5);
			xmm10 = xmm8;
			xmm8 = _mm_unpacklo_epi32(xmm8, xmm12);
			xmm10 = _mm_unpackhi_epi32(xmm10, xmm12);
			xmm14 = xmm9;
			xmm9 = _mm_unpacklo_epi32(xmm9, xmm13);
			xmm14 = _mm_unpackhi_epi32(xmm14, xmm13);

			xmm3 = xmm0;
			xmm0 = _mm_unpacklo_epi64(xmm0, xmm8);
			xmm3 = _mm_unpackhi_epi64(xmm3, xmm8);
			xmm7 = xmm2;
			xmm2 = _mm_unpacklo_epi64(xmm2, xmm10);
			xmm7 = _mm_unpackhi_epi64(xmm7, xmm10);
			xmm11 = xmm1;
			xmm1 = _mm_unpacklo_epi64(xmm1, xmm9);
			xmm11 = _mm_unpackhi_epi64(xmm11, xmm9);
			xmm15 = xmm6;
			xmm6 = _mm_unpacklo_epi64(xmm6, xmm14);
			xmm15 = _mm_unpackhi_epi64(xmm15, xmm14);

			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 0), xmm0);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 64), xmm3);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 128), xmm2);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 192), xmm7);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 256), xmm1);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 320), xmm11);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 384), xmm6);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 448), xmm15);

			xmm0 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[0]));
			xmm2 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[2]));
			xmm4 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[4]));
			xmm6 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[6]));
			xmm8 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[8]));
			xmm10 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[10]));
			xmm12 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[12]));
			xmm14 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[14]));

			xmm0 = _mm_unpacklo_epi8(xmm0,
					*(__m128i *) (score_matrix + 16 + d[1]));
			xmm2 = _mm_unpacklo_epi8(xmm2,
					*(__m128i *) (score_matrix + 16 + d[3]));
			xmm4 = _mm_unpacklo_epi8(xmm4,
					*(__m128i *) (score_matrix + 16 + d[5]));
			xmm6 = _mm_unpacklo_epi8(xmm6,
					*(__m128i *) (score_matrix + 16 + d[7]));
			xmm8 = _mm_unpacklo_epi8(xmm8,
					*(__m128i *) (score_matrix + 16 + d[9]));
			xmm10 = _mm_unpacklo_epi8(xmm10,
					*(__m128i *) (score_matrix + 16 + d[11]));
			xmm12 = _mm_unpacklo_epi8(xmm12,
					*(__m128i *) (score_matrix + 16 + d[13]));
			xmm14 = _mm_unpacklo_epi8(xmm14,
					*(__m128i *) (score_matrix + 16 + d[15]));

			xmm1 = xmm0;
			xmm0 = _mm_unpacklo_epi16(xmm0, xmm2);
			xmm1 = _mm_unpackhi_epi16(xmm1, xmm2);
			xmm5 = xmm4;
			xmm4 = _mm_unpacklo_epi16(xmm4, xmm6);
			xmm5 = _mm_unpackhi_epi16(xmm5, xmm6);
			xmm9 = xmm8;
			xmm8 = _mm_unpacklo_epi16(xmm8, xmm10);
			xmm9 = _mm_unpackhi_epi16(xmm9, xmm10);
			xmm13 = xmm12;
			xmm12 = _mm_unpacklo_epi16(xmm12, xmm14);
			xmm13 = _mm_unpackhi_epi16(xmm13, xmm14);

			xmm2 = xmm0;
			xmm0 = _mm_unpacklo_epi32(xmm0, xmm4);
			xmm2 = _mm_unpackhi_epi32(xmm2, xmm4);
			xmm6 = xmm1;
			xmm1 = _mm_unpacklo_epi32(xmm1, xmm5);
			xmm6 = _mm_unpackhi_epi32(xmm6, xmm5);
			xmm10 = xmm8;
			xmm8 = _mm_unpacklo_epi32(xmm8, xmm12);
			xmm10 = _mm_unpackhi_epi32(xmm10, xmm12);
			xmm14 = xmm9;
			xmm9 = _mm_unpacklo_epi32(xmm9, xmm13);
			xmm14 = _mm_unpackhi_epi32(xmm14, xmm13);

			xmm3 = xmm0;
			xmm0 = _mm_unpacklo_epi64(xmm0, xmm8);
			xmm3 = _mm_unpackhi_epi64(xmm3, xmm8);
			xmm7 = xmm2;
			xmm2 = _mm_unpacklo_epi64(xmm2, xmm10);
			xmm7 = _mm_unpackhi_epi64(xmm7, xmm10);
			xmm11 = xmm1;
			xmm1 = _mm_unpacklo_epi64(xmm1, xmm9);
			xmm11 = _mm_unpackhi_epi64(xmm11, xmm9);
			xmm15 = xmm6;
			xmm6 = _mm_unpacklo_epi64(xmm6, xmm14);
			xmm15 = _mm_unpackhi_epi64(xmm15, xmm14);

			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 0), xmm0);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 64), xmm3);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 128), xmm2);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 192), xmm7);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 256), xmm1);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 320),
					xmm11);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 384), xmm6);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 448),
					xmm15);
		}
	}
#ifdef HAVE_SSSE3
	inline void dprofile_shuffle7(BYTE * dprofile, BYTE * score_matrix,
			BYTE * dseq_byte)
	{
		__m128i a, b, c, d, x, y, m0, m1, m2, m3, m4, m5, m6, m7;
		__m128i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;
		__m128i u0, u1, u2, u3, u4, u5, u8, u9, u10, u11, u12, u13;
		__m128i * dseq = (__m128i *) dseq_byte;

		// 16 x 4 = 64 db symbols
		// ca 458 instructions

		// make masks

		x = _mm_set_epi8(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
				0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);

		y = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
				0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

		a = _mm_load_si128(dseq);
		t0 = _mm_and_si128(a, x);
		t1 = _mm_slli_epi16(t0, 3);
		t2 = _mm_xor_si128(t1, y);
		m0 = _mm_or_si128(a, t1);
		m1 = _mm_or_si128(a, t2);

		b = _mm_load_si128(dseq + 1);
		t3 = _mm_and_si128(b, x);
		t4 = _mm_slli_epi16(t3, 3);
		t5 = _mm_xor_si128(t4, y);
		m2 = _mm_or_si128(b, t4);
		m3 = _mm_or_si128(b, t5);

		c = _mm_load_si128(dseq + 2);
		u0 = _mm_and_si128(c, x);
		u1 = _mm_slli_epi16(u0, 3);
		u2 = _mm_xor_si128(u1, y);
		m4 = _mm_or_si128(c, u1);
		m5 = _mm_or_si128(c, u2);

		d = _mm_load_si128(dseq + 3);
		u3 = _mm_and_si128(d, x);
		u4 = _mm_slli_epi16(u3, 3);
		u5 = _mm_xor_si128(u4, y);
		m6 = _mm_or_si128(d, u4);
		m7 = _mm_or_si128(d, u5);

		/* Note: pshufb only on modern Intel cpus (SSSE3), not AMD */
		/* SSSE3: Supplemental SSE3 */

#define profline(j)                                   \
  t6  = _mm_load_si128((__m128i*)(score_matrix)+2*j);   \
  t7  = _mm_load_si128((__m128i*)(score_matrix)+2*j+1); \
  t8  = _mm_shuffle_epi8(t6, m0); \
  t9  = _mm_shuffle_epi8(t7, m1);  \
  t10 = _mm_shuffle_epi8(t6, m2); \
  t11 = _mm_shuffle_epi8(t7, m3); \
  u8  = _mm_shuffle_epi8(t6, m4); \
  u9  = _mm_shuffle_epi8(t7, m5);  \
  u10 = _mm_shuffle_epi8(t6, m6); \
  u11 = _mm_shuffle_epi8(t7, m7); \
  t12 = _mm_or_si128(t8,  t9); \
  t13 = _mm_or_si128(t10, t11); \
  u12 = _mm_or_si128(u8,  u9); \
  u13 = _mm_or_si128(u10, u11); \
  _mm_store_si128((__m128i*)(dprofile)+4*j,   t12); \
  _mm_store_si128((__m128i*)(dprofile)+4*j+1, t13); \
  _mm_store_si128((__m128i*)(dprofile)+4*j+2, u12); \
  _mm_store_si128((__m128i*)(dprofile)+4*j+3, u13)

		profline(0);
		profline(1);
		profline(2);
		profline(3);
		profline(4);
		profline(5);
		profline(6);
		profline(7);
		profline(8);
		profline(9);
		profline(10);
		profline(11);
		profline(12);
		profline(13);
		profline(14);
		profline(15);
		profline(16);
		profline(17);
		profline(18);
		profline(19);
		profline(20);
		profline(21);
		profline(22);
		profline(23);

		//  dprofile_dump7(dprofile);
	}
#else

#define dprofile_shuffle7(dprofile, score_matrix, dseq_byte) dprofile_fill7(dprofile, score_matrix, dseq_byte)

#endif	//HAVE_SSSE3


// Register usage
// rdi:   hep
// rsi:   qp
// rdx:   Qm
// rcx:   Rm
// r8:    ql
// r9:    Sm/Mm

// rax:   x, temp
// r10:   ql2
// r11:   qi
// xmm0:  H0
// xmm1:  H1
// xmm2:  H2
// xmm3:  H3
// xmm4:  F0
// xmm5:  F1
// xmm6:  F2
// xmm7:  F3
// xmm8:  N0
// xmm9:  N1
// xmm10: N2
// xmm11: N3
// xmm12: E
// xmm13: S
// xmm14: Q
// xmm15: R

#define INITIALIZE7					    \
                 "        movq      %0, rax             \n" \
		 "        movdqa    (rax), xmm13        \n" \
		 "        movdqa    (%3), xmm14         \n" \
		 "        movdqa    (%4), xmm15         \n" \
		 "        movq      %6, rax             \n" \
		 "        movdqa    (rax), xmm0         \n" \
		 "        movdqa    xmm0, xmm1          \n" \
		 "        movdqa    xmm0, xmm2          \n" \
		 "        movdqa    xmm0, xmm3          \n" \
		 "        movdqa    xmm0, xmm4          \n" \
		 "        movdqa    xmm0, xmm5          \n" \
		 "        movdqa    xmm0, xmm6          \n" \
		 "        movdqa    xmm0, xmm7          \n" \
		 "        movq      %5, r12             \n" \
		 "        shlq      $3, r12             \n" \
		 "        movq      r12, r10            \n" \
		 "        andq      $-16, r10           \n" \
		 "        xorq      r11, r11            \n"

#define ONESTEP7(H, N, F, V)	         		    \
                 "        paddsb    "V"(rax), "H"       \n" \
                 "        pmaxub    "F", "H"            \n" \
                 "        pmaxub    xmm12, "H"          \n" \
                 "        pmaxub    "H", xmm13          \n" \
		 "        psubsb    xmm15, "F"          \n" \
		 "        psubsb    xmm15, xmm12        \n" \
		 "        movdqa    "H", "N"            \n" \
		 "        psubsb    xmm14, "H"          \n" \
		 "        pmaxub    "H", xmm12          \n" \
		 "        pmaxub    "H", "F"            \n"

	inline void donormal7(__m128i * Sm, __m128i * hep, __m128i ** qp,
			__m128i * Qm, __m128i * Rm, long ql, __m128i * Zm)
{
#ifdef DEBUG
	printf("donormal\n");
	printf("Sm=%p\n", Sm);
	printf("hep=%p\n", hep);
	printf("qp=%p\n", qp);
	printf("Qm=%p\n", Qm);
	printf("Rm=%p\n", Rm);
	printf("qlen=%ld\n", ql);
	printf("Zm=%p\n", Zm);
#endif

	__asm__
		__volatile__(".att_syntax noprefix    # Change assembler syntax \n"
				INITIALIZE7
				"        jmp       2f                  \n"

				"1:      movq      0(%2,r11,1), rax    \n" // load x from qp[qi]
				"        movdqa    0(%1,r11,4), xmm8   \n"// load N0
				"        movdqa    16(%1,r11,4), xmm12 \n"// load E

				ONESTEP7("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP7("xmm1", "xmm10", "xmm5", "16")
				ONESTEP7("xmm2", "xmm11", "xmm6", "32")
				ONESTEP7("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4) \n"// save E
				"        movq      8(%2,r11,1), rax    \n"// load x from qp[qi+1]
				"        movdqa    32(%1,r11,4), xmm0  \n"// load H0
				"        movdqa    48(%1,r11,4), xmm12 \n"// load E

				ONESTEP7("xmm8", "xmm1", "xmm4", "0" )
				ONESTEP7("xmm9", "xmm2", "xmm5", "16")
				ONESTEP7("xmm10", "xmm3", "xmm6", "32")
				ONESTEP7("xmm11", "32(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 48(%1,r11,4) \n"// save E
				"        addq      $16, r11            \n"// qi++
				"2:      cmpq      r11, r10            \n"// qi = ql4 ?
				"        jne       1b                  \n"// loop

				"4:      cmpq      r11, r12            \n"
				"        je        3f                  \n"
				"        movq      0(%2,r11,1), rax    \n"// load x from qp[qi]
				"        movdqa    16(%1,r11,4), xmm12 \n"// load E

				ONESTEP7("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP7("xmm1", "xmm10", "xmm5", "16")
				ONESTEP7("xmm2", "xmm11", "xmm6", "32")
				ONESTEP7("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"3:      movq      %0, rax              \n"// save S
				"        movdqa    xmm13, (rax)         \n"
				"        .att_syntax prefix      # Change back to standard syntax"

				:
				: "m"(Sm), "r"(hep),"r"(qp), "r"(Qm), "r"(Rm), "r"(ql), "m"(Zm)

				: "xmm0", "xmm1", "xmm2", "xmm3",
		"xmm4", "xmm5", "xmm6", "xmm7",
		"xmm8", "xmm9", "xmm10", "xmm11",
		"xmm12", "xmm13", "xmm14", "xmm15",
		"rax", "r10", "r11", "r12",
		"cc"
			);
}

	inline void domasked7(__m128i * Sm, __m128i * hep, __m128i ** qp,
			__m128i * Qm, __m128i * Rm, long ql, __m128i * Zm, __m128i * Mm)
	{

#ifdef DEBUG
		printf("domasked\n");
		printf("Sm=%p\n", Sm);
		printf("hep=%p\n", hep);
		printf("qp=%p\n", qp);
		printf("Qm=%p\n", Qm);
		printf("Rm=%p\n", Rm);
		printf("qlen=%ld\n", ql);
		printf("Zm=%p\n", Zm);
		printf("Mm=%p\n", Mm);
#endif

#if 1
		__asm__
		__volatile__(".att_syntax noprefix    # Change assembler syntax \n"
				INITIALIZE7
				"        paddsb    (%7), xmm13          \n" // mask
				"        jmp       2f                   \n"

				"1:      movq      0(%2,r11,1), rax     \n"// load x from qp[qi]
				"        movdqa    0(%1,r11,4), xmm8    \n"// load N0
				"        paddsb    (%7), xmm8           \n"// mask
				"        movdqa    16(%1,r11,4), xmm12  \n"// load E
				"        paddsb    (%7), xmm12          \n"// mask

				ONESTEP7("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP7("xmm1", "xmm10", "xmm5", "16")
				ONESTEP7("xmm2", "xmm11", "xmm6", "32")
				ONESTEP7("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"        movq      8(%2,r11,1), rax     \n"// load x from qp[qi+1]
				"        movdqa    32(%1,r11,4), xmm0   \n"// load H0
				"        paddsb    (%7), xmm0           \n"// mask
				"        movdqa    48(%1,r11,4), xmm12  \n"// load E
				"        paddsb    (%7), xmm12          \n"// mask

				ONESTEP7("xmm8", "xmm1", "xmm4", "0" )
				ONESTEP7("xmm9", "xmm2", "xmm5", "16")
				ONESTEP7("xmm10", "xmm3", "xmm6", "32")
				ONESTEP7("xmm11", "32(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 48(%1,r11,4)  \n"// save E
				"        addq      $16, r11             \n"// qi++
				"2:      cmpq      r11, r10             \n"// qi = ql4 ?
				"        jne       1b                   \n"// loop

				"        cmpq      r11, r12             \n"
				"        je        3f                   \n"
				"        movq      0(%2,r11,1), rax     \n"// load x from qp[qi]
				"        movdqa    16(%1,r11,4), xmm12  \n"// load E
				"        paddsb    (%7), xmm12          \n"// mask

				ONESTEP7("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP7("xmm1", "xmm10", "xmm5", "16")
				ONESTEP7("xmm2", "xmm11", "xmm6", "32")
				ONESTEP7("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"3:      movq      %0, rax              \n"// save S
				"        movdqa    xmm13, (rax)         \n"
				"        .att_syntax prefix      # Change back to standard syntax"

				:

				: "m"(Sm), "r"(hep),"r"(qp), "r"(Qm), "r"(Rm), "r"(ql), "m"(Zm),
			"r"(Mm)

				: "xmm0", "xmm1", "xmm2", "xmm3",
			"xmm4", "xmm5", "xmm6", "xmm7",
			"xmm8", "xmm9", "xmm10", "xmm11",
			"xmm12", "xmm13", "xmm14", "xmm15",
			"rax", "r10", "r11", "r12",
			"cc"
				);
#endif

	}

inline void ONE_CELL_UPDATE_16(
		__m128i& vH,
		__m128i& vN,
		__m128i& vE,
		__m128i& vF,
		__m128i& vS,
		__m128i& vP,
		__m128i& R,
		__m128i& Q
		)
{
	vH = _mm_adds_epi16(vH, vP);//H = H + p[q]
	vH = _mm_max_epi16(vH, vF);//H = max(H, F)
	vH = _mm_max_epi16(vH, vE);//H = max(H, E)
	vS = _mm_max_epi16(vS, vH);//S = max(S, H)
	vF = _mm_subs_epi16(vF, R);//F = F - R
	vE = _mm_subs_epi16(vE, R);//E = E - R
	vN = vH;
	vH = _mm_subs_epi16(vH, Q);//H = H - Q
	vE = _mm_max_epi16(vH, vE);// E = max(H, E)
	vF = _mm_max_epi16(vH, vF);//F = ax(H, F)
}

inline void donormal16(
		__m128i * Sm, /* r9  */
		__m128i * hep, /* rdi */
		__m128i ** qp, /* rsi */
		__m128i * Qm, /* rdx */
		__m128i * Rm, /* rcx */
		long ql, /* r8  */
		__m128i * Zm)
{
	register __m128i vZero, vE, vP, vS, vHload;
	register __m128i vH[4], vN[4], vF[4];
	int i, j;

	__m128i **q_start = qp;
	__m128i *savedH = hep;
	__m128i *savedE = hep + ql;

	__m128i R = _mm_load_si128(Rm);
	__m128i Q = _mm_load_si128(Qm);


	vZero = _mm_set1_epi16(0x8000);
	vE = _mm_load_si128(savedE);
	vS = _mm_load_si128(Sm);

	/*
	 * Calculation for the first row in the tile
	 */
	for (i = 0; i < 4; i++)
	{
		vH[i] = vZero;
		vF[i] = vZero;
		//loading sub score
		vP = _mm_load_si128((q_start[0] + i));
		ONE_CELL_UPDATE_16(vH[i], vN[i], vE, vF[i], vS, vP, R, Q);
	}
	_mm_store_si128(savedE, vE);
	//Calculate along the query.
	for (j = 1; j < ql; j++)
	{
		//1st cell in the row
		//load H and then update
		vHload = _mm_load_si128(savedH + j - 1);
		_mm_store_si128(savedH + (j - 1), vN[3]);
		vE = _mm_load_si128(savedE + j);
		vP = _mm_load_si128(q_start[j]);
		ONE_CELL_UPDATE_16(vHload, vH[0], vE, vF[0], vS, vP, R, Q);
		//update remaining 3 cells in the row
		for (i = 1; i < 4; i++)
		{
			vP = _mm_load_si128(q_start[j] + i);
			ONE_CELL_UPDATE_16(vN[i-1], vH[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm_store_si128(savedE + j, vE);
		++j;
		if (j == ql)
			break;
		//swap vH and vN
		vHload = _mm_load_si128(savedH + (j - 1));
		_mm_store_si128(savedH + (j - 1), vH[3]);
		vE = _mm_load_si128(savedE + j);
		vP = _mm_load_si128(q_start[j]);
		ONE_CELL_UPDATE_16(vHload, vN[0], vE, vF[0], vS, vP, R, Q);
		for (i = 1; i < 4; i++)
		{
			vP = _mm_load_si128(q_start[j] + i);
			ONE_CELL_UPDATE_16(vH[i-1], vN[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm_store_si128(savedE + j, vE);
	}
	_mm_store_si128(Sm, vS);
}

inline void domasked16(__m128i * Sm, __m128i * hep, __m128i ** qp,
		__m128i * Qm, __m128i * Rm, long ql, __m128i * Zm, __m128i * Mm)
{
	register __m128i vZero, vE, vP, vS, vHload;
	register __m128i vH[4], vN[4], vF[4];
	int i, j;

	__m128i **q_start = qp;
	__m128i *savedH = hep;
	__m128i *savedE = hep + ql;

	__m128i R = _mm_load_si128(Rm);
	__m128i Q = _mm_load_si128(Qm);
	__m128i M = _mm_load_si128(Mm);


	vZero = _mm_set1_epi16(0x8000);
	vE = _mm_load_si128(savedE);
	vE = _mm_adds_epi16(vE, M);
	vS = _mm_load_si128(Sm);
	vS = _mm_adds_epi16(vS, M);

	/*
	 * Calculation for the first row in the tile
	 */
	for (i = 0; i < 4; i++)
	{
		vH[i] = vZero;
		vF[i] = vZero;
		//loading sub score
		vP = _mm_load_si128((q_start[0] + i));
		ONE_CELL_UPDATE_16(vH[i], vN[i], vE, vF[i], vS, vP, R, Q);
	}
	_mm_store_si128(savedE, vE);
	//Calculate along the query.
	for (j = 1; j < ql; j++)
	{
		//1st cell in the row
		//load H and then update
		vHload = _mm_load_si128(savedH + j - 1);
		vHload = _mm_adds_epi16(vHload, M);
		_mm_store_si128(savedH + (j - 1), vN[3]);
		vE = _mm_load_si128(savedE + j);
		vE = _mm_adds_epi16(vE, M);
		vP = _mm_load_si128(q_start[j]);
		ONE_CELL_UPDATE_16(vHload, vH[0], vE, vF[0], vS, vP, R, Q);
		//update remaining 3 cells in the row
		for (i = 1; i < 4; i++)
		{
			vP = _mm_load_si128(q_start[j] + i);
			ONE_CELL_UPDATE_16(vN[i-1], vH[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm_store_si128(savedE + j, vE);
		++j;
		if (j == ql)
			break;
		//swap vH and vN
		vHload = _mm_load_si128(savedH + (j - 1));
		vHload = _mm_adds_epi16(vHload, M);
		_mm_store_si128(savedH + (j - 1), vH[3]);
		vE = _mm_load_si128(savedE + j);
		vE = _mm_adds_epi16(vE, M);
		vP = _mm_load_si128(q_start[j]);
		ONE_CELL_UPDATE_16(vHload, vN[0], vE, vF[0], vS, vP, R, Q);
		for (i = 1; i < 4; i++)
		{
			vP = _mm_load_si128(q_start[j] + i);
			ONE_CELL_UPDATE_16(vH[i-1], vN[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm_store_si128(savedE + j, vE);
	}
	_mm_store_si128(Sm, vS);
}

inline void dprofile_fill16(WORD * dprofile_word, WORD * score_matrix_word,
			BYTE * dseq)
{
	__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
	__m128i xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
	__m128i xmm16, xmm17, xmm18, xmm19, xmm20, xmm21, xmm22, xmm23;
	__m128i xmm24, xmm25, xmm26, xmm27, xmm28, xmm29, xmm30, xmm31;

	for (int j = 0; j < CDEPTH; j++)
	{
		int d[N8_CHANNELS];
		for (int z = 0; z < N8_CHANNELS; z++)
			d[z] = dseq[j * N8_CHANNELS + z] << 5;

		for (int i = 0; i < 24; i += 8)
		{
			xmm0 = _mm_load_si128(
					(__m128i *) (score_matrix_word + d[0] + i));
			xmm1 = _mm_load_si128(
					(__m128i *) (score_matrix_word + d[1] + i));
			xmm2 = _mm_load_si128(
					(__m128i *) (score_matrix_word + d[2] + i));
			xmm3 = _mm_load_si128(
					(__m128i *) (score_matrix_word + d[3] + i));
			xmm4 = _mm_load_si128(
					(__m128i *) (score_matrix_word + d[4] + i));
			xmm5 = _mm_load_si128(
					(__m128i *) (score_matrix_word + d[5] + i));
			xmm6 = _mm_load_si128(
					(__m128i *) (score_matrix_word + d[6] + i));
			xmm7 = _mm_load_si128(
					(__m128i *) (score_matrix_word + d[7] + i));

			xmm8 = _mm_unpacklo_epi16(xmm0, xmm1);
			xmm9 = _mm_unpackhi_epi16(xmm0, xmm1);
			xmm10 = _mm_unpacklo_epi16(xmm2, xmm3);
			xmm11 = _mm_unpackhi_epi16(xmm2, xmm3);
			xmm12 = _mm_unpacklo_epi16(xmm4, xmm5);
			xmm13 = _mm_unpackhi_epi16(xmm4, xmm5);
			xmm14 = _mm_unpacklo_epi16(xmm6, xmm7);
			xmm15 = _mm_unpackhi_epi16(xmm6, xmm7);

			xmm16 = _mm_unpacklo_epi32(xmm8, xmm10);
			xmm17 = _mm_unpackhi_epi32(xmm8, xmm10);
			xmm18 = _mm_unpacklo_epi32(xmm12, xmm14);
			xmm19 = _mm_unpackhi_epi32(xmm12, xmm14);
			xmm20 = _mm_unpacklo_epi32(xmm9, xmm11);
			xmm21 = _mm_unpackhi_epi32(xmm9, xmm11);
			xmm22 = _mm_unpacklo_epi32(xmm13, xmm15);
			xmm23 = _mm_unpackhi_epi32(xmm13, xmm15);

			xmm24 = _mm_unpacklo_epi64(xmm16, xmm18);
			xmm25 = _mm_unpackhi_epi64(xmm16, xmm18);
			xmm26 = _mm_unpacklo_epi64(xmm17, xmm19);
			xmm27 = _mm_unpackhi_epi64(xmm17, xmm19);
			xmm28 = _mm_unpacklo_epi64(xmm20, xmm22);
			xmm29 = _mm_unpackhi_epi64(xmm20, xmm22);
			xmm30 = _mm_unpacklo_epi64(xmm21, xmm23);
			xmm31 = _mm_unpackhi_epi64(xmm21, xmm23);

			_mm_store_si128(
					(__m128i *) (dprofile_word
						+ CDEPTH * N8_CHANNELS * (i + 0)
						+ N8_CHANNELS * j), xmm24);
			_mm_store_si128(
					(__m128i *) (dprofile_word
						+ CDEPTH * N8_CHANNELS * (i + 1)
						+ N8_CHANNELS * j), xmm25);
			_mm_store_si128(
					(__m128i *) (dprofile_word
						+ CDEPTH * N8_CHANNELS * (i + 2)
						+ N8_CHANNELS * j), xmm26);
			_mm_store_si128(
					(__m128i *) (dprofile_word
						+ CDEPTH * N8_CHANNELS * (i + 3)
						+ N8_CHANNELS * j), xmm27);
			_mm_store_si128(
					(__m128i *) (dprofile_word
						+ CDEPTH * N8_CHANNELS * (i + 4)
						+ N8_CHANNELS * j), xmm28);
			_mm_store_si128(
					(__m128i *) (dprofile_word
						+ CDEPTH * N8_CHANNELS * (i + 5)
						+ N8_CHANNELS * j), xmm29);
			_mm_store_si128(
					(__m128i *) (dprofile_word
						+ CDEPTH * N8_CHANNELS * (i + 6)
						+ N8_CHANNELS * j), xmm30);
			_mm_store_si128(
					(__m128i *) (dprofile_word
						+ CDEPTH * N8_CHANNELS * (i + 7)
						+ N8_CHANNELS * j), xmm31);
		}
	}
}

void initLocks(){
	pthread_mutex_init(&idxMutex, NULL);
	pthread_mutex_init(&cntMutex, NULL);

}

void destroyLocks(){
	pthread_mutex_destroy(&idxMutex);
	pthread_mutex_destroy(&cntMutex);
} 



void Compute(
		BYTE* sseQuery,
		int   qlen,
		int  gapoe,
		int  gape,
		BYTE* score_matrix, 
		BYTE* deviceDBSeq, 
		int* map,
		size_t batchNum, 
		int* sseResult,
		int* overflow_indices,
		volatile int&       overflow_cnt,
		volatile int&       globalCounter,
		const int SCORE_LIMIT_7
		)
{
	int localIdx = -1;
	int localOverflowIdx = 0;
	
	pthread_mutex_lock(&idxMutex);
	localIdx = ++globalCounter;
	pthread_mutex_unlock(&idxMutex);

	char gap_open_penalty = gapoe;
	char gap_extend_penalty = gape;

#ifdef DEBUG	
	printf("Compute\n");
	printf("batchNum %ld\n", batchNum);
	printf("qlen %ld\n", qlen);
	printf("last deviceMapPos %dMB\n", map[batchNum * 2 - 2] >> 20);
#endif

#ifdef __INTEL_COMPILER
	/*For icc we need the profile to be aligned*/
	__declspec(align(16)) BYTE dprofile[4 * 16 * 32]; 
	__declspec(align(16)) BYTE* qtable[16 * qlen]; 
#else
	/*For GCC*/
	BYTE dprofile[4 * 16 * 32]; 
	BYTE* qtable[16 * qlen]; 
#endif
	for (int i = 0; i < qlen; ++i)
		qtable[i] = dprofile + 64 * sseQuery[i];	
	BYTE hearray[qlen * 32];

	__m128i S, Q, R, Z, T0;
	__m128i *hep, **qp;

	Z = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
	T0 = _mm_set_epi8(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80);
	Q = _mm_set_epi8(gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty);
	R = _mm_set_epi8(gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty);

	hep = (__m128i *) hearray;
	qp = (__m128i **) qtable;
	int* scores;
	while (localIdx < (int) batchNum && localIdx > -1)
	{
		//one batch at a time
		S = Z;
		memset(hearray, 0x80, qlen * 32);
		scores = sseResult + localIdx * 16;
		int pos = map[localIdx * 2];
		int dblen = map[localIdx * 2 + 1];
		char* dbseq = (char*)deviceDBSeq + pos;
		for (size_t i = 0; i < dblen >> 2; ++i)
		{
			dprofile_shuffle7(dprofile, score_matrix, (BYTE*)(dbseq + i * 64));
			donormal7(&S, hep, qp, &Q, &R, qlen, &Z);
		}

		for(int i = 0; i != 16; ++i){
			int score = ((unsigned char*) &S)[i] - 0x80;
#if 1
			if (score >= SCORE_LIMIT_7) {
				pthread_mutex_lock(&cntMutex);
				localOverflowIdx = overflow_cnt++;
				pthread_mutex_unlock(&cntMutex);
				overflow_indices[localOverflowIdx] = localIdx * 16 + i;
				score = 0;
			}
#endif
			scores[i] = score;
		}

		pthread_mutex_lock(&idxMutex);
		localIdx = ++globalCounter;
		pthread_mutex_unlock(&idxMutex);
	}
}

void ComputeRecalc(
		char* sseQuery,
		int   qlen,
		int  gapoe,
		int  gape,
		WORD* score_matrix, 
		char* d_dbseq, 
		int* map,
		size_t batchNum, 
		int* sseResult,
		volatile int&       globalCounter
		)
{
	int localIdx = -1;
	int localOverflowIdx = 0;
	
	pthread_mutex_lock(&idxMutex);
	localIdx = ++globalCounter;
	pthread_mutex_unlock(&idxMutex);

	short gap_open_penalty = gapoe;
	short gap_extend_penalty = gape;

	WORD* dprofile = (WORD*) malloc(4 * 8 * 32 * sizeof(WORD));
	WORD** qtable = (WORD**) malloc(4 * 8 * qlen * sizeof(WORD*));
	memset(dprofile, 0, sizeof(WORD) * 4 * 8 * 32);

	for (size_t i = 0; i < qlen; ++i)
		qtable[i] = dprofile + 4 * 8 * sseQuery[i];	

	WORD hearray[qlen * 16];

	__m128i S, Q, R, Z, T0;
	__m128i *hep, **qp;

	Z  = _mm_set1_epi16(0x8000);
	T0 = _mm_set1_epi16(0x0000);
	Q  = _mm_set1_epi16(gap_open_penalty);
	R  = _mm_set1_epi16(gap_extend_penalty);

	hep = (__m128i *) hearray;
	qp = (__m128i **) qtable;
	int* scores;
	while (localIdx < (int) batchNum && localIdx > -1)
	{
		//one batch at a time
		S = Z;

		for(size_t qi = 0; qi < qlen; qi++){
			hep[2 * qi] = Z;
			hep[2 * qi + 1] = Z;
		}

		scores = sseResult + localIdx * 8;
		int pos   = map[localIdx * 2];
		int dblen = map[localIdx * 2 + 1];
		//printf("pos %d\n dblen %d\n", pos, dblen);
		char* dbseq = d_dbseq + pos;

		for (size_t i = 0; i < dblen >> 2; ++i)
		{
			dprofile_fill16(dprofile, score_matrix, (dbseq + i * 64));
			donormal16(&S, hep, qp, &Q, &R, qlen, &Z);
		}

		for(int i = 0; i != N8_CHANNELS; ++i){
			int score = ((unsigned short*) &S)[i] ^ 0x8000;
			if(score > 100)
			//printf("score %d\n", score);
			scores[i] = score;
		}

		pthread_mutex_lock(&idxMutex);
		localIdx = ++globalCounter;
		pthread_mutex_unlock(&idxMutex);
	}
}

void Compute16(
		WORD* score_matrix,
		BYTE* dbSeq,
		int* vecPos,
		int numSeqs,
		int* result,
		int start_idx,
		BYTE* sseQuery,
		size_t qlen
		)
{
	if(start_idx == -1)
		return;
	__m128i S, Q, R, T, M, Z, T0;
	__m128i *hep, **qp;
	BYTE * d_begin[N8_CHANNELS];
	__m128i dseqalloc[CDEPTH];

	WORD hearray[qlen * 32];
	memset(hearray, 0, sizeof(WORD) * 2 * qlen * N8_CHANNELS);
	
	BYTE * dseq = (BYTE *) &dseqalloc;
	BYTE zero;

	WORD* dprofile = (WORD*) malloc(4 * 16 * 32 * sizeof(BYTE));
	WORD** qtable = (WORD**) malloc(16 * qlen * sizeof(BYTE*));
	memset(dprofile, 1, sizeof(BYTE) * 4 * 16 * 32);
	for (size_t i = 0; i < qlen; ++i)
		qtable[i] = dprofile + 64 * sseQuery[i] / 2;	
	
	int seq_id[N8_CHANNELS];
	int next_id = 0;
	int done;

	Z = _mm_set_epi16(0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
			0x8000);
	T0 = _mm_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
			0x8000);
	
	WORD gap_open_penalty = 12;
	WORD gap_extend_penalty = 2;
	
	Q = _mm_set_epi16(gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty);
	R = _mm_set_epi16(gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty);
	zero = 23;
	done = 0;
	S = Z;
	hep = (__m128i *) hearray;
	qp = (__m128i **) qtable;

	for(size_t qi = 0; qi < qlen; qi++){
			hep[2 * qi] = Z;
			hep[2 * qi + 1] = Z;
	}
	for(int c = 0; c < N8_CHANNELS; ++c){
		d_begin[c] = &zero;
		seq_id[c] = -1;
	}

	int easy = 0;
	int hi = 0;	
	while(1){
		if(easy){
			for (int c = 0; c < N8_CHANNELS; c++) {
				BYTE v;
				for (int j = 0; j < CDEPTH; j++) {
					v = *(d_begin[c]);
					dseq[N8_CHANNELS * j + c] = v;
					//printf("%d ", v);
					if (v != 23)
						d_begin[c]++;
				}
				if (*(d_begin[c]) == 23)
					easy = 0;
			}

			dprofile_fill16(dprofile, score_matrix, dseq);
			donormal16(&S, hep, qp, &Q, &R, qlen, &Z);
		}else{
			easy = 1;
			M = _mm_setzero_si128();
			T = T0;

			for (int c = 0; c < N8_CHANNELS; c++) {
				if (*(d_begin[c]) != 23) {
					for (int j = 0; j < CDEPTH; j++) {
						BYTE v = *(d_begin[c]);
						dseq[N8_CHANNELS * j + c] = v;
						if (v!= 23)
							d_begin[c]++;
					}

					if (*(d_begin[c]) == 23)
						easy = 0;
				} else {
					M = _mm_xor_si128(M, T);
					long cand_id = seq_id[c];
					if (cand_id >= 0) {
						int score = ((unsigned short*) &S)[c] ^ 0x8000;
						/*save the alignment score*/
						//printf("%d\n", score);
						result[cand_id + start_idx] = score;
						done++;
					}
					if (next_id < numSeqs) {
						seq_id[c] = next_id;
						d_begin[c] = dbSeq + vecPos[next_id + start_idx];
						next_id++;

						for (int j = 0; j < CDEPTH; j++) {
							BYTE v = *(d_begin[c]);
							//printf("%d ", v);
							dseq[N8_CHANNELS * j + c] = v;
							if (v != 23)
								d_begin[c]++;
						}
						if (*(d_begin[c]) == 23)
							easy = 0;
					} else {
						seq_id[c] = -1;
						d_begin[c] = &zero;
						for (int j = 0; j < CDEPTH; j++){
							//dseq[N8_CHANNELS * j + c] = 0;
							dseq[N8_CHANNELS * j + c] = 23;
						}
					}
				}
				T = _mm_slli_si128(T, 2);
			}
			if (done == numSeqs) {
				break;
			}
			dprofile_fill16(dprofile, score_matrix, dseq);
			domasked16(&S, hep, qp, &Q, &R, qlen, &Z, &M);
		}
	}
}
