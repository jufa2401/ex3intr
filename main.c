#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h> /* where intrinsics are defined */

#define CLOCK_RATE_GHZ 2.26e9

/* Time stamp counter from Lecture 2/17 */
static __inline__ unsigned long long RDTSC(void) {
    unsigned hi,lo;
    __asm__ volatile("rdtsc" : "=a"(lo),"=d"(hi));
    return ((unsigned long long) lo)| (((unsigned long long)hi) << 32);
}

int sum_naive( int n, int *a )
{
    int sum = 0;
    for( int i = 0; i < n; i++ )
        sum += a[i];
    return sum;
}
//  Further Question Assignment
void mult_naive( int n, short int *a, short int factor)
{
    for( int i = 0; i < n; i++ )
        a[i] = factor *a[i];
}
//  Further Question Assignment
void mult_vectorized( int n, short int *a, short int factor)
{
    /* WRITE YOUR VECTORIZED CODE HERE */

    int m = n/8;  //Amount of vector operations

    if(m > 0) {

        __m128i f = _mm_set_epi16(factor,factor,factor,factor,factor,factor,factor,factor);
        __m128i *p = (__m128i *) a;
        for (int i = 0; i < m; ++i) {
            __m128i b = _mm_load_si128(p);
            b = _mm_mullo_epi16 (f,b);
            _mm_storeu_si128(p++,b);
        }
    }
    for (int i = m*8; i < n; ++i) {
        a[i] = factor *a[i];
    }
}

int sum_unrolled( int n, int *a )
{
    int sum = 0;

    /* do the body of the work in a faster unrolled loop */
    for( int i = 0; i < n/4*4; i += 4 )
    {
        sum += a[i+0];
        sum += a[i+1];
        sum += a[i+2];
        sum += a[i+3];
    }

    /* handle the small tail in a usual way */
    for( int i = n/4*4; i < n; i++ )
        sum += a[i];

    return sum;
}

int sum_vectorized( int n, int *a )
{

    /* WRITE YOUR VECTORIZED CODE HERE */

    int sum = 0;
    int m = n/4;  //Amount of vector operations
    int r = n%4;  // The remainder that cannot be vectorized

    if(m > 0) {
        __m128i sumV = _mm_setzero_si128();
        __m128i *p = (__m128i *) a;
        for (int i = 0; i < m; ++i) {
            __m128i b = _mm_load_si128(p++);
            sumV = _mm_add_epi32(sumV, b);
        }
        int l[4];
        _mm_storeu_si128((__m128i *) &l, sumV);
        sum = l[0]+l[1]+l[2]+l[3];
    }
    int *l = a+m*4;
    for (int i = 0; i < r; ++i) {
        sum += *l++;
    }

    return sum;
}

int sum_vectorized_unrolled( int n, int *a )
{

    /* WRITE YOUR VECTORIZED CODE HERE */
    int factor = 16;
    int sum = 0;
    int m = n/factor;  //Amount of vector operations
    int r = n%factor;  // The remainder that cannot be vectorized

    if(m > 0) {
        __m128i sumV = _mm_setzero_si128();
        __m128i *p = (__m128i *) a;
        for (int i = 0; i < m; ++i) {
            __m128i b = _mm_load_si128(p++);
            sumV = _mm_add_epi32(sumV, b);
            b = _mm_load_si128(p++);
            sumV = _mm_add_epi32(sumV, b);
            b = _mm_load_si128(p++);
            sumV = _mm_add_epi32(sumV, b);
            b = _mm_load_si128(p++);
            sumV = _mm_add_epi32(sumV, b);

        }
        int l[4];
        _mm_storeu_si128((__m128i *) &l, sumV);
        sum = l[0]+l[1]+l[2]+l[3];
    }
    int *l = a+m*factor;
    for (int i = 0; i < r; ++i) {
        sum += *l++;
    }
    return sum;
}

void benchmark( int n, int *a, int (*computeSum)(int,int*), char *name )
{
    /* warm up */
    int sum = computeSum( n, a );

    /* measure */
    unsigned long long cycles = RDTSC();
    sum += computeSum( n, a );
    cycles = RDTSC()-cycles;

    double microseconds = cycles/CLOCK_RATE_GHZ*1e6;

    /* report */
    printf( "%20s: ", name );
    if( sum == 2*sum_naive(n,a) ) printf( "%.2f microseconds\n", microseconds );
    else	                  printf( "ERROR!\n" );
}

double benchmark3( int n, short int *a, short int f, void (*computeMul)(int,short*,short), char *name )
{
    /* measure */
    unsigned long long cycles = RDTSC();
    computeMul(n,a,f);
    cycles = RDTSC()-cycles;

    double microseconds = cycles/CLOCK_RATE_GHZ*1e6;

    /* report */
    printf( "%20s: %.2f microseconds\n", name, microseconds );
    return microseconds;
}

int main( int argc, char **argv )
{
    const int n = 7777; /* small enough to fit in cache */

    /* init the array */
    srand48( time( NULL ) );
    int a[n] __attribute__ ((aligned (16))); /* align the array in memory by 16 bytes */
    for( int i = 0; i < n; i++ ) a[i] = (short)lrand48( );

    int s1 = sum_naive(n,a);
    int s2 = sum_vectorized(n,a);
    int s3 = sum_vectorized_unrolled(n,a);
    printf("naive = %d, vectorized = %d, unrolled =%d, %s\n",s1,s2,s3,(s1==s2)?"OK":"Error");

    /* benchmark series of codes */
    benchmark( n, a, sum_naive, "naive" );
    benchmark( n, a, sum_unrolled, "unrolled" );
    benchmark( n, a, sum_vectorized, "vectorized" );
    benchmark( n, a, sum_vectorized_unrolled, "vectorized unrolled" );

    const int n2 = 7777;
    short int b[n2] __attribute__ ((aligned (16))); /* align the array in memory by 16 bytes */
    for( int i = 0; i < n2; i++ ) b[i] = lrand48( );
    short factor = 48;
    double t1 = benchmark3(n2,b,factor,mult_naive,"scale naive");
    double t2 = benchmark3(n2,b,factor,mult_vectorized,"scale vectorized");
    printf("             Speedup: %f\n",t1/t2);
    return 0;
}
