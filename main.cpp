/*

	Guaranteed Lower Area Bound for the Mandelbrot Set
	
	based on cell-mapping and interval arithmetics
	in the article: Figueiredo L.H. et.al, "Images you can trust"
	
	Marc Meidlinger
	March-April 2020

*/

/*

	2020-07 in function readHCfile: branch RESETCOLOR was without effect => activated.

*/

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "string.h"
#include "time.h"
#include "omp.h"


// Stephenson formula for values c in a period-3 or 4 component
// needs kv-library, if notavailable, comment out
#define _USESTEPHENSONTEST

#ifdef _USESTEPHENSONTEST
#include <kv/complex.hpp>
#include <kv/interval.hpp>
#include <kv/rdouble.hpp>
#endif


// number type
#define _DOUBLE
//#define _LONGDOUBLE

#ifdef _DOUBLE
typedef double NTYP;
const char NTYPSTR[]="double";
#endif

#ifdef _LONGDOUBLE
typedef long double NTYP;
const char NTYPSTR[]="long double";
#endif

// starting with the critical point's Jtiles in the Jset queue 
// or with a point-sampled derived periodic point's Jtiles
#define _USE_PS_PERIODIC

#ifdef _USESTEPHENSONTEST
// currently used for Stephenson formula test
// double is sufficient as values assigned are
// either small integers or corner coordinates
// of an Mtile or Jtile which (independent of the
// number type NTYP) are representable by double as
// maximal screen size is limited by int32_t being
// screen coordinate's number type. So conversion
// from a e.g. long double to double when initialising kv-intervals
// does not result in loss of bits
typedef kv::complex< kv::interval<double> > cplxia;
typedef kv::interval<double> ia;
#endif

// consts

// functions: for future addition

enum {
	FUNC_Z2C=0,
	
	FUNCANZ
};

const char funcname[][32] = {
	"Z2C"
};

const int32_t ANZFLOODFILL=(1 << 27);
const int32_t MAXHYPERBOLICCENTERS=(1 << 16);
const int32_t MAXBREITE=255;
const int32_t CIRCUMFERENCINGFACTOR=3;

// for Mset, at most 63
const int32_t COLOR_UNDEF=-1;
const uint8_t MTILE_GRAY=0b00;
const uint8_t MTILE_WHITE=0b01; // for future compatibility
const uint8_t MTILE_BLACK=0b10;
const uint8_t MTILE_CHECKFORBLACK=4;
const uint8_t MTILE_GRAY_DONE=6;
const uint8_t MTILE_NORMALCOLORBITS=0b00111111;

// high order bits for floodfill
const uint8_t MTILE_BIT_FLOODED    =0b10000000;
const uint8_t MTILE_BIT_ROLLBACK   =0b01000000;

// colors for Julia set
const uint8_t JTILE_GRAY=0b00;
const uint8_t JTILE_WHITE=0b01; // for future compatibility
const uint8_t JTILE_BLACK=0b10;

// result of IA-Main cardioid test or Period2 4-corner test
const int8_t INP123_MAYBE=0;
const int8_t INP123_INSIDE=1;
const int8_t INP123_OUTSIDE=2;

// general
const int32_t PS_MAXCYCLELEN=200;
const double PS_EQUALITY=1E-10;


// structs

// corner coordinates of an Mtile or Jtile
// values are accurately represented as only dyadic fractions
// are assigned
struct PlaneRect {
	NTYP x0,x1,y0,y1;
};

// flood-fill pixel coordinates
struct Int2 {
	int32_t w0,w1;
};

// as a complex number, just used for point-sampling,
// accuracy and number representability are not a concern here
struct Double2 {
	double x,y;
};

// pixel coordinates of a virtual screen
struct ScreenRect {
	int32_t x0,x1,y0,y1;
};

// constructing the Jset orbit. Stores Jtiles to follow
struct Streak {
	int32_t x0;
	uint8_t breite;
	int32_t y;
};

// list of streaks to follow
struct StreakList {
	Streak* values;
	int32_t anz;
	
	StreakList();
	virtual ~StreakList();
	void allocate(int64_t&);
	void fastEmpty(void);
	int8_t pop(Streak&);
	int8_t push(const int32_t,const int32_t,const int32_t);
};

// chunk size dependend on operating system
//#define _CHUNK512

const int32_t MAXPTR=2048;

#ifdef _CHUNK512
const uint64_t CHARMAPCHUNKSIZE=( (uint64_t)1 << 27 );
#else
// win64 => 1 GB
const uint64_t CHARMAPCHUNKSIZE=( (uint64_t)1 << 30 );
#endif

typedef unsigned char BYTE;

struct RGB3 {
	BYTE R,G,B;
};

typedef BYTE *PBYTE;

// memory manager for byte allocations
struct ArrayByteMgr {
	BYTE* current;
	int32_t allocatedIdx,freeFromIdx,allocatePerBlockIdx;
	PBYTE ptr[MAXPTR];
	int32_t anzptr;
	
	ArrayByteMgr();
	void FreeAll(void);
	virtual ~ArrayByteMgr();
	PBYTE getMemory(const int32_t);
};

// stores a full Jset or the part in a virtual screen where the
// active region of an Mset hyperbolic component resides
// data structure is equivalent to an 256-color bitmap
// data can be viewed externally
// colors used are: black = interior, gray = untested
// yellow = tested and judged gray at current resolution
struct TiledPlane {
	int64_t xlen,ylen;
	int64_t memused;
	PBYTE* cmpY;
	RGB3 palette[256];
		
	TiledPlane();
	virtual ~TiledPlane();
		
	void setlenxy(const int32_t,const int32_t);
	void save(const char*);
	void saveTwice(const char*);
	void lineHorizVert(const int32_t,const int32_t,const int32_t,const int32_t,const BYTE);
	int8_t load(const char*);
	void fillrect(const int32_t,const int32_t,const int32_t,const int32_t,const BYTE);
	void setPaletteRGB(const int32_t,const BYTE,const BYTE,const BYTE);
	void setPunkt(const int32_t,const int32_t,const BYTE);
	BYTE getPunkt(const int32_t,const int32_t);
};

// stores an active region of a hyperbolic center
struct HCenter {
	int32_t hcid;
	int32_t centerpixelx,centerpixely;
	int32_t allocatewidth;
	int8_t newlyfound;
	int8_t donotanalyze;
	ScreenRect circumsq; // circumferencing square of side length CIRCUMFERNCINGFACTOR*inscrbiedwidth
	NTYP Ax0,Ay0; // lower left pixels lower left corner's complex coordinates
	TiledPlane msetregion;
	
	void initRegion(void);
	void setMtile(const int32_t,const int32_t,const int32_t);
	int32_t getMtile(const int32_t,const int32_t);
	void save(void);
};

// JuliaSet with necessary local parameters to be used via openmp
struct JuliaSet {
	// result of last testC call for the set parameters
	int32_t erg;
	// data
	TiledPlane juliaimg;
	HCenter* ptrhc;
	// seed C interval
	NTYP seedC0re,seedC1re,seedC0im,seedC1im; 
	// position of pixel in Mset
	int32_t px,py;
	// flags for constructing the orbit and when to reset image
	int32_t UNVISITED;
	int32_t TOFOLLOW;
	int32_t FOLLOWED;
	int32_t UNVISITED0;
	int32_t ctrmaxinsl;
	// bbx parts to visit
	StreakList streaks;
	// point-sampled orbit of the critical point - merely a directional tool
	Double2 *psorbit;
	// counters for final values
	int64_t ctrtestccalled;
	int64_t ctrsettoblack;
	int64_t ctrppfound;
	int64_t ctrppfailed;
	int64_t ctrppongrid;
	int64_t ctrjtilesvisited;
	
	JuliaSet();
	void initMemory(void);
};


// globals

// function and symmetry
int8_t SYMMETRY_AT_XAXIS=0;
int8_t SYMMETRY_AT_YAXIS=1;
int32_t FUNC=FUNC_Z2C;
void (*getBoundingBoxfA)(JuliaSet&,PlaneRect&,PlaneRect&) = NULL;
void (*ps_func)(const double,const double,const double,const double,Double2&) = NULL;


// control flow flags
int8_t PRETESTGRIDDING=1; // on
int8_t produceimg=0;
int8_t listfulleventoccured=0;
int8_t _RESETCOLORS=0;
int8_t savetwice=1;

// counting
int64_t jsetmemory=0;
int64_t ctrsetp34=0;
int64_t hcmemoryused=0;
int64_t ctrnewblackregion=0;
int64_t ctrsetblackfloodfill=0;
int64_t time_at_enter_main=0; // in CLOCKS_PER_SEC
int64_t endtime=-1; // in CLOCKS_PER_SEC

// hyperbolic centers
int32_t anzcenters=0;
HCenter *centers=NULL;

// parallelized Julia set orbit construction
int32_t MAXJULIASETS=4; // parallelecity: orbits construct for Mtiles in parallel
int32_t inpipeline=0;
JuliaSet* juliasets=NULL;

// Julia set values
NTYP scaleRangePerPixelJ,scalePixelPerRangeJ;
int32_t RANGEJ0=-2,RANGEJ1=2; 
NTYP COMPLETEJ0,COMPLETEJ1;
int32_t REFINEMENTJ=10;
int32_t PS_MAXIT=25000;
NTYP *scrlowJ=NULL;

// Mset
int32_t MINGRIDSIZE=8;
int32_t MAXGRIDSIZE=64;
int32_t REFINEMENTM=10;
NTYP scaleRangePerPixelM,scalePixelPerRangeM;
int32_t SCREENWIDTHM,SCREENWIDTHJ;
int32_t RANGEM0=-2,RANGEM1=2; 
NTYP COMPLETEM0,COMPLETEM1;
NTYP *scrlowM=NULL;

// general
FILE *flog=NULL;
Int2* floodfill=NULL;
int32_t MAXSTREAKS;
ArrayByteMgr planemgr;


// forward declarations

#ifdef _USESTEPHENSONTEST
int8_t cplxia_contains_zero(const cplxia&);
int8_t ia_contains_zero(const ia&);
#endif

// orbit construction
void testC_interior(JuliaSet&);
void testC_interior(JuliaSet&,const int32_t,const int32_t);

// z2c: testing a complex number
int8_t MPointInMainCardioid_z2c(const NTYP,const NTYP);
int8_t MPointInPeriod2Bulb_z2c(const NTYP,const NTYP);
int8_t MTileInMainCardioid_z2c(PlaneRect&);
int8_t MTileInPeriod2Bulb_z2c(PlaneRect&);
int8_t MTileInPeriod3_z2c(PlaneRect&);
int8_t MTileInPeriod4_z2c(PlaneRect&);
void markP12remove_z2c(HCenter&);
int8_t markP3set_z2c(HCenter&);
int8_t markP4set_z2c(HCenter&);

// converting real-valued coordinate into virtual screen pixel position
inline int32_t scrcoord_as_lowerleftJL(const NTYP&);
inline int32_t scrcoord_as_lowerleftJR(const NTYP&);
inline int32_t scrcoord_as_lowerleftML(const NTYP&);
inline int32_t scrcoord_as_lowerleftMR(const NTYP&);

// general
void write2(FILE*,const uint8_t,const uint8_t);
void write4(FILE*,const uint8_t,const uint8_t,const uint8_t,const uint8_t);


// defines used as small expressions

// AA,BB,CC,DD are VARIABLES, not expressions
// AA..BB, CC..DD may contain zero in the interior
// faster cases: AA<=BB<=0, BB>=AA>=0 and CC,DD accordingly
// then IA multiplicaiton can be performed with 2 mul-op
// instead of all 4
#define IAMUL_SIGN(ERGMIN,ERGMAX,AA,BB,CC,DD,FUNC) \
{\
	if (((AA)>=0.0)&&((AA)<=(BB))) {\
		if (((CC)>=0.0)&&((CC)<=(DD))) {\
			ERGMIN=(AA)*(CC);\
			ERGMAX=(BB)*(DD);\
		} else if (((CC)<=(DD))&&((DD)<=0.0)) {	\
			ERGMIN=(BB)*(CC);\
			ERGMAX=(AA)*(DD);\
		} else {\
			FUNC(ERGMIN,ERGMAX,(AA)*(CC),(AA)*(DD),(BB)*(CC),(BB)*(DD));\
		}\
	} \
	else if (((AA)<=(BB))&&((BB)<=0.0)) {\
		if ((0.0<=(CC))&&((CC)<=(DD))) {\
			ERGMIN=(AA)*(DD);\
			ERGMAX=(BB)*(CC);\
		} else if (((CC)<=(DD))&&((DD)<=0.0)) {\
			ERGMIN=(BB)*(DD);\
			ERGMAX=(AA)*(CC);\
		} else {\
			FUNC(ERGMIN,ERGMAX,(AA)*(CC),(AA)*(DD),(BB)*(CC),(BB)*(DD));\
		}\
	} else {\
		FUNC(ERGMIN,ERGMAX,(AA)*(CC),(AA)*(DD),(BB)*(CC),(BB)*(DD));\
	}\
}

#define SETVALUE(RR,II) cplxia(ia(RR,RR),ia(II,II))

#define CHECKSCRCOORDJ \
{\
	if (w >= SCREENWIDTHJ) {\
		LOGMSG("Error. scrcoordJ/1");\
		exit(99);\
	}\
	if (\
		(!((scrlowJ[w] <= a) && (a < scrlowJ[w+1])))\
	) {\
		LOGMSG("Error. scrcoordJ/2.\n");\
		exit(99);\
	}\
}

#define CHECKSCRCOORDM \
{\
	if (w >= SCREENWIDTHM) {\
		LOGMSG("Error. scrcoordM/1");\
		exit(99);\
	}\
	if (\
		(!((scrlowM[w] <= a) && (a < scrlowM[w+1])))\
	) {\
		LOGMSG("Error. scrcoordM/2.\n");\
		exit(99);\
	}\
}

#define LOGMSG(TT) \
{\
	fprintf(flog,TT); fflush(flog);\
	printf(TT);\
}

#define JSUMOUT(TT,VAR) \
{\
	int64_t sum=0;\
	for(int i=0;i<MAXJULIASETS;i++) sum += juliasets[i]. VAR;\
	LOGMSG2(TT,sum);\
}

#define JMAXOUT(TT,VAR) \
{\
	int64_t sum=0;\
	for(int i=0;i<MAXJULIASETS;i++) {\
		if (juliasets[i]. VAR> sum) sum=juliasets[i]. VAR;\
	}\
	LOGMSG2(TT,sum);\
}

#define LOGMSG2(TT,AA) \
{\
	fprintf(flog,TT,AA); fflush(flog);\
	printf(TT,AA);\
}

#define LOGMSG3(TT,AA,BB) \
{\
	fprintf(flog,TT,AA,BB); fflush(flog);\
	printf(TT,AA,BB);\
}

#define TRIMM(WW) \
{\
	if (WW<0) WW=0;\
	else if (WW>=SCREENWIDTHM) WW=SCREENWIDTHM-1;\
}

#define MSETTILEINHC(XX,YY,FF) \
{\
	if (\
		((XX) >= circumsq.x0) &&\
		((XX) <= circumsq.x1) &&\
		((YY) >= circumsq.y0) &&\
		((YY) <= circumsq.y1)\
	) {\
		msetregion.setPunkt(\
			(XX)-circumsq.x0,\
			(YY)-circumsq.y0,FF);\
	}\
}

#define LOGMSG4(TT,AA,BB,CC) \
{\
	fprintf(flog,TT,AA,BB,CC); fflush(flog);\
	printf(TT,AA,BB,CC);\
}

#define LOGMSG5(TT,AA,BB,CC,DD) \
{\
	fprintf(flog,TT,AA,BB,CC,DD); fflush(flog);\
	printf(TT,AA,BB,CC,DD);\
}
	

// only used non-parallel
#define WORKOFF_RESULTS(PFNR,FFB) \
{\
	if (juliasets[PFNR].erg == JTILE_BLACK) {\
		FFB=1;\
		juliasets[PFNR].ptrhc->setMtile(\
			juliasets[PFNR].px,\
			juliasets[PFNR].py,\
			MTILE_BLACK);\
	} else {\
		juliasets[PFNR].ptrhc->setMtile(\
			juliasets[PFNR].px,\
			juliasets[PFNR].py,\
			MTILE_GRAY_DONE);\
	} \
}

#define ADDPIPELINE(XX,YY,PTR,C0RE,C1RE,C0IM,C1IM) \
{\
	if (inpipeline < MAXJULIASETS) {\
		juliasets[inpipeline].seedC0re=C0RE;\
		juliasets[inpipeline].seedC1re=C1RE;\
		juliasets[inpipeline].seedC0im=C0IM;\
		juliasets[inpipeline].seedC1im=C1IM;\
		juliasets[inpipeline].px=XX;\
		juliasets[inpipeline].py=YY;\
		juliasets[inpipeline].ptrhc=PTR;\
		inpipeline++;\
	} else {\
		LOGMSG("Error. Pipeline-C.\n");\
		exit(99);\
	}\
}

#define OUTVALUES1 \
{\
	JSUMOUT("\n%I64d Mtiles analyzed by orbit construction\n",ctrtestccalled);\
	JSUMOUT("  %I64d Mtiles judged interior\n",ctrsettoblack);\
	LOGMSG2("%I64d Mtiles set to interior by flood-fill\n",ctrsetblackfloodfill);\
	JMAXOUT("(internal) max streak used %i\n",ctrmaxinsl);\
	JMAXOUT("(internal) %I64d Jtiles visited in total\n",ctrjtilesvisited);\
	LOGMSG2("%I64d set by Stephenson formula\n",ctrsetp34);\
}

#ifdef _USE_PS_PERIODIC
#define OUTVALUES2 \
{\
	JSUMOUT("(internal) %I64d periodic points found\n",ctrppfound);\
	JSUMOUT("(internal) %I64d periodic points failed\n",ctrppfailed);\
	JSUMOUT("(internal) %I64d periodic points on grid\n",ctrppongrid);\
}
#else
#define OUTVALUES2 { }
#endif


// routines

char* chomp(char* s) {
	if (!s) return 0;
	for(int32_t i=strlen(s);i>=0;i--) if (s[i]<32) s[i]=0; else break;
	return s;
}

char* upper(char* s) {
	if (!s) return NULL;
	
	for(int32_t i=0;i<(int32_t)strlen(s);i++) {
		if ((s[i]>='a')&&(s[i]<='z')) s[i]=s[i]-'a'+'A';
	}

	return s;
}

// converting a dyadic fraction into a pixel coordinate of a virtualscreen
// either for a Jtile or an Mtile
// trimes values automatically to screen width
// outside test has been conducted before calling this function

inline int32_t scrcoord_as_lowerleftJL(const NTYP& a) {
	int32_t w;
	
	#ifdef _DOUBLE
	if (a <= COMPLETEJ0) return 0;
	if (a >= COMPLETEJ1) return (SCREENWIDTHJ-1);
	w=(int)floor( (a - COMPLETEJ0) * scalePixelPerRangeJ );
	CHECKSCRCOORDJ
	if (w >= SCREENWIDTHJ) return (SCREENWIDTHJ-1);
	if (w <= 0) return 0;

	return w;
	#endif
	
	#ifdef _LONGDOUBLE
	if (a <= COMPLETEJ0) return 0;
	if (a >= COMPLETEJ1) return (SCREENWIDTHJ-1);
	w=(int)floor( (a - COMPLETEJ0) * scalePixelPerRangeJ );
	CHECKSCRCOORDJ
	if (w >= SCREENWIDTHJ) return (SCREENWIDTHJ-1);
	if (w <= 0) return 0;

	return w;
	#endif

	LOGMSG("Error. scrcoord_as_lowerleftJL not defined for number type\n");
	exit(99);
}

inline int32_t scrcoord_as_lowerleftJR(const NTYP& a) {
	int32_t w;
	
	#ifdef _DOUBLE
	if (a <= COMPLETEJ0) return 0;
	if (a >= COMPLETEJ1) return (SCREENWIDTHJ-1);
	// 'a' is a dyadic fraction
	// 'diff' is accurately represented by double
	// scalePixelPerRangeJ is a natural number
	double diff=(a - COMPLETEJ0) * scalePixelPerRangeJ;
	double fl=floor(diff);
	w=(int)fl;
	CHECKSCRCOORDJ
	// 'a' is the right (or upper) end of the bbx
	// flooring assigns coordinates in a tile to the left and lower
	// part of a tile
	// if pixel coordinate has no fractional part => 'a' landed
	// directly on a grid line and hence is the left edge of a tile. 
	// As tiles are closed, it is sufficient to use the tile one
	// to the left (or down if 'a' is a vertical coordinate)
	// that way the orbit can be shrunk
	if (diff == fl) w--;
	if (w >= SCREENWIDTHJ) return (SCREENWIDTHJ-1);
	if (w <= 0) return 0;

	return w;
	#endif

	#ifdef _LONGDOUBLE
	if (a <= COMPLETEJ0) return 0;
	if (a >= COMPLETEJ1) return (SCREENWIDTHJ-1);
	double diff=(a - COMPLETEJ0) * scalePixelPerRangeJ;
	double fl=floor(diff);
	w=(int)fl;
	CHECKSCRCOORDJ
	if (diff == fl) w--;
	if (w >= SCREENWIDTHJ) return (SCREENWIDTHJ-1);
	if (w <= 0) return 0;

	return w;
	#endif

	LOGMSG("Error. scrcoord_as_lowerleft not defined for number type\n");
	exit(99);
}

// for mset

inline int32_t scrcoord_as_lowerleftML(const NTYP& a) {
	int32_t w;
	
	#ifdef _DOUBLE
	if (a <= COMPLETEM0) return 0;
	if (a >= COMPLETEM1) return (SCREENWIDTHM-1);
	w=(int)floor( (a - COMPLETEM0) * scalePixelPerRangeM );
	CHECKSCRCOORDM
	if (w >= SCREENWIDTHM) return (SCREENWIDTHM-1);
	if (w <= 0) return 0;

	return w;
	#endif

	#ifdef _LONGDOUBLE
	if (a <= COMPLETEM0) return 0;
	if (a >= COMPLETEM1) return (SCREENWIDTHM-1);
	w=(int)floor( (a - COMPLETEM0) * scalePixelPerRangeM );
	CHECKSCRCOORDM
	if (w >= SCREENWIDTHM) return (SCREENWIDTHM-1);
	if (w <= 0) return 0;

	return w;
	#endif

	LOGMSG("Error. scrcoord_as_lowerleft not defined for number type\n");
	exit(99);
}

inline int32_t scrcoord_as_lowerleftMR(const NTYP& a) {
	int32_t w;
	
	#ifdef _DOUBLE
	if (a <= COMPLETEM0) return 0;
	if (a >= COMPLETEM1) return (SCREENWIDTHM-1);
	// as in scrcoord_as_lowerleftJR
	double diff=(a - COMPLETEM0) * scalePixelPerRangeM;
	double fl=floor(diff);
	w=(int)fl;
	CHECKSCRCOORDM
	if (diff == fl) w--;
	if (w >= SCREENWIDTHM) return (SCREENWIDTHM-1);
	if (w <= 0) return 0;
	
	return w;
	#endif

	#ifdef _LONGDOUBLE
	if (a <= COMPLETEM0) return 0;
	if (a >= COMPLETEM1) return (SCREENWIDTHM-1);
	// as in scrcoord_as_lowerleftJR
	double diff=(a - COMPLETEM0) * scalePixelPerRangeM;
	double fl=floor(diff);
	w=(int)fl;
	CHECKSCRCOORDM
	if (diff == fl) w--;
	if (w >= SCREENWIDTHM) return (SCREENWIDTHM-1);
	if (w <= 0) return 0;
	
	return w;
	#endif

	LOGMSG("Error. scrcoord_as_lowerleft not defined for number type\n");
	exit(99);
}

inline void minimaxdAB(double& mi,double& ma,const double a,const double b) {
	if (a < b) { mi=a; ma=b; } else { mi=b; ma=a; }
}

inline void minimaxdABCD(double& mi,double& ma,const double a,const double b,const double c,const double d) {
	double miab,micd,maab,macd;
	minimaxdAB(miab,maab,a,b);
	minimaxdAB(micd,macd,c,d);
	
	if (miab < micd) mi=miab; else mi=micd;
	if (maab > macd) ma=maab; else ma=macd;
}

inline void minimaxldAB(long double& mi,long double& ma,const long double a,const long double b) {
	if (a < b) { mi=a; ma=b; } else { mi=b; ma=a; }
}

inline void minimaxldABCD(long double& mi,long double& ma,const long double a,const long double b,const long double c,const long double d) {
	long double miab,micd,maab,macd;
	minimaxldAB(miab,maab,a,b);
	minimaxldAB(micd,macd,c,d);
	
	if (miab < micd) mi=miab; else mi=micd;
	if (maab > macd) ma=maab; else ma=macd;
}

#define Z2C(NUMTYP,MINMAX2,MINMAX4) \
{\
	NUMTYP mi1,ma1;\
	MINMAX2(mi1,ma1,A.x0*A.x0,A.x1*A.x1);\
	NUMTYP mi2,ma2;\
	MINMAX2(mi2,ma2,A.y0*A.y0,A.y1*A.y1);\
	NUMTYP mi3,ma3;\
	IAMUL_SIGN(mi3,ma3,A.x0,A.x1,A.y0,A.y1,MINMAX4)\
	\
	fA.x0=jset.seedC0re+mi1-ma2;\
	fA.x1=jset.seedC1re+ma1-mi2;\
	fA.y0=jset.seedC0im+2*mi3;\
	fA.y1=jset.seedC1im+2*ma3;\
	return;\
}
	
void getBoundingBoxfA_z2c(JuliaSet& jset,PlaneRect& A,PlaneRect& fA) {
	// uses a straightforward interval extension of the component
	// functions of the complex z^2+c
	// (x+i*y)^2+(d+e*i)
	// result_x = fA.x0, fA.x1
	// result_y = fA.y0, fA.y1
	
	#ifdef _DOUBLE
	Z2C(double,minimaxdAB,minimaxdABCD)
	#endif
	
	#ifdef _LONGDOUBLE
	Z2C(long double,minimaxldAB,minimaxldABCD)
	#endif

	LOGMSG("Error. z2c func not defined for current number type\n");
	exit(99);
}

// z^4c

#define Z4C(NUMTYP,MINMAX2,MINMAX4) \
{\
	NUMTYP x02=A.x0*A.x0;\
	NUMTYP x03=x02*A.x0;\
	NUMTYP x04=x02*x02;\
	NUMTYP y02=A.y0*A.y0;\
	NUMTYP y03=y02*A.y0;\
	NUMTYP y04=y02*y02;\
	NUMTYP x12=A.x1*A.x1;\
	NUMTYP x13=x12*A.x1;\
	NUMTYP x14=x12*x12;\
	NUMTYP y12=A.y1*A.y1;\
	NUMTYP y13=y12*A.y1;\
	NUMTYP y14=y12*y12;\
	NUMTYP mi1,ma1;\
	IAMUL_SIGN(mi1,ma1,A.x0,A.x1,y03,y13,MINMAX4)\
	NUMTYP mi2,ma2;\
	IAMUL_SIGN(mi2,ma2,A.y0,A.y1,x03,x13,MINMAX4)\
	NUMTYP mi3,ma3;\
	MINMAX2(mi3,ma3,x02,x12);\
	NUMTYP mi4,ma4;\
	MINMAX2(mi4,ma4,y02,y12);\
	NUMTYP mi5,ma5;\
	MINMAX2(mi5,ma5,x04,x14);\
	NUMTYP mi6,ma6;\
	MINMAX2(mi6,ma6,y04,y14);\
	NUMTYP mi7,ma7;\
	IAMUL_SIGN(mi7,ma7,mi3,ma3,mi4,ma4,MINMAX4)\
	\
	fA.x0=mi6-(6*ma7)+mi5+jset.seedC0re;\
	fA.x1=ma6-(6*mi7)+ma5+jset.seedC1re;\
	\
	fA.y0=jset.seedC0im-(4*ma1)+4*mi2;\
	fA.y1=jset.seedC1im-(4*mi1)+4*ma2;\
	\
	return;\
}

void getBoundingBoxfA_z4c(JuliaSet& jset,PlaneRect& A,PlaneRect& fA) {
	#ifdef _DOUBLE
	Z4C(double,minimaxdAB,minimaxdABCD);
	#endif

	#ifdef _LONGDOUBLE
	Z4C(long double,minimaxldAB,minimaxldABCD);
	#endif
	
	LOGMSG("Error. z4c not supported in current number type.\n");
	exit(99);
};

void setPaletteTo(TiledPlane& md) {
	// image colors
	for(int32_t i=0;i<256;i++) md.setPaletteRGB(i,255,0,0);
	md.setPaletteRGB(MTILE_GRAY,127,127,127);
	md.setPaletteRGB(MTILE_BLACK,0,0,0);
	md.setPaletteRGB(MTILE_WHITE,255,255,255);
	md.setPaletteRGB(MTILE_GRAY_DONE,255,255,0);
	md.setPaletteRGB(MTILE_CHECKFORBLACK,0,0,255);
}

inline void minimaxQDAB(
	__float128& mi,__float128& ma,
	const __float128 a,const __float128 b
) {
	if (a < b) { mi=a; ma=b; } else { mi=b; ma=a; }
}

inline void minimaxQDABCD(
	__float128& mi,__float128& ma,
	const __float128 a,const __float128 b,
	const __float128 c,const __float128 d
) {
	__float128 miab,micd,maab,macd;
	minimaxQDAB(miab,maab,a,b);
	minimaxQDAB(micd,macd,c,d);
	
	if (miab < micd) mi=miab; else mi=micd;
	if (maab > macd) ma=maab; else ma=macd;
}

// period-4 component:Stephenson
// from "Some Explicit Formulas for the Iteration of
// Rational Functions" by Wolf Jung
// solution for complex mu
//    c^6 + 3 c^5 + (mu + 3) c^4 + (mu + 3) c^3 + (2 - mu - mu^2) c^2 + (1-mu)^3 = 0
// mit |2^4*mu| < 1

int8_t MTileInPeriod4_z2c(PlaneRect& Mtile) {
	#ifdef _USESTEPHENSONTEST
	// fast pre-test for components
	// does not need to be accurate
	// if it excludes too much => MTile will be analyzed via Orbit construction

	int32_t inone=0;
	
	if (
		(Mtile.x0 >= ((double)-29.0/128.0)) &&
		(Mtile.x1 <= ((double)-11.0/128.0)) &&
		(Mtile.y0 >= ((double)-141.0/128.0)) &&
		(Mtile.y1 <= ((double)-123.0/128.0))
	) {
		inone=1;
	}

	if (
		(Mtile.x0 >= ((double)22.0/128.0)) &&
		(Mtile.x1 <= ((double)50.0/128.0)) &&
        (Mtile.y0 >= ((double)-82.0/128.0)) &&
        (Mtile.y1 <= ((double)-54.0/128.0))
	) {
		inone=1;
	}

    if (
		(Mtile.x0 >= ((double)-183.0/128.0)) &&
		(Mtile.x1 <= ((double)-152.0/128.0)) &&
		(Mtile.y0 >= ((double)-16.0/128.0)) &&
		(Mtile.y1 <= ((double)16.0/128.0))) 
	{
		inone=1;
	}

    if (
		(Mtile.x0 >= ((double)-257.0/128.0)) &&
		(Mtile.x1 <= ((double)-240.0/128.0)) &&
		(Mtile.y0 >= ((double)-8.0/128.0)) &&
		(Mtile.y1 <= ((double)8.0/128.0))
	) {
		inone=1;
	}

    if (
		(Mtile.x0 >= ((double)22.0/128.0)) &&
		(Mtile.x1 <= ((double)50.0/128.0)) &&
        (Mtile.y0 >= ((double)54.0/128.0)) &&
        (Mtile.y1 <= ((double)82.0/128.0))
	) {
		inone=1;
	}
		
	if (
		(Mtile.x0 >= ((double)-29.0/128.0)) &&
		(Mtile.x1 <= ((double)-11.0/128.0)) &&
		(Mtile.y0 >= ((double)123.0/128.0)) &&
		(Mtile.y1 <= ((double)141.0/128.0))
	) {
		inone=1;
	}
	
	if (inone <= 0) {
		return INP123_MAYBE; // do NOT return OUTSIDE, as region might be too small
	}

	// initializiation: see comment at typedef cplxia
	ia cr=ia(Mtile.x0,Mtile.x1);
	ia ci=ia(Mtile.y0,Mtile.y1);
	cplxia c=cplxia(cr,ci);

	cplxia c2=c*c;
	cplxia c3=c2*c;
	cplxia c4=c2*c2;
	cplxia c5=c3*c2;
	cplxia c6=c3*c3;
	cplxia tmp1=SETVALUE(-1,0);
	cplxia tmp2=SETVALUE(3,0);
	cplxia tmp3=sqrt(tmp2)*SETVALUE(0,1);
	cplxia tmp4=(tmp1)/2-(tmp3)/2;
	cplxia tmp5=c+2;
	cplxia tmp6=64*c3+144*c2+108*c+135;
	cplxia tmp7=c3*(tmp5)*sqrt(tmp6);
	cplxia tmp9=2*sqrt(SETVALUE(27,0)); //2*3^(tmp8);
	cplxia tmp10=c2-3;
	cplxia tmp11=(tmp1)*(tmp10)*tmp10*tmp10;
	cplxia tmp12=-c4;
	cplxia tmp13=(tmp12)-c3+c2+3;
	cplxia tmp14=-c6;
	cplxia tmp15=(tmp14)-3*c5-3*c4-3*c3-2*c2-1;
	cplxia tmp16=(tmp10)*(tmp13)-3*(tmp15);
	cplxia tmp17=(tmp7)/(tmp9)+(tmp11)/27+(tmp16)/6;
	cplxia tmp18=SETVALUE(1,0)/SETVALUE(3,0);
	cplxia tmp19=(tmp1)*(tmp10)*tmp10;
	cplxia tmp20=(tmp19)/9+(tmp13)/3;
	cplxia tmp21=(tmp3)/2+(tmp1)/2;
	cplxia tmp22=(tmp20)*(tmp21);
	cplxia tmp23=(tmp1)*(tmp10);
	cplxia tmp24=(tmp20)*(tmp4);
	cplxia tmp17hoch18=exp(tmp18*log(tmp17));
	
	cplxia mu1=(tmp4)*tmp17hoch18-(tmp22)/tmp17hoch18+(tmp23)/3;
	cplxia mu2=(tmp21)*tmp17hoch18-(tmp24)/tmp17hoch18+(tmp23)/3;
	cplxia mu3=tmp17hoch18-(tmp20)/tmp17hoch18+(tmp23)/3;

	ia lambda1=abs(16*mu1);
	ia lambda2=abs(16*mu2);
	ia lambda3=abs(16*mu3);
	
	// if one mu is STRICTLY < 1=> return INP123_INSIDE
	if (
		(lambda1.upper() < 1.0) ||
		(lambda2.upper() < 1.0) ||
		(lambda3.upper() < 1.0)
	) return INP123_INSIDE;
	#endif
	
	// not determinable
	return INP123_MAYBE;
}

// check via closed formulas, currently P3 by Stephenson
// the 3 period-3 componentns lie in:
// north period-3-bulb:
//		real: -18/64..  2/64
//		imag: -58/64..-37/64
// cardioid minibrot
//		real: -118/64..-107/64
//		imag:   -5/64..   5/64
// south period-3 bulb
//		real: -18/64..  2/64
//		imag:  37/64.. 58/64
// values were derived from a pojnt-sampled version of the
// Stephenson formula (not error-controlled)
// if the region is too large => checking is fine
// if the region is too small => not all will be marked ebforehand
// but the rest will then be analyzed via orbit construction
// NOTE: Too small a region wwre NOT applicable for
// period1,2 as there all Mtiles therein must be removed
// as their area ius added externally, here the area of p3
// is still only added by Mtile counting
// result: INP123_INSIDE => mathematically guaranteed
// result: MAYBE => can be inside, outside or straddling

int8_t MTileInPeriod3_z2c(PlaneRect& Mtile) {
	#ifdef _USESTEPHENSONTEST
	// complex c is in any period-3 component, iff
	// c^3+2*c^2+(1-mu)*c+(1-mu)^2 = 0
	// has a complex solution mu which satisfies |8*mu| < 1
	// mu=1/2 * (2 + c +- sqrt(-c^2 * (7 + 4 * c)))

	// the mu interval DEFINITELY contains
	// a solution to the c^3..equation for every c in
	// the input rectangle MTile
	// hance if the multiplier-range of mu is STRICTLY
	// below 1, so is the multiplier of every single c in the Mtile 
	// so the whole Mtile is INSIDE a period-3 component
	
	int32_t inone=0;
	// pre-test for the regions where the parts can lie at all
	// south period-3-bulb:
	//		real: -18/64..  2/64
	//		imag: -58/64..-37/64
	if (
			(Mtile.x0 >= ((double)-18.0/64.0)) &&
			(Mtile.x1 <= ((double)  2.0/64.0)) &&
			(Mtile.y0 >= ((double)-58.0/64.0)) &&
			(Mtile.y1 <= ((double)-37.0/64.0))
	) {
		inone=1;
	}

	// cardioid minibrot
	//		real: -118/64..-107/64
	//		imag:   -5/64..   5/64
	if (inone <= 0) {
		if (
			(Mtile.x0 >= ((double)-118.0/64.0)) &&
			(Mtile.x1 <= ((double)-107.0/64.0)) &&
			(Mtile.y0 >= ((double)  -5.0/64.0)) &&
			(Mtile.y1 <= ((double)   5.0/64.0))
		) {
			inone=1;
		}
	}

	// north period-3 bulb
	//		real: -18/64..  2/64
	//		imag:  37/64.. 58/64
	if (inone <= 0) {
		if (
			(Mtile.x0 >= ((double)-18.0/64.0)) &&
			(Mtile.x1 <= ((double)  2.0/64.0)) &&
			(Mtile.y0 >= ((double) 37.0/64.0)) &&
			(Mtile.y1 <= ((double) 58.0/64.0))
		) {
			inone=1;
		}
	}
	
	if (inone <= 0) {
		return INP123_MAYBE; // do NOT return OUTSIDE, as region might be too small
	}
	
	// mu=1/2 * (2 + c +- sqrt(-c^2 * (7 + 4 * c)))

	ia cr=ia(Mtile.x0,Mtile.x1);
	ia ci=ia (Mtile.y0,Mtile.y1);
	cplxia c=cplxia(cr,ci);

	// maxima version 5.42.2
	// output automatically parsed and
	// tansformed into sequential series of temporary variables
	
	// not optimized
	
	cplxia tmp1=-4*c;
	cplxia tmp2=(tmp1)-7;
	cplxia tmp3=sqrt(tmp2)*c-c-2;
	cplxia tmp4=sqrt(tmp2)*c+c+2;

	cplxia mu1=-(tmp3)/2;
	cplxia mu2=(tmp4)/2;

	ia lambda1=abs(8*mu1);
	ia lambda2=abs(8*mu2);
	
	// if one mu is STRICTLY < 1=> return INP123_INSIDE
	if (
		(lambda1.upper() < 1.0) ||
		(lambda2.upper() < 1.0)
	) return INP123_INSIDE;
	#endif
	
	// not determinable
	return INP123_MAYBE;
}

int8_t MTileInMainCardioid_z2c(PlaneRect& A) {
	// using interval arithmetics to evaluate
	// 256*x^4+512*x^2*y^2-96*x^2+32*x+256*y^4-96*y^2-3 < 0
	// with x=[A.x0..A.x1] and y=[A.y0..A.y1]
	
	__float128 x02=(__float128)A.x0*(__float128)A.x0;
	__float128 x04=x02*x02;
	__float128 y02=(__float128)A.y0*(__float128)A.y0;
	__float128 y04=y02*y02;
	__float128 x12=(__float128)A.x1*(__float128)A.x1;
	__float128 x14=x12*x12;
	__float128 y12=(__float128)A.y1*(__float128)A.y1;
	__float128 y14=y12*y12;
	__float128 mi1,ma1;
	minimaxQDAB(mi1,ma1,x02,x12);
	__float128 mi2,ma2;
	minimaxQDAB(mi2,ma2,y02,y12);
	__float128 mi3,ma3;
	minimaxQDAB(mi3,ma3,x04,x14);
	__float128 mi4,ma4;
	minimaxQDAB(mi4,ma4,y04,y14);
	__float128 mi5,ma5;
	minimaxQDABCD(mi5,ma5,mi1*mi2,mi1*ma2,ma1*mi2,ma1*ma2);

	__float128 test_lower=256*mi3+512*mi5-96*ma1+32*(__float128)A.x0+256*mi4-96*ma2-(__float128)3.0;
	__float128 test_upper=256*ma3+512*ma5-96*mi1+32*(__float128)A.x1+256*ma4-96*mi2-(__float128)3.0;
	
	// test_upper < 0 => Mtile A lies fully /INSIDE/ main cardioid
	if (test_upper < 0.0) return INP123_INSIDE;
	
	// test_lower > 0 => Mtile A lies fully /OUTSIDE/ main cardioid
	if (test_lower > 0.0) return INP123_OUTSIDE;
	
	// else: not determinable via IA
	return INP123_MAYBE;
}

int8_t MPointInMainCardioid_z2c(const NTYP px,const NTYP py) {
	/*
		https://en.wikibooks.org/wiki/Fractals/Iterations_in_the_complex_plane/Mandelbrot_set/mandelbrot
		
		q=((x-0.25)^2+y^2) 
		q*(q+(x-0.25)) < 0.25*y^2
		((x-0.25)^2+y^2)*(((x-0.25)^2+y^2)+(x-1/4)) - 1/4*y^2 < 0
		
		quick check: rectangular region encompassing the main cardioid
		
		solve -3/256+x/8-3*x^2/8-3*y^2/8+x^4+2*x^2*y^2+y^4 < 0
		for x as input variable and as well for y as input variable
		to get (e.g. using WolframAlpha):
		
		main cardioid lies in rectangle [-3/4..3/8] x [-11/16..11/16]*i
		
	*/
	
	if (px < (double)(-0.75)) return 0;
	if (px > (double)( 0.375)) return 0;
	if (py < (double)(-0.6875)) return 0;
	if (py > (double)( 0.6875)) return 0;
	
	// 256*x^4+512*x^2*y^2-96*x^2+32*x+256*y^4-96*y^2-3 < 0
	__float128 x2=(__float128)px*(__float128)px;
	__float128 y2=(__float128)py*(__float128)py;
	
	if (
		(
			256.0*x2*x2+512.0*x2*y2-96.0*x2+32*(__float128)px+256.0*y2*y2-96.0*y2-(__float128)3.0
		)
		<
		(__float128)0.0
	) return 1;
	
	return 0;
}

int8_t MTileInPeriod2Bulb_z2c(PlaneRect& A) {
	// period-2 bulb as perfect circle
	// all 4 corner coordinates have to lie inside
	int8_t drin=0;
	if (MPointInPeriod2Bulb_z2c(A.x0,A.y0) > 0) drin++;
	if (MPointInPeriod2Bulb_z2c(A.x1,A.y0) > 0) drin++;
	if (MPointInPeriod2Bulb_z2c(A.x0,A.y1) > 0) drin++;
	if (MPointInPeriod2Bulb_z2c(A.x1,A.y1) > 0) drin++;
	
	if (drin == 0) return INP123_OUTSIDE;
	if (drin == 4) return INP123_INSIDE;
	
	return INP123_MAYBE;
}

int8_t MPointInPeriod2Bulb_z2c(const NTYP px,const NTYP py) {
	// circle of radius 0.25 around -1+0i
	// first check rectangle for faster result
	
	if (px < (double)(-1.25)) return 0;
	if (px > (double)(-0.75)) return 0;
	if (py > (double)( 0.25)) return 0;
	if (py < (double)(-0.25)) return 0;

	// (x-(-1))^2+(y-0)^2 < 0.25^2
	// x^2+2*x+1+y^2 < 1/16
	// x^2+2*x+y^2+15/16 < 0
	if ( (px*px+2.0*px+py*py+15.0/16.0) < 0.0 ) return 1;
	
	return 0;
}

void testC_interior(JuliaSet& jset,const int32_t ax,const int32_t ay) {
	jset.seedC0re=ax*scaleRangePerPixelM + COMPLETEM0;
	jset.seedC1re=jset.seedC0re + scaleRangePerPixelM;
	jset.seedC0im=ay*scaleRangePerPixelM + COMPLETEM0;
	jset.seedC1im=jset.seedC0im + scaleRangePerPixelM;
	jset.px=ax;
	jset.py=ay;
	
	testC_interior(jset);
}

void ps_func_z2c(
	const double cx,const double cy,
	const double reQ,const double imQ,
	Double2& z
) {
	z.y=2*z.x*z.y+cy;
	z.x=reQ-imQ+cx;
}

// construct a point-sampled orbit of the origin for a speicific
// Julia set: used to numerically find periodic points and use
// those as a Jtile-queue start. If falsely identified pp's
// are used does not compromosie mathematical validity of the
// area. Idea: If the origin needs a couple of iterations before
// entering an immediate basin, those preperiodic steps can be
// skipped and their bbx's need not be computed if one can start
// in an immediate basin directly
int8_t ps_periodic(JuliaSet& jset,double& ergppx,double& ergppy) {
	double cx=0.5*(jset.seedC0re+jset.seedC1re);
	double cy=0.5*(jset.seedC0im+jset.seedC1im);
	
	Double2 z;
	z.x=0.0;
	z.y=0.0;
	for(int32_t i=0;i<PS_MAXIT;i++) {
		double reQ=z.x*z.x;
		double imQ=z.y*z.y;
		if ((reQ+imQ) > 4.0) return 0; // no period present
		
		jset.psorbit[i].x=z.x;
		jset.psorbit[i].y=z.y;

		ps_func(cx,cy,reQ,imQ,z);
	} // i
	
	// is a cycle present? Check last orbit entry whether it
	// occurs previously
	// only check for last 200 values, as of now, the longest
	// detected cycle using the Figueiredo algorithm in my
	// implmenetation was 112 (as of 05/2020)
	for(int32_t i=(PS_MAXIT-1);i>=(PS_MAXIT-PS_MAXCYCLELEN);i--) {
		double dx=jset.psorbit[PS_MAXIT-1].x-jset.psorbit[i].x;
		double dy=jset.psorbit[PS_MAXIT-1].y-jset.psorbit[i].y;
		
		if (
			(fabs(dx) < PS_EQUALITY) &&
			(fabs(dy) < PS_EQUALITY)
		) {
			// found a cycle
			ergppx=jset.psorbit[i].x;
			ergppy=jset.psorbit[i].y;
			return 1;
		}
	}
	
	return 0;
}

int8_t isOnGridJ(const double aw) {
	// is px a dyadic fraction?
	// works as scrcoord_as_lowerleftJR
	// outside grid points are irrelevant
	if (aw < COMPLETEJ0) return 0;
	if (aw > COMPLETEJ1) return 0;
	// converting to (almost) memory coordinate
	// NOTE: numbertyp sufficiency is not a concern here
	// return the wrong answer only affects the number of
	// Jtiles in the queue and might return GRAY but it will
	// not have the calling routien return falsely BLACK
	double diff=(aw - COMPLETEJ0) * scalePixelPerRangeJ;
	double fl=floor(diff);
	if (diff == fl) return 1; // no fractional part left => on grid
	
	return 0;
}

void testC_interior(JuliaSet& jset) {
	// reliable routine based on cell mapping and interval arithemtics
	jset.ctrtestccalled++;
	
	// adjust UNVISIED etc.
	jset.UNVISITED=jset.FOLLOWED+1;
	jset.TOFOLLOW=jset.UNVISITED+1;
	jset.FOLLOWED=jset.TOFOLLOW+1;
	
	if (
		(jset.FOLLOWED >= 250)
	) {
		// now Juliaset data has to be reset
		jset.UNVISITED=jset.UNVISITED0; 
		jset.TOFOLLOW=jset.UNVISITED+1;
		jset.FOLLOWED=jset.TOFOLLOW+1;
		jset.juliaimg.fillrect(0,0,jset.juliaimg.xlen-1,jset.juliaimg.ylen-1,jset.UNVISITED);
	}
	
	// orbit is array of STREAKS
	// streak is tripel (x0,b,y)
	// check pixel x0 <= x <= (x0+b-1): (x,y)
	// pop back (x0+1,b-1,y) if (b-1)>0
	// if (x,y) not UNVISITED => skip
	// construct bbx => add all bbx-scr unvisited pixels
	
	jset.streaks.fastEmpty();
	ScreenRect allvisited;
	allvisited.x0=allvisited.y0=SCREENWIDTHJ;
	allvisited.x1=allvisited.y1=0;

	#define ADDST(XX0,XX1,YY) \
	{\
		if ( (XX0) < allvisited.x0) allvisited.x0=XX0;\
		if ( (XX1) > allvisited.x1) allvisited.x1=XX1;\
		if ( (YY) < allvisited.y0) allvisited.y0=YY;\
		if ( (YY) > allvisited.y1) allvisited.y1=YY;\
		int32_t x0=(XX0);\
		int32_t laenge=(XX1)-(XX0) + 1;\
		\
		while (laenge>0) {\
			int32_t ll=laenge;\
			if (ll>MAXBREITE) ll=MAXBREITE;\
			if (jset.streaks.push(x0,ll,YY) <= 0) {\
				listwasfull=1;\
			}\
			\
			jset.juliaimg.lineHorizVert(x0,YY,x0+ll-1,YY,jset.TOFOLLOW);\
			jset.ctrjtilesvisited += ll;\
			laenge -= ll;\
			x0 += ll;\
		}\
	}
	
	ScreenRect judge;
	int8_t listwasfull=0;
	int8_t ppfound=0;
	#ifdef _USE_PS_PERIODIC
	// use any (doesn't matter which one) point-sampled derived
	// periodic point, as if a cycle is present, the critical
	// point will be attracted to it in the quadratic case,
	// so the invariant set of Jtiles that only has paths amongst
	// itself, will necessarily cover all periodic points
	// Usually there's only one Jtile harbouring that periodic point,
	// but it may happen, that it is a grid corner point and hence
	// part of more than one Jtile. Then conservatively all 4 Jtiles
	// ahving the pp as a corner are put into the queue
	// NOTE: putting the wrong Jtiles in the queue does not
	// compromise the mathematical validity of the area or the
	// resulting image. If the starting Jtiles are in a detectable
	// interior region, it will find the cycle and hence
	// correctly returns JTILE_BLACK. If it lies in the exterior
	// of the Jset, the routine will return GRAY, so the Mtile
	// will not be counted, for a lower area bound not a problam.
	double px,py;
	// check the periodicity with the middle point
	// (but no special reason for that)
	if (ps_periodic(jset,px,py) > 0) {
		ppfound=1;
		jset.ctrppfound++;
		int8_t gx=isOnGridJ(px);
		int8_t gy=isOnGridJ(px);
		if ( (gx>0) || (gy>0) > 0) {
			jset.ctrppongrid++;
			if (gx>0) {
				judge.x1=scrcoord_as_lowerleftJL(px);;
				judge.x0=judge.x1-1;
			} else {
				judge.x0=judge.x1=scrcoord_as_lowerleftJL(px);;
			}
			
			if (gy>0) {
				judge.y1=scrcoord_as_lowerleftJL(px);;
				judge.y0=judge.y1-1;
			} else {
				judge.y0=judge.y1=scrcoord_as_lowerleftJL(px);;
			}
		} else {
			judge.x0=judge.x1=scrcoord_as_lowerleftJL(px);
			judge.y0=judge.y1=scrcoord_as_lowerleftJL(py);
		}
	} else {
		jset.ctrppfailed++;
	}
	#endif
	
	if (ppfound <= 0) {
		// as the origin is a grid point
		// use all 4 pixels that share the origin 
		int32_t nullnull=scrcoord_as_lowerleftJL(0.0);
		// nullnull is the Jtile where the origin is the lower left corner
		// all 4 Jtiles harbouring the origin as one corner
		judge.x0=nullnull-1;
		judge.x1=nullnull;
		judge.y0=nullnull-1;
		judge.y1=nullnull;
	}
	
	for(int32_t yj=judge.y0;yj<=judge.y1;yj++) {
		ADDST(judge.x0,judge.x1,yj);
	}

	Streak aw;
	int8_t outsidescreen=0;
	int8_t firstoccured=1;
	
	PlaneRect A,bbxfA;
	
	while (1) {
		// in case StreakList was full at some point
		// one has to go over the image to follow all
		// TOFOLLOW Jtiles
		
		while (jset.streaks.pop(aw) > 0) {
			if (jset.streaks.anz > jset.ctrmaxinsl) {
				jset.ctrmaxinsl=jset.streaks.anz;
			}
			A.y0=aw.y*scaleRangePerPixelJ + COMPLETEJ0;
			A.y1=A.y0+scaleRangePerPixelJ;
			// check whole streaks as then bbx are in a similar
			// region mostly and juliaimg-part is in cache
			for(int32_t x=(aw.x0+aw.breite-1);x>=aw.x0;x--) {
				if (jset.juliaimg.getPunkt(x,aw.y) != jset.TOFOLLOW) {
					continue;
				}
				
				jset.juliaimg.setPunkt(x,aw.y,jset.FOLLOWED);
				// compute bbx
				A.x0=x*scaleRangePerPixelJ + COMPLETEJ0;
				A.x1=A.x0+scaleRangePerPixelJ;
			
				getBoundingBoxfA(jset,A,bbxfA);
				
				// is bbx fully in COMPLETE-square ?
				if (
					!(
						(COMPLETEJ0 <= bbxfA.x0) &&
						(bbxfA.x1 <= COMPLETEJ1) &&
						(COMPLETEJ0 <= bbxfA.y0) &&
						(bbxfA.y1 <= COMPLETEJ1)
					)
				) {
					// partially or fully outside
					// ciritcal orbit not bounded in current resolution
					outsidescreen=1;
					jset.erg=JTILE_GRAY;
					return;
				}
				
				// bbx is fully in JuliaImg-Screen
				
				ScreenRect scr;
				scr.x0=scrcoord_as_lowerleftJL(bbxfA.x0);
				scr.x1=scrcoord_as_lowerleftJR(bbxfA.x1);
				scr.y0=scrcoord_as_lowerleftJL(bbxfA.y0);
				scr.y1=scrcoord_as_lowerleftJR(bbxfA.y1);
				// is already trimmed to screen
				
				// add unvisited Jtiles as streaks
				for(int32_t ty=scr.y0;ty<=scr.y1;ty++) {
					int32_t startx=-1,endx=-1;
					for(int32_t tx=scr.x0;tx<=scr.x1;tx++) {
						
						if (jset.juliaimg.getPunkt(tx,ty) <= jset.UNVISITED) {
							if (startx<0) startx=tx;
							endx=tx;
						} else {
							if (startx>=0) {
								ADDST(startx,endx,ty);
							}
							startx=endx=-1;
						}
					} // tx
						
					if (startx>=0) {
						ADDST(startx,endx,ty);
						startx=endx=-1;
					}
						
				} // ty
				
			} // x in streak
				
		} // pop
	
		// streaklist is now empty
	
		// streaklist was never full => all Jtiles
		// in iubrot were visited and followed
		if (listwasfull <= 0) break;
		
		// StreakList was full at some point
		// juliaimg can still contain jset.TOFOLLOW Jtiles
		// grab those and put in the list as long as
		// posulble and go over the list and image again
		// and again until both are done (StreakList is then
		// empty and image has no mroe jset.TOFOLLOW entries)
		
		if (firstoccured>0) {
			LOGMSG("INFO: listfullevent occured ");
			firstoccured=0;
		}

		// race conditions do not impact outcome as solely
		// the value 1 is ever written into variable
		// and it is read only outside parallel-running methods
		listfulleventoccured=1;

		int8_t raus=0;
		int8_t pushed=0;
		
		for(int32_t y=0;y<SCREENWIDTHJ;y++) {
			
			for(int32_t x=0;x<SCREENWIDTHJ;x++) {
				if (jset.juliaimg.getPunkt(x,y) == jset.TOFOLLOW) {
					pushed=1;
					if (jset.streaks.push(x,1,y) <= 0) {
						raus=1;
						break;
					}
				}
			} // x
			
			if (raus>0) break;
		} // y
		
		if (pushed <= 0) break; // no more jset.TOFOLLOW found in image
		
	} // while (1)
	
	if (outsidescreen <= 0) {
		jset.ctrsettoblack++;
		jset.erg=JTILE_BLACK;
		return;
	}
	
	jset.erg=JTILE_GRAY;
}

int8_t MPixelIsGray(const int32_t af) {
	if (
		(af == MTILE_GRAY) ||
		(af == MTILE_GRAY_DONE) ||
		(af == MTILE_CHECKFORBLACK)
	) return 1;
	
	return 0;
}

int8_t readHCfile(HCenter& ahc,const int32_t aid) {
	// try to read file with SCREENWIDTHM
	char tt[2048];
	sprintf(tt,"hc%06i.pos",aid);
	FILE *f=fopen(tt,"rb");
	if (!f) return 0; // not present
	
	// as screen coordinates are stored in int32_t
	// any complex coordinate is at finest (RANGE1-RANGE0)/2^32
	// for 2 <= RANGE1-RANGE0 <= 1024: integer part is at most 10 bits
	// fractional part is at most 31 bits
	double dx,dy;
	fread(&dx,1,sizeof(dx),f);
	fread(&dy,1,sizeof(dy),f);
	fclose(f);
	
	// convert in currently used NTYP
	ahc.Ax0=dx;
	ahc.Ay0=dy;
	
	// current resolution available as file ?
	sprintf(tt,"hc%06i_%08i.data",aid,SCREENWIDTHM);
	if (ahc.msetregion.load(tt) <= 0) return 0;
	
	if (_RESETCOLORS>0) {
		// black, white are kept, all other colors are set to gray
		for(int32_t y=0;y<ahc.msetregion.ylen;y++) {
			for(int32_t x=0;x<ahc.msetregion.xlen;x++) {
				int32_t f=ahc.msetregion.getPunkt(x,y);
				if (
					(f != MTILE_BLACK) &&
					(f != MTILE_WHITE)
				) ahc.msetregion.setPunkt(x,y,MTILE_GRAY);
			} // x
		} // y
	}
	
	// adjust values
	ahc.hcid=aid;
	ahc.circumsq.x0=scrcoord_as_lowerleftML(ahc.Ax0);
	ahc.circumsq.y0=scrcoord_as_lowerleftML(ahc.Ay0);
	ahc.allocatewidth=ahc.msetregion.xlen;
	ahc.circumsq.x1=ahc.circumsq.x0+ahc.allocatewidth-1;
	ahc.circumsq.y1=ahc.circumsq.y0+ahc.allocatewidth-1;
	
	// will overflow at REFINEMENTM > 30
	ahc.centerpixelx=(ahc.circumsq.x0 + ahc.circumsq.x1) >> 1;
	ahc.centerpixely=(ahc.circumsq.y0 + ahc.circumsq.y1) >> 1;

	return 1;
}

void readHC(void) {
	FILE *f=fopen("_bulbs_center.values","rt");
	
	anzcenters=0;
	if (f) {
		printf("\nreading hyperbolic center file ... ");
		char tmp[2048];
		while (!feof(f)) {
			if (anzcenters > (MAXHYPERBOLICCENTERS)) {
				printf("too many centers. Only first %i used.\n",MAXHYPERBOLICCENTERS);
				break;
			}

			fgets(tmp,1000,f);
			if (tmp[0]=='.') break;
			if (tmp[0]=='/') continue;
		
			chomp(tmp);
			upper(tmp);
			
			double a,b,width;
			int32_t id;
			if (sscanf(tmp,"%i,%lf,%lf,%lf,",&id,&a,&b,&width) != 4) continue; // unidentifed line
			
			// if a,b is in MainCardioid or Period2Bulb => exclude
			int32_t tmpcx=scrcoord_as_lowerleftML(a);
			int32_t tmpcy=scrcoord_as_lowerleftML(b);
			PlaneRect bA;
			bA.x0=tmpcx*scaleRangePerPixelM+COMPLETEM0;
			bA.x1=bA.x0+scaleRangePerPixelM;
			bA.y0=tmpcy*scaleRangePerPixelM+COMPLETEM0;
			bA.y1=bA.y0+scaleRangePerPixelM;
			
			if (FUNC == FUNC_Z2C) {
				// the whole Mtile in which the approximate center
				// lies is not allowed to even intersect with the
				// main cardioid or period2
			
				if (
					(MTileInMainCardioid_z2c(bA) != INP123_OUTSIDE) ||
					(MTileInPeriod2Bulb_z2c(bA) != INP123_OUTSIDE)
				) continue;
			}

			// does a file for that hc exist
			if (readHCfile(centers[anzcenters],id) > 0) {
				// mark new period1,2 Mtiles
				printf(".");
				
				centers[anzcenters].donotanalyze=0;
				if (FUNC == FUNC_Z2C) {
					// currently as it is time-consuming, only
					// period3 checked if center is in a period3 comonent
					// as markP3set SETs definite P3-Mtiles, NOT setting
					// all possible can result in not the highest
					// lower area bound, but it does NOT compromose the
					// amthematical guarantee of the bound
					#ifdef _USESTEPHENSONTEST
					if (MTileInPeriod3_z2c(bA) > 0) {
						if (markP3set_z2c(centers[anzcenters]) > 0) {
							centers[anzcenters].donotanalyze=1;
						}
					} else
					if (MTileInPeriod4_z2c(bA) > 0) {
						if (markP4set_z2c(centers[anzcenters]) > 0) {
							centers[anzcenters].donotanalyze=1;
						}
					}
					#endif
					// as last step: remove everything P12
					markP12remove_z2c(centers[anzcenters]);
				} 
				
				centers[anzcenters].newlyfound=0;
				anzcenters++;
				
				continue;
			}
			
			// non-existent file
			// if circumferencing square around hc
			// of side length CIRCUMFERENCINGFACTOR*inscribed length
			// (this covers by assumption the whole hyperbolic omponent)
			// is fully in the upper half of the Mset
			// (positve imginary part) => discardedable exploting symmetry
			// straddling the x-axis is fine, such hc is used for
			// for analysis
			centers[anzcenters].hcid=id;
			centers[anzcenters].newlyfound=1;
			int32_t cx=scrcoord_as_lowerleftML(a);;
			int32_t cy=scrcoord_as_lowerleftML(b);;
			
			// make it divisible by 4 for saving raw data in format of a bitmap
			int32_t insqw=(int32_t)ceil(width / scaleRangePerPixelM);
			insqw=(int32_t)4*ceil( (double)insqw*0.25 );
			centers[anzcenters].allocatewidth=insqw*CIRCUMFERENCINGFACTOR;
			centers[anzcenters].circumsq.x0=
				cx - (centers[anzcenters].allocatewidth >> 1);
			centers[anzcenters].circumsq.x1=centers[anzcenters].circumsq.x0 + centers[anzcenters].allocatewidth-1;
			centers[anzcenters].circumsq.y0=
				cy - (centers[anzcenters].allocatewidth >> 1);
			centers[anzcenters].circumsq.y1=centers[anzcenters].circumsq.y0 + centers[anzcenters].allocatewidth-1;
			// calculate centerpixel identically as when loading data
			// attn: possible overflow if REFINEMENTM > 30
			centers[anzcenters].centerpixelx=(centers[anzcenters].circumsq.x0 + centers[anzcenters].circumsq.x1) >> 1;
			centers[anzcenters].centerpixely=(centers[anzcenters].circumsq.y0 + centers[anzcenters].circumsq.y1) >> 1;
			
			// symmetry
			if (SYMMETRY_AT_XAXIS>0) {
				// fully in upper half => discard
				if (centers[anzcenters].circumsq.y0 > (SCREENWIDTHM >> 1)) continue;
			}
			
			if (SYMMETRY_AT_YAXIS>0) {
				// fully in right half => discard
				if (centers[anzcenters].circumsq.x0 > (SCREENWIDTHM >> 1)) continue;
			}
			
			// test, if centerMtile can be judged via CM/IA
			// as being fully inside the Mandelbrot set
			juliasets[0].ptrhc=&centers[anzcenters];
			testC_interior(
				juliasets[0],
				centers[anzcenters].centerpixelx,
				centers[anzcenters].centerpixely
			);

			if (juliasets[0].erg != JTILE_BLACK) continue; // not an active center
			
			// calculate coordinate of lower left corner of lewer left Mtile in active region
			centers[anzcenters].Ax0=centers[anzcenters].circumsq.x0*scaleRangePerPixelM+COMPLETEM0;
			centers[anzcenters].Ay0=centers[anzcenters].circumsq.y0*scaleRangePerPixelM+COMPLETEM0;

			printf(".");
			
			// valid new region
			centers[anzcenters].initRegion();
			centers[anzcenters].setMtile(
				centers[anzcenters].centerpixelx,
				centers[anzcenters].centerpixely,
				MTILE_BLACK);
			centers[anzcenters].donotanalyze=0;
			
			// remove P1,2 in quadratic case as the area are added
			// externally, hence every possibly oaanalyzed Mtile
			// has to be set back to gray for not to count 
			// Mtiles multiple times
			if (FUNC == FUNC_Z2C) {
				markP12remove_z2c(centers[anzcenters]);
			} 
			
			// Stephenson formulae are not used at the
			// emerging level, as often near the hyperbolic center
			// the multiplier calculation contains zero in some
			// intermediate intervals which prohibit division
				
			anzcenters++;
			
		} // while 
		
		fclose(f);
		LOGMSG2("\n  %i active centers loaded",anzcenters);
	} else {
		printf("no hyperbolic centers found.\n");
		anzcenters=0;
	}

}

// struct StreakList
StreakList::StreakList() {
	anz=0;
	values=NULL;
}

void StreakList::allocate(int64_t& memused) {
	values=new Streak[MAXSTREAKS];
	memused += ( (int64_t)MAXSTREAKS*sizeof(Streak) );
	if (!values) {
		LOGMSG("Memory error. StreakList\n");
		exit(99);
	}
}

StreakList::~StreakList() {
	delete[] values;
}

void StreakList::fastEmpty(void) {
	anz=0;
}

int8_t StreakList::pop(Streak& aw) {
	if ( (anz==0) || (!values) ) return 0;
	
	aw.x0=values[anz-1].x0;
	aw.breite=values[anz-1].breite;
	aw.y=values[anz-1].y;
	
	anz--;
	
	return 1;
}

int8_t StreakList::push(const int32_t ax,const int32_t ab,const int32_t ay) {	
	if (!values) return 0;
	if (anz >= (MAXSTREAKS-8)) return 0;
	
	values[anz].x0=ax;
	values[anz].breite=ab;
	values[anz].y=ay;
	
	anz++;
	
	return 1;
}

// struct JuliaSet

JuliaSet::JuliaSet() {
}

void JuliaSet::initMemory(void) {
	juliaimg.setlenxy(SCREENWIDTHJ,SCREENWIDTHJ);
	jsetmemory += (int64_t)SCREENWIDTHJ*SCREENWIDTHJ;
	setPaletteTo(juliaimg);
	ctrmaxinsl=0;
	ctrjtilesvisited=0;
	juliaimg.fillrect(0,0,juliaimg.xlen-1,juliaimg.ylen-1,0);
	ctrsettoblack=0;
	ctrtestccalled=0;
	ctrppfound=0;
	ctrppfailed=0;
	ctrppongrid=0;
	UNVISITED=256;
	TOFOLLOW=257;
	FOLLOWED=258;
	UNVISITED0=8; // larger than SQUARE_xxx JSet values
	streaks.allocate(jsetmemory);
	#ifdef _USE_PS_PERIODIC
	psorbit=new Double2[PS_MAXIT];
	jsetmemory += (int64_t)PS_MAXIT*sizeof(Double2);
	#endif
}


// struct TiledPlane

void write2(FILE *f,const BYTE a,const BYTE b) {
	fwrite(&a,1,sizeof(a),f);
	fwrite(&b,1,sizeof(b),f);
}

void write4(FILE *f,const BYTE a,const BYTE b,const BYTE c,const BYTE d) {
	fwrite(&a,1,sizeof(a),f);
	fwrite(&b,1,sizeof(b),f);
	fwrite(&c,1,sizeof(c),f);
	fwrite(&d,1,sizeof(d),f);
}

int8_t TiledPlane::load(const char* afn) {
	// 8 bit Bitmap

	FILE *fbmp=fopen(afn,"rb");
	if (!fbmp) return 0;
	
	BYTE dummypuffer[4096];
	
	#define DUMMYREAD(NR) \
		fread(dummypuffer,NR,sizeof(BYTE),fbmp);

	DUMMYREAD(2)
	
	uint32_t off;
	uint32_t filelen;
			
	fread(&filelen,1,sizeof(filelen),fbmp);
	DUMMYREAD(4)
	fread(&off,1,sizeof(off),fbmp); // offset, ab da beginnen die PIXEL
	DUMMYREAD(4)
	
	uint32_t wx,wy;
	fread(&wx,sizeof(wx),1,fbmp);
	fread(&wy,sizeof(wy),1,fbmp);
	setlenxy(wx,wy);
	DUMMYREAD(2);
	uint16_t bitsperpixel;
	fread(&bitsperpixel,sizeof(bitsperpixel),1,fbmp);
	if (bitsperpixel != 8) {
		printf("Fehler. TiledPlane::read BitsPerPixel MUSS 8 sein.\n");
		exit(99);
	}
	
	DUMMYREAD(24)
	BYTE puffer[4];
	for(int32_t i=0;i<256;i++) {
		fread(puffer,4,sizeof(BYTE),fbmp);
		palette[i].B=puffer[0];
		palette[i].G=puffer[1];
		palette[i].R=puffer[2];
	}
	
	for(int32_t y=0;y<ylen;y++) {
		cmpY[y]=planemgr.getMemory(xlen);
		fread(cmpY[y],xlen,sizeof(BYTE),fbmp);
	}

	fclose(fbmp);
	
	return 1;	
}

void TiledPlane::save(const char* afn) {
	// the data is stored in the format of an 8 bit Bitmap
	
	FILE *fbmp=fopen(afn,"wb");
	write2(fbmp,66,77); // BM
	
	uint32_t off
		=		14 // FILEHeader
			+	40 // Bitmapheader
			+	256*4 // ColorPalette
		;
	
	// filelen will overflow if image width is too large
	// but external viewers 
	// can display the image nonetheless
	uint32_t filelen
			=	off
			+	(ylen*xlen);
		;
			
	fwrite(&filelen,1,sizeof(filelen),fbmp);
	write4(fbmp,0,0,0,0);
	fwrite(&off,1,sizeof(off),fbmp); // offset, ab da beginnen die PIXEL
	write4(fbmp,40,0,0,0);
	
	uint32_t w = xlen;
	fwrite(&w,sizeof(w),1,fbmp);
	w = ylen;
	fwrite(&w,sizeof(w),1,fbmp);
	write2(fbmp,1,0); 
	write2(fbmp,8,0); // 8 bits per pixel
	write4(fbmp,0,0,0,0);
	write4(fbmp,0,0,0,0);
	write4(fbmp,19,10,0,0);
	write4(fbmp,19,10,0,0);
	write4(fbmp,0,1,0,0); // number of palette entries
	write4(fbmp,0,0,0,0);
	BYTE puffer[4];
	for(int32_t i=0;i<256;i++) {
		puffer[0]=palette[i].B;
		puffer[1]=palette[i].G;
		puffer[2]=palette[i].R;
		puffer[3]=0;
		fwrite(puffer,4,sizeof(BYTE),fbmp);
	}
	
	for(int32_t y=0;y<ylen;y++) {
		fwrite(cmpY[y],xlen,sizeof(BYTE),fbmp);
	}

	fclose(fbmp);	
}

void TiledPlane::saveTwice(const char* afn) {
	// the data is stored in the format of an 8 bit Bitmap
	// every pixel is converted to a 2x2 grid of identical colored pixels
	// this doubled data is used for the next refinement level
	// as input
	
	FILE *fbmp=fopen(afn,"wb");
	write2(fbmp,66,77); // BM
	int32_t ylen2=2*ylen;
	int32_t xlen2=2*xlen;
	
	uint32_t off
		=		14 // FILEHeader
			+	40 // Bitmapheader
			+	256*4 // ColorPalette
		;
	
	// filelen will overflow if image width is too large
	// but external viewers 
	// can display the image nonetheless
	uint32_t filelen
			=	off
			+	(xlen2*ylen2);
		;
			
	fwrite(&filelen,1,sizeof(filelen),fbmp);
	write4(fbmp,0,0,0,0);
	fwrite(&off,1,sizeof(off),fbmp); // offset, ab da beginnen die PIXEL
	write4(fbmp,40,0,0,0);
	
	uint32_t w = xlen*2;
	fwrite(&w,sizeof(w),1,fbmp);
	w = ylen*2;
	fwrite(&w,sizeof(w),1,fbmp);
	write2(fbmp,1,0); 
	write2(fbmp,8,0); // 8 bits per pixel
	write4(fbmp,0,0,0,0);
	write4(fbmp,0,0,0,0);
	write4(fbmp,19,10,0,0);
	write4(fbmp,19,10,0,0);
	write4(fbmp,0,1,0,0); // number of palette entries
	write4(fbmp,0,0,0,0);
	BYTE puffer[4];
	for(int32_t i=0;i<256;i++) {
		puffer[0]=palette[i].B;
		puffer[1]=palette[i].G;
		puffer[2]=palette[i].R;
		puffer[3]=0;
		fwrite(puffer,4,sizeof(BYTE),fbmp);
	}
	
	BYTE *zeile=new BYTE[xlen*2];
	for(int32_t y=0;y<ylen;y++) {
		for(int32_t x=0;x<xlen;x++) {
			int32_t f;
			
			// only black/white are kept, everything else is
			// set to gray to be analyzed in next refinement
			if (cmpY[y][x] == MTILE_WHITE) f=MTILE_WHITE;
			else if (cmpY[y][x] == MTILE_BLACK) f=MTILE_BLACK;
			else f=MTILE_GRAY;
			
			zeile[2*x]=zeile[2*x+1]=f;
		} // x
		
		fwrite(zeile,xlen*2,sizeof(BYTE),fbmp);
		fwrite(zeile,xlen*2,sizeof(BYTE),fbmp);
	}

	fclose(fbmp);	
}

void TiledPlane::lineHorizVert(const int32_t ax,const int32_t ay,const int32_t bx,const int32_t by,const BYTE awert) {
	if (!cmpY) return;
	
	if (ax == bx) {
		// vertical line
		int32_t y0,y1;
		if (ay < by) { y0=ay; y1=by; } else { y0=by; y1=ay; }
		for(int32_t y=y0;y<=y1;y++) {
			cmpY[y][ax]=awert;
		} // y
	} else if (ay == by) {
		// horizontal line
		int32_t x0,x1;
		if (ax < bx) { x0=ax; x1=bx; } else { x0=bx; x1=ax; }
		for(int32_t x=x0;x<=x1;x++) {
			cmpY[ay][x]=awert;
		} // y
	} else {
		// geht nicht
		printf("TiledPlane: Diagonal line not implemented.\n");
		return;
	}
}

void TiledPlane::fillrect(const int32_t ax,const int32_t ay,const int32_t bx,const int32_t by,const BYTE aw) {
	int32_t lx=ax,rx=bx;
	if (ax > bx) { lx=bx; rx=ax; }
	int32_t ly=ay,ry=by;
	if (ay > by) { ly=by; ry=ay; }
	
	for(int32_t y=ly;y<=ry;y++) {
		for(int32_t x=lx;x<=rx;x++) {
			cmpY[y][x]=aw;
		}
	}
}

void TiledPlane::setlenxy(const int32_t ax,const int32_t ay) {
	// memory itself is not deallocated here
	// as setlenxy is not called more than once
	if ((xlen>0)&&(cmpY)) {
		delete[] cmpY;
	}
	
	cmpY=new PBYTE[ay];
	xlen=ax;
	ylen=ay;
	for(int32_t y=0;y<ylen;y++) {
		cmpY[y]=planemgr.getMemory(xlen);
	}
}

TiledPlane::TiledPlane() {
	xlen=ylen=0;
	memused=0;
	cmpY=NULL;
}

TiledPlane::~TiledPlane() {
	// memory itself is deallocated when destroying the memory managaer struct
	if ((xlen>0)&&(cmpY)) {
		delete[] cmpY;
	}
}

void TiledPlane::setPaletteRGB(const int32_t pos,const BYTE ar,const BYTE ag,const BYTE ab) {
	if ((pos<0)||(pos>255)) return;
	palette[pos].R=ar;
	palette[pos].G=ag;
	palette[pos].B=ab;
}

void TiledPlane::setPunkt(const int32_t ax,const int32_t ay,const BYTE awert) {
	if (!cmpY) return;
	if (
		(ax<0) || (ax >= xlen) || (ay<0) || (ay>=ylen)
	) return;
	
	cmpY[ay][ax]=awert;
}

BYTE TiledPlane::getPunkt(const int32_t ax,const int32_t ay) {
	if (
		(!cmpY) ||
		(ax<0) || (ax >= xlen) || (ay<0) || (ay>=ylen)
	) {
		LOGMSG("Error. getPunkt for a pixel not in memory\n");
		exit(99);
	}

	return cmpY[ay][ax];
}

// struct ArrayDDByteManager

// alloctes memory in chunks for the TiledPlane
// objects to keep memory fragmentation low

ArrayByteMgr::ArrayByteMgr() {
	current=NULL;
	allocatedIdx=0;
	freeFromIdx=-1;
	anzptr=0;
	double d=CHARMAPCHUNKSIZE; d /= sizeof(BYTE);
	allocatePerBlockIdx=(int)floor(d);
}

void ArrayByteMgr::FreeAll(void) {
	for(int32_t i=0;i<anzptr;i++) {
		delete[] ptr[i];
	}
	anzptr=0;
	current=NULL;
}

ArrayByteMgr::~ArrayByteMgr() {
	FreeAll();
}

PBYTE ArrayByteMgr::getMemory(const int32_t aanz) {
	if (anzptr >= (MAXPTR-8)) {
		LOGMSG("Error, memory. ArrayByteMgr/1");
		exit(99);
	}
	if (
		(!current) ||
		((freeFromIdx + aanz + 2) >= allocatedIdx)
	) {
		ptr[anzptr]=current=new BYTE[allocatePerBlockIdx];
		anzptr++;
		if (!current) {
			LOGMSG("Error, memory. ArrayByteMgr/2");
			exit(99);
		}
		freeFromIdx=0;
		allocatedIdx=allocatePerBlockIdx;
	}
	
	PBYTE p=&current[freeFromIdx];
	freeFromIdx += aanz;
	return p;
}

int32_t getGlobalMtile(const int32_t ax,const int32_t ay) {
	// ax,ay must be converted depending on symmetry
	int32_t x=ax,y=ay;
	int32_t MID=SCREENWIDTHM >> 1;
	
	if (SYMMETRY_AT_XAXIS>0) {
		if (ay >= MID) y=SCREENWIDTHM-1-ay;
	}

	if (SYMMETRY_AT_YAXIS>0) {
		if (ax >= MID) x=SCREENWIDTHM-1-ax;
	}
	
	for(int32_t i=0;i<anzcenters;i++) {
		if (centers[i].getMtile(x,y) == MTILE_BLACK) {
			// return at first BLACK found, as hc's can overlap
			// so x,y might lie in several hc's and be judged
			// interior 
			return MTILE_BLACK;
		}
	} // i
	
	return MTILE_GRAY;
}

void area(void) {
	// ATTN: to count valid black Mtiles
	// go over a VIRTUAL image of size SCREENWIDTHM
	// unoptimized, does not use symmetry of Mandelbrot
	
	int64_t countinterior=0;
	printf("\ncounting interior Mtiles ... ");
	int32_t noch=1,noch0=SCREENWIDTHM >> 4;
	
	for(int32_t y=0;y<SCREENWIDTHM;y++) {
		if ((--noch)<=0) {
			printf("%i ",SCREENWIDTHM-y);
			noch=noch0;
		}
		
		for(int32_t x=0;x<SCREENWIDTHM;x++) {
			if (getGlobalMtile(x,y) == MTILE_BLACK) {
				countinterior++;
			}
		} // x

	} // y
	
	// COMPLETE is an integer below || < 1024 and has no fractional part
	LOGMSG4("\narea lower bound: %I64d * (%.20lg / %i)^2\n",
		countinterior,
		(double)(COMPLETEM1-COMPLETEM0),
		SCREENWIDTHM
	);
}

void construct_image(void) {
	printf("\nconstructing image ");
	char tt[2048];
	sprintf(tt,"_M%02iJ%02i_image.bmp",REFINEMENTM,REFINEMENTJ);
	FILE *fbmp=fopen(tt,"wb");
	write2(fbmp,66,77); 
	int32_t bytes_per_row=SCREENWIDTHM >> 3; // 8 Pixel per byte
	
	uint32_t off
		=		14 
			+	40 
			+	2*4
		;
	
	// filelen will overflow if image is too big
	// but external viewers 
	// can display the image nonetheless
	
	uint32_t filelen
		=	off
		+	(bytes_per_row*SCREENWIDTHM);
	;
			
	fwrite(&filelen,1,sizeof(filelen),fbmp);
	write4(fbmp,0,0,0,0);
	fwrite(&off,1,sizeof(off),fbmp); // offset, ab da beginnen die PIXEL
	write4(fbmp,40,0,0,0);
	
	int32_t  w=SCREENWIDTHM;
	fwrite(&w,sizeof(w),1,fbmp);
	fwrite(&w,sizeof(w),1,fbmp);
	write2(fbmp,1,0);
	write2(fbmp,1,0); // bits per pixel
	write4(fbmp,0,0,0,0);
	write4(fbmp,0,0,0,0);
	write4(fbmp,19,10,0,0);
	write4(fbmp,19,10,0,0);
	write4(fbmp,2,0,0,0);
	write4(fbmp,0,0,0,0);
	write4(fbmp,127,127,127,0); // non-interior
	write4(fbmp,0,0,0,0); // interior, black

	BYTE* onerow=new BYTE[bytes_per_row];
	
	printf("... ");
	int32_t noch=1,noch0=SCREENWIDTHM >> 5;
	
	for(int32_t y=0;y<SCREENWIDTHM;y++) {
		if ((--noch)<=0) {
			printf("%i ",SCREENWIDTHM-y);
			noch=noch0;
		}
		
		int32_t xpos=-1;
		for(int32_t gx=0;gx<SCREENWIDTHM;gx+=8) {
			xpos++;
			onerow[xpos]=0;
			
			for(int32_t bit=0;bit<8;bit++) {
				if (getGlobalMtile(gx+bit,y) == MTILE_BLACK) {
					onerow[xpos] |= (1 << (7-bit));
				}
			}
		} // gx
		
		fwrite(&onerow[0],bytes_per_row,sizeof(BYTE),fbmp);
	} // y
	
	fclose(fbmp);
	delete[] onerow;
	
	printf("done\n");
}

// struct HCenter

void HCenter::save(void) {
	char tt[2048];
	
	// complex coordinates of lower left corner of lower left Mtile
	sprintf(tt,"hc%06i.pos",hcid);
	FILE *f=fopen(tt,"wb");
	// as screen coordinates are stored in int32_t
	// any complex coordinate is at finest (RANGE1-RANGE0)/2^32
	// for 2 <= RANGE1-RANGE0 <= 1024: integer part is at most 10 bits
	// fractional part is at most 31 bits
	// storing always in double allows one to compute different
	// refinements in different number types necessary
	double dx=Ax0; // loss-less conversion
	double dy=Ay0; // loss-less conversion
	
	fwrite(&dx,1,sizeof(dx),f);
	fwrite(&dy,1,sizeof(dy),f);
	fclose(f);
	
	// as an 8-bit bitmap
	sprintf(tt,"hc%06i_%08i.data",hcid,SCREENWIDTHM);
	msetregion.save(tt);
	
	// double the width for refinement if intended
	if (savetwice>0) {
		sprintf(tt,"hc%06i_%08i.data",hcid,2*SCREENWIDTHM);
		msetregion.saveTwice(tt);
	}
}

void markP12remove_z2c(HCenter& ahc) {
	#pragma omp parallel for 
	for(int32_t y=ahc.circumsq.y0;y<=ahc.circumsq.y1;y++) {
		for(int32_t x=ahc.circumsq.x0;x<=ahc.circumsq.x1;x++) {
			// check every Mtile no matter the already set color
			
			PlaneRect A;
			A.x0=x*scaleRangePerPixelM+COMPLETEM0;
			A.x1=A.x0+scaleRangePerPixelM;
			A.y0=y*scaleRangePerPixelM+COMPLETEM0;
			A.y1=A.y0+scaleRangePerPixelM;
			
			if (
				(MTileInMainCardioid_z2c(A) != INP123_OUTSIDE) ||
				(MTileInPeriod2Bulb_z2c(A) != INP123_OUTSIDE)
			) {
				// set to no-longer to analyse
				ahc.setMtile(x,y,MTILE_GRAY_DONE);
			}
		} // x
	} // y
}

// set period-4 components due to Stephenson formula
int8_t markP4set_z2c(HCenter& ahc) {
	#ifndef _USESTEPHENSONTEST
	return 0;
	#endif
	#ifdef _USESTEPHENSONTEST
	printf("\np4 in hcid=%i: ",ahc.hcid);
	int8_t changed=1;
	int64_t lastset=ctrsetp34;
	int64_t init34=ctrsetp34;
	
	// this "looking near already black Mtiles"-method
	// will fail at the very first analsis of an empty screen
	// then the HC will be regularly analzyed
	// it neither finds non-connected secondary regions
	// belongig to a period3-component. Those will need
	// to be analyzed as well. Reasoning: This is a speed-up
	// to render (some) Mtiles diretly black and therefore
	// not-to-be-analyzed but it is not designed
	// to be as exhaustive as possible. This is a pending
	// future otpimization.
	
	while (changed>0) {
		changed=0;
		if (lastset < ctrsetp34) {
			printf("%I64d ",ctrsetp34-lastset);
		}
		lastset=ctrsetp34;
		
		#pragma omp parallel for reduction(+:ctrsetp34)
		for(int32_t y=ahc.circumsq.y0;y<=ahc.circumsq.y1;y++) {
			for(int32_t x=ahc.circumsq.x0;x<=ahc.circumsq.x1;x++) {\
				/* only non-black Mtiles are to check */\
				/* as the others are trustworthily either Jset orbit */\
				/* analyzed or set in previous refinements */\
				if (ahc.getMtile(x,y) != MTILE_GRAY) continue;
				
				int8_t wn=0;
					
				for(int32_t dy=-1;dy<=1;dy++) {
					for(int32_t dx=-1;dx<=1;dx++) {
						if (ahc.getMtile(x+dx,y+dy) == MTILE_BLACK) {
							wn=1;
							break;
						}
					} 
					
					if (wn > 0) break;
				} 
				
				// the outcome of wn is DEPENDENT on race conditions
				// if it is wn>0 => all fine, Mtile is checked
				// if it is wn==0 it might have been wn>0 an instant later
				// as a parallel thread might change one of the neighbours to gray
				// however, as this thread then also sets changed to 1, this gray
				// Mtile will be reanalysed in the next while-loop
				if (wn <= 0) continue;\
				
				PlaneRect A;
				A.y0=y*scaleRangePerPixelM+COMPLETEM0;
				A.y1=A.y0+scaleRangePerPixelM;
				
				A.x0=x*scaleRangePerPixelM+COMPLETEM0;
				A.x1=A.x0+scaleRangePerPixelM;
				
				if (
					(MTileInPeriod4_z2c(A) == INP123_INSIDE)
				) {
					/* set to definite interior to reduce calculation necessity */\
					// NOTE: setMtile also sets symmetric Mtiles if the hc is straddling
					// a symmetry axis. In that case, race conditions might write several
					// times the same value MTILE_BLACK into one location (so no problem)
					// ctrsetp43 however will be incremented and can have different resulting
					// values in different program runs, howveer as the value is only for
					// informational purposes, that's fine
					ahc.setMtile(x,y,MTILE_BLACK);
					ctrsetp34++;
					changed=1; /* race possible, but only value 1 is ever written */
				} 
			} 
		} // y
	} // while
	
	//LOGMSG2("\n%I64d ctrs\n",ctrs);
	//LOGMSG2("%I64d ctrfastp3\n",ctrfastp3);
	
	return (ctrsetp34>init34);
	#endif
}

// set period3 as definite interior by closed formula by Stephenson et.al.
int8_t markP3set_z2c(HCenter& ahc) {
	#ifndef _USESTEPHENSONTEST
	return 0;
	#endif
	
	#ifdef _USESTEPHENSONTEST
	printf("\np3 in hcid=%i: ",ahc.hcid);
	int8_t changed=1;
	int64_t lastset=ctrsetp34;
	int64_t init34=ctrsetp34;
	
	// thsi "looking near already black Mtiles"-method
	// will fail at the very first analsis of an empty screen
	// then the HC will be regularly analzyed
	// it neither finds non-connected secondary regions
	// belongig to a period3-component. Those will need
	// to be analyzed as well. Reasoning: This is a speed-up
	// to render (some) Mtiles diretly black and therefore
	// not-to-be-analyzed but it is not designed
	// to be as exhaustive as possible. This is a pending
	// future otpimization.
	
	// comments for P4_z2c apply (ctrsetp34, wn etc).
	
	while (changed>0) {
		changed=0;
		if (lastset < ctrsetp34) {
			printf("%I64d ",ctrsetp34-lastset);
		}
		lastset=ctrsetp34;
		
		#pragma omp parallel for reduction(+:ctrsetp34)
		for(int32_t y=ahc.circumsq.y0;y<=ahc.circumsq.y1;y++) {
			for(int32_t x=ahc.circumsq.x0;x<=ahc.circumsq.x1;x++) {
				/* only non-black Mtiles are to check */ 
				/* as the others are trustworthily either Jset orbit */ 
				/* analyzed or set in previous refinements */ 
				if (ahc.getMtile(x,y) != MTILE_GRAY) continue;
			
				int8_t wn=0;
				
				for(int32_t dy=-1;dy<=1;dy++) {
					for(int32_t dx=-1;dx<=1;dx++) {
						if (ahc.getMtile(x+dx,y+dy) == MTILE_BLACK) {
							wn=1;
							break;
						}
					}
				
					if (wn > 0) break;
				}
			
				if (wn <= 0) continue;
				
				PlaneRect A;
				A.y0=y*scaleRangePerPixelM+COMPLETEM0;
				A.y1=A.y0+scaleRangePerPixelM;
			
				A.x0=x*scaleRangePerPixelM+COMPLETEM0;
				A.x1=A.x0+scaleRangePerPixelM;
			
				if (
					(MTileInPeriod3_z2c(A) == INP123_INSIDE)
				) {
					/* set to definite interior to reduce calculation necessity */ 
					ahc.setMtile(x,y,MTILE_BLACK);
					ctrsetp34++;
					changed=1; /* only value 1 is written =)> no race conditions releavant */ 
				} 
			} // x
		} // y
		
	} // while
	
	//LOGMSG2("\n%I64d ctrs\n",ctrs);
	//LOGMSG2("%I64d ctrfastp3\n",ctrfastp3);
	
	return (ctrsetp34>init34);
	#endif
}

void HCenter::initRegion(void) {
	msetregion.setlenxy(allocatewidth,allocatewidth);
	hcmemoryused += ((int64_t)allocatewidth*allocatewidth);
	setPaletteTo(msetregion);
	msetregion.fillrect(0,0,allocatewidth-1,allocatewidth-1,MTILE_GRAY);
}

void HCenter::setMtile(const int32_t ax,const int32_t ay,const int32_t acol) {
	// exploits symmetry in the active region if straddling
	
	MSETTILEINHC(ax,ay,acol)
	if (SYMMETRY_AT_XAXIS>0) {
		MSETTILEINHC(ax,SCREENWIDTHM-1-ay,acol)
	}
	
	if (SYMMETRY_AT_YAXIS>0) {
		MSETTILEINHC(SCREENWIDTHM-1-ax,ay,acol)
		if (SYMMETRY_AT_XAXIS>0) {
			MSETTILEINHC(SCREENWIDTHM-1-ax,SCREENWIDTHM-1-ay,acol)
		}
	}
}

int32_t HCenter::getMtile(const int32_t ax,const int32_t ay) {
	if (
		(ax >= circumsq.x0) &&
		(ax <= circumsq.x1) &&
		(ay >= circumsq.y0) &&
		(ay <= circumsq.y1)
	) return msetregion.getPunkt(ax-circumsq.x0,ay-circumsq.y0);
	
	return COLOR_UNDEF;
}

int8_t floodfill_p(HCenter& ahc,const int32_t px,const int32_t py) {
	// floodfill starting from one gray pixel
	int8_t ret=0;
	int32_t anzff=0;
	
	#define ADDFF(XX,YY) \
	{\
		if (anzff >= (ANZFLOODFILL-8)) {\
			LOGMSG("Error. FloodFill too big\n");\
			exit(99);\
		}\
		floodfill[anzff].w0=XX;\
		floodfill[anzff].w1=YY;\
		if ( (XX) < ff.x0) ff.x0=XX;\
		if ( (XX) > ff.x1) ff.x1=XX;\
		if ( (YY) < ff.y0) ff.y0=YY;\
		if ( (YY) > ff.y1) ff.y1=YY;\
		anzff++;\
	}
			
	ScreenRect ff;
	ff.x0=SCREENWIDTHM;
	ff.x1=0;
	ff.y0=SCREENWIDTHM;
	ff.y1=0;
	ADDFF(px,py)
			
	int8_t rollback=0;
	int64_t sicbl=ctrsetblackfloodfill;
			
	while (anzff>0) {
		int32_t cx=floodfill[anzff-1].w0;
		int32_t cy=floodfill[anzff-1].w1;
		anzff--;
		
		int32_t f2=ahc.getMtile(cx,cy);
		if (f2 == COLOR_UNDEF) continue;
		
		int32_t f2127=f2 & MTILE_NORMALCOLORBITS;
		int32_t f2rb=f2 & MTILE_BIT_ROLLBACK;
		
		// white is used for future compatibility
		// if it meets white or a pixel already leaking
		if (
			(f2127 == MTILE_WHITE) ||
			(f2rb>0)
		) {
			rollback=1;
			break;
		}
		
		// border encountered
		if (f2 == MTILE_BLACK) continue;
		
		if (MPixelIsGray(f2) <= 0) continue;
		
		// truely gray pixel, so no higher order bits set
		f2 |= MTILE_BIT_FLOODED;
		ahc.setMtile(cx,cy,f2);
		ctrsetblackfloodfill++;
				
		#define CHECKN(XX,YY) \
		{\
			if (\
				((XX)>=ahc.circumsq.x0)&&((XX)<=ahc.circumsq.x1)&&\
				((YY)>=ahc.circumsq.y0)&&((YY)<=ahc.circumsq.y1)\
			) {\
				int32_t f3=ahc.getMtile(XX,YY);\
				if (f3 == COLOR_UNDEF) f3=MTILE_WHITE;\
				int32_t f3127=f3 & MTILE_NORMALCOLORBITS;\
				int32_t f3rb=f3 & MTILE_BIT_ROLLBACK;\
				\
				if (\
					(f3127 == MTILE_WHITE) ||\
					(f3rb>0)\
				) {\
					rollback=1;\
				} else \
				if (MPixelIsGray(f3)>0) {\
					ADDFF(XX,YY);\
				}\
			}\
		}
				
		// pixel neighbour outside region
		if (
			((cx-1)<ahc.circumsq.x0) ||
			((cx+1)>ahc.circumsq.x1) ||
			((cy-1)<ahc.circumsq.y0) ||
			((cy+1)>ahc.circumsq.y1)
		) rollback=1;
		
		if (rollback<=0) { CHECKN(cx-1,cy) }
		if (rollback<=0) { CHECKN(cx+1,cy) }
		if (rollback<=0) { CHECKN(cx,cy-1) }
		if (rollback<=0) { CHECKN(cx,cy+1) }
				
		if (rollback>0) break;
				
	} // while floodfill anzff
	
	if (rollback>0){
		// convert FLOODED-Bit everywhere visited
		// to BIT_ROLLBACK
		for(int32_t ty=ff.y0;ty<=ff.y1;ty++) {
			for(int32_t tx=ff.x0;tx<=ff.x1;tx++) {
				int32_t f=ahc.getMtile(tx,ty);
				if (f == COLOR_UNDEF) continue;
				
				if ((f & MTILE_BIT_FLOODED) >0) {
					f &= MTILE_NORMALCOLORBITS;
					f |= MTILE_BIT_ROLLBACK;
					ahc.setMtile(tx,ty,f); // symmetry
				}
			}
		}
		ctrsetblackfloodfill=sicbl;
	} else {
		// set mtile with FLOODED_BIT (independent of underlying color) 
		// to SQUARE_BLACK
		for(int32_t ty=ff.y0;ty<=ff.y1;ty++) {
			for(int32_t tx=ff.x0;tx<=ff.x1;tx++) {
				int32_t lokalf=ahc.getMtile(tx,ty);
				if (lokalf == COLOR_UNDEF) continue;
				
				if ((lokalf & MTILE_BIT_FLOODED) > 0) {
					ahc.setMtile(tx,ty,MTILE_BLACK); // symmetry
				}
			} // while tx
		} // for ty
		
		ret=1;
	}
	// now image has no FLOODED_BITs, but maybe ROLLBACK ones
	
	return ret;
}

void clearFloodedBits(HCenter& ahc) {
	// clear all high-order bits
	for(int32_t ty=ahc.circumsq.y0;ty<=ahc.circumsq.y1;ty++) {
		for(int32_t tx=ahc.circumsq.x0;tx<=ahc.circumsq.x1;tx++) {
			int32_t f=ahc.getMtile(tx,ty);
			if (f == COLOR_UNDEF) continue;
			if (f > MTILE_NORMALCOLORBITS) {
				f &= MTILE_NORMALCOLORBITS;
				ahc.setMtile(tx,ty,f); 
			}
		}
	}
			
}

void floodfill_hc(HCenter& ahc) {
	// try to fill at every gray pixel adjacent to black
	
	for(int32_t y=ahc.circumsq.y0;y<=ahc.circumsq.y1;y++) {
		for(int32_t x=ahc.circumsq.x0;x<=ahc.circumsq.x1;x++) {
			// GRAY, GRAY_DONE or CHECKFORBLACK
			if (MPixelIsGray(ahc.getMtile(x,y)) <= 0) continue;
			
			int8_t neighbourblack=0;
			for(int32_t dy=-1;dy<=1;dy++) {
				for(int32_t dx=-1;dx<=1;dx++) {
					if (ahc.getMtile(x+dx,y+dy) == MTILE_BLACK) {
						neighbourblack=1;
						break;
					}
				} // dx
				if (neighbourblack>0) break;
			} // dy
			
			if (neighbourblack>0) {
				floodfill_p(ahc,x,y);
			}
		} // x
	} // y
	
	clearFloodedBits(ahc);

}

// all MTILE_CHECKFORBLACK are orbit-analyzed
int8_t analyzeMsetToCheck(HCenter& ahc) {
	int32_t noch=1;
	int32_t noch0=(ahc.circumsq.y1-ahc.circumsq.y0+1) >> 2;
	if (SCREENWIDTHM >= 65536) noch0 >>= 1;
	int8_t foundblack=0;
	
	NTYP lokalC0re,lokalC1re;
	NTYP lokalC0im,lokalC1im;
	
	for(int32_t y=ahc.circumsq.y0;y<=ahc.circumsq.y1;y++) {
		if ((--noch)<=0) {
			printf(".");
			noch=noch0;
		}
		
		lokalC0im=y*scaleRangePerPixelM + COMPLETEM0;
		lokalC1im=lokalC0im+scaleRangePerPixelM;
			
		for(int32_t x=ahc.circumsq.x0;x<=ahc.circumsq.x1;x++) {
			int32_t mf=ahc.getMtile(x,y);
					
			if (mf != MTILE_CHECKFORBLACK) continue;
			
			lokalC0re=x*scaleRangePerPixelM + COMPLETEM0;
			lokalC1re=lokalC0re+scaleRangePerPixelM;
			
			// work off pipeline if full
			if (inpipeline >= MAXJULIASETS) {
				if (inpipeline == 1) {
					testC_interior(juliasets[0]);\
					WORKOFF_RESULTS(0,foundblack)
				} else {
					// parallel: computation. Only works on local
					// variables and ones pertaining to juliasets]
					#pragma omp parallel for num_threads(MAXJULIASETS)
					for(int32_t pnr=0;pnr<inpipeline;pnr++) {
						testC_interior(juliasets[pnr]);\
					}
					
					// seriell: set Mtile color (globally accessible)
					for(int32_t pnr=0;pnr<inpipeline;pnr++) {
						WORKOFF_RESULTS(pnr,foundblack)
					}
				} // parallel
				
				inpipeline=0;
			} // work off pipeline
			
			// add this seed to pipeline
			ADDPIPELINE(x,y,&ahc,lokalC0re,lokalC1re,lokalC0im,lokalC1im)
			
		} // x
	} // y
	
	// in case of non-empty pipeline
	if (inpipeline>0) {
		#pragma omp parallel for num_threads(MAXJULIASETS)
		for(int32_t pnr=0;pnr<inpipeline;pnr++) {
			testC_interior(juliasets[pnr]);\
		}
					
		// seriell: set Mtile color in global variable
		for(int32_t pnr=0;pnr<inpipeline;pnr++) {
			WORKOFF_RESULTS(pnr,foundblack)
		}
	}
	
	inpipeline=0;
	
	return foundblack;
}

// one active regions: Mtiles analyzed in gridding coordinates
// of decreasing pixel distance
int8_t grid_oneHC(HCenter& ahc) {
	int8_t ret=0;
	
	#define ADJUSTGRIDVALUES \
	{\
		if (x < gridx0) gridx0=x;\
		if (x > gridx1) gridx1=x;\
		if (y < gridy0) gridy0=y;\
		if (y > gridy1) gridy1=y;\
	}

	ScreenRect enc64; 
	enc64.x0=SCREENWIDTHM;
	enc64.x1=0;
	enc64.y0=SCREENWIDTHM;
	enc64.y1=0;
	
	// enc64 valid if x0 <= x1
	#define ENC64BREAK(XX,YY) \
	{\
		if (\
			(enc64.x0 <= enc64.x1) &&\
			(enc64.y0 <= enc64.y1) &&\
			(\
				( (XX) < enc64.x0 ) ||\
				( (XX) > enc64.x1 ) ||\
				( (YY) < enc64.y0 ) ||\
				( (YY) > enc64.y1 )\
			)\
		) break;\
	}
	
	// adjust enc64 values in case testC results in NON-interior
	// only in gridsize=64
	#define ENC64ADJUST(XX,YY) \
	{\
		if ( (XX) < enc64.x0 ) enc64.x0=XX;\
		if ( (XX) > enc64.x1 ) enc64.x1=XX;\
		if ( (YY) < enc64.y0 ) enc64.y0=YY;\
		if ( (YY) > enc64.y1 ) enc64.y1=YY;\
	}
	
	// pretest: jump horizontal and vertical line
	// through the center pixel in 64-size steps to roughly get
	// the rectangular region in which the current hc 
	// shows interior at the current M-level
	// this also excludes other hyperbolic components that
	// reside at the border of ahc.circumsq
	
	// CAVE: If the hyperbolic component is not partially circular in nature
	// or the center's complex coordinates are way off the true hyperbolic
	// center, this MAXGRIDSIZE-jump might actually cut off part of
	// the possible interior. The same can happen if the jump is on the cusp-line of a cardioid.
	// If this happens multiple times, enc64 should be set to ahc.circumsq instead
	// at the expense of more "scanning" orbit constructions
	
	juliasets[0].ptrhc=&ahc;
	
	if (PRETESTGRIDDING>0) {
		for(int32_t x=ahc.centerpixelx;x<=ahc.circumsq.x1;x+=MAXGRIDSIZE) {
			ENC64ADJUST(x,ahc.centerpixely);
			int32_t f=ahc.getMtile(x,ahc.centerpixely);
			if (f == MTILE_GRAY_DONE) break;
			if (f != MTILE_GRAY) continue; 
			
			// now analyze
			testC_interior(juliasets[0],x,ahc.centerpixely);
			if (juliasets[0].erg == MTILE_BLACK) {
				ahc.setMtile(x,ahc.centerpixely,MTILE_BLACK);
				ret=1;
				continue;
			} else {
				ahc.setMtile(x,ahc.centerpixely,MTILE_GRAY_DONE);
				break;
			}
		} // x

		for(int32_t x=ahc.centerpixelx;x>=ahc.circumsq.x0;x-=MAXGRIDSIZE) {
			ENC64ADJUST(x,ahc.centerpixely);
			int32_t f=ahc.getMtile(x,ahc.centerpixely);
			if (f == MTILE_GRAY_DONE) break;
			if (f != MTILE_GRAY) continue; 
		
			testC_interior(juliasets[0],x,ahc.centerpixely);
			if (juliasets[0].erg == MTILE_BLACK) {
				ahc.setMtile(x,ahc.centerpixely,MTILE_BLACK);
				ret=1;
				continue;
			} else {
				ahc.setMtile(x,ahc.centerpixely,MTILE_GRAY_DONE);
				break;
			}
		} // X

		for(int32_t y=ahc.centerpixely;y<=ahc.circumsq.y1;y+=MAXGRIDSIZE) {
			ENC64ADJUST(ahc.centerpixelx,y);
			int32_t f=ahc.getMtile(ahc.centerpixelx,y);
			if (f == MTILE_GRAY_DONE) break;
			if (f != MTILE_GRAY) continue; 
				
			testC_interior(juliasets[0],ahc.centerpixelx,y);
			if (juliasets[0].erg == MTILE_BLACK) {
				ahc.setMtile(ahc.centerpixelx,y,MTILE_BLACK);
				ret=1;
				continue;
			} else {
				ahc.setMtile(ahc.centerpixelx,y,MTILE_GRAY_DONE);
				break;
			}
		} // y

		for(int32_t y=ahc.centerpixely;y>=ahc.circumsq.y0;y-=MAXGRIDSIZE) {
			ENC64ADJUST(ahc.centerpixelx,y)
			int32_t f=ahc.getMtile(ahc.centerpixelx,y);
			if (f == MTILE_GRAY_DONE) break;
			if (f != MTILE_GRAY) continue; 
		
			testC_interior(juliasets[0],ahc.centerpixelx,y);
			if (juliasets[0].erg == MTILE_BLACK) {
				ahc.setMtile(ahc.centerpixelx,y,MTILE_BLACK);
				ret=1;
				continue;
			} else {
				ahc.setMtile(ahc.centerpixelx,y,MTILE_GRAY_DONE);
				break;
			}
		} // y
	
		// widen the enc64 32 pixels in all directions, so the centrqal region
		// can still grow
		enc64.x0 -= (MAXGRIDSIZE >>1);
		if (enc64.x0 < ahc.circumsq.x0) enc64.x0=ahc.circumsq.x0;
		enc64.x1 += (MAXGRIDSIZE >>1);
		if (enc64.x1 > ahc.circumsq.x1) enc64.x1=ahc.circumsq.x1;
		enc64.y0 -= (MAXGRIDSIZE >>1);
		if (enc64.y0 < ahc.circumsq.y0) enc64.y0=ahc.circumsq.y0;
		enc64.y1 += (MAXGRIDSIZE >>1);
		if (enc64.y1 > ahc.circumsq.y1) enc64.y1=ahc.circumsq.y1;
	} else {
		enc64.x0=ahc.circumsq.x0;
		enc64.x1=ahc.circumsq.x1;
		enc64.y0=ahc.circumsq.y0;
		enc64.y1=ahc.circumsq.y1;
	}
	
	// grid at different jump-sizes
	int8_t first=1;
	for(int32_t gridsize=MAXGRIDSIZE;gridsize>=MINGRIDSIZE;gridsize>>=1) {
		if (
			(gridsize < MAXGRIDSIZE) &&
			(SCREENWIDTHJ >= ((int32_t)1 << 16)) &&
			(ahc.allocatewidth >= 2200)
		) {
			printf(" saving ... ");
			ahc.save();
			printf("done");
		}
		int32_t gridx0=ahc.centerpixelx;
		int32_t gridx1=ahc.centerpixelx;
		int32_t gridy0=ahc.centerpixely;
		int32_t gridy1=ahc.centerpixely;

		// upper/lower left quadrant of hc
		for(int32_t x=ahc.centerpixelx;x<=ahc.circumsq.x1;x+=gridsize) {
			
			// go upwards
			for(int32_t y=ahc.centerpixely;y<=ahc.circumsq.y1;y+=gridsize) {
				ENC64BREAK(x,y)
				int32_t f=ahc.getMtile(x,y);
				if (f == MTILE_GRAY_DONE) break;
				if (f == MTILE_BLACK) {
					// adjust gridx,y values
					ADJUSTGRIDVALUES
					continue;
				}
				if (f != MTILE_GRAY) continue; // next outward
				
				// now analyze
				testC_interior(juliasets[0],x,y);
				if (juliasets[0].erg == MTILE_BLACK) {
					ahc.setMtile(x,y,MTILE_BLACK);
					ADJUSTGRIDVALUES
					ret=1;
					continue;
				} else {
					ahc.setMtile(x,y,MTILE_GRAY_DONE);
					
					break;
				}
			} // y
			
			// go downwards
			for(int32_t y=ahc.centerpixely;y>=ahc.circumsq.y0;y-=gridsize) {
				ENC64BREAK(x,y)
				int32_t f=ahc.getMtile(x,y);
				if (f == MTILE_GRAY_DONE) {
					
					break; // more outwards isprobably not interior anyways
				}
				if (f == MTILE_BLACK) {
					// adjust gridx,y values
					ADJUSTGRIDVALUES
					continue;
				}
				if (f != MTILE_GRAY) continue; // next outward
				
				// now analyze
				testC_interior(juliasets[0],x,y);
				if (juliasets[0].erg == MTILE_BLACK) {
					ahc.setMtile(x,y,MTILE_BLACK);
					ret=1;
					ADJUSTGRIDVALUES
					continue;
				} else {
					ahc.setMtile(x,y,MTILE_GRAY_DONE);
					
					break;
				}
			} // y
			
		} // x
		
		// upper/lower right  quadrant of hc
		for(int32_t x=ahc.centerpixelx;x>=ahc.circumsq.x0;x-=gridsize) {
			
			// go upwards
			for(int32_t y=ahc.centerpixely;y<=ahc.circumsq.y1;y+=gridsize) {
				ENC64BREAK(x,y)
				int32_t f=ahc.getMtile(x,y);
				if (f == MTILE_GRAY_DONE) break; // more outwards isprobably not interior anyways
				if (f == MTILE_BLACK) {
					// adjust gridx,y values
					ADJUSTGRIDVALUES
					continue;
				}
				if (f != MTILE_GRAY) continue; // next outward
				
				// now analyze
				testC_interior(juliasets[0],x,y);
				if (juliasets[0].erg == MTILE_BLACK) {
					ahc.setMtile(x,y,MTILE_BLACK);
					ret=1;
					ADJUSTGRIDVALUES
					continue;
				} else {
					ahc.setMtile(x,y,MTILE_GRAY_DONE);
					break;
				}
			} // y
			
			// go downwards
			for(int32_t y=ahc.centerpixely;y>=ahc.circumsq.y0;y-=gridsize) {
				ENC64BREAK(x,y)
				int32_t f=ahc.getMtile(x,y);
				if (f == MTILE_GRAY_DONE) break;
				if (f == MTILE_BLACK) {
					// adjust gridx,y values
					ADJUSTGRIDVALUES
					continue;
				}
				if (f != MTILE_GRAY) continue; // next outward
				
				// now analyze
				testC_interior(juliasets[0],x,y);
				if (juliasets[0].erg == MTILE_BLACK) {
					ahc.setMtile(x,y,MTILE_BLACK);
					ret=1;
					ADJUSTGRIDVALUES
					continue;
				} else {
					ahc.setMtile(x,y,MTILE_GRAY_DONE);
					break;
				}
			} // y
			
		} // x
		
		// now vertical line
		for(int32_t y=ahc.centerpixely;y<=ahc.circumsq.y1;y+=gridsize) {
			
			// go right
			// lastinterior now used as x-coordinate
			for(int32_t x=ahc.centerpixelx;x<=ahc.circumsq.x1;x+=gridsize) {
				ENC64BREAK(x,y)
				int32_t f=ahc.getMtile(x,y);
				if (f == MTILE_GRAY_DONE) break;
				if (f == MTILE_BLACK) {
					// adjust gridx,y values
					ADJUSTGRIDVALUES
					continue;
				}
				if (f != MTILE_GRAY) continue; // next outward
				
				// now analyze
				testC_interior(juliasets[0],x,y);
				if (juliasets[0].erg == MTILE_BLACK) {
					ahc.setMtile(x,y,MTILE_BLACK);
					ADJUSTGRIDVALUES
					ret=1;
					continue;
				} else {
					ahc.setMtile(x,y,MTILE_GRAY_DONE);
					break;
				}
			} // x
			
			// go left
			for(int32_t x=ahc.centerpixelx;x>=ahc.circumsq.x0;x-=gridsize) {
				ENC64BREAK(x,y)
				int32_t f=ahc.getMtile(x,y);
				if (f == MTILE_GRAY_DONE) break;
				if (f == MTILE_BLACK) {
					// adjust gridx,y values
					ADJUSTGRIDVALUES
					continue;
				}
				if (f != MTILE_GRAY) continue; // next outward
				
				// now analyze
				testC_interior(juliasets[0],x,y);
				if (juliasets[0].erg == MTILE_BLACK) {
					ahc.setMtile(x,y,MTILE_BLACK);
					ret=1;
					ADJUSTGRIDVALUES
					continue;
				} else {
					ahc.setMtile(x,y,MTILE_GRAY_DONE);
					break;
				}
			} // x
			
		} // y
		
		for(int32_t y=ahc.centerpixely;y>=ahc.circumsq.y0;y-=gridsize) {
			
			// go right
			for(int32_t x=ahc.centerpixelx;x<=ahc.circumsq.x1;x+=gridsize) {
				ENC64BREAK(x,y)
				int32_t f=ahc.getMtile(x,y);
				if (f == MTILE_GRAY_DONE) break;
				if (f == MTILE_BLACK) {
					ADJUSTGRIDVALUES
					continue;
				}
				if (f != MTILE_GRAY) continue;
				
				testC_interior(juliasets[0],x,y);
				if (juliasets[0].erg == MTILE_BLACK) {
					ahc.setMtile(x,y,MTILE_BLACK);
					ret=1;
					ADJUSTGRIDVALUES
					continue;
				} else {
					ahc.setMtile(x,y,MTILE_GRAY_DONE);
					break;
				}
			} // x
			
			// go left
			for(int32_t x=ahc.centerpixelx;x>=ahc.circumsq.x0;x-=gridsize) {
				ENC64BREAK(x,y)
				int32_t f=ahc.getMtile(x,y);
				if (f == MTILE_GRAY_DONE) break;
				if (f == MTILE_BLACK) {
					ADJUSTGRIDVALUES
					continue;
				}
				if (f != MTILE_GRAY) continue;
				
				testC_interior(juliasets[0],x,y);
				if (juliasets[0].erg == MTILE_BLACK) {
					ahc.setMtile(x,y,MTILE_BLACK);
					ret=1;
					ADJUSTGRIDVALUES
					continue;
				} else {
					ahc.setMtile(x,y,MTILE_GRAY_DONE);
					break;
				}
			} // x
			
		} // y
		
		// now look at any black grid point
		// connecting vertical/horizontal line: analyze those pixels as well
		
		int64_t callanalyze=0;
		for(int32_t y=gridy0;y<=gridy1;y+=gridsize) {
			for(int32_t x=gridx0;x<=gridx1;x+=gridsize) {
				if (ahc.getMtile(x,y) != MTILE_BLACK) continue;
				
				// neighbour
				for(int32_t dy=-1;dy<=1;dy++) {
					for(int32_t dx=-1;dx<=1;dx++) {
						if ( (dy==0) && (dx==0) ) continue;
						
						// diagonally: future work
						if ( (dy!=0) && (dx!=0) ) continue;
						
						if (ahc.getMtile(x+dx*gridsize,y+dy*gridsize) != MTILE_BLACK) continue;
						
						int32_t linex0,linex1,liney0,liney1;
						if (dx < 0) { linex0=x-gridsize; linex1=x; }
						else { linex0=x; linex1=x+gridsize; }
						if (dy < 0) { liney0=y-gridsize; liney1=y; }
						else { liney0=y; liney1=y+gridsize; }
						
						if (
							(linex0 < ahc.circumsq.x0) ||
							(linex1 > ahc.circumsq.x1) ||
							(liney0 < ahc.circumsq.y0) ||
							(liney1 > ahc.circumsq.y1)
						) continue;
						
						if ( (dx==0) && (dy!=0) ) {
							// vertical line
							for(int32_t testy=liney0;testy<=liney1;testy++) {
								if (ahc.getMtile(linex0,testy) == MTILE_GRAY) {
									ahc.setMtile(linex0,testy,MTILE_CHECKFORBLACK);
									callanalyze++;
								}
							} // testy
						} else if ( (dx!=0) && (dy==0) ) {
							// horizontal line
							for(int32_t testx=linex0;testx<=linex1;testx++) {
								if (ahc.getMtile(testx,liney0) == MTILE_GRAY) {
									ahc.setMtile(testx,liney0,MTILE_CHECKFORBLACK);
									callanalyze++;
								}
							} // testy
						}
						
					} // dx
				} // dy
			} // x
		} // y
		
		int8_t changed=0;
		if (callanalyze>0) {
			if (first>0) {
				printf("check %I64d Mtiles ",callanalyze);
				first=0;
			} else {
				printf(" %I64d ",callanalyze);
			}
			if (analyzeMsetToCheck(ahc) > 0) {
				changed=1;
				ret=1;
			}
		}
		
		// run a flood-fill here, so maybe smaller grids are not needed
		if (changed>0) {
			floodfill_hc(ahc);
		}
		
	} // gridsize
	
	return ret;
}

void gridHC(void) {
	printf("\ngridding ...");
	for(int32_t i=0;i<anzcenters;i++) {
		
		if (centers[i].donotanalyze > 0) {
			// save it again and if wanted, double the size
			centers[i].save();
			continue;
		}

		printf("\n  %i. hc id=%i ... ",i+1,centers[i].hcid);
		
		(void)grid_oneHC(centers[i]);
		
		// save the results
		printf(" saving temporary ... ");
		centers[i].save();
		printf("done");
		
	} // i
	
}

int32_t getFuncId(const char* s) {
	char tt[2048];
	strcpy(tt,s);
	upper(tt);
	
	for(int32_t i=0;i<FUNCANZ;i++) {
		if (!strcmp(funcname[i],tt)) return i;
	} // i
	
	return FUNC_Z2C;
}

void setFunc(void) {
	switch (FUNC) {
		default: {
			getBoundingBoxfA=getBoundingBoxfA_z2c;
			ps_func=ps_func_z2c;
			SYMMETRY_AT_XAXIS=1;
			SYMMETRY_AT_YAXIS=0;
			break;
		};
	}; // switch
}

int32_t makePower2(const int32_t aa) {
	double b=log(aa) / log(2.0);
	int32_t bits=(int32_t)ceil(b);
	return ( (int32_t)1 << bits );
}

#ifdef _USESTEPHENSONTEST
inline int8_t ia_contains_zero(const ia& cp) {
	if ( (cp.lower() <= 0.0) && (cp.upper() >= 0.0) ) return 1;
	return 0;
}

inline int8_t cplxia_contains_zero(const cplxia& cp) {
	if (
		(ia_contains_zero(cp.real()) > 0) ||
		(ia_contains_zero(cp.imag()) > 0) 
	) return 1;
	
	return 0;
}
#endif

// main

int32_t main(int32_t argc,char** argv) {
	time_at_enter_main=clock();
	anzcenters=0;
	
	floodfill=new Int2[ANZFLOODFILL];
	centers=new HCenter[MAXHYPERBOLICCENTERS];
	
	flog=fopen("areatsa.log.txt","at");
	fprintf(flog,"\n-----------------\n");
	printf("areatsa\n");
	#ifdef _USE_PS_PERIODIC	
	LOGMSG("\nperiodic point used as starting Jtiles\n");
	#else
	LOGMSG("\norigin used as starting Jtiles\n");
	#endif
	// standard values
	// Mset and Jsets are computed in the 2-square
	RANGEM0=-2;
	RANGEM1=2;
	RANGEJ0=-2;
	RANGEJ1=2;
	// standard values in case non given
	REFINEMENTJ=10;
	REFINEMENTM=12;
	SCREENWIDTHM=(1 << REFINEMENTM);
	SCREENWIDTHJ=(1 << REFINEMENTJ);
	
	for(int32_t i=1;i<argc;i++) {
		upper(argv[i]);

		if (strstr(argv[i],"IMG")) {
			// construct image
			produceimg=1;
		} else
		if (strstr(argv[i],"RESET")) {
			// resetdata to only contain black, white and gray
			// i.e. GRAY_DONE is reverted
			_RESETCOLORS=1;
		} else
		if (strstr(argv[i],"FUNC=")) {
			int32_t a=getFuncId(&argv[i][5]);
			if (a >= 0) FUNC=a;
		} else
		if (strstr(argv[i],"NO64JUMP")) {
			// disable pre-test 64-jump
			PRETESTGRIDDING=0;
		} else
		if (strstr(argv[i],"PARALLEL=")) {
			// number of Mtiles to be analyzed simultaneously
			int32_t a;
			if (sscanf(&argv[i][9],"%i",&a) == 1) {
				if (a < 1) a=1; else if (a > 8) a=8;
				MAXJULIASETS=a;
			}
		} else
		if (strstr(argv[i],"NOTWICE")) {
			// disables saving double the size
			savetwice=0;
		} else
		if (strstr(argv[i],"LENJ=")==argv[i]) {
			// refinement level for the JuliaSet
			int a;
			if (sscanf(&argv[i][5],"%i",&a) == 1) {
				if (a < 8) a=8;
				REFINEMENTJ=a;
				SCREENWIDTHJ=(1 << a);
			}
		} else
		if (strstr(argv[i],"GRID=")==argv[i]) {
			// a,b
			int a,b;
			if (sscanf(&argv[i][5],"%i,%i",&a,&b) == 2) {
				MINGRIDSIZE=a;
				MAXGRIDSIZE=b;
			}
		} else
		if (strstr(argv[i],"LENM=")==argv[i]) {
			// refinement level for the Mandelbrot set, i.e. the active region
			int a;
			if (sscanf(&argv[i][5],"%i",&a) == 1) {
				if (a < 8) a=8;
				REFINEMENTM=a;
				SCREENWIDTHM=(1 << a);
			}
		} else 
		if (strstr(argv[i],"RANGEM=")==argv[i]) {
			int a;
			if (sscanf(&argv[i][7],"%i",&a) == 1) {
				if (a < 0) a=-a;
				if (a > 256) a=256;
				int32_t a2=makePower2(a);
				RANGEM0=-a2;
				RANGEM1= a2;
			}
		} else
		if (strstr(argv[i],"RANGEJ=")==argv[i]) {
			int a;
			if (sscanf(&argv[i][7],"%i",&a) == 1) {
				if (a < 0) a=-a;
				if (a > 256) a=256;
				int32_t a2=makePower2(a);
				RANGEJ0=-a2;
				RANGEJ1= a2;
			}
		}
	} // i
	
	LOGMSG2("number type %s\n",NTYPSTR);
	LOGMSG2("maximal parallel Julia sets %i\n",MAXJULIASETS);
	LOGMSG2("circumferencing factor=%i\n",CIRCUMFERENCINGFACTOR);

	juliasets=new JuliaSet[MAXJULIASETS];
	
	setFunc();
	LOGMSG4("function=%s sym xaxis=%i sym yaxis=%i\n",
		funcname[FUNC],SYMMETRY_AT_XAXIS,SYMMETRY_AT_YAXIS);
	
	// set AFTER setFunc call
	COMPLETEJ0=RANGEJ0;
	COMPLETEJ1=RANGEJ1;
	COMPLETEM0=RANGEM0;
	COMPLETEM1=RANGEM1;
	
	LOGMSG2("RANGEJ=%i\n",RANGEJ1);
	LOGMSG2("RANGEM=%i\n",RANGEM1);
	LOGMSG2("maxgridsize=%i\n",MAXGRIDSIZE);
	LOGMSG2("mingridsize=%i\n",MINGRIDSIZE);
	LOGMSG2("PRETESTGRIDDING=%i\n",PRETESTGRIDDING);

	// calculate scale values
	NTYP w=(RANGEJ1-RANGEJ0) / (double)SCREENWIDTHJ;
	scaleRangePerPixelJ=w; // dyadic fraction

	w=(NTYP)SCREENWIDTHJ / (RANGEJ1-RANGEJ0);
	scalePixelPerRangeJ=w; // natural number
	
	w=(RANGEM1-RANGEM0) / (NTYP)SCREENWIDTHM;
	scaleRangePerPixelM=w; // dyadic fraction

	w=(NTYP)SCREENWIDTHM / (RANGEM1-RANGEM0);
	scalePixelPerRangeM=w; // natural number

	LOGMSG2("M%02i",REFINEMENTM);
	LOGMSG2("J%02i\n",REFINEMENTJ);
	
	// scrlow
	scrlowM=new NTYP[SCREENWIDTHM+1];
	for(int32_t i=0;i<=SCREENWIDTHM;i++) {
		scrlowM[i]=i*scaleRangePerPixelM + COMPLETEM0;
	}
	scrlowJ=new NTYP[SCREENWIDTHJ+1];
	for(int32_t i=0;i<=SCREENWIDTHJ;i++) {
		scrlowJ[i]=i*scaleRangePerPixelJ + COMPLETEJ0;
	}

	#ifdef _CHUNK512
	MAXSTREAKS=-2 + ((int64_t)1 << 27) / sizeof(Streak);
	#else
	MAXSTREAKS=-2 + ((int64_t)1 << 32) / sizeof(Streak);
	#endif
	
	// init JuliaSets for parallel computation
	printf("\ninitialising Julia set memory ... ");
	for(int32_t i=0;i<MAXJULIASETS;i++) {
		printf("%i ",MAXJULIASETS-i);
		juliasets[i].initMemory();
	}
	printf("... %I64d GB used",1+(jsetmemory >> 30));
	
	// reads hc, activates them if possible
	readHC();
	
	LOGMSG2("\n  %I64d GB components memory used\n",1+(hcmemoryused >> 30));
	
	// //////////////////////////////////////
	//
	// main routine
	//
	// grid active hc's
	// connect grid points
	// flood fill if closed regions occur
	
	gridHC();
	
	// output result values
	OUTVALUES1
	OUTVALUES2
	
	if (listfulleventoccured>0) {
		LOGMSG("Info: SteakList was full at some point.\n  Increasing MAXSTREAKS is recommended for faster speed.\n");
	}

	// area calculation
	area();
	
	// overview image
	if (produceimg>0) {
		construct_image();
	}
	
	int32_t c1=clock();\
	LOGMSG2("\nduration %.0lf sec\n",(double)(c1-time_at_enter_main)/CLOCKS_PER_SEC);\
	
	fclose(flog);
	
	delete[] floodfill;
	delete[] centers;
	delete[] juliasets;
	
    return 0;
}

