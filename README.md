![bulb3](./_overviewbulb3.gif)

# area-tsa-lower
Guaranteed lower area bound for the Mandelbrot set

Summary
A method based on interval arithmetics and cell mapping is presented to compute
a lower bound on the area of the Mandelbrot set of 1.4932 that comes with a mathematical
guarantee. The reliable computational steps are guided through pointsampled
data towards positions in the complex plane where possible Mandelbrot
interior resides. Simple connectivity of the set is exploited to 
ood-ll the inside
of closed reliably analyzed region borders. Due to the highly parallelizable
nature of the algorithm and as all computations were performed on a personal
computer, tighter bounds are fairly easy to obtain using more equipped hardware.

A full description of the algorithm is found in the pdf file in the current directory.

The code is optimized for Windows 10 64 bit and demands a large amount of memory. If using win32, comment in define CHUNK512 in the source code.


command-line parameters for AREATSA.EXE

1. Parameters
2. Output
3. Input


=============
1. refinement
=============

LENJ=j - refinement j for the orbit construction, i.e. 2^j x 2^j Jtiles

LENM=m - refinement m for the active region, i.e. 2^m x 2^m Mtiles


parallelicity
=============

PARALLEL=n


Flags to control routine calls:
===============================

NOTWICE - active region not stored in twice the size for future refinements

NO64JUMP - disables pre-test for gridding: 64-jump

IMG - saves a 2-color bitmap showing interior-identified Mtiles (black)

RESET - reads active regions and resets every non-black, non-white pixel to gray


=========
2. OUTPUT
=========

values for Mtiles analyzed, Mtiles set to black via CM/IA orbit construction or flood-filled
lower bound for the area of the Mandelbrot set as a rational number

file: hcIIIIII.pos - stores two double values = the lower left corner coordinate of
the lower left Mtile in the active region with internal id IIIIII (as a natural number with leading zeros)

file: hcIIIIII_MMMMMMMM.data - the raw data for the Mset resolution MMMMMMMM (a natural number). File format is
an 8-bit bitmap. If renamed, the data can be viewed externally. Gray colored pixel = untested, black = identiefied as Mset-interior,
yellow = analyzed but not possible to judge as Mset-interior.

The software also saves a file with double the size: every Mtile issplit into a grid of 2x2 (rewfinement) with the
same color: only black, gray are present, yellow is converted to gray. This is the input file for the next higher refinement
level M+1.
CAVE: There is no check whether this double-sized file already exits. If so, it will be overwritten.


========
2. INPUT
========

command-line parameters

file _bulbs_center.values: list of approximated hyperbolic center values and size estimates
format: 
one line:
ID,REAL-PART,IMAG-PART,SIZE,
0,-0.1583251953125,1.0328369140625,0.0072021484375,

already activated centers with file names as in output described


