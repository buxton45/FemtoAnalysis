///////////////////////////////////////////////////////////////////////////
// Types_ThermBgdParams:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "Types_ThermBgdParams.h"
#include <cassert>




const double  cLamK0_ThermBgdParamValues_0010[7] = {1.00698258, -0.02291412,  0.01312921, -0.00875929,  0.00185616,  0.00336635, -0.00129966};
const double  cLamK0_ThermBgdParamValues_1030[7] = {1.00073498,  0.04608827, -0.17458269,  0.11667504,  0.01847243, -0.03675633,  0.00855837};
const double  cLamK0_ThermBgdParamValues_3050[7] = {0.99519167,  0.12422358, -0.41231021,  0.28483665,  0.04387198, -0.09666038,  0.02375852};
  //-----------------------------------------------------------
const double*  cLamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};
const double*  cALamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};

//******************************************************************************************************************************************************
/*
//----- When LamKchP and ALamKchM only fit together (w. Full tag)
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00219879, -0.00268644, -0.01149827,  0.00650626,  0.00232835, -0.00129431,  0.00003672};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00561843,  0.01002434, -0.09349890,  0.06827250,  0.01508669, -0.02403859,  0.00531314};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01115048,  0.02327916, -0.23074565,  0.26927380, -0.11240518,  0.01733010, -0.00022366};
*/
/*
//----- When all (A)LamKchPM fit together (w. Full tag)
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00393853, -0.01448417,  0.02298048, -0.05442036,  0.06202911, -0.03010840,  0.00533448};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00738912, -0.00339132, -0.05754277,  0.01839328,  0.05296848, -0.03902484,  0.00772781};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01516858, -0.01012887, -0.12648928,  0.10471344,  0.02304729, -0.03787704,  0.00853539};
*/

/*
//----- When LamKchP and ALamKchM only fit together (w. PrimaryOnly tag)
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00199595, -0.00140720, -0.01362711,  0.00924591,  0.00259215, -0.00404227,  0.00123590};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00280348,  0.02621849, -0.12299037,  0.08825577,  0.01649416, -0.03146107,  0.00761372};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01581060, -0.05309941,  0.09551001, -0.33238860,  0.42778942, -0.21453488,  0.03772539};
*/

//----- When all (A)LamKchPM fit together (w. PrimaryOnly tag)
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00246060, -0.00155776, -0.01371020,  0.00787551,  0.00340504, -0.00310941,  0.00068475};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00411940,  0.01811882, -0.10615387,  0.07622336,  0.01552898, -0.02700245,  0.00629627};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01132102,  0.01220438, -0.16140574,  0.12207285,  0.02744645, -0.04395974,  0.00983796};

/*
//----- When LamKchP and ALamKchM only fit together (w. PrimaryAndShortDecays tag)
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00436074, -0.00892923, -0.00783283,  0.01144900, -0.00090738, -0.00305952,  0.00115675};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {0.99750988,  0.10032047, -0.41521056,  0.58831610, -0.40445082,  0.14015030, -0.01948778};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01361366,  0.01250787, -0.18784270,  0.15551294,  0.02630949, -0.05774623,  0.01445487};
*/

/*
//----- When all (A)LamKchPM fit together (w. PrimaryAndShortDecays tag)
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00375153, -0.00709952, -0.00690389,  0.00607043,  0.00160830, -0.00156365,  0.00029582};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00144856,  0.05223412, -0.23703079,  0.29627037, -0.16582429,  0.04503645, -0.00475687};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01057880,  0.04888972, -0.33638979,  0.45117775, -0.26636590,  0.08051541, -0.01031557};
*/
  //-----------------------------------------------------------
const double*  cLamKchP_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};
const double*  cALamKchM_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};

//******************************************************************************************************************************************************
/*
//----- When LamKchM and ALamKchP only fit together (w. Full tag)
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00597097, -0.03087241,  0.07669474, -0.15003231,  0.15233191, -0.07192563,  0.01275871};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00892050, -0.01464958, -0.02908122, -0.01888769,  0.07999158, -0.04941018,  0.00938512};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.02011740, -0.05318403,  0.01406824, -0.12389058,  0.21547553, -0.11776764,  0.02142437};
*/
/*
//----- When all (A)LamKchPM fit together (w. Full tag)
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00393853, -0.01448417,  0.02298048, -0.05442036,  0.06202911, -0.03010840,  0.00533448};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00738912, -0.00339132, -0.05754277,  0.01839328,  0.05296848, -0.03902484,  0.00772781};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01516858, -0.01012887, -0.12648928,  0.10471344,  0.02304729, -0.03787704,  0.00853539};
*/

/*
//----- When LamKchM and ALamKchP only fit together (w. PrimaryOnly tag)
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00454099, -0.02037230,  0.05806382, -0.11938320,  0.11426761, -0.04888295,  0.00778938};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00540760,  0.01015666, -0.08954536,  0.06431447,  0.01458193, -0.02257636,  0.00498479};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.00547852,  0.09240566, -0.47478592,  0.67508115, -0.45921848,  0.16342620, -0.02412154};
*/

//----- When all (A)LamKchPM fit together (w. PrimaryOnly tag)
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00246060, -0.00155776, -0.01371020,  0.00787551,  0.00340504, -0.00310941,  0.00068475};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00411940,  0.01811882, -0.10615387,  0.07622336,  0.01552898, -0.02700245,  0.00629627};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01132102,  0.01220438, -0.16140574,  0.12207285,  0.02744645, -0.04395974,  0.00983796};

/*
//----- When LamKchM and ALamKchP only fit together (w. PrimaryAndShortDecays tag)
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00530864, -0.03035763,  0.09098118, -0.16976300,  0.15359121, -0.06367303,  0.00988521};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00450264,  0.01404024, -0.09662215,  0.07038899,  0.01476665, -0.02532595,  0.00589435};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.00970931,  0.05934611, -0.38214026,  0.56192771, -0.39341606,  0.14689111, -0.02305879};
*/

/*
//----- When all (A)LamKchPM fit together (w. PrimaryAndShortDecays tag)
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00375153, -0.00709952, -0.00690389,  0.00607043,  0.00160830, -0.00156365,  0.00029582};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00144856,  0.05223412, -0.23703079,  0.29627037, -0.16582429,  0.04503645, -0.00475687};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01057880,  0.04888972, -0.33638979,  0.45117775, -0.26636590,  0.08051541, -0.01031557};
*/
  //-----------------------------------------------------------
const double*  cLamKchM_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};
const double*  cALamKchP_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};

//******************************************************************************************************************************************************

const double** cThermBgdParamValues[6] = {cLamK0_ThermBgdParamValues, cALamK0_ThermBgdParamValues, 
                                           cLamKchP_ThermBgdParamValues, cALamKchM_ThermBgdParamValues, 
                                           cLamKchM_ThermBgdParamValues, cALamKchP_ThermBgdParamValues};





