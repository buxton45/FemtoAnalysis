///////////////////////////////////////////////////////////////////////////
// Types_ThermBgdParams:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "Types_ThermBgdParams.h"
#include <cassert>




const double  cLamK0_ThermBgdParamValues_0010[7] = {1.00698258, -0.02291412,  0.01312921, -0.00875929,  0.00185616,  0.00336635, -0.00129966};
const double  cLamK0_ThermBgdParamValues_1030[7] = {1.00073498,  0.04608827, -0.17458269,  0.11667504,  0.01847243, -0.03675633,  0.00855837};
const double  cLamK0_ThermBgdParamValues_3050[7] = {0.99519167,  0.12422358, -0.41231021,  0.28483665,  0.04387198, -0.09666038,  0.02375852};

const double*  cLamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};
const double*  cALamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};

  //-----------------------------------------------------------
//----- When LamKchP and ALamKchM only fit together
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00219879, -0.00268644, -0.01149827,  0.00650626,  0.00232835, -0.00129431,  0.00003672};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00561843,  0.01002434, -0.09349890,  0.06827250,  0.01508669, -0.02403859,  0.00531314};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01115048,  0.02327916, -0.23074565,  0.26927380, -0.11240518,  0.01733010, -0.00022366};
/*
//----- When all (A)LamKchPM fit together
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00393853, -0.01448417,  0.02298048, -0.05442036,  0.06202911, -0.03010840,  0.00533448};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00738912, -0.00339132, -0.05754277,  0.01839328,  0.05296848, -0.03902484,  0.00772781};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01516858, -0.01012887, -0.12648928,  0.10471344,  0.02304729, -0.03787704,  0.00853539};
*/
const double*  cLamKchP_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};
const double*  cALamKchM_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};

  //-----------------------------------------------------------
//----- When LamKchM and ALamKchP only fit together
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00597097, -0.03087241,  0.07669474, -0.15003231,  0.15233191, -0.07192563,  0.01275871};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00892050, -0.01464958, -0.02908122, -0.01888769,  0.07999158, -0.04941018,  0.00938512};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.02011740, -0.05318403,  0.01406824, -0.12389058,  0.21547553, -0.11776764,  0.02142437};
/*
//----- When all (A)LamKchPM fit together
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00393853, -0.01448417,  0.02298048, -0.05442036,  0.06202911, -0.03010840,  0.00533448};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00738912, -0.00339132, -0.05754277,  0.01839328,  0.05296848, -0.03902484,  0.00772781};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01516858, -0.01012887, -0.12648928,  0.10471344,  0.02304729, -0.03787704,  0.00853539};
*/
const double*  cLamKchM_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};
const double*  cALamKchP_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};

  //-----------------------------------------------------------

const double** cThermBgdParamValues[6] = {cLamK0_ThermBgdParamValues, cALamK0_ThermBgdParamValues, 
                                           cLamKchP_ThermBgdParamValues, cALamKchM_ThermBgdParamValues, 
                                           cLamKchM_ThermBgdParamValues, cALamKchP_ThermBgdParamValues};





