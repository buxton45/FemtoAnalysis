///////////////////////////////////////////////////////////////////////////
// Types_ThermBgdParams:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "Types_ThermBgdParams.h"
#include <cassert>




const double  cLamK0_ThermBgdParamValues_0010[7] = {1.00619, -0.020829,  0.0125535, -0.0112537, 0.00128651, 0.00636532, -0.00237831};
const double  cLamK0_ThermBgdParamValues_1030[7] = {1.00353,  0.024711, -0.123425,   0.0792277, 0.0142085, -0.0225213,   0.00461667};
const double  cLamK0_ThermBgdParamValues_3050[7] = {0.99171,  0.157127, -0.495511,   0.346828,  0.0498605, -0.119323,    0.0301585};
const double*  cLamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};

//const double  cALamK0_ThermBgdParamValues_0010[7];
//const double  cALamK0_ThermBgdParamValues_1030[7];
//const double  cALamK0_ThermBgdParamValues_3050[7];
const double*  cALamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};

  //-----------------------------------------------------------

const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00084, 0.0141217, -0.0806719, 0.133697, -0.112349, 0.0482764, -0.00820189};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00664, -0.0064652, -0.0448696, 0.0370136, 0.00573111, -0.0120334, 0.00284469};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01408, 0.00289488, -0.201177, 0.32152, -0.229079, 0.0846125, -0.0128624};
const double*  cLamKchP_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};

//const double  cALamKchM_ThermBgdParamValues_0010[7];
//const double  cALamKchM_ThermBgdParamValues_1030[7];
//const double  cALamKchM_ThermBgdParamValues_3050[7];
const double*  cALamKchM_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};

  //-----------------------------------------------------------

const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00363, -0.00616401, -0.0176615, 0.0183406, 0.00174698, -0.00672259, 0.00186687};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00647, 0.00226052, -0.091388, 0.127954, -0.0772218, 0.0236897, -0.00300451};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.02442, -0.0999212, 0.15968, -0.286472, 0.292335, -0.134547, 0.0228895};
const double*  cLamKchM_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};

//const double  cALamKchP_ThermBgdParamValues_0010[7];
//const double  cALamKchP_ThermBgdParamValues_1030[7];
//const double  cALamKchP_ThermBgdParamValues_3050[7];
const double*  cALamKchP_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};

  //-----------------------------------------------------------

const double** cThermBgdParamValues[6] = {cLamK0_ThermBgdParamValues, cALamK0_ThermBgdParamValues, 
                                           cLamKchP_ThermBgdParamValues, cALamKchM_ThermBgdParamValues, 
                                           cLamKchM_ThermBgdParamValues, cALamKchP_ThermBgdParamValues};





