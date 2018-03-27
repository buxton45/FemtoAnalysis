///////////////////////////////////////////////////////////////////////////
// Types_ThermBgdParams:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "Types_ThermBgdParams.h"
#include <cassert>




const double  cLamK0_ThermBgdParamValues_0010[7] = {1.00530031, -0.01452789,  0.00278913, -0.01805203,  0.02653437, -0.01179845,  0.00169353};
const double  cLamK0_ThermBgdParamValues_1030[7] = {1.00022827,  0.05843885, -0.23789186,  0.25472520, -0.11899432,  0.02614068, -0.00218247};
const double  cLamK0_ThermBgdParamValues_3050[7] = {0.98400987,  0.24885546, -0.86208158,  1.01453018, -0.55721039,  0.14853543, -0.01544907};

const double*  cLamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};
const double*  cALamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};

  //-----------------------------------------------------------
//----- When LamKchP and ALamKchM only fit together
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00134582,  0.00584625, -0.03979721,  0.04717569, -0.02471905,  0.00652870, -0.00070812};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00525959,  0.00863245, -0.10047383,  0.13040648, -0.07248078,  0.01970768, -0.00212067};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01560715, -0.01599201, -0.12304742,  0.17498130, -0.09290104,  0.02368836, -0.00242570};
/*
//----- When all (A)LamKchPM fit together
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00256972,  0.00019213, -0.03325485,  0.04509279, -0.02563115,  0.00720430, -0.00081415};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00632110,  0.00154589, -0.08313435,  0.10891484, -0.05844228,  0.01518648, -0.00156089};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01783697, -0.03269580, -0.08009869,  0.11909762, -0.05525593,  0.01119210, -0.00081785};
*/
const double*  cLamKchP_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};
const double*  cALamKchM_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};

  //-----------------------------------------------------------
//----- When LamKchM and ALamKchP only fit together
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00360811, -0.00458890, -0.02825906,  0.04434643, -0.02714457,  0.00801474, -0.00093208};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00703760, -0.00398266, -0.06839438,  0.08948639, -0.04522686,  0.01082094, -0.00101188};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01998869, -0.04868902, -0.03935482,  0.06633902, -0.01981400, -0.00055505,  0.00069224};
/*
//----- When all (A)LamKchPM fit together
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00256972,  0.00019213, -0.03325485,  0.04509279, -0.02563115,  0.00720430, -0.00081415};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00632110,  0.00154589, -0.08313435,  0.10891484, -0.05844228,  0.01518648, -0.00156089};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01783697, -0.03269580, -0.08009869,  0.11909762, -0.05525593,  0.01119210, -0.00081785};
*/
const double*  cLamKchM_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};
const double*  cALamKchP_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};

  //-----------------------------------------------------------

const double** cThermBgdParamValues[6] = {cLamK0_ThermBgdParamValues, cALamK0_ThermBgdParamValues, 
                                           cLamKchP_ThermBgdParamValues, cALamKchM_ThermBgdParamValues, 
                                           cLamKchM_ThermBgdParamValues, cALamKchP_ThermBgdParamValues};





