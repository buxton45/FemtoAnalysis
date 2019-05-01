///////////////////////////////////////////////////////////////////////////
// Types_ThermBgdParams:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "Types_ThermBgdParams.h"
#include <cassert>




const double  cLamK0_ThermBgdParamValues_0010[7] = {1.00423520,  0.00000000, -0.05046272,  0.07674462, -0.05822510,  0.02489065, -0.00444368};
const double  cLamK0_ThermBgdParamValues_1030[7] = {1.00601903,  0.00000000, -0.01510665, -0.14392437,  0.23584743, -0.12607015,  0.02288242};
const double  cLamK0_ThermBgdParamValues_3050[7] = {1.01009361,  0.00000000, -0.01130339, -0.33020513,  0.52459198, -0.28158232,  0.05156574};
  //-----------------------------------------------------------
const double*  cLamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};
const double*  cALamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};

//******************************************************************************************************************************************************

const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00271112,  0.00000000, -0.02938766,  0.03422496, -0.01717411,  0.00603068, -0.00112565};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00671532,  0.00000000, -0.04870300, -0.02435201,  0.10585054, -0.06503261,  0.01226726};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01391997,  0.00000000, -0.12382734,  0.02951566,  0.13881934, -0.10231629,  0.02092779};
  //-----------------------------------------------------------
const double*  cLamKchP_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};
const double*  cALamKchM_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};

//******************************************************************************************************************************************************

const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00286011,  0.00000000, -0.02993600,  0.02524732,  0.00134525, -0.00610047,  0.00146731};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00673993,  0.00000000, -0.06044350,  0.00438327,  0.07884293, -0.05397173,  0.01063575};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01455025,  0.00000000, -0.14387853,  0.07994079,  0.08737466, -0.07891218,  0.01701091};
  //-----------------------------------------------------------
const double*  cLamKchM_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};
const double*  cALamKchP_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};

//******************************************************************************************************************************************************

const double** cThermBgdParamValues[6] = {cLamK0_ThermBgdParamValues, cALamK0_ThermBgdParamValues, 
                                          cLamKchP_ThermBgdParamValues, cALamKchM_ThermBgdParamValues, 
                                          cLamKchM_ThermBgdParamValues, cALamKchP_ThermBgdParamValues};





