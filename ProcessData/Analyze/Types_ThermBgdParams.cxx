///////////////////////////////////////////////////////////////////////////
// Types_ThermBgdParams:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "Types_ThermBgdParams.h"
#include <cassert>




const double  cLamK0_ThermBgdParamValues_0010[7] = {1.00685371, -0.02283374,  0.01796114, -0.02615002,  0.02491367, -0.00953017,  0.00126252};
const double  cLamK0_ThermBgdParamValues_1030[7] = {0.99948146,  0.06618974, -0.26299477,  0.28401306, -0.13496627,  0.03024689, -0.00258031};
const double  cLamK0_ThermBgdParamValues_3050[7] = {0.99072158,  0.18640644, -0.68128008,  0.79389433, -0.42581996,  0.11052764, -0.01116572};

const double*  cLamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};
const double*  cALamK0_ThermBgdParamValues[3] = {cLamK0_ThermBgdParamValues_0010, cLamK0_ThermBgdParamValues_1030, cLamK0_ThermBgdParamValues_3050};

  //-----------------------------------------------------------
//----- When LamKchP and ALamKchM only fit together
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00201685,  0.00213347, -0.02511291,  0.01530264,  0.00463876, -0.00448999,  0.00072070};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00290306,  0.03691869, -0.18444741,  0.20665661, -0.09153570,  0.01705584, -0.00093664};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.00945247,  0.04105291, -0.28787474,  0.33678902, -0.14843588,  0.02709794, -0.00142652};
/*
//----- When all (A)LamKchPM fit together
const double  cLamKchP_ThermBgdParamValues_0010[7] = {1.00299127, -0.00098046, -0.02740270,  0.02587529, -0.00410908, -0.00184383,  0.00048164};
const double  cLamKchP_ThermBgdParamValues_1030[7] = {1.00500904,  0.02222310, -0.15044241,  0.16842872, -0.06930795,  0.01066954, -0.00022459};
const double  cLamKchP_ThermBgdParamValues_3050[7] = {1.01270736,  0.01639683, -0.22112452,  0.24814525, -0.08867010,  0.00762095,  0.00097910};
*/
const double*  cLamKchP_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};
const double*  cALamKchM_ThermBgdParamValues[3] = {cLamKchP_ThermBgdParamValues_0010, cLamKchP_ThermBgdParamValues_1030, cLamKchP_ThermBgdParamValues_3050};

  //-----------------------------------------------------------
//----- When LamKchM and ALamKchP only fit together
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00406785, -0.00562252, -0.02418825,  0.02822291, -0.00699019, -0.00117052,  0.00049263};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00702551,  0.00789538, -0.11696622,  0.13053939, -0.04719024,  0.00430984,  0.00048219};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01596225, -0.00706925, -0.16005701,  0.16949828, -0.03711488, -0.00867313,  0.00291430};
/*
//----- When all (A)LamKchPM fit together
const double  cLamKchM_ThermBgdParamValues_0010[7] = {1.00299127, -0.00098046, -0.02740270,  0.02587529, -0.00410908, -0.00184383,  0.00048164};
const double  cLamKchM_ThermBgdParamValues_1030[7] = {1.00500904,  0.02222310, -0.15044241,  0.16842872, -0.06930795,  0.01066954, -0.00022459};
const double  cLamKchM_ThermBgdParamValues_3050[7] = {1.01270736,  0.01639683, -0.22112452,  0.24814525, -0.08867010,  0.00762095,  0.00097910};
*/
const double*  cLamKchM_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};
const double*  cALamKchP_ThermBgdParamValues[3] = {cLamKchM_ThermBgdParamValues_0010, cLamKchM_ThermBgdParamValues_1030, cLamKchM_ThermBgdParamValues_3050};

  //-----------------------------------------------------------

const double** cThermBgdParamValues[6] = {cLamK0_ThermBgdParamValues, cALamK0_ThermBgdParamValues, 
                                           cLamKchP_ThermBgdParamValues, cALamKchM_ThermBgdParamValues, 
                                           cLamKchM_ThermBgdParamValues, cALamKchP_ThermBgdParamValues};





