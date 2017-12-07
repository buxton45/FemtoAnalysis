///////////////////////////////////////////////////////////////////////////
// Types_FitParamValues:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "Types_FitParamValues.h"
#include <cassert>


//------------------------------------------------------------------------------------------------------------------------------
//****************************************** No Residuals **********************************************************************
//------------------------------------------------------------------------------------------------------------------------------
const double cLamK0_ReF0_NoRes = -0.16;
const double cLamK0_ImF0_NoRes =  0.18;
const double cLamK0_D0_NoRes   =  3.57;

const double cLamK0_Radius_NoRes_0010 = 3.02;
const double cLamK0_Radius_NoRes_1030 = 2.27;
const double cLamK0_Radius_NoRes_3050 = 1.67;

//-----

const double  cLamK0_FitParamValues_NoRes_0010[6] = {0.40, cLamK0_Radius_NoRes_0010, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, 0.1};
const double  cLamK0_FitParamValues_NoRes_1030[6] = {0.40, cLamK0_Radius_NoRes_1030, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, 0.1};
const double  cLamK0_FitParamValues_NoRes_3050[6] = {0.40, cLamK0_Radius_NoRes_3050, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, 0.1};
const double* cLamK0_FitParamValues_NoRes[3] = {cLamK0_FitParamValues_NoRes_0010, cLamK0_FitParamValues_NoRes_1030, cLamK0_FitParamValues_NoRes_3050};

const double  cALamK0_FitParamValues_NoRes_0010[6] = {0.40, cLamK0_Radius_NoRes_0010, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, 0.1};
const double  cALamK0_FitParamValues_NoRes_1030[6] = {0.40, cLamK0_Radius_NoRes_1030, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, 0.1};
const double  cALamK0_FitParamValues_NoRes_3050[6] = {0.40, cLamK0_Radius_NoRes_3050, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, 0.1};
const double* cALamK0_FitParamValues_NoRes[3] = {cALamK0_FitParamValues_NoRes_0010, cALamK0_FitParamValues_NoRes_1030, cALamK0_FitParamValues_NoRes_3050};

//------------------------------------------------------------------------------------------

const double cLamKchP_ReF0_NoRes = -0.69;
const double cLamKchP_ImF0_NoRes =  0.39;
const double cLamKchP_D0_NoRes   =  0.64;

const double cLamKchP_Radius_NoRes_0010 = 4.04;
const double cLamKchP_Radius_NoRes_1030 = 3.92;
const double cLamKchP_Radius_NoRes_3050 = 3.72;

//-----

const double  cLamKchP_FitParamValues_NoRes_0010[6] = {0.38, cLamKchP_Radius_NoRes_0010, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, 0.1};
const double  cLamKchP_FitParamValues_NoRes_1030[6] = {0.48, cLamKchP_Radius_NoRes_1030, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, 0.1};
const double  cLamKchP_FitParamValues_NoRes_3050[6] = {0.64, cLamKchP_Radius_NoRes_3050, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, 0.1};
const double* cLamKchP_FitParamValues_NoRes[3] = {cLamKchP_FitParamValues_NoRes_0010, cLamKchP_FitParamValues_NoRes_1030, cLamKchP_FitParamValues_NoRes_3050};

const double  cALamKchM_FitParamValues_NoRes_0010[6] = {0.37, cLamKchP_Radius_NoRes_0010, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, 0.1};
const double  cALamKchM_FitParamValues_NoRes_1030[6] = {0.41, cLamKchP_Radius_NoRes_1030, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, 0.1};
const double  cALamKchM_FitParamValues_NoRes_3050[6] = {0.62, cLamKchP_Radius_NoRes_3050, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, 0.1};
const double* cALamKchM_FitParamValues_NoRes[3] = {cALamKchM_FitParamValues_NoRes_0010, cALamKchM_FitParamValues_NoRes_1030, cALamKchM_FitParamValues_NoRes_3050};

//------------------------------------------------------------------------------------------

const double cLamKchM_ReF0_NoRes =  0.18;
const double cLamKchM_ImF0_NoRes =  0.45;
const double cLamKchM_D0_NoRes   =  -5.29;

const double cLamKchM_Radius_NoRes_0010 = 4.79;
const double cLamKchM_Radius_NoRes_1030 = 4.00;
const double cLamKchM_Radius_NoRes_3050 = 2.11;

//-----

const double  cLamKchM_FitParamValues_NoRes_0010[6] = {0.45, cLamKchM_Radius_NoRes_0010, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, 0.1};
const double  cLamKchM_FitParamValues_NoRes_1030[6] = {0.40, cLamKchM_Radius_NoRes_1030, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, 0.1};
const double  cLamKchM_FitParamValues_NoRes_3050[6] = {0.20, cLamKchM_Radius_NoRes_3050, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, 0.1};
const double* cLamKchM_FitParamValues_NoRes[3] = {cLamKchM_FitParamValues_NoRes_0010, cLamKchM_FitParamValues_NoRes_1030, cLamKchM_FitParamValues_NoRes_3050};

const double  cALamKchP_FitParamValues_NoRes_0010[6] = {0.48, cLamKchM_Radius_NoRes_0010, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, 0.1};
const double  cALamKchP_FitParamValues_NoRes_1030[6] = {0.49, cLamKchM_Radius_NoRes_1030, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, 0.1};
const double  cALamKchP_FitParamValues_NoRes_3050[6] = {0.22, cLamKchM_Radius_NoRes_3050, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, 0.1};
const double* cALamKchP_FitParamValues_NoRes[3] = {cALamKchP_FitParamValues_NoRes_0010, cALamKchP_FitParamValues_NoRes_1030, cALamKchP_FitParamValues_NoRes_3050};

//------------------------------------------------------------------------------------------

const double** cFitParamValues_NoRes[6] = {cLamK0_FitParamValues_NoRes, cALamK0_FitParamValues_NoRes, 
                                           cLamKchP_FitParamValues_NoRes, cALamKchM_FitParamValues_NoRes, 
                                           cLamKchM_FitParamValues_NoRes, cALamKchP_FitParamValues_NoRes};




//------------------------------------------------------------------------------------------------------------------------------
//****************************************** 10 Residuals **********************************************************************
//------------------------------------------------------------------------------------------------------------------------------
const double cLamK0_ReF0_10Res = -0.26;
const double cLamK0_ImF0_10Res =  0.17;
const double cLamK0_D0_10Res   =  2.53;

const double cLamK0_Radius_10Res_0010 = 2.97;
const double cLamK0_Radius_10Res_1030 = 2.30;
const double cLamK0_Radius_10Res_3050 = 1.70;

//-----

const double  cLamK0_FitParamValues_10Res_0010[6] = {0.60, cLamK0_Radius_10Res_0010, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, 0.1};
const double  cLamK0_FitParamValues_10Res_1030[6] = {0.60, cLamK0_Radius_10Res_1030, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, 0.1};
const double  cLamK0_FitParamValues_10Res_3050[6] = {0.60, cLamK0_Radius_10Res_3050, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, 0.1};
const double* cLamK0_FitParamValues_10Res[3] = {cLamK0_FitParamValues_10Res_0010, cLamK0_FitParamValues_10Res_1030, cLamK0_FitParamValues_10Res_3050};

const double  cALamK0_FitParamValues_10Res_0010[6] = {0.60, cLamK0_Radius_10Res_0010, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, 0.1};
const double  cALamK0_FitParamValues_10Res_1030[6] = {0.60, cLamK0_Radius_10Res_1030, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, 0.1};
const double  cALamK0_FitParamValues_10Res_3050[6] = {0.60, cLamK0_Radius_10Res_3050, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, 0.1};
const double* cALamK0_FitParamValues_10Res[3] = {cALamK0_FitParamValues_10Res_0010, cALamK0_FitParamValues_10Res_1030, cALamK0_FitParamValues_10Res_3050};

//------------------------------------------------------------------------------------------

const double cLamKchP_ReF0_10Res = -1.51;
const double cLamKchP_ImF0_10Res =  0.65;
const double cLamKchP_D0_10Res   =  1.13;

const double cLamKchP_Radius_10Res_0010 = 4.98;
const double cLamKchP_Radius_10Res_1030 = 4.76;
const double cLamKchP_Radius_10Res_3050 = 3.55;

//-----

const double  cLamKchP_FitParamValues_10Res_0010[6] = {0.96, cLamKchP_Radius_10Res_0010, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, 0.1};
const double  cLamKchP_FitParamValues_10Res_1030[6] = {1.18, cLamKchP_Radius_10Res_1030, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, 0.1};
const double  cLamKchP_FitParamValues_10Res_3050[6] = {1.01, cLamKchP_Radius_10Res_3050, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, 0.1};
const double* cLamKchP_FitParamValues_10Res[3] = {cLamKchP_FitParamValues_10Res_0010, cLamKchP_FitParamValues_10Res_1030, cLamKchP_FitParamValues_10Res_3050};

const double  cALamKchM_FitParamValues_10Res_0010[6] = {0.94, cLamKchP_Radius_10Res_0010, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, 0.1};
const double  cALamKchM_FitParamValues_10Res_1030[6] = {0.99, cLamKchP_Radius_10Res_1030, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, 0.1};
const double  cALamKchM_FitParamValues_10Res_3050[6] = {0.98, cLamKchP_Radius_10Res_3050, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, 0.1};
const double* cALamKchM_FitParamValues_10Res[3] = {cALamKchM_FitParamValues_10Res_0010, cALamKchM_FitParamValues_10Res_1030, cALamKchM_FitParamValues_10Res_3050};

//------------------------------------------------------------------------------------------

const double cLamKchM_ReF0_10Res =  0.45;
const double cLamKchM_ImF0_10Res =  0.52;
const double cLamKchM_D0_10Res   =  -4.81;

const double cLamKchM_Radius_10Res_0010 = 6.21;
const double cLamKchM_Radius_10Res_1030 = 4.86;
const double cLamKchM_Radius_10Res_3050 = 2.86;

//-----

const double  cLamKchM_FitParamValues_10Res_0010[6] = {1.50, cLamKchM_Radius_10Res_0010, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, 0.1};
const double  cLamKchM_FitParamValues_10Res_1030[6] = {1.15, cLamKchM_Radius_10Res_1030, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, 0.1};
const double  cLamKchM_FitParamValues_10Res_3050[6] = {1.07, cLamKchM_Radius_10Res_3050, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, 0.1};
const double* cLamKchM_FitParamValues_10Res[3] = {cLamKchM_FitParamValues_10Res_0010, cLamKchM_FitParamValues_10Res_1030, cLamKchM_FitParamValues_10Res_3050};

const double  cALamKchP_FitParamValues_10Res_0010[6] = {1.49, cLamKchM_Radius_10Res_0010, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, 0.1};
const double  cALamKchP_FitParamValues_10Res_1030[6] = {1.41, cLamKchM_Radius_10Res_1030, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, 0.1};
const double  cALamKchP_FitParamValues_10Res_3050[6] = {0.80, cLamKchM_Radius_10Res_3050, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, 0.1};
const double* cALamKchP_FitParamValues_10Res[3] = {cALamKchP_FitParamValues_10Res_0010, cALamKchP_FitParamValues_10Res_1030, cALamKchP_FitParamValues_10Res_3050};

//------------------------------------------------------------------------------------------

const double** cFitParamValues_10Res[6] = {cLamK0_FitParamValues_10Res, cALamK0_FitParamValues_10Res, 
                                           cLamKchP_FitParamValues_10Res, cALamKchM_FitParamValues_10Res, 
                                           cLamKchM_FitParamValues_10Res, cALamKchP_FitParamValues_10Res};



//------------------------------------------------------------------------------------------------------------------------------
//****************************************** 3 Residuals ***********************************************************************
//------------------------------------------------------------------------------------------------------------------------------
const double cLamK0_ReF0_3Res = -0.27;
const double cLamK0_ImF0_3Res =  0.21;
const double cLamK0_D0_3Res   =  2.66;

const double cLamK0_Radius_3Res_0010 = 2.91;
const double cLamK0_Radius_3Res_1030 = 2.22;
const double cLamK0_Radius_3Res_3050 = 1.64;

//-----

const double  cLamK0_FitParamValues_3Res_0010[6] = {0.60, cLamK0_Radius_3Res_0010, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, 0.1};
const double  cLamK0_FitParamValues_3Res_1030[6] = {0.60, cLamK0_Radius_3Res_1030, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, 0.1};
const double  cLamK0_FitParamValues_3Res_3050[6] = {0.60, cLamK0_Radius_3Res_3050, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, 0.1};
const double* cLamK0_FitParamValues_3Res[3] = {cLamK0_FitParamValues_3Res_0010, cLamK0_FitParamValues_3Res_1030, cLamK0_FitParamValues_3Res_3050};

const double  cALamK0_FitParamValues_3Res_0010[6] = {0.60, cLamK0_Radius_3Res_0010, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, 0.1};
const double  cALamK0_FitParamValues_3Res_1030[6] = {0.60, cLamK0_Radius_3Res_1030, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, 0.1};
const double  cALamK0_FitParamValues_3Res_3050[6] = {0.60, cLamK0_Radius_3Res_3050, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, 0.1};
const double* cALamK0_FitParamValues_3Res[3] = {cALamK0_FitParamValues_3Res_0010, cALamK0_FitParamValues_3Res_1030, cALamK0_FitParamValues_3Res_3050};

//------------------------------------------------------------------------------------------

const double cLamKchP_ReF0_3Res = -1.24;
const double cLamKchP_ImF0_3Res =  0.50;
const double cLamKchP_D0_3Res   =  1.11;

const double cLamKchP_Radius_3Res_0010 = 4.43;
const double cLamKchP_Radius_3Res_1030 = 4.34;
const double cLamKchP_Radius_3Res_3050 = 3.38;

//-----

const double  cLamKchP_FitParamValues_3Res_0010[6] = {0.84, cLamKchP_Radius_3Res_0010, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, 0.1};
const double  cLamKchP_FitParamValues_3Res_1030[6] = {1.09, cLamKchP_Radius_3Res_1030, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, 0.1};
const double  cLamKchP_FitParamValues_3Res_3050[6] = {1.02, cLamKchP_Radius_3Res_3050, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, 0.1};
const double* cLamKchP_FitParamValues_3Res[3] = {cLamKchP_FitParamValues_3Res_0010, cLamKchP_FitParamValues_3Res_1030, cLamKchP_FitParamValues_3Res_3050};

const double  cALamKchM_FitParamValues_3Res_0010[6] = {0.82, cLamKchP_Radius_3Res_0010, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, 0.1};
const double  cALamKchM_FitParamValues_3Res_1030[6] = {0.90, cLamKchP_Radius_3Res_1030, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, 0.1};
const double  cALamKchM_FitParamValues_3Res_3050[6] = {0.97, cLamKchP_Radius_3Res_3050, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, 0.1};
const double* cALamKchM_FitParamValues_3Res[3] = {cALamKchM_FitParamValues_3Res_0010, cALamKchM_FitParamValues_3Res_1030, cALamKchM_FitParamValues_3Res_3050};

//------------------------------------------------------------------------------------------

const double cLamKchM_ReF0_3Res =  0.34;
const double cLamKchM_ImF0_3Res =  0.42;
const double cLamKchM_D0_3Res   =  -5.72;

const double cLamKchM_Radius_3Res_0010 = 6.02;
const double cLamKchM_Radius_3Res_1030 = 4.74;
const double cLamKchM_Radius_3Res_3050 = 2.75;

//-----

const double  cLamKchM_FitParamValues_3Res_0010[6] = {1.55, cLamKchM_Radius_3Res_0010, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, 0.1};
const double  cLamKchM_FitParamValues_3Res_1030[6] = {1.19, cLamKchM_Radius_3Res_1030, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, 0.1};
const double  cLamKchM_FitParamValues_3Res_3050[6] = {1.08, cLamKchM_Radius_3Res_3050, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, 0.1};
const double* cLamKchM_FitParamValues_3Res[3] = {cLamKchM_FitParamValues_3Res_0010, cLamKchM_FitParamValues_3Res_1030, cLamKchM_FitParamValues_3Res_3050};

const double  cALamKchP_FitParamValues_3Res_0010[6] = {1.54, cLamKchM_Radius_3Res_0010, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, 0.1};
const double  cALamKchP_FitParamValues_3Res_1030[6] = {1.46, cLamKchM_Radius_3Res_1030, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, 0.1};
const double  cALamKchP_FitParamValues_3Res_3050[6] = {0.80, cLamKchM_Radius_3Res_3050, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, 0.1};
const double* cALamKchP_FitParamValues_3Res[3] = {cALamKchP_FitParamValues_3Res_0010, cALamKchP_FitParamValues_3Res_1030, cALamKchP_FitParamValues_3Res_3050};

//------------------------------------------------------------------------------------------

const double** cFitParamValues_3Res[6] = {cLamK0_FitParamValues_3Res, cALamK0_FitParamValues_3Res, 
                                           cLamKchP_FitParamValues_3Res, cALamKchM_FitParamValues_3Res, 
                                           cLamKchM_FitParamValues_3Res, cALamKchP_FitParamValues_3Res};

//------------------------------------------------------------------------------------------------------------------------------
//******************************************************************************************************************************
//------------------------------------------------------------------------------------------------------------------------------


const double*** cFitParamValues[3] = {cFitParamValues_NoRes, cFitParamValues_10Res, cFitParamValues_3Res};



