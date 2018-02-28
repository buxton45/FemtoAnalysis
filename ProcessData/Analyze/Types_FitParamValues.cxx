///////////////////////////////////////////////////////////////////////////
// Types_FitParamValues:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "Types_FitParamValues.h"
#include <cassert>

const double cNormArray[3] = {1.0, 0.0, 0.0};
//------------------------------------------------------------------------------------------------------------------------------
//****************************************** No Residuals **********************************************************************
//------------------------------------------------------------------------------------------------------------------------------
const double cLamK0_Lambda_NoRes_0010[3] = {0.40, 0.19, 0.12};
const double cLamK0_Lambda_NoRes_1030[3] = {0.40, 0.19, 0.12};
const double cLamK0_Lambda_NoRes_3050[3] = {0.40, 0.19, 0.12};

const double cALamK0_Lambda_NoRes_0010[3] = {0.40, 0.19, 0.12};
const double cALamK0_Lambda_NoRes_1030[3] = {0.40, 0.19, 0.12};
const double cALamK0_Lambda_NoRes_3050[3] = {0.40, 0.19, 0.12};

const double cLamK0_Radius_NoRes_0010[3] = {3.02, 0.54, 0.33};
const double cLamK0_Radius_NoRes_1030[3] = {2.27, 0.41, 0.32};
const double cLamK0_Radius_NoRes_3050[3] = {1.67, 0.31, 0.28};

const double cLamK0_ReF0_NoRes[3] = {-0.16, 0.03, 0.04};
const double cLamK0_ImF0_NoRes[3] =  {0.18, 0.08, 0.06};
const double cLamK0_D0_NoRes[3]   =  {3.57, 0.95, 2.84};

//-----

const double*  cLamK0_FitParamValues_NoRes_0010[6] = {cLamK0_Lambda_NoRes_0010, cLamK0_Radius_NoRes_0010, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, cNormArray};
const double*  cLamK0_FitParamValues_NoRes_1030[6] = {cLamK0_Lambda_NoRes_1030, cLamK0_Radius_NoRes_1030, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, cNormArray};
const double*  cLamK0_FitParamValues_NoRes_3050[6] = {cLamK0_Lambda_NoRes_3050, cLamK0_Radius_NoRes_3050, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, cNormArray};
const double** cLamK0_FitParamValues_NoRes[3] = {cLamK0_FitParamValues_NoRes_0010, cLamK0_FitParamValues_NoRes_1030, cLamK0_FitParamValues_NoRes_3050};

const double*  cALamK0_FitParamValues_NoRes_0010[6] = {cALamK0_Lambda_NoRes_0010, cLamK0_Radius_NoRes_0010, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, cNormArray};
const double*  cALamK0_FitParamValues_NoRes_1030[6] = {cALamK0_Lambda_NoRes_1030, cLamK0_Radius_NoRes_1030, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, cNormArray};
const double*  cALamK0_FitParamValues_NoRes_3050[6] = {cALamK0_Lambda_NoRes_3050, cLamK0_Radius_NoRes_3050, cLamK0_ReF0_NoRes, cLamK0_ImF0_NoRes, cLamK0_D0_NoRes, cNormArray};
const double** cALamK0_FitParamValues_NoRes[3] = {cALamK0_FitParamValues_NoRes_0010, cALamK0_FitParamValues_NoRes_1030, cALamK0_FitParamValues_NoRes_3050};

//------------------------------------------------------------------------------------------

const double cLamKchP_Lambda_NoRes_0010[3] = {0.38, 0.09, 0.22};
const double cLamKchP_Lambda_NoRes_1030[3] = {0.48, 0.13, 0.24};
const double cLamKchP_Lambda_NoRes_3050[3] = {0.64, 0.20, 0.20};

const double cALamKchM_Lambda_NoRes_0010[3] = {0.37, 0.08, 0.22};
const double cALamKchM_Lambda_NoRes_1030[3] = {0.41, 0.11, 0.20};
const double cALamKchM_Lambda_NoRes_3050[3] = {0.62, 0.19, 0.20};

const double cLamKchP_Radius_NoRes_0010[3] = {4.04, 0.38, 0.83};
const double cLamKchP_Radius_NoRes_1030[3] = {3.92, 0.45, 0.66};
const double cLamKchP_Radius_NoRes_3050[3] = {3.72, 0.55, 0.42};

const double cLamKchP_ReF0_NoRes[3] = {-0.69, 0.16, 0.22};
const double cLamKchP_ImF0_NoRes[3] =  {0.39, 0.14, 0.11};
const double cLamKchP_D0_NoRes[3]   =  {0.64, 0.53, 1.62};

//-----

const double*  cLamKchP_FitParamValues_NoRes_0010[6] = {cLamKchP_Lambda_NoRes_0010, cLamKchP_Radius_NoRes_0010, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, cNormArray};
const double*  cLamKchP_FitParamValues_NoRes_1030[6] = {cLamKchP_Lambda_NoRes_1030, cLamKchP_Radius_NoRes_1030, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, cNormArray};
const double*  cLamKchP_FitParamValues_NoRes_3050[6] = {cLamKchP_Lambda_NoRes_3050, cLamKchP_Radius_NoRes_3050, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, cNormArray};
const double** cLamKchP_FitParamValues_NoRes[3] = {cLamKchP_FitParamValues_NoRes_0010, cLamKchP_FitParamValues_NoRes_1030, cLamKchP_FitParamValues_NoRes_3050};

const double*  cALamKchM_FitParamValues_NoRes_0010[6] = {cALamKchM_Lambda_NoRes_0010, cLamKchP_Radius_NoRes_0010, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, cNormArray};
const double*  cALamKchM_FitParamValues_NoRes_1030[6] = {cALamKchM_Lambda_NoRes_1030, cLamKchP_Radius_NoRes_1030, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, cNormArray};
const double*  cALamKchM_FitParamValues_NoRes_3050[6] = {cALamKchM_Lambda_NoRes_3050, cLamKchP_Radius_NoRes_3050, cLamKchP_ReF0_NoRes, cLamKchP_ImF0_NoRes, cLamKchP_D0_NoRes, cNormArray};
const double** cALamKchM_FitParamValues_NoRes[3] = {cALamKchM_FitParamValues_NoRes_0010, cALamKchM_FitParamValues_NoRes_1030, cALamKchM_FitParamValues_NoRes_3050};

//------------------------------------------------------------------------------------------

const double cLamKchM_Lambda_NoRes_0010[3] = {0.45, 0.16, 0.19};
const double cLamKchM_Lambda_NoRes_1030[3] = {0.40, 0.15, 0.20};
const double cLamKchM_Lambda_NoRes_3050[3] = {0.20, 0.08, 0.13};

const double cALamKchP_Lambda_NoRes_0010[3] = {0.48, 0.17, 0.15};
const double cALamKchP_Lambda_NoRes_1030[3] = {0.49, 0.18, 0.15};
const double cALamKchP_Lambda_NoRes_3050[3] = {0.22, 0.08, 0.11};

const double cLamKchM_Radius_NoRes_0010[3] = {4.79, 0.79, 1.38};
const double cLamKchM_Radius_NoRes_1030[3] = {4.00, 0.72, 0.98};
const double cLamKchM_Radius_NoRes_3050[3] = {2.11, 0.52, 0.46};

const double cLamKchM_ReF0_NoRes[3] =  {0.18, 0.13, 0.10};
const double cLamKchM_ImF0_NoRes[3] =  {0.45, 0.18, 0.18};
const double cLamKchM_D0_NoRes[3]   =  {-5.29, 2.90, 7.66};

//-----

const double*  cLamKchM_FitParamValues_NoRes_0010[6] = {cLamKchM_Lambda_NoRes_0010, cLamKchM_Radius_NoRes_0010, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, cNormArray};
const double*  cLamKchM_FitParamValues_NoRes_1030[6] = {cLamKchM_Lambda_NoRes_1030, cLamKchM_Radius_NoRes_1030, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, cNormArray};
const double*  cLamKchM_FitParamValues_NoRes_3050[6] = {cLamKchM_Lambda_NoRes_3050, cLamKchM_Radius_NoRes_3050, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, cNormArray};
const double** cLamKchM_FitParamValues_NoRes[3] = {cLamKchM_FitParamValues_NoRes_0010, cLamKchM_FitParamValues_NoRes_1030, cLamKchM_FitParamValues_NoRes_3050};

const double*  cALamKchP_FitParamValues_NoRes_0010[6] = {cALamKchP_Lambda_NoRes_0010, cLamKchM_Radius_NoRes_0010, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, cNormArray};
const double*  cALamKchP_FitParamValues_NoRes_1030[6] = {cALamKchP_Lambda_NoRes_1030, cLamKchM_Radius_NoRes_1030, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, cNormArray};
const double*  cALamKchP_FitParamValues_NoRes_3050[6] = {cALamKchP_Lambda_NoRes_3050, cLamKchM_Radius_NoRes_3050, cLamKchM_ReF0_NoRes, cLamKchM_ImF0_NoRes, cLamKchM_D0_NoRes, cNormArray};
const double** cALamKchP_FitParamValues_NoRes[3] = {cALamKchP_FitParamValues_NoRes_0010, cALamKchP_FitParamValues_NoRes_1030, cALamKchP_FitParamValues_NoRes_3050};

//------------------------------------------------------------------------------------------

const double*** cFitParamValues_NoRes[6] = {cLamK0_FitParamValues_NoRes, cALamK0_FitParamValues_NoRes, 
                                            cLamKchP_FitParamValues_NoRes, cALamKchM_FitParamValues_NoRes, 
                                            cLamKchM_FitParamValues_NoRes, cALamKchP_FitParamValues_NoRes};




//------------------------------------------------------------------------------------------------------------------------------
//****************************************** 10 Residuals **********************************************************************
//------------------------------------------------------------------------------------------------------------------------------
const double cLamK0_Lambda_10Res_0010[3] = {0.60, 0.63, 0.17};
const double cLamK0_Lambda_10Res_1030[3] = {0.60, 0.63, 0.17};
const double cLamK0_Lambda_10Res_3050[3] = {0.60, 0.63, 0.17};

const double cALamK0_Lambda_10Res_0010[3] = {0.60, 0.63, 0.17};
const double cALamK0_Lambda_10Res_1030[3] = {0.60, 0.63, 0.17};
const double cALamK0_Lambda_10Res_3050[3] = {0.60, 0.63, 0.17};

const double cLamK0_Radius_10Res_0010[3] = {2.94, 0.45, 0.35};
const double cLamK0_Radius_10Res_1030[3] = {2.39, 0.38, 0.25};
const double cLamK0_Radius_10Res_3050[3] = {1.81, 0.29, 0.12};

const double cLamK0_ReF0_10Res[3] = {-0.40, 0.12, 0.17};
const double cLamK0_ImF0_10Res[3] =  {0.17, 0.08, 0.12};
const double cLamK0_D0_10Res[3]   =  {1.94, 0.47, 0.77};

//-----

const double*  cLamK0_FitParamValues_10Res_0010[6] = {cLamK0_Lambda_10Res_0010, cLamK0_Radius_10Res_0010, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, cNormArray};
const double*  cLamK0_FitParamValues_10Res_1030[6] = {cLamK0_Lambda_10Res_1030, cLamK0_Radius_10Res_1030, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, cNormArray};
const double*  cLamK0_FitParamValues_10Res_3050[6] = {cLamK0_Lambda_10Res_3050, cLamK0_Radius_10Res_3050, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, cNormArray};
const double** cLamK0_FitParamValues_10Res[3] = {cLamK0_FitParamValues_10Res_0010, cLamK0_FitParamValues_10Res_1030, cLamK0_FitParamValues_10Res_3050};

const double*  cALamK0_FitParamValues_10Res_0010[6] = {cALamK0_Lambda_10Res_0010, cLamK0_Radius_10Res_0010, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, cNormArray};
const double*  cALamK0_FitParamValues_10Res_1030[6] = {cALamK0_Lambda_10Res_1030, cLamK0_Radius_10Res_1030, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, cNormArray};
const double*  cALamK0_FitParamValues_10Res_3050[6] = {cALamK0_Lambda_10Res_3050, cLamK0_Radius_10Res_3050, cLamK0_ReF0_10Res, cLamK0_ImF0_10Res, cLamK0_D0_10Res, cNormArray};
const double** cALamK0_FitParamValues_10Res[3] = {cALamK0_FitParamValues_10Res_0010, cALamK0_FitParamValues_10Res_1030, cALamK0_FitParamValues_10Res_3050};

//------------------------------------------------------------------------------------------

const double cLamKchP_Lambda_10Res_0010[3] = {1.51, 0.56, 0.27};
const double cLamKchP_Lambda_10Res_1030[3] = {1.47, 0.55, 0.31};
const double cLamKchP_Lambda_10Res_3050[3] = {1.10, 0.30, 0.27};

const double cALamKchM_Lambda_10Res_0010[3] = {1.52, 0.58, 0.33};
const double cALamKchM_Lambda_10Res_1030[3] = {1.28, 0.47, 0.25};
const double cALamKchM_Lambda_10Res_3050[3] = {1.06, 0.28, 0.16};

const double cLamKchP_Radius_10Res_0010[3] = {5.92, 1.08, 0.51};
const double cLamKchP_Radius_10Res_1030[3] = {4.98, 0.86, 0.40};
const double cLamKchP_Radius_10Res_3050[3] = {3.38, 0.45, 0.28};

const double cLamKchP_ReF0_10Res[3] = {-1.38, 0.32, 0.34};
const double cLamKchP_ImF0_10Res[3] =  {0.61, 0.34, 0.20};
const double cLamKchP_D0_10Res[3]   =  {0.97, 0.66, 0.42};

//-----

const double*  cLamKchP_FitParamValues_10Res_0010[6] = {cLamKchP_Lambda_10Res_0010, cLamKchP_Radius_10Res_0010, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, cNormArray};
const double*  cLamKchP_FitParamValues_10Res_1030[6] = {cLamKchP_Lambda_10Res_1030, cLamKchP_Radius_10Res_1030, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, cNormArray};
const double*  cLamKchP_FitParamValues_10Res_3050[6] = {cLamKchP_Lambda_10Res_3050, cLamKchP_Radius_10Res_3050, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, cNormArray};
const double** cLamKchP_FitParamValues_10Res[3] = {cLamKchP_FitParamValues_10Res_0010, cLamKchP_FitParamValues_10Res_1030, cLamKchP_FitParamValues_10Res_3050};

const double*  cALamKchM_FitParamValues_10Res_0010[6] = {cALamKchM_Lambda_10Res_0010, cLamKchP_Radius_10Res_0010, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, cNormArray};
const double*  cALamKchM_FitParamValues_10Res_1030[6] = {cALamKchM_Lambda_10Res_1030, cLamKchP_Radius_10Res_1030, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, cNormArray};
const double*  cALamKchM_FitParamValues_10Res_3050[6] = {cALamKchM_Lambda_10Res_3050, cLamKchP_Radius_10Res_3050, cLamKchP_ReF0_10Res, cLamKchP_ImF0_10Res, cLamKchP_D0_10Res, cNormArray};
const double** cALamKchM_FitParamValues_10Res[3] = {cALamKchM_FitParamValues_10Res_0010, cALamKchM_FitParamValues_10Res_1030, cALamKchM_FitParamValues_10Res_3050};

//------------------------------------------------------------------------------------------

const double cLamKchM_Lambda_10Res_0010[3] = {1.72, 0.61, 0.28};
const double cLamKchM_Lambda_10Res_1030[3] = {1.24, 0.43, 0.25};
const double cLamKchM_Lambda_10Res_3050[3] = {1.34, 0.75, 0.42};

const double cALamKchP_Lambda_10Res_0010[3] = {1.72, 0.58, 0.31};
const double cALamKchP_Lambda_10Res_1030[3] = {1.33, 0.46, 0.26};
const double cALamKchP_Lambda_10Res_3050[3] = {0.84, 0.31, 0.31};

const double cLamKchM_Radius_10Res_0010[3] = {6.54, 1.22, 0.90};
const double cLamKchM_Radius_10Res_1030[3] = {4.90, 0.94, 0.64};
const double cLamKchM_Radius_10Res_3050[3] = {3.10, 0.67, 0.40};

const double cLamKchM_ReF0_10Res[3] =  {0.53, 0.20, 0.15};
const double cLamKchM_ImF0_10Res[3] =  {0.57, 0.17, 0.11};
const double cLamKchM_D0_10Res[3]   =  {-4.13, 1.74, 1.53};

//-----

const double*  cLamKchM_FitParamValues_10Res_0010[6] = {cLamKchM_Lambda_10Res_0010, cLamKchM_Radius_10Res_0010, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, cNormArray};
const double*  cLamKchM_FitParamValues_10Res_1030[6] = {cLamKchM_Lambda_10Res_1030, cLamKchM_Radius_10Res_1030, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, cNormArray};
const double*  cLamKchM_FitParamValues_10Res_3050[6] = {cLamKchM_Lambda_10Res_3050, cLamKchM_Radius_10Res_3050, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, cNormArray};
const double** cLamKchM_FitParamValues_10Res[3] = {cLamKchM_FitParamValues_10Res_0010, cLamKchM_FitParamValues_10Res_1030, cLamKchM_FitParamValues_10Res_3050};

const double*  cALamKchP_FitParamValues_10Res_0010[6] = {cALamKchP_Lambda_10Res_0010, cLamKchM_Radius_10Res_0010, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, cNormArray};
const double*  cALamKchP_FitParamValues_10Res_1030[6] = {cALamKchP_Lambda_10Res_1030, cLamKchM_Radius_10Res_1030, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, cNormArray};
const double*  cALamKchP_FitParamValues_10Res_3050[6] = {cALamKchP_Lambda_10Res_3050, cLamKchM_Radius_10Res_3050, cLamKchM_ReF0_10Res, cLamKchM_ImF0_10Res, cLamKchM_D0_10Res, cNormArray};
const double** cALamKchP_FitParamValues_10Res[3] = {cALamKchP_FitParamValues_10Res_0010, cALamKchP_FitParamValues_10Res_1030, cALamKchP_FitParamValues_10Res_3050};

//------------------------------------------------------------------------------------------

const double*** cFitParamValues_10Res[6] = {cLamK0_FitParamValues_10Res, cALamK0_FitParamValues_10Res, 
                                            cLamKchP_FitParamValues_10Res, cALamKchM_FitParamValues_10Res, 
                                            cLamKchM_FitParamValues_10Res, cALamKchP_FitParamValues_10Res};



//------------------------------------------------------------------------------------------------------------------------------
//****************************************** 3 Residuals ***********************************************************************
//------------------------------------------------------------------------------------------------------------------------------
const double cLamK0_Lambda_3Res_0010[3] = {0.60, 0.63, 0.16};
const double cLamK0_Lambda_3Res_1030[3] = {0.60, 0.63, 0.16};
const double cLamK0_Lambda_3Res_3050[3] = {0.60, 0.63, 0.16};

const double cALamK0_Lambda_3Res_0010[3] = {0.60, 0.63, 0.16};
const double cALamK0_Lambda_3Res_1030[3] = {0.60, 0.63, 0.16};
const double cALamK0_Lambda_3Res_3050[3] = {0.60, 0.63, 0.16};

const double cLamK0_Radius_3Res_0010[3] = {2.78, 0.45, 0.33};
const double cLamK0_Radius_3Res_1030[3] = {2.22, 0.37, 0.23};
const double cLamK0_Radius_3Res_3050[3] = {1.68, 0.28, 0.11};

const double cLamK0_ReF0_3Res[3] = {-0.41, 0.10, 0.16};
const double cLamK0_ImF0_3Res[3] =  {0.20, 0.10, 0.13};
const double cLamK0_D0_3Res[3]   =  {2.08, 0.39, 0.62};

//-----

const double*  cLamK0_FitParamValues_3Res_0010[6] = {cLamK0_Lambda_3Res_0010, cLamK0_Radius_3Res_0010, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, cNormArray};
const double*  cLamK0_FitParamValues_3Res_1030[6] = {cLamK0_Lambda_3Res_1030, cLamK0_Radius_3Res_1030, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, cNormArray};
const double*  cLamK0_FitParamValues_3Res_3050[6] = {cLamK0_Lambda_3Res_3050, cLamK0_Radius_3Res_3050, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, cNormArray};
const double** cLamK0_FitParamValues_3Res[3] = {cLamK0_FitParamValues_3Res_0010, cLamK0_FitParamValues_3Res_1030, cLamK0_FitParamValues_3Res_3050};

const double*  cALamK0_FitParamValues_3Res_0010[6] = {cALamK0_Lambda_3Res_0010, cLamK0_Radius_3Res_0010, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, cNormArray};
const double*  cALamK0_FitParamValues_3Res_1030[6] = {cALamK0_Lambda_3Res_1030, cLamK0_Radius_3Res_1030, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, cNormArray};
const double*  cALamK0_FitParamValues_3Res_3050[6] = {cALamK0_Lambda_3Res_3050, cLamK0_Radius_3Res_3050, cLamK0_ReF0_3Res, cLamK0_ImF0_3Res, cLamK0_D0_3Res, cNormArray};
const double** cALamK0_FitParamValues_3Res[3] = {cALamK0_FitParamValues_3Res_0010, cALamK0_FitParamValues_3Res_1030, cALamK0_FitParamValues_3Res_3050};

//------------------------------------------------------------------------------------------

const double cLamKchP_Lambda_3Res_0010[3] = {1.53, 0.56, 0.28};
const double cLamKchP_Lambda_3Res_1030[3] = {1.62, 0.58, 0.36};
const double cLamKchP_Lambda_3Res_3050[3] = {1.21, 0.31, 0.31};

const double cALamKchM_Lambda_3Res_0010[3] = {1.53, 0.57, 0.33};
const double cALamKchM_Lambda_3Res_1030[3] = {1.39, 0.49, 0.29};
const double cALamKchM_Lambda_3Res_3050[3] = {1.17, 0.30, 0.19};

const double cLamKchP_Radius_3Res_0010[3] = {5.43, 1.09, 0.54};
const double cLamKchP_Radius_3Res_1030[3] = {4.75, 0.82, 0.42};
const double cLamKchP_Radius_3Res_3050[3] = {3.22, 0.41, 0.32};

const double cLamKchP_ReF0_3Res[3] = {-1.16, 0.25, 0.36};
const double cLamKchP_ImF0_3Res[3] =  {0.51, 0.28, 0.23};
const double cLamKchP_D0_3Res[3]   =  {1.08, 0.43, 0.53};

//-----

const double*  cLamKchP_FitParamValues_3Res_0010[6] = {cLamKchP_Lambda_3Res_0010, cLamKchP_Radius_3Res_0010, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, cNormArray};
const double*  cLamKchP_FitParamValues_3Res_1030[6] = {cLamKchP_Lambda_3Res_1030, cLamKchP_Radius_3Res_1030, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, cNormArray};
const double*  cLamKchP_FitParamValues_3Res_3050[6] = {cLamKchP_Lambda_3Res_3050, cLamKchP_Radius_3Res_3050, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, cNormArray};
const double** cLamKchP_FitParamValues_3Res[3] = {cLamKchP_FitParamValues_3Res_0010, cLamKchP_FitParamValues_3Res_1030, cLamKchP_FitParamValues_3Res_3050};

const double*  cALamKchM_FitParamValues_3Res_0010[6] = {cALamKchM_Lambda_3Res_0010, cLamKchP_Radius_3Res_0010, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, cNormArray};
const double*  cALamKchM_FitParamValues_3Res_1030[6] = {cALamKchM_Lambda_3Res_1030, cLamKchP_Radius_3Res_1030, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, cNormArray};
const double*  cALamKchM_FitParamValues_3Res_3050[6] = {cALamKchM_Lambda_3Res_3050, cLamKchP_Radius_3Res_3050, cLamKchP_ReF0_3Res, cLamKchP_ImF0_3Res, cLamKchP_D0_3Res, cNormArray};
const double** cALamKchM_FitParamValues_3Res[3] = {cALamKchM_FitParamValues_3Res_0010, cALamKchM_FitParamValues_3Res_1030, cALamKchM_FitParamValues_3Res_3050};

//------------------------------------------------------------------------------------------

const double cLamKchM_Lambda_3Res_0010[3] = {1.91, 0.60, 0.24};
const double cLamKchM_Lambda_3Res_1030[3] = {1.39, 0.43, 0.27};
const double cLamKchM_Lambda_3Res_3050[3] = {1.57, 0.82, 0.57};

const double cALamKchP_Lambda_3Res_0010[3] = {1.90, 0.57, 0.27};
const double cALamKchP_Lambda_3Res_1030[3] = {1.50, 0.46, 0.26};
const double cALamKchP_Lambda_3Res_3050[3] = {0.92, 0.31, 0.37};

const double cLamKchM_Radius_3Res_0010[3] = {6.25, 1.08, 0.81};
const double cLamKchM_Radius_3Res_1030[3] = {4.74, 0.86, 0.60};
const double cLamKchM_Radius_3Res_3050[3] = {2.98, 0.61, 0.38};

const double cLamKchM_ReF0_3Res[3] =  {0.41, 0.18, 0.14};
const double cLamKchM_ImF0_3Res[3] =  {0.47, 0.15, 0.11};
const double cLamKchM_D0_3Res[3]   =  {-4.89, 2.16, 1.33};

//-----

const double*  cLamKchM_FitParamValues_3Res_0010[6] = {cLamKchM_Lambda_3Res_0010, cLamKchM_Radius_3Res_0010, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, cNormArray};
const double*  cLamKchM_FitParamValues_3Res_1030[6] = {cLamKchM_Lambda_3Res_1030, cLamKchM_Radius_3Res_1030, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, cNormArray};
const double*  cLamKchM_FitParamValues_3Res_3050[6] = {cLamKchM_Lambda_3Res_3050, cLamKchM_Radius_3Res_3050, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, cNormArray};
const double** cLamKchM_FitParamValues_3Res[3] = {cLamKchM_FitParamValues_3Res_0010, cLamKchM_FitParamValues_3Res_1030, cLamKchM_FitParamValues_3Res_3050};

const double*  cALamKchP_FitParamValues_3Res_0010[6] = {cALamKchP_Lambda_3Res_0010, cLamKchM_Radius_3Res_0010, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, cNormArray};
const double*  cALamKchP_FitParamValues_3Res_1030[6] = {cALamKchP_Lambda_3Res_1030, cLamKchM_Radius_3Res_1030, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, cNormArray};
const double*  cALamKchP_FitParamValues_3Res_3050[6] = {cALamKchP_Lambda_3Res_3050, cLamKchM_Radius_3Res_3050, cLamKchM_ReF0_3Res, cLamKchM_ImF0_3Res, cLamKchM_D0_3Res, cNormArray};
const double** cALamKchP_FitParamValues_3Res[3] = {cALamKchP_FitParamValues_3Res_0010, cALamKchP_FitParamValues_3Res_1030, cALamKchP_FitParamValues_3Res_3050};

//------------------------------------------------------------------------------------------

const double*** cFitParamValues_3Res[6] = {cLamK0_FitParamValues_3Res, cALamK0_FitParamValues_3Res, 
                                           cLamKchP_FitParamValues_3Res, cALamKchM_FitParamValues_3Res, 
                                           cLamKchM_FitParamValues_3Res, cALamKchP_FitParamValues_3Res};

//------------------------------------------------------------------------------------------------------------------------------
//******************************************************************************************************************************
//------------------------------------------------------------------------------------------------------------------------------


const double**** cFitParamValues[3] = {cFitParamValues_NoRes, cFitParamValues_10Res, cFitParamValues_3Res};



