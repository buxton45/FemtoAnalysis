///////////////////////////////////////////////////////////////////////////
// Types_ThermBgdParams:                                                 //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef TYPES_THERMBGDPARAMS_H
#define TYPES_THERMBGDPARAMS_H

//#include "TString.h"

#include <vector>
#include <complex>
#include <cassert>
#include <iostream>
using std::vector;
using namespace std;

  //-----------------------------------------------------------


  extern const double  cLamK0_ThermBgdParamValues_0010[7];
  extern const double  cLamK0_ThermBgdParamValues_1030[7];
  extern const double  cLamK0_ThermBgdParamValues_3050[7];
  extern const double*  cLamK0_ThermBgdParamValues[3];

//  extern const double  cALamK0_ThermBgdParamValues_0010[7];
//  extern const double  cALamK0_ThermBgdParamValues_1030[7];
//  extern const double  cALamK0_ThermBgdParamValues_3050[7];
  extern const double*  cALamK0_ThermBgdParamValues[3];

  //-----------------------------------------------------------

  extern const double  cLamKchP_ThermBgdParamValues_0010[7];
  extern const double  cLamKchP_ThermBgdParamValues_1030[7];
  extern const double  cLamKchP_ThermBgdParamValues_3050[7];
  extern const double*  cLamKchP_ThermBgdParamValues[3];

//  extern const double  cALamKchM_ThermBgdParamValues_0010[7];
//  extern const double  cALamKchM_ThermBgdParamValues_1030[7];
//  extern const double  cALamKchM_ThermBgdParamValues_3050[7];
  extern const double*  cALamKchM_ThermBgdParamValues[3];

  //-----------------------------------------------------------

  extern const double  cLamKchM_ThermBgdParamValues_0010[7];
  extern const double  cLamKchM_ThermBgdParamValues_1030[7];
  extern const double  cLamKchM_ThermBgdParamValues_3050[7];
  extern const double*  cLamKchM_ThermBgdParamValues[3];

//  extern const double  cALamKchP_ThermBgdParamValues_0010[7];
//  extern const double  cALamKchP_ThermBgdParamValues_1030[7];
//  extern const double  cALamKchP_ThermBgdParamValues_3050[7];
  extern const double*  cALamKchP_ThermBgdParamValues[3];

  //-----------------------------------------------------------

  extern const double** cThermBgdParamValues[6];  // = cThermBgdParamValues[6][3][7]
                                                  // first  index = enum AnalysisType
                                                  // second index = enum CentralityType
                                                  // third  index = int ParamIndex

#endif



