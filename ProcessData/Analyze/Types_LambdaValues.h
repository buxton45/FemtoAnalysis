///////////////////////////////////////////////////////////////////////////
// Types_LambdaValues.h:                                                 //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef TYPES_LAMBDAVALUES_H
#define TYPES_LAMBDAVALUES_H


#include <vector>
#include <complex>
#include <cassert>
#include <iostream>
using std::vector;
using namespace std;


  extern const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay0[85];
  extern const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay4[85];
  extern const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay5[85];
  extern const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay6[85];
  extern const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay10[85];
  extern const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay100[85];

  //Index with enum ResPrimMaxDecayType
  extern const double *cAnalysisLambdaFactors_NoRes[6];

  //-----------------------------------

  extern const double cAnalysisLambdaFactors_10Res_MaxPrimDecay0[85];
  extern const double cAnalysisLambdaFactors_10Res_MaxPrimDecay4[85];  //TODO
  extern const double cAnalysisLambdaFactors_10Res_MaxPrimDecay5[85];
  extern const double cAnalysisLambdaFactors_10Res_MaxPrimDecay6[85];  //TODO
  extern const double cAnalysisLambdaFactors_10Res_MaxPrimDecay10[85];
  extern const double cAnalysisLambdaFactors_10Res_MaxPrimDecay100[85];

  //Index with enum ResPrimMaxDecayType
  extern const double *cAnalysisLambdaFactors_10Res[6];

  //-----------------------------------


  extern const double cAnalysisLambdaFactors_3Res_MaxPrimDecay0[85];
  extern const double cAnalysisLambdaFactors_3Res_MaxPrimDecay4[85]; //TODO
  extern const double cAnalysisLambdaFactors_3Res_MaxPrimDecay5[85];
  extern const double cAnalysisLambdaFactors_3Res_MaxPrimDecay6[85]; //TODO
  extern const double cAnalysisLambdaFactors_3Res_MaxPrimDecay10[85];
  extern const double cAnalysisLambdaFactors_3Res_MaxPrimDecay100[85];

  //Index with enum ResPrimMaxDecayType
  extern const double *cAnalysisLambdaFactors_3Res[6];

  //-----------------------------------

  //First Index with enum IncludeResidualsType
  //Second Index with enum ResPrimMaxDecayType
  //Thirs Index with enum AnalysisType
  extern const double **cAnalysisLambdaFactorsArr[3];



#endif


