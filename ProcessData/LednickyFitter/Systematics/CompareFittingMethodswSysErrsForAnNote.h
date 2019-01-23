#ifndef COMPAREFITTINGMETHODSWSYSERRSFORANNOTE_H_
#define COMPAREFITTINGMETHODSWSYSERRSFORANNOTE_H_

#include "CompareFittingMethodswSysErrs.cxx"


//NOTE:
//      LamKchSTD = LamKch_3Res_PolyBgd_10fm (and, STD w/o other ammendments implies dualie shared radii and dualie shared lambda)
//      LamK0STD  = LamK0_3Res_LinrBgd_10fm


//For refernce:
//  static TString BuildFitInfoTString(bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection, NonFlatBgdFitType aNonFlatBgdFitType, 
//                                     IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType=k4fm, 
//                                     ChargedResidualsType aChargedResidualsType=kUseXiDataAndCoulombOnlyInterp, bool aFixD0=false,
//                                     bool aUseStavCf=false, bool aFixAllLambdaTo1=false, bool aFixAllNormTo1=false, bool aFixRadii=false, bool aFixAllScattParams=false, 
//                                     bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aUsemTScalingOfResidualRadii=false, bool aIsDualie=false, 
//                                     bool aDualieShareLambda=false, bool aDualieShareRadii=false);


//############################################### LAMKCH ######################################################################################

  TString tFIS_LamKchSTD =                          FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);


  //-------------------------------- Vary number of residuals --------------------------------------------------
  TString tFIS_LamKch_10Res_PolyBgd_4fm =           FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude10Residuals, k4fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

  TString tFIS_LamKch_NoRes_PolyBgd =               FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kIncludeNoResiduals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

  //-------------------------------- Vary background treatment --------------------------------------------------
  TString tFIS_LamKch_3Res_LinrBgd_10fm =           FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

  TString tFIS_LamKch_3Res_StavCf_10fm =            FitValuesWriter::BuildFitInfoTString(true, false, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         true, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);

  //-------------------------------- Separate radii between LamKchP and LamKchM ----------------------------------------------
  TString tFIS_LamKchSTD_SepR =                     FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, false);


  //-------------------------------- Fix lambda --------------------------------------------------
  TString tFIS_LamKchSTD_FixLam =                   FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, true, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, true);




  TString tFIS_LamKchSTD_SepR_FixLam =              FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, true, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         true, false);


  //-------------------------------- Vary lambda sharing --------------------------------------------------
  TString tFIS_LamKchSTD_ShareLamConj =             FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         false, true);

  TString tFIS_LamKchSTD_UniqueLam =                FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         false, false, false, true, 
                                                                                         false, true);


  TString tFIS_LamKchSTD_ShareLamConj_SepR =        FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         true, false, false, true, 
                                                                                         false, false);

  TString tFIS_LamKchSTD_UniqueLam_SepR =           FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                         kInclude3Residuals, k10fm, 
                                                                                         kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                         false, false, false, false, false, 
                                                                                         false, false, false, true, 
                                                                                         false, false);


//############################################### LAMK0 #######################################################################################

  TString tFIS_LamK0STD =                      FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  //-------------------------------- Vary number of residuals --------------------------------------------------
  TString tFIS_LamK0_10Res_LinrBgd_4fm =       FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kInclude10Residuals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFIS_LamK0_NoRes_LinrBgd =           FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kIncludeNoResiduals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  //-------------------------------- Vary background treatment --------------------------------------------------
  TString tFIS_LamK0_3Res_PolyBgd_10fm =       FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFIS_LamK0_3Res_StavCf_10fm =        FitValuesWriter::BuildFitInfoTString(true, false, kLinear, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    true, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  //-------------------------------- Fix lambda --------------------------------------------------
  TString tFIS_LamK0STD_FixLam =               FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, true, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  //-------------------------------- Vary lambda sharing --------------------------------------------------
  TString tFIS_LamK0STD_ShareLamConj =         FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    true, false, false, false, 
                                                                                    false, false);

  TString tFIS_LamK0STD_UniqueLam =            FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kInclude3Residuals, k10fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, false, false, false, 
                                                                                    false, false);


//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_Comp3An_3Res_10fm =               {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                        "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                                       FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                        "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, true), 
                                                                       FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD, 
                                                                                        "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};

//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_Comp3An_WithBgdVsStav_10fm =      {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                        "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                                       FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                        "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd", tColorLamKchM, 20, tMarkerSize, true), 
                                                                       FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD, 
                                                                                        "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false),
                                                                       FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_3Res_StavCf_10fm, 
                                                                                        "3 Residuals (Suppress Markers)", tColorLamKchP, 34, tMarkerSize, true),
                                                                       FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_3Res_StavCf_10fm, 
                                                                                        "Stav. Cf", tColorLamKchM, 34, tMarkerSize, true), 
                                                                       FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0_3Res_StavCf_10fm, 
                                                                                        "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 34, tMarkerSize, false)};

//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_Comp3An_LinrPolyStav_10fm =      {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Poly. Bgd", tColorLamKchM, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0_3Res_PolyBgd_10fm, 
                                                                                       "Poly. Bgd", tColorLamK0, 20, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_3Res_LinrBgd_10fm, 
                                                                                       "Linr. Bgd", tColorLamKchP, 21, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_3Res_LinrBgd_10fm, 
                                                                                       "Linr. Bgd", tColorLamKchM, 21, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD, 
                                                                                       "Linr. Bgd", tColorLamK0, 21, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_3Res_StavCf_10fm, 
                                                                                       "Stav. Cf", tColorLamKchP, 34, tMarkerSize, true),
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_3Res_StavCf_10fm, 
                                                                                       "Stav. Cf", tColorLamKchM, 34, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0_3Res_StavCf_10fm, 
                                                                                       "Stav. Cf", tColorLamK0, 34, tMarkerSize, false)};

//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_CompNRes_Std =                   {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "3 Residuals", tColorLamKchP, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "3 Residuals", tColorLamKchM, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD, 
                                                                                       "3 Residuals", tColorLamK0, 20, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_10Res_PolyBgd_4fm, 
                                                                                       "10 Residuals", tColorLamKchP, 21, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_10Res_PolyBgd_4fm, 
                                                                                       "10 Residuals", tColorLamKchM, 21, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0_10Res_LinrBgd_4fm, 
                                                                                       "10 Residuals", tColorLamK0, 21, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_NoRes_PolyBgd, 
                                                                                       "No Residuals", tColorLamKchP, 34, tMarkerSize, true),
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKch_NoRes_PolyBgd, 
                                                                                       "No Residuals", tColorLamKchM, 34, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0_NoRes_LinrBgd, 
                                                                                       "No Residuals", tColorLamK0, 34, tMarkerSize, false)};

//---------------------------------------------------------------------



  vector<FitValWriterInfo> tFVWIVec_CompFreevsFixedLam_Std =         {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Free #lambda", tColorLamKchP, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Free #lambda", tColorLamKchM, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD, 
                                                                                       "Free #lambda", tColorLamK0, 20, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_FixLam, 
                                                                                       "Fix #lambda", tColorLamKchP, 21, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_FixLam, 
                                                                                       "Fix #lambda", tColorLamKchM, 21, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD_FixLam, 
                                                                                       "Fix #lambda", tColorLamK0, 21, tMarkerSize, false)};

//---------------------------------------------------------------------



  vector<FitValWriterInfo> tFVWIVec_CompFreevsFixedLam_SepR_Std =    {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_SepR, 
                                                                                       "Free #lambda", tColorLamKchP, 20, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_SepR, 
                                                                                       "Free #lambda", tColorLamKchM, 20, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD, 
                                                                                       "Free #lambda", tColorLamK0, 20, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_SepR_FixLam, 
                                                                                       "Fix #lambda", tColorLamKchP, 21, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_SepR_FixLam, 
                                                                                       "Fix #lambda", tColorLamKchM, 21, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD_FixLam, 
                                                                                       "Fix #lambda", tColorLamK0, 21, tMarkerSize, false)};

//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_CompSharedvsUniqueLam_Std =      {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Share #lambda", tColorLamKchP, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Share #lambda", tColorLamKchM, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD, 
                                                                                       "Share #lambda", tColorLamK0, 20, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_ShareLamConj, 
                                                                                       "Share #lambda_{Conj}", tColorLamKchP, 21, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_ShareLamConj, 
                                                                                       "Share #lambda_{Conj}", tColorLamKchM, 21, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD_ShareLamConj, 
                                                                                       "Share #lambda_{Conj}", tColorLamK0, 21, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_UniqueLam, 
                                                                                       "Unique #lambda", tColorLamKchP, 34, tMarkerSize, true),
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_UniqueLam, 
                                                                                       "Unique #lambda", tColorLamKchM, 34, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD_UniqueLam, 
                                                                                       "Unique #lambda", tColorLamK0, 34, tMarkerSize, false)};

//---------------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_CompSharedvsUniqueLam_SepR_Std = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_SepR, 
                                                                                       "Share #lambda", tColorLamKchP, 20, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_SepR, 
                                                                                       "Share #lambda", tColorLamKchM, 20, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD, 
                                                                                       "Share #lambda", tColorLamK0, 20, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_ShareLamConj_SepR, 
                                                                                       "Share #lambda_{Conj}", tColorLamKchP, 21, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_ShareLamConj_SepR, 
                                                                                       "Share #lambda_{Conj}", tColorLamKchM, 21, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD_ShareLamConj, 
                                                                                       "Share #lambda_{Conj}", tColorLamK0, 21, tMarkerSize, false),

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_UniqueLam_SepR, 
                                                                                       "Unique #lambda", tColorLamKchP, 34, tMarkerSize, false),
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_UniqueLam_SepR, 
                                                                                       "Unique #lambda", tColorLamKchM, 34, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFIS_LamK0STD_UniqueLam, 
                                                                                       "Unique #lambda", tColorLamK0, 34, tMarkerSize, false)};

//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_CompSharedvsSepRLam_Std =        {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Shared Radii, Shared #lambda", tColorLamKchP, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Shared Radii, Shared #lambda", tColorLamKchM, 20, tMarkerSize, true), 

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_ShareLamConj_SepR, 
                                                                                       "Separate Radii, Separate #lambda", tColorLamKchP, 21, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_ShareLamConj_SepR, 
                                                                                       "Separate Radii, Separate #lambda", tColorLamKchM, 21, tMarkerSize, false)};

//---------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_CompSharedvsSepR_ShareLamConj_Std = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Shared Radii, Shared #lambda", tColorLamKchP, 20, tMarkerSize, true), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD, 
                                                                                       "Shared Radii, Shared #lambda", tColorLamKchM, 20, tMarkerSize, true), 

                                                                      FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_SepR, 
                                                                                       "Separate Radii, Shared #lambda", tColorLamKchP, 21, tMarkerSize, false), 
                                                                      FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFIS_LamKchSTD_SepR, 
                                                                                       "Separate Radii, Shared #lambda", tColorLamKchM, 21, tMarkerSize, false)};

//---------------------------------------------------------------------


#endif
