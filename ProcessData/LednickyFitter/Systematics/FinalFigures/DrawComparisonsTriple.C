#include "CompareFittingMethodswSysErrs.cxx"

  //--------------------------------------------------------------------------
//TRIPLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  vector<NonFlatBgdFitType> tNonFlatBgdFitTypesStd{kLinear, kLinear,
                                                   kPolynomial, kPolynomial, kPolynomial, kPolynomial};
  double aMarkerSize = 2.0; //1.5 = default from CompareFittingMethods.h


  vector<FitValWriterInfo> tFVWIVec_Comp3An_Triple = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamKchP, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamKchM, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamK0, 20, aMarkerSize, false, true)};

  //--------------------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_3v10vNo = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "3 Residuals", tColorLamKchP, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude10Residuals, k4fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "10 Residuals", tColorLamKchP, 28, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kIncludeNoResiduals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "No Residuals", tColorLamKchP, 26, aMarkerSize, false, true), 



                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "3 Residuals", tColorLamKchM, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude10Residuals, k4fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "10 Residuals", tColorLamKchM, 28, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kIncludeNoResiduals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "No Residuals", tColorLamKchM, 26, aMarkerSize, false, true), 




                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "3 Residuals", tColorLamK0, 20, aMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude10Residuals, k4fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "10 Residuals", tColorLamK0, 28, aMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kIncludeNoResiduals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "No Residuals", tColorLamK0, 26, aMarkerSize, false, true)};
  //--------------------------------------------------------------------------
  vector<FitValWriterInfo> tFVWIVec_FreevFixlam = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #lambda", tColorLamKchP, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#lambda=1", tColorLamKchP, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #lambda", tColorLamKchM, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#lambda=1", tColorLamKchM, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #lambda", tColorLamK0, 20, aMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#lambda=1", tColorLamK0, 24, aMarkerSize, false, true)};

  //--------------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_NormvStav = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Normal", tColorLamKchP, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, false, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       true, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Stav. Method", tColorLamKchP, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Normal", tColorLamKchM, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, false, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       true, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Stav. Method", tColorLamKchM, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Normal", tColorLamK0, 20, aMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, false, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       true, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Stav. Method", tColorLamK0, 24, aMarkerSize, false, true)};

  //--------------------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_SharevSepR = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "All share R", tColorLamKchP, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} share R", tColorLamKchP, 28, aMarkerSize, true, false), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "All share R", tColorLamKchM, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} share R", tColorLamKchM, 28, aMarkerSize, true, false), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "All share R", tColorLamK0, 20, aMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                 FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                                                      kInclude3Residuals, k10fm, 
                                                                                                                      kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                      false, false, false, false, false, 
                                                                                                                      false, true, false, false, 
                                                                                                                      false, false), 
                                                                                 "#LambdaK^{0}_{S} fit alone", tColorLamK0, 26, aMarkerSize, false, false)};

  //--------------------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_ExpvSimXi = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Exp. #Xi in Residual", tColorLamKchP, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseCoulombOnlyInterpForAll, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Sim. #Xi in Residual", tColorLamKchP, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Exp. #Xi in Residual", tColorLamKchM, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseCoulombOnlyInterpForAll, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Sim. #Xi in Residual", tColorLamKchM, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Exp. #Xi in Residual", tColorLamK0, 20, aMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseCoulombOnlyInterpForAll, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Sim. #Xi in Residual", tColorLamK0, 24, aMarkerSize, false, true)};

  //--------------------------------------------------------------------------
  vector<FitValWriterInfo> tFVWIVec_FreevFixR_FreeLam = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #it{R}_{inv}", tColorLamKchP, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, true, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Fixed #it{R}_{inv}", tColorLamKchP, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #it{R}_{inv}", tColorLamKchM, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, true, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Fixed #it{R}_{inv}", tColorLamKchM, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #it{R}_{inv}", tColorLamK0, 20, aMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, true, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Fixed #it{R}_{inv}", tColorLamK0, 24, aMarkerSize, false, true)};

  //--------------------------------------------------------------------------
  vector<FitValWriterInfo> tFVWIVec_FreevFixR_FixLam = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #it{R}_{inv} (#lambda=1)", tColorLamKchP, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, true, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Fixed #it{R}_{inv} (#lambda=1)", tColorLamKchP, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #it{R}_{inv} (#lambda=1)", tColorLamKchM, 20, aMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, true, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Fixed #it{R}_{inv} (#lambda=1)", tColorLamKchM, 24, aMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #it{R}_{inv} (#lambda=1)", tColorLamK0, 20, aMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, true, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Fixed #it{R}_{inv} (#lambda=1)", tColorLamK0, 24, aMarkerSize, false, true)};

  //--------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  bool bSaveFigures = false ;
  TString tSaveFileType = "pdf";  //Needs to be pdf for systematics to be transparent!

  vector<TString> tStatOnlyTags = {"", "_StatOnly"};
  vector<TString> tVerticalTags = {"", "_Vertical"};

  bool bDrawStatOnly = true;
  TString tSystematicsFileLocation = "";

  bool bDrawPredictions = false;


  TString tSaveDirBase = "/home/jesse/Analysis/Dissertation/Appendices/Appendix_Results/App_ResultsLamK_FitMethodComparisons/Figures/";



/*
List of FVWIVecs for convenience
  vector<FitValWriterInfo> tFVWIVec_Comp3An_Triple  //Mainly just here to help in creation of others (serves as starting point)
  vector<FitValWriterInfo> tFVWIVec_3v10vNo
  vector<FitValWriterInfo> tFVWIVec_FreevFixlam
  vector<FitValWriterInfo> tFVWIVec_NormvStav
  vector<FitValWriterInfo> tFVWIVec_SharevSepR
  vector<FitValWriterInfo> tFVWIVec_ExpvSimXi
  vector<FitValWriterInfo> tFVWIVec_FreevFixR_FreeLam
  vector<FitValWriterInfo> tFVWIVec_FreevFixR_FixLam
*/


  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_3v10vNo;            TString tCanNameMod = TString("_3v10vNo");
//  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_FreevFixlam;        TString tCanNameMod = TString("_FreevFixlam");
//  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_NormvStav;          TString tCanNameMod = TString("_NormvStav");
//  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_SharevSepR;         TString tCanNameMod = TString("_SharevSepR");
//  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_ExpvSimXi;          TString tCanNameMod = TString("_ExpvSimXi");
//  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_FreevFixR_FreeLam;          TString tCanNameMod = TString("_FreevFixR_FreeLam");
//  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_FreevFixR_FixLam;          TString tCanNameMod = TString("_FreevFixR_FixLam");

  TCanvas* tCanAll = CompareAllTweak(tFVWIVec, tSystematicsFileLocation, tSystematicsFileLocation, bDrawPredictions, tCanNameMod, bDrawStatOnly);
  if(bSaveFigures) tCanAll->SaveAs(TString::Format("%sComparisons%s%s.%s", tSaveDirBase.Data(), tCanNameMod.Data(), tStatOnlyTags[bDrawStatOnly].Data(), tSaveFileType.Data()));



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.



  cout << "DONE" << endl;
  return 0;
}








