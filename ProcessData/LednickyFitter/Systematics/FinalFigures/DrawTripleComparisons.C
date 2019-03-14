#include "CompareFittingMethodswSysErrs.cxx"

  //--------------------------------------------------------------------------
//TRIPLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  vector<NonFlatBgdFitType> tNonFlatBgdFitTypesStd{kLinear, kLinear,
                                                   kPolynomial, kPolynomial, kPolynomial, kPolynomial};


  vector<FitValWriterInfo> tFVWIVec_Comp3An_Triple = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamKchP, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamKchM, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} and #LambdaK^{0}_{S} fit together", tColorLamK0, 20, tMarkerSize, false, true)};

  //--------------------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_3v10vNo = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "3 Residuals", tColorLamKchP, 34, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude10Residuals, k4fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "10 Residuals", tColorLamKchP, 47, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kIncludeNoResiduals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "No Residuals", tColorLamKchP, 20, tMarkerSize, false, true), 



                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "3 Residuals", tColorLamKchM, 34, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude10Residuals, k4fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "10 Residuals", tColorLamKchM, 47, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kIncludeNoResiduals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "No Residuals", tColorLamKchM, 20, tMarkerSize, false, true), 




                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "3 Residuals", tColorLamK0, 34, tMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude10Residuals, k4fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "10 Residuals", tColorLamK0, 47, tMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kIncludeNoResiduals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "No Residuals", tColorLamK0, 20, tMarkerSize, false, true)};
  //--------------------------------------------------------------------------
  vector<FitValWriterInfo> tFVWIVec_FreevFixlam = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #lambda", tColorLamKchP, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#lambda=1", tColorLamKchP, 24, tMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #lambda", tColorLamKchM, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#lambda=1", tColorLamKchM, 24, tMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Free #lambda", tColorLamK0, 20, tMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, true, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#lambda=1", tColorLamK0, 24, tMarkerSize, false, true)};

  //--------------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_NormvStav = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Normal", tColorLamKchP, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, false, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       true, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Stav. Method", tColorLamKchP, 24, tMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Normal", tColorLamKchM, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, false, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       true, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Stav. Method", tColorLamKchM, 24, tMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Normal", tColorLamK0, 20, tMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, false, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       true, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Stav. Method", tColorLamK0, 24, tMarkerSize, false, true)};

  //--------------------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_SharevSepR = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "All share R", tColorLamKchP, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} share R", tColorLamKchP, 24, tMarkerSize, true, false), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "All share R", tColorLamKchM, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "#LambdaK^{#pm} share R", tColorLamKchM, 24, tMarkerSize, true, false), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "All share R", tColorLamK0, 20, tMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                 FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                                                      kInclude3Residuals, k10fm, 
                                                                                                                      kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                      false, false, false, false, false, 
                                                                                                                      false, true, false, false, 
                                                                                                                      false, false), 
                                                                                 "#LambdaK^{0}_{S} fit alone", tColorLamK0, 30, tMarkerSize, false, false)};

  //--------------------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_ExpvSimXi = {
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Exp. #Xi in Residual", tColorLamKchP, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseCoulombOnlyInterpForAll, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Sim. #Xi in Residual", tColorLamKchP, 24, tMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Exp. #Xi in Residual", tColorLamKchM, 20, tMarkerSize, false, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate,
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseCoulombOnlyInterpForAll, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Sim. #Xi in Residual", tColorLamKchM, 24, tMarkerSize, false, true), 


                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Exp. #Xi in Residual", tColorLamK0, 20, tMarkerSize, false, true),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, 
                                                                                  FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                                       kInclude3Residuals, k10fm, 
                                                                                                                       kUseCoulombOnlyInterpForAll, false, 
                                                                                                                       false, false, false, false, false, 
                                                                                                                       true, false, false, true, 
                                                                                                                       true, true), 
                                                                                 "Sim. #Xi in Residual", tColorLamK0, 24, tMarkerSize, false, true)};

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

  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";  //Needs to be pdf for systematics to be transparent!

  vector<TString> tStatOnlyTags = {"", "_StatOnly"};
  vector<TString> tVerticalTags = {"", "_Vertical"};

  bool bDrawStatOnly = true;
  TString tSystematicsFileLocation = "";

  bool bDrawPredictions = false;


  TString tSaveDirBase = "";



/*
List of FVWIVecs for convenience
  vector<FitValWriterInfo> tFVWIVec_Comp3An_Triple
  vector<FitValWriterInfo> tFVWIVec_3v10vNo
  vector<FitValWriterInfo> tFVWIVec_FreevFixlam
  vector<FitValWriterInfo> tFVWIVec_NormvStav
  vector<FitValWriterInfo> tFVWIVec_SharevSepR
  vector<FitValWriterInfo> tFVWIVec_ExpvSimXi
*/



  

  vector<FitValWriterInfo> tFVWIVec = tFVWIVec_ExpvSimXi;
  TString tCanNameMod = TString("_Comp3An_Triple");

  TCanvas* tCanAll = CompareAllTweak(tFVWIVec, tSystematicsFileLocation, tSystematicsFileLocation, bDrawPredictions, tCanNameMod, bDrawStatOnly);
  if(bSaveFigures) tCanAll->SaveAs(TString::Format("%sFinalResults_Comp3An%s%s.pdf", tSaveDirBase.Data(), tStatOnlyTags[bDrawStatOnly].Data()));



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.



  cout << "DONE" << endl;
  return 0;
}








