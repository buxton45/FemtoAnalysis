#include "TripleFitGenerator.h"
class TripleFitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  //******** Notes on tShareLambdaParams and tTripleShareLambda ********
  // tShareLambdaParams controls how lambda are shared within a given FitGenerator object (across a given centrality)
  //      i.e. between (LamKchP and ALamKchM) or between (LamKchM and ALamKchP)
  // tTripleShareLambda controls how lambda are shared across the two FitGenerator objects (across a given centrality)
  //      i.e. between (LamKchP and LamKchM) or (ALamKchM and ALamKchP)
  //
  // Here are the different scenarios with explanations of consequences
  // (1) tShareLambdaParams=true  &&  tTripleShareLambda=true
  //       There will be a single lambda parameter for each centrality
  //       3 lambda parameters in total for MB analysis
  //
  // (2) tShareLambdaParams=true  &&  tTripleShareLambda=false
  //       Conjugate pairs in each centrality WILL share a lambda, but the different FitGenerators WILL NOT
  //       2 lambda parameters per centrality (lam1:LamKchP & ALamKchM and lam2:LamKchM & ALamKchP)
  //       6 lambda parameters in total for MB analysis
  //
  // (3) tShareLambdaParams=false &&  tTripleShareLambda=true
  //       Conjugate pairs in each centrality WILL NOT share a lambda, but the different FitGenerators WILL
  //       2 lambda parameters per centrality (lam1:LamKchP & LamKchM and lam2:ALamKchM & ALamKchP)
  //       6 lambda parameters in total for MB analysis
  //       THIS OPTION PROBABLY DOES NOT MAKE SENSE TO USE
  //
  // (4) tShareLambdaParams=false &&  tTripleShareLambda=false
  //       Conjugate pairs in each centrality WILL NOT share a lambda, and the different FitGenerators WILL NOT
  //       4 lambda parameters per centrality (lam1:LamKchP & lam2:ALamKchM & lam3:LamKchM & lam4:ALamKchP)
  //       12 lambda parameters in total for MB analysis
  //********************************************************************

  //--Rarely change---------------------
  AnalysisRunType tAnRunType_LamKch = kTrain, tAnRunType_LamK0 = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB/*k0010*/;
  FitGeneratorType tGenType = kPairwConj;
  FitType tFitType = kChi2PML;
  //------------------------------------

  //*****************************************
  bool bDoFit = true;
  bool bGenerateContours = false;

  TString tResultsDate = "20180505";

  double tMaxFitKStar=0.3;

  bool bUseStavCf=false;
  //*****************************************

  //--Save options
  bool SaveImages = false;
  TString tSaveFileType = "pdf";
  bool bWriteToMasterFitValuesFile = false;
  bool SaveImagesInRootFile = false;

  //--Sharing lambda
  bool tShareLambdaParams = true;          //If true, only share lambda parameters across like-centralities
  bool tAllShareSingleLambdaParam = false;

  //--Triple sharing options
  bool tTripleShareLambda = true;
  bool tTripleShareRadii = true;

  //--Corrections
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  if(bUseStavCf) ApplyNonFlatBackgroundCorrection = false;
  NonFlatBgdFitType tNonFlatBgdFitType_LamKch = kPolynomial;
  NonFlatBgdFitType tNonFlatBgdFitType_LamK0  = kLinear;
  vector<NonFlatBgdFitType> tNonFlatBgdFitTypes{tNonFlatBgdFitType_LamK0, tNonFlatBgdFitType_LamK0, 
                                                tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch};

  bool UseNewBgdTreatment = false;  //TODO For now, and maybe forever, should always be FALSE!!!!!
    if(UseNewBgdTreatment) tMaxFitKStar = 0.5;

  //--Residuals
  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp/*kUseCoulombOnlyInterpForAll*/;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;

  //--Bound lambda
  bool UnboundLambda = true;
  double aLambdaMin=0.6, aLambdaMax=1.1;
  if(UnboundLambda) {aLambdaMin=0.; aLambdaMax=0.;}

  //--Fix parameters
  bool FixRadii = false;
  bool FixD0 = false;
  bool FixAllScattParams = false;
  bool FixAllLambdaTo1 = false;
  if(FixAllLambdaTo1) tAllShareSingleLambdaParam = true;
  bool FixAllNormTo1 = false;

  //--mT scaling
  bool UsemTScalingOfResidualRadii = false;
  double mTScalingPowerOfResidualRadii = -0.5;

  //--Plotting options
  bool bZoomROP = false;
  bool bDrawResiduals = false;
  bool bDrawResv2 = true;
  bool bDrawPartAn = false;

  bool bDrawSysErrs = false;

  bool tSuppressFitInfoOutput=false;
  bool tLabelLines=false;
  if(tLabelLines) tSuppressFitInfoOutput=true;

//-----------------------------------------------------------------------------
  if(tShareLambdaParams==false && tTripleShareLambda==true)
  {
    cout << "!!!!!!!!!!!!!!! tShareLambdaParams==false && tTripleShareLambda==true !!!!!!!!!!!!!!!" << endl;
    cout << "This means Conjugate pairs in each centrality WILL NOT share a lambda, but the different FitGenerators WILL" << endl;
    cout << "2 lambda parameters per centrality (lam1:LamKchP & LamKchM and lam2:ALamKchM & ALamKchP)" << endl;
    cout << "6 lambda parameters in total for MB analysis" << endl;
    cout << "Is this really what you want? (0=No  1=Yes)" << endl;

    int tResponse;
    cin >> tResponse;
    assert(tResponse);
  }

//-----------------------------------------------------------------------------

  TString tDirectoryBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  TString tFileLocationBase_LamKch = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_LamKch.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_LamKch = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_LamKch.Data(),tResultsDate.Data());

  TString tSaveDirectoryBase_LamKch = tDirectoryBase_LamKch;

  TString tLocationMasterFitResults_LamKch = TString::Format("%sMasterFitResults_%s.txt", tDirectoryBase_LamKch.Data(), tResultsDate.Data());

  //-----

  TString tDirectoryBase_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate.Data());
  TString tFileLocationBase_LamK0 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_LamK0.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_LamK0 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_LamK0.Data(),tResultsDate.Data());

  TString tSaveDirectoryBase_LamK0 = tDirectoryBase_LamK0;

  TString tLocationMasterFitResults_LamK0 = TString::Format("%sMasterFitResults_%s.txt", tDirectoryBase_LamK0.Data(), tResultsDate.Data());

//-----------------------------------------------------------------------------

  TString tSaveNameModifier = LednickyFitter::BuildSaveNameModifier(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, FixD0, bUseStavCf, FixAllLambdaTo1, FixAllNormTo1, FixRadii, FixAllScattParams, tShareLambdaParams, tAllShareSingleLambdaParam, UsemTScalingOfResidualRadii, true, tTripleShareLambda, tTripleShareRadii);

  //For Triple analyses, systematics only currently stored in LamKch directory
  TString tSystematicsFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s.txt", tResultsDate.Data(), tSaveNameModifier.Data(), tSaveNameModifier.Data());


//-----------------------------------------------------------------------------
  bool bExistsCurrentSysFile;
  ifstream tFileIn;
  tFileIn.open(tSystematicsFileLocation);
  if(tFileIn) bExistsCurrentSysFile=true;
  else bExistsCurrentSysFile = false;
  tFileIn.close();
  if(!bExistsCurrentSysFile) cout << "WARNING!!!!!!!!!!!!!!!!!!!!!" << endl << "!bExistsCurrentSysFile, so syst. errs. on fit parameters not precisely accurate" << endl << endl;

//-----------------------------------------------------------------------------

  TripleFitGenerator* tTriple = new TripleFitGenerator(tFileLocationBase_LamKch, tFileLocationBaseMC_LamKch, tFileLocationBase_LamK0, tFileLocationBaseMC_LamK0, tCentType, tAnRunType_LamKch, tAnRunType_LamK0, tNPartialAnalysis, tGenType, tShareLambdaParams, tAllShareSingleLambdaParam, "", "", bUseStavCf);
  tTriple->SetSaveLocationBase(tSaveDirectoryBase_LamKch, tSaveDirectoryBase_LamK0,tSaveNameModifier);
  tTriple->SetSaveFileType(tSaveFileType);

  tTriple->SetFitType(tFitType);
  tTriple->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tTriple->SetNonFlatBgdFitTypes(tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamK0);
  tTriple->SetUseNewBgdTreatment(UseNewBgdTreatment);
  tTriple->SetApplyMomResCorrection(ApplyMomResCorrection);
  tTriple->SetIncludeResidualCorrelationsType(tIncludeResidualsType, aLambdaMin, aLambdaMax);  //TODO fix this in FitGenerator
  if(!UnboundLambda) tTriple->SetAllLambdaParamLimits(aLambdaMin, aLambdaMax);
  tTriple->SetChargedResidualsType(tChargedResidualsType);
  tTriple->SetResPrimMaxDecayType(tResPrimMaxDecayType);
  tTriple->SetMasterFileLocation(tLocationMasterFitResults_LamKch, tLocationMasterFitResults_LamK0);
  if(bExistsCurrentSysFile) tTriple->SetSystematicsFileLocation(tSystematicsFileLocation, tSystematicsFileLocation);

  if(FixRadii) 
  {
    tTriple->SetRadiusLimits({{4.13, 4.13}, {3.34, 3.34}, {2.59, 2.59}});
  }
/*
  if(FixRadii) 
  {
    if(tAnType==kLamK0 || tAnType==kALamK0) tTriple->SetRadiusLimits({{3.25, 3.25}, {2.75, 2.75}, {2.25, 2.25}});
    else tTriple->SetRadiusLimits({{3.5, 3.5}, {3.25, 3.25}, {2.5, 2.5}});
  }
  if(FixD0) tTriple->SetScattParamStartValue(0., kd0, true);
  if(FixAllScattParams)
  {
    if     (tAnType==kLamKchP || tAnType==kALamKchM) tTriple->SetScattParamStartValues(-1.02, 0.08, 0.92, true);
    else if(tAnType==kLamKchM || tAnType==kALamKchP) tTriple->SetScattParamStartValues(0.23, 0.64, 1.81, true);
    else if(tAnType==kLamK0 || tAnType==kALamK0)     tTriple->SetScattParamStartValues(-0.11, 0.10, -0.73, true);
    else assert(0);
  }
*/
  if(FixAllLambdaTo1) tTriple->SetLambdaParamStartValue(1.0, false, kMB, true);
  if(FixAllNormTo1) tTriple->SetFixNormParams(FixAllNormTo1);
  if(UsemTScalingOfResidualRadii) tTriple->SetUsemTScalingOfResidualRadii(UsemTScalingOfResidualRadii, mTScalingPowerOfResidualRadii);

//  if(ApplyNonFlatBackgroundCorrection && tNonFlatBgdFitType_LamKch == kPolynomial) tTriple->SetMinMaxBgdFit(kLamKchP, 0.32, 0.80);
//  if(ApplyNonFlatBackgroundCorrection && tNonFlatBgdFitType_LamK0 == kPolynomial)  tTriple->SetMinMaxBgdFit(kLamK0, 0.32, 0.80);
  if(ApplyNonFlatBackgroundCorrection && tNonFlatBgdFitType_LamKch == kPolynomial && tNonFlatBgdFitType_LamK0 == kPolynomial) 
  {
    tTriple->SetMinMaxBgdFit(kLamKchP, 0.32, 0.80);
    tTriple->SetMinMaxBgdFit(kLamK0, 0.32, 0.80);
  }


/*
  if(ApplyNonFlatBackgroundCorrection && tNonFlatBgdFitType_LamKch != kLinear)
  {
//    tTriple->SetKStarMinMaxNorm(kLamKchP, 0.5,0.6);
    tTriple->SetMinMaxBgdFit(kLamKchP, 0.45, 0.95);
    tTriple->SetAllRadiiLimits(1., 10.);

    if(tIncludeResidualsType == kIncludeNoResiduals) tTriple->SetAllLambdaParamLimits(0.1,1.0);
  }
  if(ApplyNonFlatBackgroundCorrection && tNonFlatBgdFitType_LamKch != kLinear)
  {
//    tTriple->SetKStarMinMaxNorm(kLamK0, 0.5,0.6);
    tTriple->SetMinMaxBgdFit(kLamK0, 0.45, 0.95);
    tTriple->SetAllRadiiLimits(1., 10.);

    if(tIncludeResidualsType == kIncludeNoResiduals) tTriple->SetAllLambdaParamLimits(0.1,1.0);
  }
*/


//  if(tNonFlatBgdFitType==kPolynomial) tTriple->SetMinMaxBgdFit(0.3, 1.99);
/*
  if(tAnType==kLamKchP && tIncludeResidualsType==kIncludeNoResiduals && tResultsDate.EqualTo("20171227"))
  {
//    tTriple->SetAllLambdaParamLimits(0.35,0.65);

    tTriple->SetLambdaParamLimits(0.30, 0.50, false, k0010);
    tTriple->SetLambdaParamLimits(0.30, 0.50, true, k0010);

    tTriple->SetLambdaParamLimits(0.30, 0.50, false, k1030);
    tTriple->SetLambdaParamLimits(0.30, 0.50, true, k1030);

    tTriple->SetLambdaParamLimits(0.50, 0.70, false, k3050);
    tTriple->SetLambdaParamLimits(0.50, 0.70, true, k3050);

  }
*/

//-------------------------------------------------------------------------------
  if(bDoFit)
  {
    tTriple->DoFit(tTripleShareLambda, tTripleShareRadii, tMaxFitKStar);
    if(bWriteToMasterFitValuesFile) tTriple->WriteToMasterFitValuesFile(tLocationMasterFitResults_LamKch, tLocationMasterFitResults_LamK0, tResultsDate);

//    TObjArray* tKStarwFitsCan = tTriple->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitTypes,SaveImages,bDrawSysErrs,bZoomROP);
    TObjArray* tKStarwFitsCan_Zoom = tTriple->DrawKStarCfswFits(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bDrawSysErrs, true, tSuppressFitInfoOutput, tLabelLines);
    TObjArray* tKStarwFitsCan_UnZoom = tTriple->DrawKStarCfswFits(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bDrawSysErrs, false, tSuppressFitInfoOutput, tLabelLines);

    TCanvas* tKStarwFitsCan_CombineConj_AllAn_Zoom = tTriple->DrawKStarCfswFits_CombineConj_AllAn(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bDrawSysErrs, true, tSuppressFitInfoOutput, tLabelLines);
    TCanvas* tKStarwFitsCan_CombineConj_AllAn_UnZoom = tTriple->DrawKStarCfswFits_CombineConj_AllAn(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bDrawSysErrs, false, tSuppressFitInfoOutput, tLabelLines);

    if(bDrawPartAn)
    {
      TObjArray* tKStarwFitsCan_FemtoMinus = tTriple->DrawKStarCfswFits_PartAn(kFemtoMinus,ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitTypes,SaveImages,bZoomROP);
      TObjArray* tKStarwFitsCan_FemtoPlus = tTriple->DrawKStarCfswFits_PartAn(kFemtoPlus,ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitTypes,SaveImages,bZoomROP);
    }

//    TObjArray* tKStarCfs = tTriple->DrawKStarCfs(SaveImages);
//    TObjArray* tModelKStarCfs = tTriple->DrawModelKStarCfs(SaveImages);
//    tTriple->FindGoodInitialValues(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);

    //-------------------------------------------------------------------------------
    TObjArray* tAllCanLamKchP;
    TObjArray* tCanPrimwFitsAndResidual;
    TObjArray* tAllResWithTransMatrices;

    bool aOutputCheckCorrectedCf = false;
    bool aZoomResiduals = true;
    TObjArray* tAllSingleKStarCfwFitAndResiduals;
    TObjArray* tAllSingleKStarCfwFitAndResiduals_FemtoMinus;
    TObjArray* tAllSingleKStarCfwFitAndResiduals_FemtoPlus;

    if(tIncludeResidualsType != kIncludeNoResiduals && bDrawResiduals)
    {
//      tAllCanLamKchP = tTriple->DrawAllResiduals(SaveImages);


      tCanPrimwFitsAndResidual = tTriple->DrawKStarCfswFitsAndResiduals(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitTypes,SaveImages,bDrawSysErrs,bZoomROP,aZoomResiduals);

      tAllResWithTransMatrices = tTriple->DrawAllResidualsWithTransformMatrices(SaveImages, bDrawResv2);

      bool bDrawData = false;

      tAllSingleKStarCfwFitAndResiduals = tTriple->DrawAllSingleKStarCfwFitAndResiduals(bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bDrawSysErrs, bZoomROP, aOutputCheckCorrectedCf);

      if(bDrawPartAn)
      {
        tAllSingleKStarCfwFitAndResiduals_FemtoMinus = tTriple->DrawAllSingleKStarCfwFitAndResiduals_PartAn(kFemtoMinus, bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bZoomROP, aOutputCheckCorrectedCf);
        tAllSingleKStarCfwFitAndResiduals_FemtoPlus = tTriple->DrawAllSingleKStarCfwFitAndResiduals_PartAn(kFemtoPlus, bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bZoomROP, aOutputCheckCorrectedCf);
      }
    }

  }


//-------------------------------------------------------------------------------

  if(bGenerateContours)
  {
    bool bFixAllOthers = false;
    tTriple->GenerateContourPlots(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, {4, 1}, "_All", bFixAllOthers, tTripleShareLambda, tTripleShareRadii, tMaxFitKStar);

  }


//-------------------------------------------------------------------------------

  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
