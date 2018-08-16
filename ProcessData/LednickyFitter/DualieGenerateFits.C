#include "DualieFitGenerator.h"
class DualieFitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  //******** Notes on tShareLambdaParams and tDualieShareLambda ********
  // tShareLambdaParams controls how lambda are shared within a given FitGenerator object (across a given centrality)
  //      i.e. between (LamKchP and ALamKchM) or between (LamKchM and ALamKchP)
  // tDualieShareLambda controls how lambda are shared across the two FitGenerator objects (across a given centrality)
  //      i.e. between (LamKchP and LamKchM) or (ALamKchM and ALamKchP)
  //
  // Here are the different scenarios with explanations of consequences
  // (1) tShareLambdaParams=true  &&  tDualieShareLambda=true
  //       There will be a single lambda parameter for each centrality
  //       3 lambda parameters in total for MB analysis
  //
  // (2) tShareLambdaParams=true  &&  tDualieShareLambda=false
  //       Conjugate pairs in each centrality WILL share a lambda, but the different FitGenerators WILL NOT
  //       2 lambda parameters per centrality (lam1:LamKchP & ALamKchM and lam2:LamKchM & ALamKchP)
  //       6 lambda parameters in total for MB analysis
  //
  // (3) tShareLambdaParams=false &&  tDualieShareLambda=true
  //       Conjugate pairs in each centrality WILL NOT share a lambda, but the different FitGenerators WILL
  //       2 lambda parameters per centrality (lam1:LamKchP & LamKchM and lam2:ALamKchM & ALamKchP)
  //       6 lambda parameters in total for MB analysis
  //       THIS OPTION PROBABLY DOES NOT MAKE SENSE TO USE
  //
  // (4) tShareLambdaParams=false &&  tDualieShareLambda=false
  //       Conjugate pairs in each centrality WILL NOT share a lambda, and the different FitGenerators WILL NOT
  //       4 lambda parameters per centrality (lam1:LamKchP & lam2:ALamKchM & lam3:LamKchM & lam4:ALamKchP)
  //       12 lambda parameters in total for MB analysis
  //********************************************************************

  //--Rarely change---------------------
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB/*k0010*/;
  FitGeneratorType tGenType = kPairwConj;
  FitType tFitType = kChi2PML;
  //------------------------------------

  //*****************************************
  bool bDoFit = true;

  TString tResultsDate = "20180505";
  AnalysisType tAnType = kLamKchP;

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

  //--Dualie sharing options
  bool tDualieShareLambda = true;
  bool tDualieShareRadii = true;

  //--Corrections
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kPolynomial;
    if(tNonFlatBgdFitType==kDivideByTherm)
    {
      tFitType = kChi2;
      ApplyNonFlatBackgroundCorrection = false;
    }
  bool UseNewBgdTreatment = false;  //TODO For now, and maybe forever, should always be FALSE!!!!!
    if(UseNewBgdTreatment) tMaxFitKStar = 0.5;

  //--Residuals
  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp/*kUseCoulombOnlyInterpForAll*/;
  ResPrimMaxDecayType tResPrimMaxDecayType = k4fm;

  //--Bound lambda
  bool UnboundLambda = true;
  double aLambdaMin=0., aLambdaMax=1.;
  if(UnboundLambda) aLambdaMax=0.;

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
  bool bDrawPartAn = false;

  bool bDrawSysErrs = false;

//-----------------------------------------------------------------------------
  if(tShareLambdaParams==false && tDualieShareLambda==true)
  {
    cout << "!!!!!!!!!!!!!!! tShareLambdaParams==false && tDualieShareLambda==true !!!!!!!!!!!!!!!" << endl;
    cout << "This means Conjugate pairs in each centrality WILL NOT share a lambda, but the different FitGenerators WILL" << endl;
    cout << "2 lambda parameters per centrality (lam1:LamKchP & LamKchM and lam2:ALamKchM & ALamKchP)" << endl;
    cout << "6 lambda parameters in total for MB analysis" << endl;
    cout << "Is this really what you want? (0=No  1=Yes)" << endl;

    int tResponse;
    cin >> tResponse;
    assert(tResponse);
  }

//-----------------------------------------------------------------------------

  if(tAnType==kLamK0)
  {
    tAllShareSingleLambdaParam = true;
    UnboundLambda = false;
    aLambdaMin = 0.4;  //TODO currently, if tIncludeResidualsType = kIncludeNoResiduals, this does nothing
    aLambdaMax = 0.6;  //TODO "                                                                          "

    if(tIncludeResidualsType != kIncludeNoResiduals)
    {
      aLambdaMin = 0.6;
      aLambdaMax = 1.5;
    }
  }

//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

//  TString tSaveDirectoryBase = TString::Format("/home/jesse/Analysis/Presentations/AliFemto/20180627/Figures/Fits/Dualie/%s/", cAnalysisBaseTags[tAnType]);
  TString tSaveDirectoryBase = tDirectoryBase;

  TString tLocationMasterFitResults = TString::Format("%sMasterFitResults_%s.txt", tDirectoryBase.Data(), tResultsDate.Data());

//-----------------------------------------------------------------------------

  TString tSaveNameModifier = LednickyFitter::BuildSaveNameModifier(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitType, tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, FixD0, bUseStavCf, FixAllLambdaTo1, FixAllNormTo1, FixRadii, FixAllScattParams, tShareLambdaParams, tAllShareSingleLambdaParam, UsemTScalingOfResidualRadii, true, tDualieShareLambda, tDualieShareRadii);

//-----------------------------------------------------------------------------

  DualieFitGenerator* tDualie = new DualieFitGenerator(tFileLocationBase, tFileLocationBaseMC, tAnType, tCentType, tAnRunType, tNPartialAnalysis, tGenType, tShareLambdaParams, tAllShareSingleLambdaParam, "", bUseStavCf);
  tDualie->SetSaveLocationBase(tSaveDirectoryBase,tSaveNameModifier);
  tDualie->SetSaveFileType(tSaveFileType);

  tDualie->SetFitType(tFitType);
  tDualie->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tDualie->SetNonFlatBgdFitType(tNonFlatBgdFitType);
  tDualie->SetUseNewBgdTreatment(UseNewBgdTreatment);
  tDualie->SetApplyMomResCorrection(ApplyMomResCorrection);
  tDualie->SetIncludeResidualCorrelationsType(tIncludeResidualsType, aLambdaMin, aLambdaMax);  //TODO fix this in FitGenerator
  if(!UnboundLambda) tDualie->SetAllLambdaParamLimits(aLambdaMin, aLambdaMax);
  tDualie->SetChargedResidualsType(tChargedResidualsType);
  tDualie->SetResPrimMaxDecayType(tResPrimMaxDecayType);

  if(FixRadii) 
  {
    if(tAnType==kLamK0 || tAnType==kALamK0) tDualie->SetRadiusLimits({{3.25, 3.25}, {2.75, 2.75}, {2.25, 2.25}});
    else tDualie->SetRadiusLimits({{3.5, 3.5}, {3.25, 3.25}, {2.5, 2.5}});
  }
  if(FixD0) tDualie->SetScattParamStartValue(0., kd0, true);
  if(FixAllScattParams)
  {
    if     (tAnType==kLamKchP || tAnType==kALamKchM) tDualie->SetScattParamStartValues(-1.02, 0.08, 0.92, true);
    else if(tAnType==kLamKchM || tAnType==kALamKchP) tDualie->SetScattParamStartValues(0.23, 0.64, 1.81, true);
    else if(tAnType==kLamK0 || tAnType==kALamK0)     tDualie->SetScattParamStartValues(-0.11, 0.10, -0.73, true);
    else assert(0);
  }

  if(FixAllLambdaTo1) tDualie->SetLambdaParamStartValue(1.0, false, kMB, true);
  if(FixAllNormTo1) tDualie->SetFixNormParams(FixAllNormTo1);
  if(UsemTScalingOfResidualRadii) tDualie->SetUsemTScalingOfResidualRadii(UsemTScalingOfResidualRadii, mTScalingPowerOfResidualRadii);


  if(ApplyNonFlatBackgroundCorrection && tNonFlatBgdFitType != kLinear)
  {
//    tDualie->SetKStarMinMaxNorm(0.5,0.6);
    tDualie->SetMinMaxBgdFit(0.45, 0.95);
    tDualie->SetAllRadiiLimits(1., 10.);

    if(tIncludeResidualsType == kIncludeNoResiduals) tDualie->SetAllLambdaParamLimits(0.1,1.0);
//    else tDualie->SetAllLambdaParamLimits(0.1,2.0);
      //Don't seem to need lambda limits with residuals.  In fact, when limits in place, the fit doesn't converge
      // Without limits, the fit converges (with lambda values within limits!)
  }


//  if(tNonFlatBgdFitType==kPolynomial) tDualie->SetMinMaxBgdFit(0.3, 1.99);

  if(tAnType==kLamKchP && tIncludeResidualsType==kIncludeNoResiduals && tResultsDate.EqualTo("20171227"))
  {
//    tDualie->SetAllLambdaParamLimits(0.35,0.65);

    tDualie->SetLambdaParamLimits(0.30, 0.50, false, k0010);
    tDualie->SetLambdaParamLimits(0.30, 0.50, true, k0010);

    tDualie->SetLambdaParamLimits(0.30, 0.50, false, k1030);
    tDualie->SetLambdaParamLimits(0.30, 0.50, true, k1030);

    tDualie->SetLambdaParamLimits(0.50, 0.70, false, k3050);
    tDualie->SetLambdaParamLimits(0.50, 0.70, true, k3050);

  }


//-------------------------------------------------------------------------------
  if(bDoFit)
  {
    tDualie->DoFit(tDualieShareLambda, tDualieShareRadii, tMaxFitKStar);
    if(bWriteToMasterFitValuesFile) tDualie->WriteToMasterFitValuesFile(tLocationMasterFitResults, tResultsDate);

//    TObjArray* tKStarwFitsCan = tDualie->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bDrawSysErrs,bZoomROP);
    TObjArray* tKStarwFitsCan_Zoom = tDualie->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bDrawSysErrs,true);
    TObjArray* tKStarwFitsCan_UnZoom = tDualie->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bDrawSysErrs,false);

    if(bDrawPartAn)
    {
      TObjArray* tKStarwFitsCan_FemtoMinus = tDualie->DrawKStarCfswFits_PartAn(kFemtoMinus,ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bZoomROP);
      TObjArray* tKStarwFitsCan_FemtoPlus = tDualie->DrawKStarCfswFits_PartAn(kFemtoPlus,ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bZoomROP);
    }

//    TObjArray* tKStarCfs = tDualie->DrawKStarCfs(SaveImages);
//    TObjArray* tModelKStarCfs = tDualie->DrawModelKStarCfs(SaveImages);
//    tDualie->FindGoodInitialValues(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);

    //-------------------------------------------------------------------------------
    TObjArray* tAllCanLamKchP;
    TObjArray* tCanPrimwFitsAndResidual;
    TObjArray* tAllResWithTransMatrices;

    bool aOutputCheckCorrectedCf = true;
    bool aZoomResiduals = true;
    TObjArray* tAllSingleKStarCfwFitAndResiduals;
    TObjArray* tAllSingleKStarCfwFitAndResiduals_FemtoMinus;
    TObjArray* tAllSingleKStarCfwFitAndResiduals_FemtoPlus;

    if(tIncludeResidualsType != kIncludeNoResiduals && bDrawResiduals)
    {
//      tAllCanLamKchP = tDualie->DrawAllResiduals(SaveImages);


      tCanPrimwFitsAndResidual = tDualie->DrawKStarCfswFitsAndResiduals(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bDrawSysErrs,bZoomROP,aZoomResiduals);

      tAllResWithTransMatrices = tDualie->DrawAllResidualsWithTransformMatrices(SaveImages);

      bool bDrawData = false;

      tAllSingleKStarCfwFitAndResiduals = tDualie->DrawAllSingleKStarCfwFitAndResiduals(bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitType, SaveImages, bDrawSysErrs, bZoomROP, aOutputCheckCorrectedCf);

      if(bDrawPartAn)
      {
        tAllSingleKStarCfwFitAndResiduals_FemtoMinus = tDualie->DrawAllSingleKStarCfwFitAndResiduals_PartAn(kFemtoMinus, bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitType, SaveImages, bZoomROP, aOutputCheckCorrectedCf);
        tAllSingleKStarCfwFitAndResiduals_FemtoPlus = tDualie->DrawAllSingleKStarCfwFitAndResiduals_PartAn(kFemtoPlus, bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitType, SaveImages, bZoomROP, aOutputCheckCorrectedCf);
      }
    }

    //-------------------------------------------------------------------------------
/*
    if(SaveImagesInRootFile)
    {
      TFile *tFile = new TFile(tDualie->GetSaveLocationBase() + TString(cAnalysisBaseTags[tAnType]) + TString("Plots") + tDualie->GetSaveNameModifier() + TString(".root"), "RECREATE");
//      tKStarwFitsCan->Write();
      tKStarwFitsCan_Zoom->Write();
      tKStarwFitsCan_UnZoom->Write();
      if(tIncludeResidualsType != kIncludeNoResiduals && bDrawResiduals)
      {
        for(int i=0; i<tAllCanLamKchP->GetEntries(); i++) (TCanvas*)tAllCanLamKchP->At(i)->Write();
        tCanPrimwFitsAndResidual->Write();
      }
      tFile->Close();
    }
*/
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
