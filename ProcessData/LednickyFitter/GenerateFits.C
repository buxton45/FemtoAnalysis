#include "FitGeneratorAndDraw.h"
class FitGeneratorAndDraw;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------

  //--Rarely change---------------------
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB/*k0010*/;
  FitGeneratorType tGenType = kPairwConj;
  FitType tFitType = kChi2PML;
  //------------------------------------

  //*****************************************
  bool bDoFit = true;
  bool bGenerateContours = false;

  TString tResultsDate = "20171227"/*"20161027"*//*"20171220_onFlyStatusFalse"*//*"20171227_LHC10h"*//*"20180104_useIsProbableElectronMethodTrue"*//*"20180104_useIsProbableElectronMethodFalse"*/;
  AnalysisType tAnType = kLamKchP;

  double tMaxFitKStar=0.3;
  //*****************************************

  //--Save options
  bool SaveImages = false;
  TString tSaveFileType = "pdf";
  bool SaveImagesInRootFile = false;

  //--Sharing lambda
  bool tShareLambdaParams = false;          //If true, only share lambda parameters across like-centralities
  bool tAllShareSingleLambdaParam = false;

  //--Corrections
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;
  bool UseNewBgdTreatment = false;
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

  //--mT scaling
  bool UsemTScalingOfResidualRadii = false;
  double mTScalingPowerOfResidualRadii = -0.5;

  //--Plotting options
  bool bZoomROP = false;
  bool bDrawResiduals = false;
  bool bDrawPartAn = false;

  bool bDrawSysErrs = true;

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

//  TString tSaveDirectoryBase = TString::Format("/home/jesse/Analysis/Presentations/AliFemto/20171108/Figures/%s/", cAnalysisBaseTags[tAnType]);
  TString tSaveDirectoryBase = tDirectoryBase;

//-----------------------------------------------------------------------------

  TString tSaveNameModifier = "";
  LednickyFitter::AppendFitInfo(tSaveNameModifier, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, FixD0);

  if(FixAllLambdaTo1) tSaveNameModifier += TString("_FixAllLambdaTo1");
  if(FixRadii) tSaveNameModifier += TString("_FixedRadii");
  if(FixAllScattParams) tSaveNameModifier += TString("_FixedScattParams");
  if(tShareLambdaParams) tSaveNameModifier += TString("_ShareLamAcrossCent");
  if(tAllShareSingleLambdaParam && !FixAllLambdaTo1) tSaveNameModifier += TString("_SingleLamParam");
  if(UsemTScalingOfResidualRadii) tSaveNameModifier += TString::Format("_UsingmTScalingOfResidualRadii");

//-----------------------------------------------------------------------------

  FitGeneratorAndDraw* tLamKchP = new FitGeneratorAndDraw(tFileLocationBase,tFileLocationBaseMC,tAnType, tCentType,tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);
//  FitGeneratorAndDraw* tLamKchP = new FitGeneratorAndDraw(tFileLocationBase,tFileLocationBaseMC,tAnType,{k0010,k1030},tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);
//  tLamKchP->SetRadiusStartValues({3.0,4.0,5.0});
//  tLamKchP->SetRadiusLimits({{0.,10.},{0.,10.},{0.,10.}});
  tLamKchP->SetSaveLocationBase(tSaveDirectoryBase,tSaveNameModifier);
  //tLamKchP->SetFitType(kChi2);
  tLamKchP->SetSaveFileType(tSaveFileType);


//  TCanvas* tKStarCan = tLamKchP->DrawKStarCfs();

  //tLamKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tLamKchP->SetFitType(tFitType);
  tLamKchP->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tLamKchP->SetNonFlatBgdFitType(tNonFlatBgdFitType);
  tLamKchP->SetUseNewBgdTreatment(UseNewBgdTreatment);
  tLamKchP->SetApplyMomResCorrection(ApplyMomResCorrection);
  tLamKchP->SetIncludeResidualCorrelationsType(tIncludeResidualsType, aLambdaMin, aLambdaMax);  //TODO fix this in FitGenerator
  if(!UnboundLambda) tLamKchP->SetAllLambdaParamLimits(aLambdaMin, aLambdaMax);
  tLamKchP->SetChargedResidualsType(tChargedResidualsType);
  tLamKchP->SetResPrimMaxDecayType(tResPrimMaxDecayType);
  if(FixRadii) 
  {
    if(tAnType==kLamK0 || tAnType==kALamK0) tLamKchP->SetRadiusLimits({{3.25, 3.25}, {2.75, 2.75}, {2.25, 2.25}});
    else tLamKchP->SetRadiusLimits({{3.5, 3.5}, {3.25, 3.25}, {2.5, 2.5}});
  }
  if(FixD0) tLamKchP->SetScattParamStartValue(0., kd0, true);
  if(FixAllScattParams)
  {
    if     (tAnType==kLamKchP || tAnType==kALamKchM) tLamKchP->SetScattParamStartValues(-1.02, 0.08, 0.92, true)/*tLamKchP->SetScattParamStartValues(-0.76, 0.12, 0., true)*/;
    else if(tAnType==kLamKchM || tAnType==kALamKchP) tLamKchP->SetScattParamStartValues(0.23, 0.64, 1.81, true)/*tLamKchP->SetScattParamStartValues(0.25, 0.71, 0., true)*/;
    else if(tAnType==kLamK0 || tAnType==kALamK0)     tLamKchP->SetScattParamStartValues(-0.11, 0.10, -0.73, true)/*tLamKchP->SetScattParamStartValues(-0.12, 0.16, 0., true)*/;
    else assert(0);

  }
  if(FixAllLambdaTo1) tLamKchP->SetLambdaParamStartValue(1.0, false, kMB, true);
  if(UsemTScalingOfResidualRadii) tLamKchP->SetUsemTScalingOfResidualRadii(UsemTScalingOfResidualRadii, mTScalingPowerOfResidualRadii);

  if(ApplyNonFlatBackgroundCorrection && tNonFlatBgdFitType != kLinear)
  {
//    tLamKchP->SetKStarMinMaxNorm(0.5,0.6);
    tLamKchP->SetMinMaxBgdFit(0.45, 0.95);
    tLamKchP->SetAllRadiiLimits(1., 10.);

    if(tIncludeResidualsType == kIncludeNoResiduals) tLamKchP->SetAllLambdaParamLimits(0.1,1.0);
//    else tLamKchP->SetAllLambdaParamLimits(0.1,2.0);
      //Don't seem to need lambda limits with residuals.  In fact, when limits in place, the fit doesn't converge
      // Without limits, the fit converges (with lambda values within limits!)
  }

//  if(tNonFlatBgdFitType==kPolynomial) tLamKchP->SetMinMaxBgdFit(0.3, 1.99);

  if(tAnType==kLamKchP && tIncludeResidualsType==kIncludeNoResiduals && tResultsDate.EqualTo("20171227"))
  {
//    tLamKchP->SetAllLambdaParamLimits(0.35,0.65);

    tLamKchP->SetLambdaParamLimits(0.30, 0.50, false, k0010);
    tLamKchP->SetLambdaParamLimits(0.30, 0.50, true, k0010);

    tLamKchP->SetLambdaParamLimits(0.30, 0.50, false, k1030);
    tLamKchP->SetLambdaParamLimits(0.30, 0.50, true, k1030);

    tLamKchP->SetLambdaParamLimits(0.50, 0.70, false, k3050);
    tLamKchP->SetLambdaParamLimits(0.50, 0.70, true, k3050);

  }


//-------------------------------------------------------------------------------
  if(bDoFit)
  {
    tLamKchP->DoFit(tMaxFitKStar);
//    TCanvas* tKStarwFitsCan = tLamKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bDrawSysErrs,bZoomROP);
    TCanvas* tKStarwFitsCan_Zoom = tLamKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bDrawSysErrs,true);
    TCanvas* tKStarwFitsCan_UnZoom = tLamKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bDrawSysErrs,false);

    if(bDrawPartAn)
    {
      TCanvas* tKStarwFitsCan_FemtoMinus = tLamKchP->DrawKStarCfswFits_PartAn(kFemtoMinus,ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bZoomROP);
      TCanvas* tKStarwFitsCan_FemtoPlus = tLamKchP->DrawKStarCfswFits_PartAn(kFemtoPlus,ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bZoomROP);
    }

//    TCanvas* tKStarCfs = tLamKchP->DrawKStarCfs(SaveImages);
//    TCanvas* tModelKStarCfs = tLamKchP->DrawModelKStarCfs(SaveImages);
//    tLamKchP->FindGoodInitialValues(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);

    //-------------------------------------------------------------------------------
    TObjArray* tAllCanLamKchP;
    TCanvas* tCanPrimwFitsAndResidual;
    TObjArray* tAllResWithTransMatrices;

    bool aOutputCheckCorrectedCf = true;
    bool aZoomResiduals = true;
    TObjArray* tAllSingleKStarCfwFitAndResiduals;
    TObjArray* tAllSingleKStarCfwFitAndResiduals_FemtoMinus;
    TObjArray* tAllSingleKStarCfwFitAndResiduals_FemtoPlus;

    if(tIncludeResidualsType != kIncludeNoResiduals && bDrawResiduals)
    {
//      tAllCanLamKchP = tLamKchP->DrawAllResiduals(SaveImages);


      tCanPrimwFitsAndResidual = tLamKchP->DrawKStarCfswFitsAndResiduals(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,bDrawSysErrs,bZoomROP,aZoomResiduals);

//      tAllResWithTransMatrices = tLamKchP->DrawAllResidualsWithTransformMatrices(SaveImages);

      bool bDrawData = false;

      tAllSingleKStarCfwFitAndResiduals = tLamKchP->DrawAllSingleKStarCfwFitAndResiduals(bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitType, SaveImages, bDrawSysErrs, bZoomROP, aOutputCheckCorrectedCf);

      if(bDrawPartAn)
      {
        tAllSingleKStarCfwFitAndResiduals_FemtoMinus = tLamKchP->DrawAllSingleKStarCfwFitAndResiduals_PartAn(kFemtoMinus, bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitType, SaveImages, bZoomROP, aOutputCheckCorrectedCf);
        tAllSingleKStarCfwFitAndResiduals_FemtoPlus = tLamKchP->DrawAllSingleKStarCfwFitAndResiduals_PartAn(kFemtoPlus, bDrawData, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitType, SaveImages, bZoomROP, aOutputCheckCorrectedCf);
      }
    }

    //-------------------------------------------------------------------------------
    if(SaveImagesInRootFile)
    {
      TFile *tFile = new TFile(tLamKchP->GetSaveLocationBase() + TString(cAnalysisBaseTags[tAnType]) + TString("Plots") + tLamKchP->GetSaveNameModifier() + TString(".root"), "RECREATE");
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

  }
//-------------------------------------------------------------------------------

  if(bGenerateContours)
  {
    bool bFixAllOthers = false;

    if(FixRadii && FixD0)
    {
      if(tAnType != kLamK0)
      {
        tLamKchP->GenerateContourPlots(10, {0, 1, 9, 10}, {4, 1}, "_0010", bFixAllOthers);
//        tLamKchP->GenerateContourPlots(10, {2, 3, 9, 10}, {4, 1}, "_1030", bFixAllOthers);
//        tLamKchP->GenerateContourPlots(10, {4, 5, 9, 10}, {4, 1}, "_3050", bFixAllOthers);
      }
      else
      {
        tLamKchP->GenerateContourPlots(10, {0, 4, 5}, {4, 1}, "_0010_1030_3050", bFixAllOthers);
      }
    }
    else if(FixRadii)
    {
      if(tAnType != kLamK0)
      {
        tLamKchP->GenerateContourPlots(10, {0, 1, 9, 10, 11}, {4, 1}, "_0010", bFixAllOthers);
//        tLamKchP->GenerateContourPlots(10, {2, 3, 9, 10, 11}, {4, 1}, "_1030", bFixAllOthers);
//        tLamKchP->GenerateContourPlots(10, {4, 5, 9, 10, 11}, {4, 1}, "_3050", bFixAllOthers);
      }
      else
      {
        tLamKchP->GenerateContourPlots(10, {0, 4, 5, 6}, {4, 1}, "_0010_1030_3050", bFixAllOthers);
      }
    }
    else if(FixD0)
    {
      if(tAnType != kLamK0)
      {
        tLamKchP->GenerateContourPlots(10, {0, 1, 6, 9, 10}, {4, 1}, "_0010", bFixAllOthers);
//        tLamKchP->GenerateContourPlots(10, {2, 3, 7, 9, 10}, {4, 1}, "_1030", bFixAllOthers);
//        tLamKchP->GenerateContourPlots(10, {4, 5, 8, 9, 10}, {4, 1}, "_3050", bFixAllOthers);
      }
      else
      {
        tLamKchP->GenerateContourPlots(10, {0, 1, 4, 5}, {4, 1}, "_0010", bFixAllOthers);
//        tLamKchP->GenerateContourPlots(10, {0, 2, 4, 5}, {4, 1}, "_1030", bFixAllOthers);
//        tLamKchP->GenerateContourPlots(10, {0, 3, 4, 5}, {4, 1}, "_3050", bFixAllOthers);
      }
    }
    else
    {
      tLamKchP->GenerateContourPlots(10, k0010, {4, 1}, bFixAllOthers);
//      tLamKchP->GenerateContourPlots(10, {0, 1, 6, 9, 10, 11}, {4, 1}, "Custom", bFixAllOthers);
//      tLamKchP->GenerateContourPlots(10, {2, 3, 7, 9, 10, 11}, {4, 1}, "Custom", bFixAllOthers);
//      tLamKchP->GenerateContourPlots(10, {4, 5, 8, 9, 10, 11}, {4, 1}, "Custom", bFixAllOthers);
    }
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
