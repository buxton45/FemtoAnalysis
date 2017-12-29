#include "FitGenerator.h"
class FitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  TString tResultsDate = "20161027";
//  TString tResultsDate = "20171220_onFlyStatusFalse";
//  TString tResultsDate = "20171227";
//  TString tResultsDate = "20171227_LHC10h";

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
//  CentralityType tCentType = k0010;  //TODO
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPairwConj;
  FitType tFitType = kChi2PML;
  bool tShareLambdaParams = false;
  bool tAllShareSingleLambdaParam = false;

  bool SaveImages = false;
  bool SaveImagesInRootFile = false;
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;

  IncludeResidualsType tIncludeResidualsType = kIncludeNoResiduals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;
  ResPrimMaxDecayType tResPrimMaxDecayType = k5fm;

  bool UnboundLambda = true;

  bool FixRadii = false;
  bool FixD0 = false;
  bool FixAllScattParams = false;
  bool FixAllLambdaTo1 = false;
  if(FixAllLambdaTo1) tAllShareSingleLambdaParam = true;

  bool UsemTScalingOfResidualRadii = false;
  double mTScalingPowerOfResidualRadii = -0.5;

  double aLambdaMin=0., aLambdaMax=1.;
  if(UnboundLambda) aLambdaMax=0.;

  bool bZoomROP = true;
  bool bDrawResiduals = false;


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

  TString tSaveNameModifier = "";
  if(ApplyMomResCorrection) tSaveNameModifier += TString("_MomResCrctn");
  if(ApplyNonFlatBackgroundCorrection) tSaveNameModifier += TString("_NonFlatBgdCrctn");

  if(tAllShareSingleLambdaParam && !FixAllLambdaTo1) tSaveNameModifier += TString("_SingleLamParam");
  if(FixAllLambdaTo1) tSaveNameModifier += TString("_FixAllLambdaTo1");

  if(FixRadii) tSaveNameModifier += TString("_FixedRadii");
  if(FixD0) tSaveNameModifier += TString("_FixedD0");
  if(FixAllScattParams) tSaveNameModifier += TString("_FixedScattParams");

  tSaveNameModifier += cIncludeResidualsTypeTags[tIncludeResidualsType];
  if(tIncludeResidualsType != kIncludeNoResiduals)
  {
    tSaveNameModifier += cResPrimMaxDecayTypeTags[tResPrimMaxDecayType];
    tSaveNameModifier += cChargedResidualsTypeTags[tChargedResidualsType];
  }

  if(UsemTScalingOfResidualRadii) tSaveNameModifier += TString::Format("_UsingmTScalingOfResidualRadii");


  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType, tCentType,tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);
//  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType,{k0010,k1030},tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);
//  tLamKchP->SetRadiusStartValues({3.0,4.0,5.0});
//  tLamKchP->SetRadiusLimits({{0.,10.},{0.,10.},{0.,10.}});
  tLamKchP->SetSaveLocationBase(tSaveDirectoryBase,tSaveNameModifier);
  //tLamKchP->SetFitType(kChi2);


//  TCanvas* tKStarCan = tLamKchP->DrawKStarCfs();

  //tLamKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tLamKchP->SetFitType(tFitType);
  tLamKchP->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tLamKchP->SetNonFlatBgdFitType(tNonFlatBgdFitType);
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






  tLamKchP->DoFit();
  TCanvas* tKStarwFitsCan = tLamKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages,false,bZoomROP);
//  TCanvas* tKStarCfs = tLamKchP->DrawKStarCfs(SaveImages);
//  TCanvas* tModelKStarCfs = tLamKchP->DrawModelKStarCfs(SaveImages);
//  tLamKchP->FindGoodInitialValues(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);

//-------------------------------------------------------------------------------
  TObjArray* tAllCanLamKchP;
  TCanvas* tCanPrimwFitsAndResidual;
  TObjArray* tAllResWithTransMatrices;

  if(tIncludeResidualsType != kIncludeNoResiduals && bDrawResiduals)
  {
    TCanvas* tCanLamKchP = tLamKchP->DrawResiduals(0,k0010,cAnalysisBaseTags[tAnType]);

    tAllCanLamKchP = tLamKchP->DrawAllResiduals(SaveImages);

//    TCanvas* tCanPrimWithRes = tLamKchP->DrawPrimaryWithResiduals(0,k0010,TString("PrimaryWithResidual_")+TString(cAnalysisBaseTags[tAnType]));
    tCanPrimwFitsAndResidual = tLamKchP->DrawKStarCfswFitsAndResiduals(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages);

    tAllResWithTransMatrices = tLamKchP->DrawAllResidualsWithTransformMatrices(SaveImages);
  }

//-------------------------------------------------------------------------------
  if(SaveImagesInRootFile)
  {
    TFile *tFile = new TFile(tLamKchP->GetSaveLocationBase() + TString(cAnalysisBaseTags[tAnType]) + TString("Plots") + tLamKchP->GetSaveNameModifier() + TString(".root"), "RECREATE");
    tKStarwFitsCan->Write();
    if(tIncludeResidualsType != kIncludeNoResiduals && bDrawResiduals)
    {
      for(int i=0; i<tAllCanLamKchP->GetEntries(); i++) (TCanvas*)tAllCanLamKchP->At(i)->Write();
      tCanPrimwFitsAndResidual->Write();
    }
    tFile->Close();
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
