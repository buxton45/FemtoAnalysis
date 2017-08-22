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

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
//  CentralityType tCentType = k0010;  //TODO
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPairwConj;
  bool tShareLambdaParams = false;
  bool tAllShareSingleLambdaParam = false;

  bool SaveImages = false;
  bool SaveImagesInRootFile = false;
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;
  bool IncludeResiduals = true;

  bool bDrawResiduals = true;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());


  TString tSaveNameModifier = "";
  if(ApplyMomResCorrection) tSaveNameModifier += TString("_MomResCrctn");
  if(ApplyNonFlatBackgroundCorrection) tSaveNameModifier += TString("_NonFlatBgdCrctn");
  if(tAllShareSingleLambdaParam) tSaveNameModifier += TString("_SingleLamParam");
  if(IncludeResiduals) tSaveNameModifier += TString("_ResidualsIncluded");
  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType, tCentType,tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);
//  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType,{k0010,k1030},tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);
//  tLamKchP->SetRadiusStartValues({3.0,4.0,5.0});
//  tLamKchP->SetRadiusLimits({{0.,10.},{0.,10.},{0.,10.}});
  tLamKchP->SetSaveLocationBase(tDirectoryBase,tSaveNameModifier);
  //tLamKchP->SetFitType(kChi2);


//  TCanvas* tKStarCan = tLamKchP->DrawKStarCfs();

  //tLamKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tLamKchP->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tLamKchP->SetNonFlatBgdFitType(tNonFlatBgdFitType);
  tLamKchP->SetApplyMomResCorrection(ApplyMomResCorrection);
  tLamKchP->SetIncludeResidualCorrelations(IncludeResiduals);

  tLamKchP->DoFit();
  TCanvas* tKStarwFitsCan = tLamKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages);
//  TCanvas* tKStarCfs = tLamKchP->DrawKStarCfs(SaveImages);
//  TCanvas* tModelKStarCfs = tLamKchP->DrawModelKStarCfs(SaveImages);
//  tLamKchP->FindGoodInitialValues(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);

//-------------------------------------------------------------------------------
  TObjArray* tAllCanLamKchP;
  TCanvas* tCanPrimwFitsAndResidual;

  if(IncludeResiduals && bDrawResiduals)
  {
    TCanvas* tCanLamKchP = tLamKchP->DrawResiduals(0,k0010,cAnalysisBaseTags[tAnType]);

    tAllCanLamKchP = tLamKchP->DrawAllResiduals(SaveImages);

//    TCanvas* tCanPrimWithRes = tLamKchP->DrawPrimaryWithResiduals(0,k0010,TString("PrimaryWithResidual_")+TString(cAnalysisBaseTags[tAnType]));
    tCanPrimwFitsAndResidual = tLamKchP->DrawKStarCfswFitsAndResiduals(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages);
  }

//-------------------------------------------------------------------------------
  if(SaveImagesInRootFile)
  {
    TFile *tFile = new TFile(tLamKchP->GetSaveLocationBase() + TString(cAnalysisBaseTags[tAnType]) + TString("Plots") + tLamKchP->GetSaveNameModifier() + TString(".root"), "RECREATE");
    tKStarwFitsCan->Write();
    if(IncludeResiduals && bDrawResiduals)
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
