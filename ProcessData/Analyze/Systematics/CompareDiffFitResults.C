#include "SystematicAnalysis.h"
class SystematicAnalysis;

#include "Types_SysFileInfo.h"

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  TString tParentResultsDate = "20161027";  //Parent analysis these systematics are to accompany

  AnalysisType tAnType = kALamK0;
  CentralityType tCentType = k0010;
  int tFileInfoNumber = -3;
  int tCutValIndex1 = 1;
  int tCutValIndex2 = 2;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  bool bWriteToFile = true;

  SystematicsFileInfo tFileInfo = GetFileInfo_LamK(tFileInfoNumber, tParentResultsDate);
    TString tResultsDate = tFileInfo.resultsDate;
    TString tDirNameModifierBase1 = tFileInfo.dirNameModifierBase1;
    vector<double> tModifierValues1 = tFileInfo.modifierValues1;
    TString tDirNameModifierBase2 = tFileInfo.dirNameModifierBase2;
    vector<double> tModifierValues2 = tFileInfo.modifierValues2;
    bool tAllCent = tFileInfo.allCentralities;

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics_LamK_%s/Results_%s_Systematics%s", tParentResultsDate, tGeneralAnTypeName.Data(), tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tDirectoryBase.Remove(TString::kTrailing,'_');
    tDirectoryBase += tDirNameModifierBase2;
  }
  tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

  TString tFileLocationBase = tDirectoryBase + TString::Format("Results_%s_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
  TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_%sMC_Systematics%s",tGeneralAnTypeName.Data(),tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tFileLocationBase.Remove(TString::kTrailing,'_');
    tFileLocationBaseMC.Remove(TString::kTrailing,'_');

    tFileLocationBase += tDirNameModifierBase2;
    tFileLocationBaseMC += tDirNameModifierBase2;
  }
  tFileLocationBase += tResultsDate;
  tFileLocationBaseMC += tResultsDate;



  TString tDirNameModifier1 = tDirNameModifierBase1 + TString::Format("%0.6f",tModifierValues1[tCutValIndex1]);
  if(!tDirNameModifierBase2.IsNull()) tDirNameModifier1 += tDirNameModifierBase2 + TString::Format("%0.6f",tModifierValues2[tCutValIndex1]);

  TString tDirNameModifier2 = tDirNameModifierBase1 + TString::Format("%0.6f",tModifierValues1[tCutValIndex2]);
  if(!tDirNameModifierBase2.IsNull()) tDirNameModifier2 += tDirNameModifierBase2 + TString::Format("%0.6f",tModifierValues2[tCutValIndex2]);

  Analysis* tAnalysis1 = new Analysis(tFileLocationBase,tAnType,tCentType,kTrainSys,2,tDirNameModifier1);
  tAnalysis1->BuildKStarHeavyCf();
  TH1* tHist1 = tAnalysis1->GetKStarHeavyCf()->GetHeavyCfClone();

  Analysis* tAnalysis2 = new Analysis(tFileLocationBase,tAnType,tCentType,kTrainSys,2,tDirNameModifier2);
  tAnalysis2->BuildKStarHeavyCf();
  TH1* tHist2 = tAnalysis2->GetKStarHeavyCf()->GetHeavyCfClone();

  TH1* tDiffHist = SystematicAnalysis::GetDiffHist(tHist1,tHist2);
  TF1* tFitFull = SystematicAnalysis::FitDiffHist(tDiffHist,SystematicAnalysis::kExpDecay,false);
    tFitFull->SetLineColor(2);
  TF1* tFitSimple = SystematicAnalysis::FitDiffHist(tDiffHist,SystematicAnalysis::kExpDecay,true);
    tFitSimple->SetLineColor(4);

  double tSigma = 2;

  cout << "tFitFull" << endl;
  bool tIsSignificantFull;
  if(TMath::Abs(tFitFull->GetParameter(0)/tFitFull->GetParError(0)) > tSigma) tIsSignificantFull = true;
  else tIsSignificantFull = false;
  for(int iPar=0; iPar<tFitFull->GetNpar(); iPar++)
  {
    cout << std::scientific << "par[" << iPar << "]: Value = " << tFitFull->GetParameter(iPar) << "\t Error = " << tFitFull->GetParError(iPar) << endl;
  }
  cout << "Is Signficant? " << tIsSignificantFull << endl << endl;

  cout << "tFitSimple" << endl;
  bool tIsSignificantSimple;
  if(TMath::Abs(tFitSimple->GetParameter(0)/tFitSimple->GetParError(0)) > tSigma) tIsSignificantSimple = true;
  for(int iPar=0; iPar<tFitSimple->GetNpar(); iPar++)
  {
    cout << std::scientific << "par[" << iPar << "]: Value = " << tFitSimple->GetParameter(iPar) << "\t Error = " << tFitSimple->GetParError(iPar) << endl;
  }
  cout << "Is Signficant? " << tIsSignificantSimple << endl << endl;


  TCanvas* tCan = new TCanvas("tCan","tCan");
  tCan->cd();


  tDiffHist->Draw();
  tFitFull->Draw("same");
  tFitSimple->Draw("same");

cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
