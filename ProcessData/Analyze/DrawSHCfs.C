#include <iostream>
#include <iomanip>

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

#include "CorrFctnDirectYlmTherm.h"



//_________________________________________________________________________________________
void DrawSHCfComponent(TPad* aPad, Analysis* aAnaly, YlmComponent aComponent, int al, int am, int aRebin/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/)
{
  aPad->cd();

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  double tXLow=0., tXHigh=0.3;
  double tYLow, tYHigh;
  if(al==0 && am==0)
  {
    tYLow = 0.86;
    tYHigh = 1.07;
  }
  else
  {
    tYLow = -0.03;
    tYHigh = 0.03;
  }

  //--------------------------------------------------------------

  vector<TString> tReImVec{"#Rgothic", "#Jgothic"};
  int tColor;
  if(aAnaly->GetAnalysisType()==kLamK0 || aAnaly->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aAnaly->GetAnalysisType()==kLamKchP || aAnaly->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aAnaly->GetAnalysisType()==kLamKchM || aAnaly->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  //--------------------------------------------------------------
//  TH1D* tTestCfn = tAnaly0010->GetYlmCfnHist(kYlmReal, 1, 1); 

  TH1D* tSHCf = (TH1D*)aAnaly->GetYlmCfnHist(aComponent, al, am, aRebin);
  tSHCf->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf->SetMarkerStyle(20);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(tColor);
  tSHCf->SetLineColor(tColor);

  tSHCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));

  tSHCf->Draw();

  //--------------------------------------------------------------

  TPaveText* tText = new TPaveText(0.60, 0.70, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(tColor);
    tText->AddText(cAnalysisRootTags[aAnaly->GetAnalysisType()]);
    tText->AddText(TString::Format("%sC_{%d%d} (%s)", tReImVec[(int)aComponent].Data(), al, am, cPrettyCentralityTags[aAnaly->GetCentralityType()]));
  tText->Draw();

}

//________________________________________________________________________________________________________________
CorrFctnDirectYlmTherm* GetYlmCfTherm(TString aFileLocation, int aImpactParam, AnalysisType aAnType, int aMaxl, int aNbins, double aKStarMin, double aKStarMax, int aRebin, double aNumScale=0.)
{
  CorrFctnDirectYlmTherm* tCfYlmTherm = new CorrFctnDirectYlmTherm(aFileLocation, aImpactParam, aAnType, aMaxl, aNbins, aKStarMin, aKStarMax, aRebin, aNumScale);
  return tCfYlmTherm;
}

//_________________________________________________________________________________________
void DrawSHCfThermComponent(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmTherm, YlmComponent aComponent, int al, int am/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/, int aMarkerStyle=20, int aColor=1)
{
  aPad->cd();

  TH1D* tSHCf = (TH1D*)aCfYlmTherm->GetYlmHist(aComponent, kYlmCf, al, am);

  tSHCf->SetMarkerStyle(aMarkerStyle);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(aColor);
  tSHCf->SetLineColor(aColor);

  tSHCf->Draw("same");
}



//_________________________________________________________________________________________
//*****************************************************************************************
//_________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  //--Rarely change---------------------
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;
  //------------------------------------
  TString tResultsDate = "20181205";
  AnalysisType tAnType = kLamKchP;

  bool bDrawThermCfs = false;
  bool bSaveFigures = false;

  int tl = 1;
  int tm = 1;
  YlmComponent tComponent = kYlmReal;

  double tMinNorm=0.32;
  double tMaxNorm=0.40;
  int tRebin=2;

//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

//  TString tSaveDirectoryBase = TString::Format("/home/jesse/Analysis/Presentations/AliFemto/20180627/Figures/Fits/%s/", cAnalysisBaseTags[tAnType]);
  TString tSaveDirectoryBase = tDirectoryBase;

//-----------------------------------------------------------------------------

  Analysis* tAnaly0010 = new Analysis(tFileLocationBase, tAnType, k0010, tAnRunType, 2, "", false);
  Analysis* tAnaly1030 = new Analysis(tFileLocationBase, tAnType, k1030, tAnRunType, 2, "", false);
  Analysis* tAnaly3050 = new Analysis(tFileLocationBase, tAnType, k3050, tAnRunType, 2, "", false);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->Divide(2, 3);

  DrawSHCfComponent((TPad*)tCan->cd(1), tAnaly0010, tComponent, 0, 0, tRebin);
  DrawSHCfComponent((TPad*)tCan->cd(2), tAnaly0010, tComponent, 1, 1, tRebin);

  DrawSHCfComponent((TPad*)tCan->cd(3), tAnaly1030, tComponent, 0, 0, tRebin);
  DrawSHCfComponent((TPad*)tCan->cd(4), tAnaly1030, tComponent, 1, 1, tRebin);

  DrawSHCfComponent((TPad*)tCan->cd(5), tAnaly3050, tComponent, 0, 0, tRebin);
  DrawSHCfComponent((TPad*)tCan->cd(6), tAnaly3050, tComponent, 1, 1, tRebin);

  if(bDrawThermCfs)
  {
    int tImpactParam = 2;
    TString aCfDescriptor = "Full";

    TString tFileNameBaseTherm = "CorrelationFunctions_DrawRStarFromGaussian_BuildCfYlm_BuildAliFemtoCfYlm_PairOnly_cLamcKchMuOut3_cLamK0MuOut3_KchPKchPR538";
    TString tFileNameModifierTherm = "";

    TString tFileNameTherm = TString::Format("%s%s.root", tFileNameBaseTherm.Data(), tFileNameModifierTherm.Data());

    TString tFileDirTherm = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
    TString tFileLocationTherm = TString::Format("%s%s", tFileDirTherm.Data(), tFileNameTherm.Data());

    CorrFctnDirectYlmTherm* tCfYlmTherm = GetYlmCfTherm(tFileLocationTherm, tImpactParam, tAnType, 2, 300, 0., 3., tRebin);

    DrawSHCfThermComponent((TPad*)tCan->cd(1), tCfYlmTherm, tComponent, 0, 0, 29, kOrange);
    DrawSHCfThermComponent((TPad*)tCan->cd(2), tCfYlmTherm, tComponent, 1, 1, 29, kOrange);
  }

/*
  Analysis* tAnaly0010 = new Analysis(tFileLocationBase, tAnType, k0010, tAnRunType, 2, "", false);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();


  TH1D* tTestCfn = tAnaly0010->GetYlmCfnHist(kYlmReal, 1, 1); 

  tTestCfn->Draw();
*/
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
