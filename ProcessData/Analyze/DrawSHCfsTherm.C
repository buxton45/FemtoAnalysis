#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TH3.h"

#include "PartialAnalysis.h"
#include "Types.h"
#include "CorrFctnDirectYlmTherm.h"

#include "ThermCf.h"

using std::cout;
using std::endl;
using std::vector;


//________________________________________________________________________________________________________________
CorrFctnDirectYlmTherm* GetYlmCfTherm(TString aFileLocation, int aImpactParam, AnalysisType aAnType, int aMaxl, int aNbins, double aKStarMin, double aKStarMax, int aRebin, double aNumScale=0.)
{
  CorrFctnDirectYlmTherm* tCfYlmTherm = new CorrFctnDirectYlmTherm(aFileLocation, aImpactParam, aAnType, aMaxl, aNbins, aKStarMin, aKStarMax, aRebin, aNumScale);
  return tCfYlmTherm;
}

//_________________________________________________________________________________________
void DrawSHCfThermComponent(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmTherm, YlmComponent aComponent, int al, int am/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/)
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
  if(aCfYlmTherm->GetAnalysisType()==kLamK0 || aCfYlmTherm->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aCfYlmTherm->GetAnalysisType()==kLamKchP || aCfYlmTherm->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aCfYlmTherm->GetAnalysisType()==kLamKchM || aCfYlmTherm->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  //--------------------------------------------------------------
//  TH1D* tTestCfn = tAnaly0010->GetYlmCfnHist(kYlmReal, 1, 1); 

  TH1D* tSHCf = (TH1D*)aCfYlmTherm->GetYlmHist(aComponent, kYlmCf, al, am);
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
    tText->AddText(cAnalysisRootTags[aCfYlmTherm->GetAnalysisType()]);
    tText->AddText(TString::Format("%sC_{%d%d} (b%d)", tReImVec[(int)aComponent].Data(), al, am, aCfYlmTherm->GetImpactParam()));
  tText->Draw();

}

//________________________________________________________________________________________________________________
//ThermCf already knows the default directory, so it only needs the name of the file, not the complete path
void Draw1DCf(TPad* aPad, TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam=2, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, double aAdditionalScale=1., int aMarkerStyle=20, int aColor=1, bool aUseStavCf=false)
{
  TH1* aThermCf = ThermCf::GetThermCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, aEventsType, aRebin, aMinNorm, aMaxNorm, aMarkerStyle, aColor, aUseStavCf);
  double tThermCfScale = aThermCf->Integral(aMinNorm, aMaxNorm);
  aThermCf->Scale(aAdditionalScale/tThermCfScale);

  aPad->cd();
  aThermCf->Draw("same");
}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  AnalysisType tAnType = kLamKchP;

  bool aDrawNormalCfOnC00 = true;
  bool bCombineConjugates = false;

  bool bSaveFigures = false;
  TString tSaveDir = "/home/jesse/Analysis/Presentations/GroupMeetings/20181204/Figures/";

  int tRebin=2;
  double tMinNorm = /*0.80*//*0.80*/0.32;
  double tMaxNorm = /*0.99*//*0.99*/0.40;

  int tImpactParam = 2;

  TString aCfDescriptor = "Full";
//  TString aCfDescriptor = "PrimaryOnly";

  int tl = 1;
  int tm = 1;
  YlmComponent tComponent = kYlmReal;

  TString tFileNameBase = "CorrelationFunctions_DrawRStarFromGaussian_BuildCfYlm_BuildAliFemtoCfYlm_PairOnly_cLamcKchMuOut3_cLamK0MuOut3_KchPKchPR538";

  TString tFileNameModifier = "";
//  TString tFileNameModifier = "_WeightParentsInteraction";
//  TString tFileNameModifier = "_WeightParentsInteraction_OnlyWeightLongDecayParents";
//  TString tFileNameModifier = "_WeightParentsInteraction_NoCharged";

  //--------------------------------------------

  TString tFileName = TString::Format("%s%s.root", tFileNameBase.Data(), tFileNameModifier.Data());

  TString tFileDir = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
  TString tFileLocation = TString::Format("%s%s", tFileDir.Data(), tFileName.Data());


  //--------------------------------------------
  CorrFctnDirectYlmTherm* tCfYlmTherm = GetYlmCfTherm(tFileLocation, tImpactParam, tAnType, 2, 300, 0., 3., tRebin);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->Divide(2,1);
  DrawSHCfThermComponent((TPad*)tCan->cd(1), tCfYlmTherm, tComponent, 0, 0);
  if(aDrawNormalCfOnC00) 
  {
    //TODO Not sure how to set normalization range for CfYlm's, so I will simply scale the regular Cf to the CfYlm
    TH1D* tTempSHCf00 = (TH1D*)tCfYlmTherm->GetYlmHist(tComponent, kYlmCf, 0, 0);
    double tTempNorm = tTempSHCf00->Integral(tMinNorm, tMaxNorm);

    Draw1DCf((TPad*)tCan->cd(1), tFileName, aCfDescriptor, tAnType, tImpactParam, bCombineConjugates, kMe, tRebin, tMinNorm, tMaxNorm, tTempNorm, 30, kBlack);    
  }

  DrawSHCfThermComponent((TPad*)tCan->cd(2), tCfYlmTherm, tComponent, 1, 1);

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
