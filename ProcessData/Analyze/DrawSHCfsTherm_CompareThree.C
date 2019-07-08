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
#include "TLatex.h"

#include "PartialAnalysis.h"
#include "Types.h"
#include "CorrFctnDirectYlmTherm.h"

#include "ThermCf.h"

using std::cout;
using std::endl;
using std::vector;


//________________________________________________________________________________________________________________
TH1* CombineTwoHists(TH1* aHist1, TH1* aHist2, double aNorm1, double aNorm2)
{
  TString aReturnName = TString::Format("%s_and_%s", aHist1->GetName(), aHist2->GetName());

  if(!aHist1->GetSumw2N()) aHist1->Sumw2();
  if(!aHist2->GetSumw2N()) aHist2->Sumw2();

  TH1* tReturnHist = (TH1*)aHist1->Clone(aReturnName);
    tReturnHist->Scale(aNorm1);
  tReturnHist->Add(aHist2, aNorm2);
  tReturnHist->Scale(1./(aNorm1+aNorm2));

  return tReturnHist;
}

//________________________________________________________________________________________________________________
CorrFctnDirectYlmTherm* GetYlmCfTherm(TString aFileLocation, int aImpactParam, AnalysisType aAnType, int aMaxl, int aNbins, double aKStarMin, double aKStarMax, int aRebin, double aNumScale=0.)
{
  CorrFctnDirectYlmTherm* tCfYlmTherm = new CorrFctnDirectYlmTherm(aFileLocation, aImpactParam, aAnType, aMaxl, aNbins, aKStarMin, aKStarMax, aRebin, aNumScale);
  return tCfYlmTherm;
}

//_________________________________________________________________________________________
void DrawSHCfThermComponent(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmTherm, YlmComponent aComponent, int al, int am, int aMarkerStyle=20, int aColor=1)
{
  aPad->cd();
  aPad->SetLeftMargin(0.15);
  aPad->SetRightMargin(0.025);

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
  //--------------------------------------------------------------
//  TH1D* tTestCfn = tAnaly0010->GetYlmCfnHist(kYlmReal, 1, 1); 

  TH1D* tSHCf = (TH1D*)aCfYlmTherm->GetYlmHist(aComponent, kYlmCf, al, am);
  tSHCf->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf->SetMarkerStyle(aMarkerStyle);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(aColor);
  tSHCf->SetLineColor(aColor);

  tSHCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));

  tSHCf->Draw("same");

  //--------------------------------------------------------------
/*
  TPaveText* tText = new TPaveText(0.60, 0.70, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(aColor);
    tText->AddText(cAnalysisRootTags[aCfYlmTherm->GetAnalysisType()]);
    tText->AddText(TString::Format("%s#it{C}_{%d%d} (b%d)", tReImVec[(int)aComponent].Data(), al, am, aCfYlmTherm->GetImpactParam()));
  tText->Draw();
*/
}

//_________________________________________________________________________________________
void DrawThreeSHCfThermComponents(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmTherm1, CorrFctnDirectYlmTherm* aCfYlmTherm2, CorrFctnDirectYlmTherm* aCfYlmTherm3, YlmComponent aComponent, int al, int am)
{
  aPad->cd();
  aPad->SetLeftMargin(0.15);
  aPad->SetRightMargin(0.025);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  double tXLow=0., tXHigh=0.3;
  double tYLow, tYHigh;
  if(al==0 && am==0)
  {
    tYLow = 0.86;
    tYHigh = 1.07;

    if     (aCfYlmTherm1->GetAnalysisType()==kLamKchP || aCfYlmTherm1->GetAnalysisType()==kALamKchM)
    {
      tYLow = 0.79;
      tYHigh = 1.03;
    }
    else if(aCfYlmTherm1->GetAnalysisType()==kLamKchM || aCfYlmTherm1->GetAnalysisType()==kALamKchP)
    {
      tYLow = 0.97;
      tYHigh = 1.05;
    }
    else if(aCfYlmTherm1->GetAnalysisType()==kLamK0 || aCfYlmTherm1->GetAnalysisType()==kALamK0)
    {
      tYLow = 0.85;
      tYHigh = 1.03;
    }
  }
  else
  {
    tYLow = -0.03;
    tYHigh = 0.03;
  }

  //--------------------------------------------------------------
  vector<TString> tReImVec{"#Rgothic", "#Jgothic"};
  //--------------------------------------------------------------
  double tMarkerSize = 0.75;

  int tMarkerStyle1 = 20;
  int tColor1 = kBlack;

  int tMarkerStyle2 = 25;
  int tColor2 = kBlue;

  int tMarkerStyle3 = 22;
  int tColor3 = kRed;


  //--------------------------------------------------------------

  TH1D* tSHCf1 = (TH1D*)aCfYlmTherm1->GetYlmHist(aComponent, kYlmCf, al, am);
  tSHCf1->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf1->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf1->SetMarkerStyle(tMarkerStyle1);
  tSHCf1->SetMarkerSize(tMarkerSize);
  tSHCf1->SetMarkerColor(tColor1);
  tSHCf1->SetLineColor(tColor1);

  tSHCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf1->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));

  //-----

  TH1D* tSHCf2 = (TH1D*)aCfYlmTherm2->GetYlmHist(aComponent, kYlmCf, al, am);

  tSHCf2->SetMarkerStyle(tMarkerStyle2);
  tSHCf2->SetMarkerSize(tMarkerSize);
  tSHCf2->SetMarkerColor(tColor2);
  tSHCf2->SetLineColor(tColor2);

  //-----

  TH1D* tSHCf3 = (TH1D*)aCfYlmTherm3->GetYlmHist(aComponent, kYlmCf, al, am);

  tSHCf3->SetMarkerStyle(tMarkerStyle3);
  tSHCf3->SetMarkerSize(tMarkerSize);
  tSHCf3->SetMarkerColor(tColor3);
  tSHCf3->SetLineColor(tColor3);

  //--------------------------------------------------------------
  tSHCf1->Draw();
  tSHCf2->Draw("same");
  tSHCf3->Draw("same");

  //--------------------------------------------------------------

  TPaveText* tText = new TPaveText(0.60, 0.70, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(kBlack);
    tText->AddText(cAnalysisRootTags[aCfYlmTherm1->GetAnalysisType()]);
    tText->AddText(TString::Format("%s#it{C}_{%d%d} (#it{b} = %d fm)", tReImVec[(int)aComponent].Data(), al, am, aCfYlmTherm1->GetImpactParam()));
  tText->Draw();

}

//_________________________________________________________________________________________
void DrawThreeSHCfThermComponents_CombConj(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmTherm1a,CorrFctnDirectYlmTherm* aCfYlmTherm1b, CorrFctnDirectYlmTherm* aCfYlmTherm2a, CorrFctnDirectYlmTherm* aCfYlmTherm2b, CorrFctnDirectYlmTherm* aCfYlmTherm3a, CorrFctnDirectYlmTherm* aCfYlmTherm3b, YlmComponent aComponent, int al, int am)
{
  aPad->cd();
  aPad->SetLeftMargin(0.15);
  aPad->SetRightMargin(0.025);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  double tXLow=0., tXHigh=0.3;
  double tYLow, tYHigh;
  if(al==0 && am==0)
  {
    tYLow = 0.79;
    tYHigh = 1.07;
/*
    if     (aCfYlmTherm1a->GetAnalysisType()==kLamKchP)
    {
      tYLow = 0.79;
      tYHigh = 1.05;
    }
    else if(aCfYlmTherm1a->GetAnalysisType()==kLamKchM)
    {
      tYLow = 0.97;
      tYHigh = 1.05;
    }
    else if(aCfYlmTherm1a->GetAnalysisType()==kLamK0)
    {
      tYLow = 0.90;
      tYHigh = 1.05;
    }
*/
  }
  else
  {
    tYLow = -0.03;
    tYHigh = 0.01;
  }

  //--------------------------------------------------------------
  vector<TString> tReImVec{"#Rgothic", "#Jgothic"};
  //--------------------------------------------------------------
  double tMarkerSize = 0.75;

  int tMarkerStyle1 = 20;
  int tColor1 = kBlack;

  int tMarkerStyle2 = 25;
  int tColor2 = kBlue;

  int tMarkerStyle3 = 22;
  int tColor3 = kRed;


  //--------------------------------------------------------------

  TH1D* tSHCf1a = (TH1D*)aCfYlmTherm1a->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf1b = (TH1D*)aCfYlmTherm1b->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf1 = (TH1D*)CombineTwoHists(tSHCf1a, tSHCf1b, aCfYlmTherm1a->GetNumScale(), aCfYlmTherm1b->GetNumScale());

  tSHCf1->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf1->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf1->SetMarkerStyle(tMarkerStyle1);
  tSHCf1->SetMarkerSize(tMarkerSize);
  tSHCf1->SetMarkerColor(tColor1);
  tSHCf1->SetLineColor(tColor1);

  tSHCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf1->GetXaxis()->SetTitleSize(0.070);
  tSHCf1->GetXaxis()->SetTitleOffset(0.9);
  tSHCf1->GetXaxis()->SetLabelSize(0.045);

  tSHCf1->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));
  tSHCf1->GetYaxis()->SetTitleSize(0.070);
  tSHCf1->GetYaxis()->SetTitleOffset(1.0);
  tSHCf1->GetYaxis()->SetLabelSize(0.045);

  //-----

  TH1D* tSHCf2a = (TH1D*)aCfYlmTherm2a->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf2b = (TH1D*)aCfYlmTherm2b->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf2 = (TH1D*)CombineTwoHists(tSHCf2a, tSHCf2b, aCfYlmTherm2a->GetNumScale(), aCfYlmTherm2b->GetNumScale());

  tSHCf2->SetMarkerStyle(tMarkerStyle2);
  tSHCf2->SetMarkerSize(tMarkerSize);
  tSHCf2->SetMarkerColor(tColor2);
  tSHCf2->SetLineColor(tColor2);

  //-----

  TH1D* tSHCf3a = (TH1D*)aCfYlmTherm3a->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf3b = (TH1D*)aCfYlmTherm3b->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf3 = (TH1D*)CombineTwoHists(tSHCf3a, tSHCf3b, aCfYlmTherm3a->GetNumScale(), aCfYlmTherm3b->GetNumScale());

  tSHCf3->SetMarkerStyle(tMarkerStyle3);
  tSHCf3->SetMarkerSize(tMarkerSize);
  tSHCf3->SetMarkerColor(tColor3);
  tSHCf3->SetLineColor(tColor3);

  //--------------------------------------------------------------
  tSHCf1->Draw();
  tSHCf2->Draw("same");
  tSHCf3->Draw("same");

  //--------------------------------------------------------------
/*
  TPaveText* tText = new TPaveText(0.60, 0.70, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(kBlack);
    tText->AddText(cAnalysisRootTags[aCfYlmTherm1a->GetAnalysisType()]);
    tText->AddText(TString::Format("%s#it{C}_{%d%d} (#it{b} = %d fm)", tReImVec[(int)aComponent].Data(), al, am, aCfYlmTherm1a->GetImpactParam()));
  tText->Draw();
*/

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(11);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.070);

  double tText1X = 0.0175;
  double tText2X = 0.21;

  double tTextY = 1.035;

  if(al==1 && am==1) tTextY = 0.0051;

  tTex->DrawLatex(tText1X, tTextY, TString::Format("%s#it{C}_{%d%d} (#it{b} = %d fm)", tReImVec[(int)aComponent].Data(), al, am, aCfYlmTherm1a->GetImpactParam()));
  tTex->DrawLatex(tText2X, tTextY, TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[aCfYlmTherm1a->GetAnalysisType()], cAnalysisRootTags[aCfYlmTherm1b->GetAnalysisType()]));

  //--------------------------------------------------------------

  if(al==0 && am==0)
  {
    TLegend* tLeg = new TLegend(0.50, 0.25, 0.95, 0.50);
      tLeg->SetFillColor(0);
      tLeg->SetBorderSize(0);
      tLeg->SetTextAlign(12);
    tLeg->SetHeader("#it{R}_{O} = #it{R}_{S} = #it{R}_{L} = 5 fm");
    tLeg->AddEntry(tSHCf1, "#mu_{O} = 1 fm", "p");
    tLeg->AddEntry(tSHCf2, "#mu_{O} = 3 fm", "p");
    tLeg->AddEntry(tSHCf3, "#mu_{O} = 6 fm", "p");

    tLeg->Draw();
  }
}

//________________________________________________________________________________________________________________
//ThermCf already knows the default directory, so it only needs the name of the file, not the complete path
double GetNorm(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam=2, bool aCombineConj=false, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40)
{
  TH1* aThermCf = ThermCf::GetThermCf(aFileName, aCfDescriptor, aAnType, aImpactParam, aCombineConj, aEventsType, aRebin, aMinNorm, aMaxNorm, 20, kBlack, false);
  double tThermCfScale = aThermCf->Integral(aThermCf->FindBin(aMinNorm), aThermCf->FindBin(aMaxNorm));

  return tThermCfScale;
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
  AnalysisType tConjType = static_cast<AnalysisType>(tAnType+1);

  bool bCombineConjugates = true;

  bool bSaveFigures = false;
  TString tSaveDir = "/home/jesse/Analysis/Presentations/GroupMeetings/20190108/Figures/";

  int tRebin=2;
  double tMinNorm = /*0.80*//*0.80*/0.32;
  double tMaxNorm = /*0.99*//*0.99*/0.40;

  int tImpactParam = 2;

  TString aCfDescriptor = "Full";
//  TString aCfDescriptor = "PrimaryOnly";

  int tl = 1;
  int tm = 1;
  YlmComponent tComponent = kYlmReal;

  TString tFileNameBase1 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_1.0_0.0_0.0_BuildCfYlm";
  TString tFileNameBase2 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_3.0_0.0_0.0_BuildCfYlm";
  TString tFileNameBase3 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_6.0_0.0_0.0_BuildCfYlm";

  TString tFileNameModifier = "";

  //--------------------------------------------

  TString tFileDir = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);

  TString tFileName1 = TString::Format("%s%s.root", tFileNameBase1.Data(), tFileNameModifier.Data());
  TString tFileLocation1 = TString::Format("%s%s", tFileDir.Data(), tFileName1.Data());

  TString tFileName2 = TString::Format("%s%s.root", tFileNameBase2.Data(), tFileNameModifier.Data());
  TString tFileLocation2 = TString::Format("%s%s", tFileDir.Data(), tFileName2.Data());

  TString tFileName3 = TString::Format("%s%s.root", tFileNameBase3.Data(), tFileNameModifier.Data());
  TString tFileLocation3 = TString::Format("%s%s", tFileDir.Data(), tFileName3.Data());

  if(tAnType==kKchPKchP || tAnType==kK0K0 || tAnType==kLamLam) bCombineConjugates = false;
  if(tFileNameBase1.Contains("PairOnly") || tFileNameBase2.Contains("PairOnly") || tFileNameBase3.Contains("PairOnly")) bCombineConjugates = false;

  //--------------------------------------------
  CorrFctnDirectYlmTherm* tCfYlmTherm1a = GetYlmCfTherm(tFileLocation1, tImpactParam, tAnType, 2, 300, 0., 3., tRebin, 
                                                        GetNorm(tFileName1, aCfDescriptor, tAnType, tImpactParam, false, kMe, tRebin, tMinNorm, tMaxNorm));
  CorrFctnDirectYlmTherm* tCfYlmTherm1b = GetYlmCfTherm(tFileLocation1, tImpactParam, tConjType, 2, 300, 0., 3., tRebin, 
                                                        GetNorm(tFileName1, aCfDescriptor, tConjType, tImpactParam, false, kMe, tRebin, tMinNorm, tMaxNorm));


  CorrFctnDirectYlmTherm* tCfYlmTherm2a = GetYlmCfTherm(tFileLocation2, tImpactParam, tAnType, 2, 300, 0., 3., tRebin,
                                                        GetNorm(tFileName2, aCfDescriptor, tAnType, tImpactParam, false, kMe, tRebin, tMinNorm, tMaxNorm));
  CorrFctnDirectYlmTherm* tCfYlmTherm2b = GetYlmCfTherm(tFileLocation2, tImpactParam, tConjType, 2, 300, 0., 3., tRebin,
                                                        GetNorm(tFileName2, aCfDescriptor, tConjType, tImpactParam, false, kMe, tRebin, tMinNorm, tMaxNorm));


  CorrFctnDirectYlmTherm* tCfYlmTherm3a = GetYlmCfTherm(tFileLocation3, tImpactParam, tAnType, 2, 300, 0., 3., tRebin,
                                                        GetNorm(tFileName3, aCfDescriptor, tAnType, tImpactParam, false, kMe, tRebin, tMinNorm, tMaxNorm));
  CorrFctnDirectYlmTherm* tCfYlmTherm3b = GetYlmCfTherm(tFileLocation3, tImpactParam, tConjType, 2, 300, 0., 3., tRebin,
                                                        GetNorm(tFileName3, aCfDescriptor, tConjType, tImpactParam, false, kMe, tRebin, tMinNorm, tMaxNorm));

  //--------------------------------------------
  TString tCanC00C11Name;
  if(!bCombineConjugates) tCanC00C11Name = TString::Format("CanCompThreeThermCfYlmReC00C11_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
  else tCanC00C11Name = TString::Format("CanCompThreeThermCfYlmReC00C11_%s_%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType], cAnalysisBaseTags[tConjType]);
  TCanvas* tCanC00C11 = new TCanvas(tCanC00C11Name, tCanC00C11Name, 1400, 500);
  tCanC00C11->Divide(2,1);
    tCanC00C11->cd(1)->SetTopMargin(0.02);
    tCanC00C11->cd(2)->SetTopMargin(0.02);
    tCanC00C11->cd(1)->SetBottomMargin(0.125);
    tCanC00C11->cd(2)->SetBottomMargin(0.125);

  if(!bCombineConjugates)
  {
    DrawThreeSHCfThermComponents((TPad*)tCanC00C11->cd(1), tCfYlmTherm1a, tCfYlmTherm2a, tCfYlmTherm3a, tComponent, 0, 0);
    DrawThreeSHCfThermComponents((TPad*)tCanC00C11->cd(2), tCfYlmTherm1a, tCfYlmTherm2a, tCfYlmTherm3a, tComponent, 1, 1);
  }
  else
  {
    DrawThreeSHCfThermComponents_CombConj((TPad*)tCanC00C11->cd(1), tCfYlmTherm1a, tCfYlmTherm1b, tCfYlmTherm2a, tCfYlmTherm2b, tCfYlmTherm3a, tCfYlmTherm3b, tComponent, 0, 0);
    DrawThreeSHCfThermComponents_CombConj((TPad*)tCanC00C11->cd(2), tCfYlmTherm1a, tCfYlmTherm1b, tCfYlmTherm2a, tCfYlmTherm2b, tCfYlmTherm3a, tCfYlmTherm3b, tComponent, 1, 1);
  }

  if(bSaveFigures) tCanC00C11->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tCanC00C11Name.Data()));
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
