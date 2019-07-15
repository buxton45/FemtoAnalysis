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

#include "Analysis.h"
class Analysis;

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
void DrawFourSHCfThermComponents(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmTherm0, CorrFctnDirectYlmTherm* aCfYlmTherm1, CorrFctnDirectYlmTherm* aCfYlmTherm2, CorrFctnDirectYlmTherm* aCfYlmTherm3, YlmComponent aComponent, int al, int am)
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

  int tMarkerStyle0 = 20;
  int tColor0 = kBlack;

  int tMarkerStyle1 = 25;
  int tColor1 = kBlue;

  int tMarkerStyle2 = 22;
  int tColor2 = kGreen;

  int tMarkerStyle3 = 28;
  int tColor3 = kRed;


  //--------------------------------------------------------------

  TH1D* tSHCf0 = (TH1D*)aCfYlmTherm0->GetYlmHist(aComponent, kYlmCf, al, am);
  tSHCf0->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf0->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf0->SetMarkerStyle(tMarkerStyle0);
  tSHCf0->SetMarkerSize(tMarkerSize);
  tSHCf0->SetMarkerColor(tColor0);
  tSHCf0->SetLineColor(tColor0);

  tSHCf0->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf0->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));

  //-----

  TH1D* tSHCf1 = (TH1D*)aCfYlmTherm1->GetYlmHist(aComponent, kYlmCf, al, am);

  tSHCf1->SetMarkerStyle(tMarkerStyle1);
  tSHCf1->SetMarkerSize(tMarkerSize);
  tSHCf1->SetMarkerColor(tColor1);
  tSHCf1->SetLineColor(tColor1);

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
  tSHCf0->Draw();
  tSHCf1->Draw("same");
  tSHCf2->Draw("same");
  tSHCf3->Draw("same");

  //--------------------------------------------------------------

  TPaveText* tText = new TPaveText(0.60, 0.70, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(kBlack);
    tText->AddText(cAnalysisRootTags[aCfYlmTherm0->GetAnalysisType()]);
    tText->AddText(TString::Format("%s#it{C}_{%d%d} (#it{b} = %d fm)", tReImVec[(int)aComponent].Data(), al, am, aCfYlmTherm0->GetImpactParam()));
  tText->Draw();

}

//_________________________________________________________________________________________
void DrawFourSHCfThermComponents_CombConj(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmTherm0a,CorrFctnDirectYlmTherm* aCfYlmTherm0b, CorrFctnDirectYlmTherm* aCfYlmTherm1a,CorrFctnDirectYlmTherm* aCfYlmTherm1b, CorrFctnDirectYlmTherm* aCfYlmTherm2a, CorrFctnDirectYlmTherm* aCfYlmTherm2b, CorrFctnDirectYlmTherm* aCfYlmTherm3a, CorrFctnDirectYlmTherm* aCfYlmTherm3b, YlmComponent aComponent, int al, int am)
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

  int tMarkerStyle0 = 20;
  int tColor0 = kBlack;

  int tMarkerStyle1 = 25;
  int tColor1 = kBlue;

  int tMarkerStyle2 = 22;
  int tColor2 = kGreen;

  int tMarkerStyle3 = 28;
  int tColor3 = kRed;


  //--------------------------------------------------------------

  TH1D* tSHCf0a = (TH1D*)aCfYlmTherm0a->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf0b = (TH1D*)aCfYlmTherm0b->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf0 = (TH1D*)CombineTwoHists(tSHCf0a, tSHCf0b, aCfYlmTherm0a->GetNumScale(), aCfYlmTherm0b->GetNumScale());

  tSHCf0->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf0->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf0->SetMarkerStyle(tMarkerStyle0);
  tSHCf0->SetMarkerSize(tMarkerSize);
  tSHCf0->SetMarkerColor(tColor0);
  tSHCf0->SetLineColor(tColor0);

  tSHCf0->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf0->GetXaxis()->SetTitleSize(0.070);
  tSHCf0->GetXaxis()->SetTitleOffset(0.9);
  tSHCf0->GetXaxis()->SetLabelSize(0.045);

  tSHCf0->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));
  tSHCf0->GetYaxis()->SetTitleSize(0.070);
  tSHCf0->GetYaxis()->SetTitleOffset(1.0);
  tSHCf0->GetYaxis()->SetLabelSize(0.045);

  //-----

  TH1D* tSHCf1a = (TH1D*)aCfYlmTherm1a->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf1b = (TH1D*)aCfYlmTherm1b->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCf1 = (TH1D*)CombineTwoHists(tSHCf1a, tSHCf1b, aCfYlmTherm1a->GetNumScale(), aCfYlmTherm1b->GetNumScale());

  tSHCf1->SetMarkerStyle(tMarkerStyle1);
  tSHCf1->SetMarkerSize(tMarkerSize);
  tSHCf1->SetMarkerColor(tColor1);
  tSHCf1->SetLineColor(tColor1);

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
  tSHCf0->Draw();
  tSHCf1->Draw("same");
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

  tTex->DrawLatex(tText1X, tTextY, TString::Format("%s#it{C}_{%d%d} (#it{b} = %d fm)", tReImVec[(int)aComponent].Data(), al, am, aCfYlmTherm0a->GetImpactParam()));
  tTex->DrawLatex(tText2X, tTextY, TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[aCfYlmTherm0a->GetAnalysisType()], cAnalysisRootTags[aCfYlmTherm0b->GetAnalysisType()]));

  //--------------------------------------------------------------

  if(al==0 && am==0)
  {
    TLegend* tLeg = new TLegend(0.50, 0.25, 0.95, 0.50);
      tLeg->SetFillColor(0);
      tLeg->SetBorderSize(0);
      tLeg->SetTextAlign(12);
    tLeg->SetHeader("#it{R}_{O} = #it{R}_{S} = #it{R}_{L} = 5 fm");
    tLeg->AddEntry(tSHCf0, "#mu_{O} = 0 fm", "p");
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
TH1D* GetDataYlmCfwSysErrors(TString aDate, AnalysisType aAnalysisType, CentralityType aCentralityType, YlmComponent aComponent, int al, int am)
{
  vector<TString> tReImVec{"Re", "Im"};

  TString tGenAnType;
  if(aAnalysisType==kLamKchP || aAnalysisType==kALamKchM || aAnalysisType==kLamKchM || aAnalysisType==kALamKchP) tGenAnType = TString("cLamcKch");
  else if(aAnalysisType==kLamK0 || aAnalysisType==kALamK0) tGenAnType = TString("cLamK0");
  else assert(0);
  TString tDirName = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/", tGenAnType.Data(), aDate.Data());

  TString tFileLocation = TString::Format("%sSystematicResults_%s%s_%s.root",tDirName.Data(),cAnalysisBaseTags[aAnalysisType],cCentralityTags[aCentralityType],aDate.Data());
  TString tHistName = TString::Format("%s%s_%sC%d%d_wSysErrors", cAnalysisRootTags[aAnalysisType], cCentralityTags[aCentralityType], tReImVec[aComponent].Data(), al, am);

  TFile tFile(tFileLocation);
  TH1D* tReturnHist = (TH1D*)tFile.Get(tHistName);
  assert(tReturnHist);
    tReturnHist->SetDirectory(0);

  return tReturnHist;
}


//_________________________________________________________________________________________
void DrawDataSHCfComponent(TPad* aPad, Analysis* aAnaly, YlmComponent aComponent, int al, int am, int aRebin, bool aDrawSysErrs=false/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/)
{
  aPad->cd();
  int tColor;
  if(aAnaly->GetAnalysisType()==kLamK0 || aAnaly->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aAnaly->GetAnalysisType()==kLamKchP || aAnaly->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aAnaly->GetAnalysisType()==kLamKchM || aAnaly->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  int tColorTransparent = TColor::GetColorTransparent(tColor,0.2);
  //--------------------------------------------------------------

  TH1D* tSHCf = (TH1D*)aAnaly->GetYlmCfnHist(aComponent, al, am, aRebin);

  tSHCf->SetMarkerStyle(20);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(tColor);
  tSHCf->SetLineColor(tColor);

  tSHCf->Draw("same");

  //--------------------------------------------------------------
  if(aDrawSysErrs)
  {
    TH1D* tSHCfwSysErrs = GetDataYlmCfwSysErrors(TString("20181205"), aAnaly->GetAnalysisType(), aAnaly->GetCentralityType(), aComponent, al, am);
    //TODO for some reason, 0-10% data does not match perfectly
    for(int i=1; i<tSHCfwSysErrs->GetNbinsX()+1; i++) tSHCfwSysErrs->SetBinContent(i, tSHCf->GetBinContent(i));

      tSHCfwSysErrs->SetFillColor(tColorTransparent);
      tSHCfwSysErrs->SetFillStyle(1000);
      tSHCfwSysErrs->SetLineColor(0);
      tSHCfwSysErrs->SetLineWidth(0);

      tSHCfwSysErrs->Draw("e2psame");
  }

}

//_________________________________________________________________________________________
void DrawDataSHCfComponent(TPad* aPad, Analysis* aAnaly, Analysis* aConjAnaly, YlmComponent aComponent, int al, int am, int aRebin, bool aDrawSysErrs=false/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/, bool aPrintAliceInfo=false)
{
  aPad->cd();
  int tColor;
  if(aAnaly->GetAnalysisType()==kLamK0 || aAnaly->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aAnaly->GetAnalysisType()==kLamKchP || aAnaly->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aAnaly->GetAnalysisType()==kLamKchM || aAnaly->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  int tColorTransparent = TColor::GetColorTransparent(tColor,0.2);
  //--------------------------------------------------------------
  vector<CorrFctnDirectYlmLite*> tYlmLiteCollAn = aAnaly->GetYlmCfHeavy(aRebin)->GetYlmCfLiteCollection();
  vector<CorrFctnDirectYlmLite*> tYlmLiteCollConjAn = aConjAnaly->GetYlmCfHeavy(aRebin)->GetYlmCfLiteCollection();

  double tOverallScale = 0.;
  TH1D* tSHCf = tYlmLiteCollAn[0]->GetYlmHist(aComponent, kYlmCf, al, am);
  tSHCf->Scale(tYlmLiteCollAn[0]->GetNumScale());
  tOverallScale += tYlmLiteCollAn[0]->GetNumScale();

  if(!tSHCf->GetSumw2N()) tSHCf->Sumw2();

  for(unsigned int i=1; i<tYlmLiteCollAn.size(); i++)
  {
    tSHCf->Add(tYlmLiteCollAn[i]->GetYlmHist(aComponent, kYlmCf, al, am), tYlmLiteCollAn[i]->GetNumScale());
    tOverallScale += tYlmLiteCollAn[i]->GetNumScale();
  }
  for(unsigned int i=0; i<tYlmLiteCollConjAn.size(); i++)
  {
    tSHCf->Add(tYlmLiteCollConjAn[i]->GetYlmHist(aComponent, kYlmCf, al, am), tYlmLiteCollConjAn[i]->GetNumScale());
    tOverallScale += tYlmLiteCollConjAn[i]->GetNumScale();
  }
  tSHCf->Scale(1./tOverallScale);

  tSHCf->SetMarkerStyle(20);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(tColor);
  tSHCf->SetLineColor(tColor);

  tSHCf->Draw("same");

  //--------------------------------------------------------------

  if(aPrintAliceInfo && al==0 && am==0)
  {
    TLatex *   tex = new TLatex(0.125,0.88,"ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    tex->SetTextFont(42);
    tex->SetTextSize(0.055);
    tex->SetLineWidth(2);
    tex->Draw();
  }

  //--------------------------------------------------------------
  if(aDrawSysErrs)
  {
    TH1D* tSHCfwSysErrs_An = GetDataYlmCfwSysErrors(TString("20181205"), aAnaly->GetAnalysisType(), aAnaly->GetCentralityType(), aComponent, al, am);
    TH1D* tSHCfwSysErrs_Conj = GetDataYlmCfwSysErrors(TString("20181205"), aConjAnaly->GetAnalysisType(), aConjAnaly->GetCentralityType(), aComponent, al, am);

    assert(tYlmLiteCollAn.size()==2);
    TH1D* tSHCfwSysErrs = (TH1D*)CombineTwoHists(tSHCfwSysErrs_An, tSHCfwSysErrs_Conj, tYlmLiteCollAn[0]->GetNumScale(), tYlmLiteCollAn[1]->GetNumScale());

    //TODO for some reason, 0-10% data does not match perfectly
    for(int i=1; i<tSHCfwSysErrs->GetNbinsX()+1; i++) tSHCfwSysErrs->SetBinContent(i, tSHCf->GetBinContent(i));

      tSHCfwSysErrs->SetFillColor(tColorTransparent);
      tSHCfwSysErrs->SetFillStyle(1000);
      tSHCfwSysErrs->SetLineColor(0);
      tSHCfwSysErrs->SetLineWidth(0);

      tSHCfwSysErrs->Draw("e2psame");
  }

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
  bool bIncludeData = true;

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

  TString tFileNameBase0 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_0.0_0.0_0.0_BuildCfYlm";
  TString tFileNameBase1 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_1.0_0.0_0.0_BuildCfYlm";
  TString tFileNameBase2 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_3.0_0.0_0.0_BuildCfYlm";
  TString tFileNameBase3 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_6.0_0.0_0.0_BuildCfYlm";

  TString tFileNameModifier = "";

  //--------------------------------------------

  TString tFileDir = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);

  TString tFileName0 = TString::Format("%s%s.root", tFileNameBase0.Data(), tFileNameModifier.Data());
  TString tFileLocation0 = TString::Format("%s%s", tFileDir.Data(), tFileName0.Data());

  TString tFileName1 = TString::Format("%s%s.root", tFileNameBase1.Data(), tFileNameModifier.Data());
  TString tFileLocation1 = TString::Format("%s%s", tFileDir.Data(), tFileName1.Data());

  TString tFileName2 = TString::Format("%s%s.root", tFileNameBase2.Data(), tFileNameModifier.Data());
  TString tFileLocation2 = TString::Format("%s%s", tFileDir.Data(), tFileName2.Data());

  TString tFileName3 = TString::Format("%s%s.root", tFileNameBase3.Data(), tFileNameModifier.Data());
  TString tFileLocation3 = TString::Format("%s%s", tFileDir.Data(), tFileName3.Data());

  if(tAnType==kKchPKchP || tAnType==kK0K0 || tAnType==kLamLam) bCombineConjugates = false;
  if(tFileNameBase0.Contains("PairOnly") || tFileNameBase1.Contains("PairOnly") || 
     tFileNameBase2.Contains("PairOnly") || tFileNameBase3.Contains("PairOnly")) bCombineConjugates = false;

  //--------------------------------------------
  CorrFctnDirectYlmTherm* tCfYlmTherm0a = GetYlmCfTherm(tFileLocation0, tImpactParam, tAnType, 2, 300, 0., 3., tRebin, 
                                                        GetNorm(tFileName0, aCfDescriptor, tAnType, tImpactParam, false, kMe, tRebin, tMinNorm, tMaxNorm));
  CorrFctnDirectYlmTherm* tCfYlmTherm0b = GetYlmCfTherm(tFileLocation0, tImpactParam, tConjType, 2, 300, 0., 3., tRebin, 
                                                        GetNorm(tFileName0, aCfDescriptor, tConjType, tImpactParam, false, kMe, tRebin, tMinNorm, tMaxNorm));


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
    DrawFourSHCfThermComponents((TPad*)tCanC00C11->cd(1), tCfYlmTherm0a, tCfYlmTherm1a, tCfYlmTherm2a, tCfYlmTherm3a, tComponent, 0, 0);
    DrawFourSHCfThermComponents((TPad*)tCanC00C11->cd(2), tCfYlmTherm0a, tCfYlmTherm1a, tCfYlmTherm2a, tCfYlmTherm3a, tComponent, 1, 1);
  }
  else
  {
    DrawFourSHCfThermComponents_CombConj((TPad*)tCanC00C11->cd(1), tCfYlmTherm0a, tCfYlmTherm0b, tCfYlmTherm1a, tCfYlmTherm1b, tCfYlmTherm2a, tCfYlmTherm2b, tCfYlmTherm3a, tCfYlmTherm3b, tComponent, 0, 0);
    DrawFourSHCfThermComponents_CombConj((TPad*)tCanC00C11->cd(2), tCfYlmTherm0a, tCfYlmTherm0b, tCfYlmTherm1a, tCfYlmTherm1b, tCfYlmTherm2a, tCfYlmTherm2b, tCfYlmTherm3a, tCfYlmTherm3b, tComponent, 1, 1);
  }

  if(bIncludeData)
  {
    TString tResultsDate = "20181205";

    TString tGeneralAnTypeName;
    if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
    else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
    else assert(0);

    TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
    TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

    Analysis* tAnaly0010 = new Analysis(tFileLocationBase, tAnType, k0010, kTrain, 2, "", false);
    Analysis* tConjAnaly0010 = new Analysis(tFileLocationBase, tConjType, k0010, kTrain, 2, "", false);

    if(!bCombineConjugates) 
    {
      DrawDataSHCfComponent((TPad*)tCanC00C11->cd(1), tAnaly0010, tComponent, 0, 0, tRebin, true);
      DrawDataSHCfComponent((TPad*)tCanC00C11->cd(2), tAnaly0010, tComponent, 1, 1, tRebin, true);
    }
    else
    {
      DrawDataSHCfComponent((TPad*)tCanC00C11->cd(1), tAnaly0010, tConjAnaly0010, tComponent, 0, 0, tRebin, true, true);
      DrawDataSHCfComponent((TPad*)tCanC00C11->cd(2), tAnaly0010, tConjAnaly0010, tComponent, 1, 1, tRebin, true, true);
    }
  }



  if(bSaveFigures) tCanC00C11->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tCanC00C11Name.Data()));
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
