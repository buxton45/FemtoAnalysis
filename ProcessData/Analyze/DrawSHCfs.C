#include <iostream>
#include <iomanip>

#include "TApplication.h"
#include "TSystem.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

#include "CorrFctnDirectYlmTherm.h"
#include "HistInfoPrinter.h"
#include "CanvasPartition.h"


//________________________________________________________________________________________________________________
TH1D* CombineTwoHists(TH1D* aHist1, TH1D* aHist2, double aNorm1, double aNorm2)
{
  TString aReturnName = TString::Format("%s_and_%s", aHist1->GetName(), aHist2->GetName());

  if(!aHist1->GetSumw2N()) aHist1->Sumw2();
  if(!aHist2->GetSumw2N()) aHist2->Sumw2();

  TH1D* tReturnHist = (TH1D*)aHist1->Clone(aReturnName);
    tReturnHist->Scale(aNorm1);
  tReturnHist->Add(aHist2, aNorm2);
  tReturnHist->Scale(1./(aNorm1+aNorm2));

  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* GetYlmCfwSysErrors(TString aDate, AnalysisType aAnalysisType, CentralityType aCentralityType, YlmComponent aComponent, int al, int am)
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
void DrawSHCfComponent(TPad* aPad, Analysis* aAnaly, YlmComponent aComponent, int al, int am, int aRebin, bool aDrawSysErrs=false/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/)
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
    tYLow = -0.02;
    tYHigh = 0.02;
  }

  //--------------------------------------------------------------

  vector<TString> tReImVec{"#Rgothic", "#Jgothic"};
  int tColor;
  if(aAnaly->GetAnalysisType()==kLamK0 || aAnaly->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aAnaly->GetAnalysisType()==kLamKchP || aAnaly->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aAnaly->GetAnalysisType()==kLamKchM || aAnaly->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  int tColorTransparent = TColor::GetColorTransparent(tColor,0.3);
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

  tSHCf->Draw("ex0");

  //--------------------------------------------------------------

  TPaveText* tText = new TPaveText(0.60, 0.70, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(tColor);
    tText->AddText(cAnalysisRootTags[aAnaly->GetAnalysisType()]);
    tText->AddText(TString::Format("%sC_{%d%d} (%s)", tReImVec[(int)aComponent].Data(), al, am, cPrettyCentralityTags[aAnaly->GetCentralityType()]));
  tText->Draw();

  //--------------------------------------------------------------
  if(aDrawSysErrs)
  {
    TH1D* tSHCfwSysErrs = GetYlmCfwSysErrors(TString("20181205"), aAnaly->GetAnalysisType(), aAnaly->GetCentralityType(), aComponent, al, am);

    for(int i=1; i<tSHCfwSysErrs->GetNbinsX()+1; i++) 
    {
      assert(tSHCfwSysErrs->GetBinWidth(i)==tSHCf->GetBinWidth(i));
      double tFracDiffConj = (tSHCfwSysErrs->GetBinContent(i) - tSHCf->GetBinContent(i))/tSHCf->GetBinContent(i);
      assert(fabs(tFracDiffConj) < 0.025);

      tSHCfwSysErrs->SetBinContent(i, tSHCf->GetBinContent(i));
    }

      tSHCfwSysErrs->SetFillColor(tColorTransparent);
      tSHCfwSysErrs->SetFillStyle(1000);
      tSHCfwSysErrs->SetLineColor(0);
      tSHCfwSysErrs->SetLineWidth(0);

      tSHCfwSysErrs->Draw("e2psame");
  }

}

//_________________________________________________________________________________________
void DrawSHCfComponent(TPad* aPad, Analysis* aAnaly, Analysis* aConjAnaly, YlmComponent aComponent, int al, int am, int aRebin, bool aDrawSysErrs=false/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/, bool aPrintAliceInfo=false, double aMarkerSize=0.75)
{
  aPad->cd();

  aPad->SetTopMargin(0.02);
  aPad->SetBottomMargin(0.15);
  aPad->SetLeftMargin(0.15);
  aPad->SetRightMargin(0.02);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  double tXLow=0., tXHigh=0.329;
  double tYLow, tYHigh;
  if(al==0 && am==0)
  {
    tYLow = 0.86;
    tYHigh = 1.07;
  }
  else
  {
    tYLow = -0.018;
    tYHigh = 0.02;
    if(aAnaly->GetAnalysisType()==kLamKchP && aAnaly->GetCentralityType()==k0010)
    {
      tYHigh = 0.0045;
    }
  }

  //--------------------------------------------------------------

  vector<TString> tReImVec{"#Rgothic", "#Jgothic"};
  if(al==0 && am==0) tReImVec = vector<TString>{"", ""};
  int tColor;
  if(aAnaly->GetAnalysisType()==kLamK0 || aAnaly->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aAnaly->GetAnalysisType()==kLamKchP || aAnaly->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aAnaly->GetAnalysisType()==kLamKchM || aAnaly->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  int tColorTransparent = TColor::GetColorTransparent(tColor,0.3);
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


  tSHCf->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf->SetMarkerStyle(20);
  tSHCf->SetMarkerSize(aMarkerSize);
  tSHCf->SetMarkerColor(tColor);
  tSHCf->SetLineColor(tColor);

  tSHCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf->GetXaxis()->SetTitleOffset(1.02);
  tSHCf->GetXaxis()->SetTitleSize(0.07);
  tSHCf->GetXaxis()->SetLabelSize(0.06);
  tSHCf->GetXaxis()->SetLabelOffset(0.01);  
  tSHCf->GetXaxis()->SetNdivisions(505);

  tSHCf->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));
  tSHCf->GetYaxis()->SetTitleOffset(1.125);
  if(am==1 && al==1) tSHCf->GetYaxis()->SetTitleOffset(0.9);
  tSHCf->GetYaxis()->SetTitleSize(0.07);
  tSHCf->GetYaxis()->SetLabelSize(0.06);
  tSHCf->GetYaxis()->SetLabelOffset(0.0125);  
  tSHCf->GetYaxis()->SetNdivisions(505);

  tSHCf->Draw("ex0");

  //--------------------------------------------------------------
/*
  TPaveText* tText = new TPaveText(0.70, 0.80, 0.95, 0.95, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(tColor);
    tText->AddText(TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[aAnaly->GetAnalysisType()], cAnalysisRootTags[aConjAnaly->GetAnalysisType()]));
    tText->AddText(TString::Format("%sC_{%d%d} (%s)", tReImVec[(int)aComponent].Data(), al, am, cPrettyCentralityTags[aAnaly->GetCentralityType()]));
  tText->Draw();
*/

  TString tTextSys = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", 
                                     cAnalysisRootTags[aAnaly->GetAnalysisType()], 
                                     cAnalysisRootTags[aConjAnaly->GetAnalysisType()]);
  TString tCfAndCentInfo = TString::Format("%s#it{C}_{%d%d} (%s)", tReImVec[(int)aComponent].Data(), al, am, cPrettyCentralityTags[aAnaly->GetCentralityType()]);

//  TLegend* tLeg = new TLegend(0.50, 0.175, 0.95, 0.475, tCfAndCentInfo.Data());
  TLegend* tLeg = new TLegend(0.60, 0.275, 0.95, 0.475, tCfAndCentInfo.Data());
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
    tLeg->SetFillStyle(0);
    tLeg->SetTextSize(0.07);    
//  tLeg->AddEntry(tSHCf, TString::Format("%s, stat. errors", tTextSys.Data()), "PE");
  tLeg->AddEntry(tSHCf, tTextSys.Data(), "P");

  if(aPrintAliceInfo && al==0 && am==0)
  {
    TLatex *   tex = new TLatex(0.02,1.04,"ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    tex->SetTextFont(42);
    tex->SetTextSize(0.070);
    tex->SetLineWidth(2);
    tex->Draw();
  }

  //--------------------------------------------------------------
  if(aDrawSysErrs)
  {
    TH1D* tSHCfwSysErrs_An = GetYlmCfwSysErrors(TString("20181205"), aAnaly->GetAnalysisType(), aAnaly->GetCentralityType(), aComponent, al, am);
    TH1D* tSHCfwSysErrs_Conj = GetYlmCfwSysErrors(TString("20181205"), aConjAnaly->GetAnalysisType(), aConjAnaly->GetCentralityType(), aComponent, al, am);

    assert(tYlmLiteCollAn.size()==2);
    TH1D* tSHCfwSysErrs = CombineTwoHists(tSHCfwSysErrs_An, tSHCfwSysErrs_Conj, 
                                          tYlmLiteCollAn[0]->GetNumScale()+tYlmLiteCollAn[1]->GetNumScale(), 
                                          tYlmLiteCollConjAn[0]->GetNumScale()+tYlmLiteCollConjAn[1]->GetNumScale());

    for(int i=1; i<tSHCfwSysErrs->GetNbinsX()+1; i++) 
    {
      assert(tSHCfwSysErrs->GetBinWidth(i)==tSHCf->GetBinWidth(i));
      double tFracDiff = (tSHCfwSysErrs->GetBinContent(i) - tSHCf->GetBinContent(i))/tSHCf->GetBinContent(i);
      assert(fabs(tFracDiff) < 0.025);

      tSHCfwSysErrs->SetBinContent(i, tSHCf->GetBinContent(i));
    }

      tSHCfwSysErrs->SetFillColor(tColorTransparent);
      tSHCfwSysErrs->SetFillStyle(1000);
      tSHCfwSysErrs->SetLineColor(0);
      tSHCfwSysErrs->SetLineWidth(0);

      tSHCfwSysErrs->Draw("e2psame");

      //tLeg->AddEntry(tSHCfwSysErrs, "syst. errors", "F");
      
/*
      FILE* tOutput = stdout;
      HistInfoPrinter::PrintHistInfowStatAndSystYAML(tSHCf, tSHCfwSysErrs, tOutput, tXLow, tXHigh);
*/      
      
  }

  tLeg->Draw();
}

//_________________________________________________________________________________________
void DrawSHCfComponent_ForVertical(TPad* aPad, Analysis* aAnaly, Analysis* aConjAnaly, YlmComponent aComponent, int al, int am, int aRebin, bool aDrawSysErrs=false/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/, bool aPrintAliceInfo=false, double aMarkerSize=0.75)
{
  aPad->cd();

  aPad->SetTopMargin(0.02);
  aPad->SetBottomMargin(0.15);
  aPad->SetLeftMargin(0.20);
  aPad->SetRightMargin(0.02);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  double tXLow=0., tXHigh=0.329;
  double tYLow, tYHigh;
  if(al==0 && am==0)
  {
    tYLow = 0.86;
    tYHigh = 1.02;
  }
  else
  {
    tYLow = -0.018;
    tYHigh = 0.02;
    if(aAnaly->GetAnalysisType()==kLamKchP && aAnaly->GetCentralityType()==k0010)
    {
      tYHigh = 0.0045;
    }
  }

  //--------------------------------------------------------------

  vector<TString> tReImVec{"#Rgothic", "#Jgothic"};
  if(al==0 && am==0) tReImVec = vector<TString>{"", ""};
  int tColor;
  if(aAnaly->GetAnalysisType()==kLamK0 || aAnaly->GetAnalysisType()==kALamK0) tColor=kBlack;
  else if(aAnaly->GetAnalysisType()==kLamKchP || aAnaly->GetAnalysisType()==kALamKchM) tColor=kRed+1;
  else if(aAnaly->GetAnalysisType()==kLamKchM || aAnaly->GetAnalysisType()==kALamKchP) tColor=kBlue+1;
  else tColor = kGray;

  int tColorTransparent = TColor::GetColorTransparent(tColor,0.3);
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


  tSHCf->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tSHCf->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tSHCf->SetMarkerStyle(20);
  tSHCf->SetMarkerSize(aMarkerSize);
  tSHCf->SetMarkerColor(tColor);
  tSHCf->SetLineColor(tColor);

  tSHCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tSHCf->GetXaxis()->SetTitleOffset(0.85);
  tSHCf->GetXaxis()->SetTitleSize(0.085);
  tSHCf->GetXaxis()->SetLabelSize(0.06);
  tSHCf->GetXaxis()->SetLabelOffset(0.01);  
  tSHCf->GetXaxis()->SetNdivisions(505);

  tSHCf->GetYaxis()->SetTitle(TString::Format("%s#it{C}_{%d%d}(#it{k}*)", tReImVec[(int)aComponent].Data(), al, am));
  tSHCf->GetYaxis()->SetTitleOffset(0.9);
  tSHCf->GetYaxis()->SetTitleSize(0.095);
  tSHCf->GetYaxis()->SetLabelSize(0.06);
  tSHCf->GetYaxis()->SetLabelOffset(0.0125);  
  tSHCf->GetYaxis()->SetNdivisions(505);

  tSHCf->Draw("ex0");

  //--------------------------------------------------------------
/*
  TPaveText* tText = new TPaveText(0.70, 0.80, 0.95, 0.95, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(tColor);
    tText->AddText(TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[aAnaly->GetAnalysisType()], cAnalysisRootTags[aConjAnaly->GetAnalysisType()]));
    tText->AddText(TString::Format("%sC_{%d%d} (%s)", tReImVec[(int)aComponent].Data(), al, am, cPrettyCentralityTags[aAnaly->GetCentralityType()]));
  tText->Draw();
*/

  TString tTextSys = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", 
                                     cAnalysisRootTags[aAnaly->GetAnalysisType()], 
                                     cAnalysisRootTags[aConjAnaly->GetAnalysisType()]);
  TString tCfAndCentInfo = TString::Format("%s#it{C}_{%d%d} (%s)", tReImVec[(int)aComponent].Data(), al, am, cPrettyCentralityTags[aAnaly->GetCentralityType()]);

//  TLegend* tLeg = new TLegend(0.50, 0.175, 0.95, 0.475, tCfAndCentInfo.Data());
  TLegend* tLeg = new TLegend(0.55, 0.40, 0.90, 0.60, tCfAndCentInfo.Data());
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
    tLeg->SetFillStyle(0);
    tLeg->SetTextSize(0.08);    
//  tLeg->AddEntry(tSHCf, TString::Format("%s, stat. errors", tTextSys.Data()), "PE");
  tLeg->AddEntry(tSHCf, tTextSys.Data(), "P");

  if(aPrintAliceInfo && al==0 && am==0)
  {
    TLatex *   tex = new TLatex(0.03,0.8735,"ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    tex->SetTextFont(42);
    tex->SetTextSize(0.075);
    tex->SetLineWidth(2);
    tex->Draw();
  }

  //--------------------------------------------------------------
  if(aDrawSysErrs)
  {
    TH1D* tSHCfwSysErrs_An = GetYlmCfwSysErrors(TString("20181205"), aAnaly->GetAnalysisType(), aAnaly->GetCentralityType(), aComponent, al, am);
    TH1D* tSHCfwSysErrs_Conj = GetYlmCfwSysErrors(TString("20181205"), aConjAnaly->GetAnalysisType(), aConjAnaly->GetCentralityType(), aComponent, al, am);

    assert(tYlmLiteCollAn.size()==2);
    TH1D* tSHCfwSysErrs = CombineTwoHists(tSHCfwSysErrs_An, tSHCfwSysErrs_Conj, 
                                          tYlmLiteCollAn[0]->GetNumScale()+tYlmLiteCollAn[1]->GetNumScale(), 
                                          tYlmLiteCollConjAn[0]->GetNumScale()+tYlmLiteCollConjAn[1]->GetNumScale());

    for(int i=1; i<tSHCfwSysErrs->GetNbinsX()+1; i++) 
    {
      assert(tSHCfwSysErrs->GetBinWidth(i)==tSHCf->GetBinWidth(i));
      double tFracDiff = (tSHCfwSysErrs->GetBinContent(i) - tSHCf->GetBinContent(i))/tSHCf->GetBinContent(i);
      assert(fabs(tFracDiff) < 0.025);

      tSHCfwSysErrs->SetBinContent(i, tSHCf->GetBinContent(i));
    }

      tSHCfwSysErrs->SetFillColor(tColorTransparent);
      tSHCfwSysErrs->SetFillStyle(1000);
      tSHCfwSysErrs->SetLineColor(0);
      tSHCfwSysErrs->SetLineWidth(0);

      tSHCfwSysErrs->Draw("e2psame");

      //tLeg->AddEntry(tSHCfwSysErrs, "syst. errors", "F");
      
/*
      FILE* tOutput = stdout;
      HistInfoPrinter::PrintHistInfowStatAndSystYAML(tSHCf, tSHCfwSysErrs, tOutput, tXLow, tXHigh);
*/      
      
  }

  tLeg->Draw();
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

  tSHCf->Draw("ex0same");
}

//_________________________________________________________________________________________
void DrawSHCfThermComponent(TPad* aPad, CorrFctnDirectYlmTherm* aCfYlmThermAn, CorrFctnDirectYlmTherm* aCfYlmThermConjAn, YlmComponent aComponent, int al, int am/*, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=2*/, int aMarkerStyle=20, int aColor=1)
{
  aPad->cd();

  TH1D* tSHCf = (TH1D*)aCfYlmThermAn->GetYlmHist(aComponent, kYlmCf, al, am);
  TH1D* tSHCfConjAn = (TH1D*)aCfYlmThermConjAn->GetYlmHist(aComponent, kYlmCf, al, am);
  tSHCf->Add(tSHCfConjAn);
  tSHCf->Scale(0.5);

  tSHCf->SetMarkerStyle(aMarkerStyle);
  tSHCf->SetMarkerSize(0.75);
  tSHCf->SetMarkerColor(aColor);
  tSHCf->SetLineColor(aColor);

  tSHCf->Draw("ex0same");
}

//_________________________________________________________________________________________
TCanvas* DrawFirstSixComponents(Analysis* aAnaly, Analysis* aConjAnaly, YlmComponent aComponent, int aRebin)
{
  vector<TString> tRealOrImag{"Re", "Im"};
  TString tCanName = TString::Format("CanCfYlm%sFirstSixComps_%s%s%s", tRealOrImag[aComponent].Data(), cAnalysisBaseTags[aAnaly->GetAnalysisType()], cAnalysisBaseTags[aConjAnaly->GetAnalysisType()], cCentralityTags[aAnaly->GetCentralityType()]);
  TCanvas *tReturnCan = new TCanvas(tCanName, tCanName);
  tReturnCan->Divide(3,3);

  int tCan=0;
  for(unsigned int il=0; il<3; il++)
  {
    for(unsigned int im=0; im<=il; im++)
    {
      tCan = il*3 + im +1;
      DrawSHCfComponent((TPad*)tReturnCan->cd(tCan), aAnaly, aConjAnaly, aComponent, il, im, aRebin);
    }
  }

  return tReturnCan;
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
  AnalysisType tConjAnType;
  if((int)tAnType %2 == 0) tConjAnType = static_cast<AnalysisType>((int)tAnType+1);
  else                     tConjAnType = static_cast<AnalysisType>((int)tAnType-1);

  bool bCombineConjugates = true;
  bool bDrawThermCfs = false;
  bool bDrawFirstSix = false;
  bool bDrawSysErrs = true;
  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";

  int tl = 1;
  int tm = 1;
//  YlmComponent tComponent = kYlmReal;

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

  TString tSaveDirectoryBase = TString::Format("%sSphericalHarmonics/%s/", tDirectoryBase.Data(), cAnalysisBaseTags[tAnType]);
//  TString tSaveDirectoryBase = "/home/jesse/Analysis/Presentations/AliFemto/20190116/Figures/";

  if(bSaveFigures) gSystem->mkdir(tSaveDirectoryBase, true);
//-----------------------------------------------------------------------------

  Analysis* tAnaly0010 = new Analysis(tFileLocationBase, tAnType, k0010, tAnRunType, 2, "", false);
  Analysis* tAnaly1030 = new Analysis(tFileLocationBase, tAnType, k1030, tAnRunType, 2, "", false);
  Analysis* tAnaly3050 = new Analysis(tFileLocationBase, tAnType, k3050, tAnRunType, 2, "", false);

  Analysis* tConjAnaly0010 = new Analysis(tFileLocationBase, tConjAnType, k0010, tAnRunType, 2, "", false);
  Analysis* tConjAnaly1030 = new Analysis(tFileLocationBase, tConjAnType, k1030, tAnRunType, 2, "", false);
  Analysis* tConjAnaly3050 = new Analysis(tFileLocationBase, tConjAnType, k3050, tAnRunType, 2, "", false);

  //----------
  TString tCanNameAll, tCanName0010;
  TString tCanName0010_C00, tCanName0010_ReC11;
  TCanvas *tCanAll, *tCan0010;
  TCanvas *tCan0010_C00, *tCan0010_ReC11;
  bool draw_vert = true;
  if(!bCombineConjugates)
  {
    tCanNameAll = TString::Format("CanCfYlmReC00C11_%s_All", cAnalysisBaseTags[tAnType]);
    if(bDrawSysErrs) tCanNameAll += TString("_wSysErr");
    tCanAll = new TCanvas(tCanNameAll, tCanNameAll);
    tCanAll->Divide(2, 3);

    cout << "BUILDING LARGE 6 PANEL FIGURE" << endl;
    DrawSHCfComponent((TPad*)tCanAll->cd(1), tAnaly0010, kYlmReal, 0, 0, tRebin, bDrawSysErrs);
    DrawSHCfComponent((TPad*)tCanAll->cd(2), tAnaly0010, kYlmReal, 1, 1, tRebin, bDrawSysErrs);

    DrawSHCfComponent((TPad*)tCanAll->cd(3), tAnaly1030, kYlmReal, 0, 0, tRebin, bDrawSysErrs);
    DrawSHCfComponent((TPad*)tCanAll->cd(4), tAnaly1030, kYlmReal, 1, 1, tRebin, bDrawSysErrs);

    DrawSHCfComponent((TPad*)tCanAll->cd(5), tAnaly3050, kYlmReal, 0, 0, tRebin, bDrawSysErrs);
    DrawSHCfComponent((TPad*)tCanAll->cd(6), tAnaly3050, kYlmReal, 1, 1, tRebin, bDrawSysErrs);

    //----------

    tCanName0010 = TString::Format("CanCfYlmReC00C11_%s_0010", cAnalysisBaseTags[tAnType]);
    if(bDrawSysErrs) tCanName0010 += TString("_wSysErr");
    tCan0010 = new TCanvas(tCanName0010, tCanName0010);
    tCan0010->Divide(2, 1);

    cout << "BUILDING SMALL 2 PANEL FIGURE" << endl;
    DrawSHCfComponent((TPad*)tCan0010->cd(1), tAnaly0010, kYlmReal, 0, 0, tRebin, bDrawSysErrs);
    DrawSHCfComponent((TPad*)tCan0010->cd(2), tAnaly0010, kYlmReal, 1, 1, tRebin, bDrawSysErrs);
  }
  else
  {
    tCanNameAll = TString::Format("CanCfYlmReC00C11_%s%s_All", cAnalysisBaseTags[tAnType], cAnalysisBaseTags[tConjAnType]);
    if(bDrawSysErrs) tCanNameAll += TString("_wSysErr");
    tCanAll = new TCanvas(tCanNameAll, tCanNameAll);
    tCanAll->Divide(2, 3);

    DrawSHCfComponent((TPad*)tCanAll->cd(1), tAnaly0010, tConjAnaly0010, kYlmReal, 0, 0, tRebin, bDrawSysErrs);
    DrawSHCfComponent((TPad*)tCanAll->cd(2), tAnaly0010, tConjAnaly0010, kYlmReal, 1, 1, tRebin, bDrawSysErrs);

    DrawSHCfComponent((TPad*)tCanAll->cd(3), tAnaly1030, tConjAnaly1030, kYlmReal, 0, 0, tRebin, bDrawSysErrs);
    DrawSHCfComponent((TPad*)tCanAll->cd(4), tAnaly1030, tConjAnaly1030, kYlmReal, 1, 1, tRebin, bDrawSysErrs);

    DrawSHCfComponent((TPad*)tCanAll->cd(5), tAnaly3050, tConjAnaly3050, kYlmReal, 0, 0, tRebin, bDrawSysErrs);
    DrawSHCfComponent((TPad*)tCanAll->cd(6), tAnaly3050, tConjAnaly3050, kYlmReal, 1, 1, tRebin, bDrawSysErrs);

    //----------

    tCanName0010 = TString::Format("CanCfYlmReC00C11_%s%s_0010", cAnalysisBaseTags[tAnType], cAnalysisBaseTags[tConjAnType]);
    tCanName0010_C00 = TString::Format("CanCfYlmC00_%s%s_0010", cAnalysisBaseTags[tAnType], cAnalysisBaseTags[tConjAnType]);
    tCanName0010_ReC11 = TString::Format("CanCfYlmReC11_%s%s_0010", cAnalysisBaseTags[tAnType], cAnalysisBaseTags[tConjAnType]);
    if(bDrawSysErrs) 
    {
      tCanName0010 += TString("_wSysErr");
      tCanName0010_C00 += TString("_wSysErr");
      tCanName0010_ReC11 += TString("_wSysErr");
    }
    if(draw_vert)
    {
      tCan0010 = new TCanvas(tCanName0010, tCanName0010, 700, 1000);
      tCan0010->Divide(1, 2);
      
      tCan0010_C00 = new TCanvas(tCanName0010_C00, tCanName0010_C00, 700, 500);
      tCan0010_ReC11 = new TCanvas(tCanName0010_ReC11, tCanName0010_ReC11, 700, 500);
    }
    else
    {
      tCan0010 = new TCanvas(tCanName0010, tCanName0010, 1400, 500);
      tCan0010->Divide(2, 1);    
    }
    ((TPad*)tCan0010->cd(1))->SetTicks(1,1);
    ((TPad*)tCan0010->cd(2))->SetTicks(1,1);

    if(draw_vert)
    {
      DrawSHCfComponent_ForVertical((TPad*)tCan0010->cd(1), tAnaly0010, tConjAnaly0010, kYlmReal, 0, 0, tRebin, bDrawSysErrs, true, 1.0);
      DrawSHCfComponent_ForVertical((TPad*)tCan0010->cd(2), tAnaly0010, tConjAnaly0010, kYlmReal, 1, 1, tRebin, bDrawSysErrs, true, 1.0);
      
      DrawSHCfComponent_ForVertical(tCan0010_C00, tAnaly0010, tConjAnaly0010, kYlmReal, 0, 0, tRebin, bDrawSysErrs, true, 1.25);
      DrawSHCfComponent_ForVertical(tCan0010_ReC11, tAnaly0010, tConjAnaly0010, kYlmReal, 1, 1, tRebin, bDrawSysErrs, true, 1.25);
    }
    else
    {
      DrawSHCfComponent((TPad*)tCan0010->cd(1), tAnaly0010, tConjAnaly0010, kYlmReal, 0, 0, tRebin, bDrawSysErrs, true, 1.0);
      DrawSHCfComponent((TPad*)tCan0010->cd(2), tAnaly0010, tConjAnaly0010, kYlmReal, 1, 1, tRebin, bDrawSysErrs, true, 1.0);    
    }
    
    //------ For Phys Rev C final
    TLatex* tLaText;

    double tXLett_LaTex=0.20;
    double tYLett_LaTex=0.225;
    bool tIsNDC_LaTex=true;    
    
    int tTextAlign_LaTex = 11;
    double tLineWidth_LaTex=2;
    int tTextFont_LaTex = 62;
    double tTextSize_LaTex = 0.10;
    double tScaleFactor_LaTex = 1.0;
    
    if(draw_vert)
    {
      tXLett_LaTex=0.25;
      tYLett_LaTex = 0.85;
    }

    tLaText = CanvasPartition::BuildTLatex(TString("(a)"), tXLett_LaTex, tYLett_LaTex, tTextAlign_LaTex, tLineWidth_LaTex, tTextFont_LaTex, tTextSize_LaTex, tScaleFactor_LaTex, tIsNDC_LaTex);
    (TPad*)tCan0010->cd(1)->cd();
    tLaText->Draw();
    tCan0010_C00->cd();
    tLaText->Draw();
    
    tLaText = CanvasPartition::BuildTLatex(TString("(b)"), tXLett_LaTex, tYLett_LaTex, tTextAlign_LaTex, tLineWidth_LaTex, tTextFont_LaTex, tTextSize_LaTex, tScaleFactor_LaTex, tIsNDC_LaTex);
    (TPad*)tCan0010->cd(2)->cd();
    tLaText->Draw();
    tCan0010_ReC11->cd();
    tLaText->Draw();
  }

  if(bDrawThermCfs)
  {
    int tImpactParam = 2;
    TString aCfDescriptor = "Full";

    TString tFileNameBaseTherm = "CorrelationFunctions_wOtherPairs_BuildCfYlm";
    TString tFileNameModifierTherm = "";

    TString tFileNameTherm = TString::Format("%s%s.root", tFileNameBaseTherm.Data(), tFileNameModifierTherm.Data());

    TString tFileDirTherm = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
    TString tFileLocationTherm = TString::Format("%s%s", tFileDirTherm.Data(), tFileNameTherm.Data());

    CorrFctnDirectYlmTherm* tCfYlmThermAn = GetYlmCfTherm(tFileLocationTherm, tImpactParam, tAnType, 2, 300, 0., 3., tRebin);
    CorrFctnDirectYlmTherm* tCfYlmThermConjAn = GetYlmCfTherm(tFileLocationTherm, tImpactParam, tConjAnType, 2, 300, 0., 3., tRebin);

    if(!bCombineConjugates)
    {
      DrawSHCfThermComponent((TPad*)tCanAll->cd(1), tCfYlmThermAn, kYlmReal, 0, 0, 29, kOrange);
      DrawSHCfThermComponent((TPad*)tCanAll->cd(2), tCfYlmThermAn, kYlmReal, 1, 1, 29, kOrange);
      //----------
      DrawSHCfThermComponent((TPad*)tCan0010->cd(1), tCfYlmThermAn, kYlmReal, 0, 0, 29, kOrange);
      DrawSHCfThermComponent((TPad*)tCan0010->cd(2), tCfYlmThermAn, kYlmReal, 1, 1, 29, kOrange);
    }
    else
    {
      DrawSHCfThermComponent((TPad*)tCanAll->cd(1), tCfYlmThermAn, tCfYlmThermConjAn, kYlmReal, 0, 0, 29, kOrange);
      DrawSHCfThermComponent((TPad*)tCanAll->cd(2), tCfYlmThermAn, tCfYlmThermConjAn, kYlmReal, 1, 1, 29, kOrange);
      //----------
      DrawSHCfThermComponent((TPad*)tCan0010->cd(1), tCfYlmThermAn, tCfYlmThermConjAn, kYlmReal, 0, 0, 29, kOrange);
      DrawSHCfThermComponent((TPad*)tCan0010->cd(2), tCfYlmThermAn, tCfYlmThermConjAn, kYlmReal, 1, 1, 29, kOrange);
    }
  }

  //-----------------------------------------------------------------------------
  if(bDrawFirstSix)
  {
    TCanvas* tCanFirstSixReal = DrawFirstSixComponents(tAnaly0010, tConjAnaly0010, kYlmReal, tRebin);
    TCanvas* tCanFirstSixImag = DrawFirstSixComponents(tAnaly0010, tConjAnaly0010, kYlmImag, tRebin);
    if(bSaveFigures)
    {
      tCanFirstSixReal->SaveAs(TString::Format("%s%s.%s", tSaveDirectoryBase.Data(), tCanFirstSixReal->GetName(), tSaveFileType.Data()));
      tCanFirstSixImag->SaveAs(TString::Format("%s%s.%s", tSaveDirectoryBase.Data(), tCanFirstSixImag->GetName(), tSaveFileType.Data()));
    }
  }



  if(bSaveFigures)
  {
    vector<TString> twThermTagVec{TString(""), TString("_wTherm")};

    tCanAll->SaveAs(TString::Format("%s%s%s.%s", tSaveDirectoryBase.Data(), tCanNameAll.Data(), twThermTagVec[bDrawThermCfs].Data(), tSaveFileType.Data()));
    if(draw_vert) 
    {
      tCan0010->SaveAs(TString::Format("%s%s%s_vert.%s", tSaveDirectoryBase.Data(), tCanName0010.Data(), twThermTagVec[bDrawThermCfs].Data(), tSaveFileType.Data()));
      tCan0010_C00->SaveAs(TString::Format("%s%s%s.%s", tSaveDirectoryBase.Data(), tCanName0010_C00.Data(), twThermTagVec[bDrawThermCfs].Data(), tSaveFileType.Data()));
      tCan0010_ReC11->SaveAs(TString::Format("%s%s%s.%s", tSaveDirectoryBase.Data(), tCanName0010_ReC11.Data(), twThermTagVec[bDrawThermCfs].Data(), tSaveFileType.Data()));
    }
    else tCan0010->SaveAs(TString::Format("%s%s%s.%s", tSaveDirectoryBase.Data(), tCanName0010.Data(), twThermTagVec[bDrawThermCfs].Data(), tSaveFileType.Data()));
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
