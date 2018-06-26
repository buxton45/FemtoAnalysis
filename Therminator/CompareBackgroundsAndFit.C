#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TF1.h"
#include "TLatex.h"

#include "ThermCf.h"
class ThermCf;

bool gRejectOmega=false;
//________________________________________________________________________________________________________________
double FitFunctionPolynomial(double *x, double *par)
{
  if(gRejectOmega && x[0]>0.19 && x[0]<0.23)
  {
    TF1::RejectPoint();
    return 0;
  }
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4) + par[5]*pow(x[0],5) + par[6]*pow(x[0],6);
}

//________________________________________________________________________________________________________________
TF1* FitBackground(TH1* aBgdOnlyCf, int aPower=6, double aMinBgdFit=0., double aMaxBgdFit=3.)
{
  cout << endl << endl << "Fitting: " << aBgdOnlyCf->GetName() << " with FitBackground" << endl;

  assert(aPower <= 6);  //Up to 6th order polynomial

  TString tFitName = TString::Format("BgdFit_%s", aBgdOnlyCf->GetName());
  TF1* tBgdFit = new TF1(tFitName, FitFunctionPolynomial, 0., 3., 7);

  tBgdFit->SetParameter(0, 1.);
  tBgdFit->SetParameter(1, 0.);
  tBgdFit->SetParameter(2, 0.);
  tBgdFit->SetParameter(3, 0.);
  tBgdFit->SetParameter(4, 0.);
  tBgdFit->SetParameter(5, 0.);
  tBgdFit->SetParameter(6, 0.);

  if(aPower<6)
  {
    for(int i=6; i>aPower; i--)
    {
      cout << "tBgdFit->FixParameter(" << i << ", 0.);" << endl;
      tBgdFit->FixParameter(i, 0.);
    }
  }

  aBgdOnlyCf->Fit(tFitName, "0q", "", aMinBgdFit, aMaxBgdFit);
  //-------------------------------------------------------------------
  cout << "Chi2 = " << tBgdFit->GetChisquare() << endl;
  cout << "Parameters:" << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
  cout << "Single Line: " << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << TString::Format("% 11.8f, ", tBgdFit->GetParameter(i));
  cout << endl << endl;
  //-------------------------------------------------------------------
  return tBgdFit;
}

//________________________________________________________________________________________________________________
double FitFunctionPolynomialwNorm(double *x, double *par)
{
  if(gRejectOmega && x[0]>0.19 && x[0]<0.23)
  {
    TF1::RejectPoint();
    return 0;
  }
  return par[7]*FitFunctionPolynomial(x, par);
}

//________________________________________________________________________________________________________________
TF1* FitBackgroundwNorm(TF1* aThermBgdFit, TH1D* aBgdOnlyCf, int aPower=6, double aMinBgdFit=0., double aMaxBgdFit=3.)
{
//  cout << endl << endl << "Fitting: " << aBgdOnlyCf->GetName() << " with FitBackgroundwNorm" << endl;

  assert(aPower <= 6);  //Up to 6th order polynomial

  TString tFitName = TString::Format("BgdFitwNorm_%s", aBgdOnlyCf->GetName());
  TF1* tBgdFit = new TF1(tFitName, FitFunctionPolynomialwNorm, 0., 3., 8);

  tBgdFit->FixParameter(0, aThermBgdFit->GetParameter(0));
  tBgdFit->FixParameter(1, aThermBgdFit->GetParameter(1));
  tBgdFit->FixParameter(2, aThermBgdFit->GetParameter(2));
  tBgdFit->FixParameter(3, aThermBgdFit->GetParameter(3));
  tBgdFit->FixParameter(4, aThermBgdFit->GetParameter(4));
  tBgdFit->FixParameter(5, aThermBgdFit->GetParameter(5));
  tBgdFit->FixParameter(6, aThermBgdFit->GetParameter(6));

  tBgdFit->SetParameter(7, 1.);
/*
  if(aPower<6)
  {
    for(int i=6; i>aPower; i--)
    {
      cout << "tBgdFitwNorm->FixParameter(" << i << ", 0.);" << endl;
      tBgdFit->FixParameter(i, 0.);
    }
  }
*/
  aBgdOnlyCf->Fit(tFitName, "0q", "", aMinBgdFit, aMaxBgdFit);
  //-------------------------------------------------------------------
/*
  cout << "Chi2 = " << tBgdFit->GetChisquare() << endl;
  cout << "Parameters:" << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
  cout << "Single Line: " << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << TString::Format("% 11.8f, ", tBgdFit->GetParameter(i));
  cout << endl << endl;
*/
  //-------------------------------------------------------------------
  return tBgdFit;
}

//________________________________________________________________________________________________________________
void PrintFitParams(TPad* aPad, TF1* aFit, double aTextSize=0.035)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  double tPar0, tPar1, tPar2, tPar3, tPar4, tPar5, tPar6;
  tPar0 = aFit->GetParameter(0);
  tPar1 = aFit->GetParameter(1);
  tPar2 = aFit->GetParameter(2);
  tPar3 = aFit->GetParameter(3);
  tPar4 = aFit->GetParameter(4);
  tPar5 = aFit->GetParameter(5);
  tPar6 = aFit->GetParameter(6);

  tTex->DrawLatex(0.15, 0.92, TString::Format("#color[800]{Bgd} = %0.3f + %0.3fx + %0.3fx^{2} + ..." , tPar0, tPar1, tPar2));
  tTex->DrawLatex(0.30, 0.91, TString::Format("... + %0.3fx^{3} + %0.3fx^{4} + ..." , tPar3, tPar4));
  tTex->DrawLatex(0.30, 0.90, TString::Format("... + %0.3fx^{5} + %0.3fx^{6} + ..." , tPar5, tPar6));
}

//________________________________________________________________________________________________________________
void PrintInfo(TPad* aPad, AnalysisType aAnType, double aTextSize=0.04)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  tTex->DrawLatex(0.2, 1.05, "THERMINATOR");
  tTex->DrawLatex(1.4, 1.05, cAnalysisRootTags[aAnType]);
}
//________________________________________________________________________________________________________________
void PrintInfo(TPad* aPad, TString aOverallDescriptor, double aTextSize=0.04)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  tTex->DrawLatex(0.2, 1.05, "THERMINATOR");
  tTex->DrawLatex(1.2, 1.05, aOverallDescriptor);

  //For CompareBackgroundReductionMethods when using ArtificialV3Signal-1
  //tTex->DrawLatex(0.3, 3.05, "THERMINATOR");
  //tTex->DrawLatex(1.2, 3.05, aOverallDescriptor);

}

//________________________________________________________________________________________________________________
CentralityType GetCentralityType(int aImpactParam)
{
  CentralityType tCentType=kMB;
  if(aImpactParam==3) tCentType=k0010;
  else if(aImpactParam==5 || aImpactParam==7) tCentType=k1030;
  else if(aImpactParam==8 || aImpactParam==9) tCentType=k3050;
  else assert(0);

  return tCentType;
}

//________________________________________________________________________________________________________________
AnalysisType GetConjAnType(AnalysisType aAnType)
{
  AnalysisType tConjAnType;
  if     (aAnType==kLamK0)    {tConjAnType=kALamK0;}
  else if(aAnType==kALamK0)   {tConjAnType=kLamK0;}
  else if(aAnType==kLamKchP)  {tConjAnType=kALamKchM;}
  else if(aAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(aAnType==kLamKchM)  {tConjAnType=kALamKchP;}
  else if(aAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);

  return tConjAnType;
}

//________________________________________________________________________________________________________________
int GetColor(AnalysisType aAnType)
{
  int tReturnColor;

  if(aAnType==kLamK0 || aAnType==kALamK0) tReturnColor = kBlack;
  else if(aAnType==kLamKchP || aAnType==kALamKchM) tReturnColor = kRed+1;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tReturnColor = kBlue+1;
  else assert(0);

  return tReturnColor;
}


//________________________________________________________________________________________________________________
TH1D* GetQuickData(AnalysisType aAnType, CentralityType aCentType, bool aCombineConjugates, TString aResultsData="20180307")
{
  TString tFileName = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Therminator/QuickData/QuickDataCfs_%s.root", aResultsData.Data());
  TFile tFile(tFileName);

  TString tReturnHistName;
  if(!aCombineConjugates) tReturnHistName = TString::Format("KStarHeavyCf_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
  else tReturnHistName = TString::Format("KStarHeavyCf_%s%s%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[GetConjAnType(aAnType)], cCentralityTags[aCentType]);
  TH1D* tReturnHist = (TH1D*)tFile.Get(tReturnHistName);
  tReturnHist->SetDirectory(0);

  return tReturnHist;
}


//________________________________________________________________________________________________________________
void Draw1vs2vs3(TPad* aPad, AnalysisType aAnType, TH1* aCf1, TH1* aCf2, TH1* aCf3, TString aDescriptor1, TString aDescriptor2, TString aDescriptor3, TString aOverallDescriptor, bool aFitBgd=true, double aMaxFit=3.0)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  if(aFitBgd) aDescriptor3 += TString(" (w. Fit)");
  //---------------------------------------------------------------
  aCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  aCf1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  aCf1->GetXaxis()->SetRangeUser(0.,2.0);
  aCf1->GetYaxis()->SetRangeUser(0.86, 1.07);

  aCf1->Draw();
  aCf2->Draw("same");
  aCf3->Draw("same");

  //---------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(aCf1, aDescriptor1.Data());
  tLeg->AddEntry(aCf2, aDescriptor2.Data());
  tLeg->AddEntry(aCf3, aDescriptor3.Data());

  //---------------------------------------------------------------

  TF1 *tBgdFit, *tBgdFitDraw;
  if(aFitBgd)
  {
    cout << "**************************************************" << endl;
    cout << "Fitting call from: Draw1vs2vs3" << endl;
    int tPower = 6;
    tBgdFit = FitBackground(aCf3, tPower, 0., aMaxFit);
    cout << "**************************************************" << endl;
    if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
    {
      gRejectOmega=false;
      tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 2., 7);
      for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
    }
    else tBgdFitDraw = tBgdFit;

    tBgdFitDraw->SetLineColor(aCf3->GetLineColor());
    tBgdFitDraw->Draw("lsame");
  }


  //---------------------------------------------------------------
  TH1D* tCfRatio = (TH1D*)aCf2->Clone();
  tCfRatio->Divide(aCf3);
  ThermCf::SetStyleAndColor(tCfRatio, 20, kCyan);
  tCfRatio->Draw("same");

  TH1D* tCfDiff = (TH1D*)aCf2->Clone();
  tCfDiff->Add(aCf3, -1.);
  TH1D* tUnity = (TH1D*)aCf2->Clone();
  for(int i=1; i<=tUnity->GetNbinsX(); i++) tUnity->SetBinContent(i, 1.);
  tCfDiff->Add(tUnity, 1.);
  ThermCf::SetStyleAndColor(tCfDiff, 24, kMagenta);
  tCfDiff->Draw("same");

  tLeg->AddEntry(tCfRatio, "Ratio (B/C)");
  tLeg->AddEntry(tCfDiff, "1+Diff (B-C)");

  //---------------------------------------------------------------

  aCf1->Draw("same");
  tLeg->Draw();


  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo(aPad, aOverallDescriptor, 0.04);
  if(aFitBgd) PrintFitParams(aPad, tBgdFitDraw, 0.035);
}


//________________________________________________________________________________________________________________
TCanvas* CompareCfWithAndWithoutBgd(TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmega=true;
  else gRejectOmega=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmega=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
  TString tFileNameCfs1 = "CorrelationFunctions.root";
  TString tFileNameCfs2 = "CorrelationFunctions_RandomEPs.root";
  TString tFileNameCfs3 = "CorrelationFunctions_RandomEPs_NumWeight1.root";

  TString tDescriptor1 = "Cf w/o Bgd (A)";
  TString tDescriptor2 = "Cf w. Bgd (B)";
  TString tDescriptor3 = "Bgd (C)";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 21;
  int tMarkerStyle3 = 26;

  int tColor1 = kBlack;
  int tColor2 = kGreen+1;
  int tColor3 = kOrange;

  //--------------------------------------------

  ThermCf* tThermCf1 = new ThermCf(tFileNameCfs1, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf2 = new ThermCf(tFileNameCfs2, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf3 = new ThermCf(tFileNameCfs3, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);

  if(!aCombineImpactParams)
  {
    tThermCf1->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf2->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf3->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  }
  TH1* tCf1 = tThermCf1->GetThermCf(tMarkerStyle1, tColor1, 0.75);
  TH1* tCf2 = tThermCf2->GetThermCf(tMarkerStyle2, tColor2, 0.75);
  TH1* tCf3 = tThermCf3->GetThermCf(tMarkerStyle3, tColor3, 0.75);

//-------------------------------------------------------------------------------
  TString tCanCfsName;
  tCanCfsName = TString::Format("CompareBgds_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanCfsName += TString("wConj");
  if(!aCombineImpactParams) tCanCfsName += TString::Format("_b%d", aImpactParam);
  else tCanCfsName += TString(cCentralityTags[tCentType]);

  TCanvas* tCanCfs = new TCanvas(tCanCfsName, tCanCfsName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  Draw1vs2vs3((TPad*)tCanCfs, aAnType, tCf1, tCf2, tCf3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  return tCanCfs;
}


//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmega=true;
  else gRejectOmega=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmega=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
  TString tDescriptor = "THERM. Bgd (w. Fit)";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = kOrange;

  //--------------------------------------------
  ThermCf* tThermCf = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  if(!aCombineImpactParams) tThermCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  TH1* tCf = tThermCf->GetThermCf(tMarkerStyle, tColor, 0.75);
//-------------------------------------------------------------------------------
  TString tCanBgdwFitName = "BgdwFitOnly";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanBgdwFitName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanBgdwFitName += TString("_RandomEPs");
  else tCanBgdwFitName += TString("");

  tCanBgdwFitName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanBgdwFitName += TString(cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanBgdwFitName += TString("wConj");
  if(!aCombineImpactParams) tCanBgdwFitName += TString::Format("_b%d", aImpactParam);
  else tCanBgdwFitName += TString(cCentralityTags[tCentType]);

//-------------------------------------------------------------------------------

  TCanvas* tCanBgdwFit = new TCanvas(tCanBgdwFitName, tCanBgdwFitName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCf->GetXaxis()->SetRangeUser(0.,3.0);
  tCf->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf->Draw();
  //---------------------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call(1) from: DrawBgdwFit" << endl;
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  tBgdFit = FitBackground(tCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmega=false;
    tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
  }
  else tBgdFitDraw = tBgdFit;

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->SetRange(0., aMaxBgdFit);
  tBgdFitDraw->Draw("lsame");

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanBgdwFit, tOverallDescriptor, 0.04);
  PrintFitParams((TPad*)tCanBgdwFit, tBgdFitDraw, 0.035);

  //---------------------------------------------------------------
  TLegend* tLeg = new TLegend(0.55, 0.15, 0.90, 0.30);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf, tDescriptor.Data());

  //---------------------------------------------------------------

  CentralityType tCentTypeData = GetCentralityType(aImpactParam);

  TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates);
  ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));

//  cout << "**************************************************" << endl;
//  cout << "Fitting call(2) from: DrawBgdwFit" << endl;
  TF1 *tBgdFitData, *tBgdFitDataDraw;
  tBgdFitData = FitBackgroundwNorm(tBgdFit, tData, tPower, 0.6, 0.9);
//  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmega=false;
    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNorm, 0., 3., 8);
    for(int i=0; i<tBgdFitData->GetNpar(); i++) tBgdFitDataDraw->SetParameter(i, tBgdFitData->GetParameter(i));
  }
  else tBgdFitDataDraw = tBgdFitData;

  tBgdFitDataDraw->SetLineColor(tData->GetLineColor());
  tBgdFitDataDraw->SetRange(0., aMaxBgdFit);

  tData->Draw("same");
  tBgdFitDataDraw->Draw("lsame");

  tCf->Draw("same");
  tBgdFitDraw->Draw("lsame");

  tLeg->AddEntry(tData, TString::Format("Data (%s)", cPrettyCentralityTags[tCentTypeData]));
  //-------------------------------------
  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.035);

  tTex->DrawLatex(0.15, 0.88, TString::Format("Bgd = %0.3f*#color[800]{Bgd}" , tBgdFitData->GetParameter(7)));
  //---------------------------------------------------------------

  tLeg->Draw();
  return tCanBgdwFit;
}

//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit_AllCent(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = DrawBgdwFit(aCfDescriptor, aFileNameCfs, aAnType, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  tCan_1030 = DrawBgdwFit(aCfDescriptor, aFileNameCfs, aAnType, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  tCan_3050 = DrawBgdwFit(aCfDescriptor, aFileNameCfs, aAnType, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1, 3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;
}


//________________________________________________________________________________________________________________
TCanvas* DrawLamKchPMBgdwFit(TString aCfDescriptor, TString aFileNameCfs, int aImpactParam, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  gRejectOmega=true;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmega=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  TString tDescriptor = "Bgd (w. Fit)";

  TString tOverallDescriptor = TString::Format("(#bar{#Lambda})#LambdaK^{#pm} (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = kOrange;

  //--------------------------------------------
  TH1 *tCf = ThermCf::GetLamKchPMCombinedThermCfs(aFileNameCfs, aCfDescriptor, tCentType, aEventsType, aRebin, aMinNorm, aMaxNorm, tMarkerStyle, tColor, aUseStavCf);
//-------------------------------------------------------------------------------
  TString tCanBgdwFitName = "LamKchPMBgdwFitOnly";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanBgdwFitName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanBgdwFitName += TString("_RandomEPs");
  else tCanBgdwFitName += TString("");

  tCanBgdwFitName += TString::Format("_%s", aCfDescriptor.Data());
  tCanBgdwFitName += TString(cCentralityTags[tCentType]);

//-------------------------------------------------------------------------------

  TCanvas* tCanBgdwFit = new TCanvas(tCanBgdwFitName, tCanBgdwFitName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCf->GetXaxis()->SetRangeUser(0.,3.0);
  tCf->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf->Draw();
  //---------------------------------------------------------------
  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf, tDescriptor.Data());
  //---------------------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call from: DrawLamKchPMBgdwFit" << endl;
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  tBgdFit = FitBackground(tCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;

  gRejectOmega=false;
  tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
  for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->SetRange(0., aMaxBgdFit);
  tBgdFitDraw->Draw("lsame");

  tLeg->Draw();

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanBgdwFit, tOverallDescriptor, 0.04);
  PrintFitParams((TPad*)tCanBgdwFit, tBgdFitDraw, 0.035);

  //---------------------------------------------------------------

  return tCanBgdwFit;
}


//________________________________________________________________________________________________________________
TCanvas* DrawLamKchPMBgdwFit_AllCent(TString aCfDescriptor, TString aFileNameCfs, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  tCan_0010 = DrawLamKchPMBgdwFit(aCfDescriptor, aFileNameCfs, 3, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  tCan_1030 = DrawLamKchPMBgdwFit(aCfDescriptor, aFileNameCfs, 5, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  tCan_3050 = DrawLamKchPMBgdwFit(aCfDescriptor, aFileNameCfs, 8, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1, 3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* CompareAnalyses(TString aCfDescriptor, TString aFileNameCfs, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------

  TString tDescriptor1 = cAnalysisRootTags[kLamK0];
  TString tDescriptor2 = cAnalysisRootTags[kLamKchP];
  TString tDescriptor3 = cAnalysisRootTags[kLamKchM];

  if(aCombineConjugates)
  {
    tDescriptor1 += TString::Format(" & %s", cAnalysisRootTags[kALamK0]);
    tDescriptor2 += TString::Format(" & %s", cAnalysisRootTags[kALamKchM]);
    tDescriptor3 += TString::Format(" & %s", cAnalysisRootTags[kALamKchP]);
  }

  TString tOverallDescriptor;
  if(!aCombineImpactParams) tOverallDescriptor = TString::Format("b=%d", aImpactParam);
  else tOverallDescriptor = TString::Format("%s", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 20;
  int tMarkerStyle3 = 20;

  int tColor1 = kBlack;
  int tColor2 = kRed+1;
  int tColor3 = kBlue+1;

  //--------------------------------------------

  ThermCf* tThermCf1 = new ThermCf(aFileNameCfs, aCfDescriptor, kLamK0, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf2 = new ThermCf(aFileNameCfs, aCfDescriptor, kLamKchP, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf3 = new ThermCf(aFileNameCfs, aCfDescriptor, kLamKchM, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);

  if(!aCombineImpactParams)
  {
    tThermCf1->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf2->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf3->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  }
  TH1* tCf1 = tThermCf1->GetThermCf(tMarkerStyle1, tColor1, 0.75);
  TH1* tCf2 = tThermCf2->GetThermCf(tMarkerStyle2, tColor2, 0.75);
  TH1* tCf3 = tThermCf3->GetThermCf(tMarkerStyle3, tColor3, 0.75);
//-------------------------------------------------------------------------------
  TString tCanCfsName;
  tCanCfsName = "CompareAnalyses";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanCfsName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanCfsName += TString("_RandomEPs");
  else tCanCfsName += TString("");

  tCanCfsName += TString::Format("_%s_", aCfDescriptor.Data());

  if(aCombineConjugates) tCanCfsName += TString("wConj");
  if(!aCombineImpactParams) tCanCfsName += TString::Format("_b%d", aImpactParam);
  else tCanCfsName += TString(cCentralityTags[tCentType]);

  TCanvas* tCanCfs = new TCanvas(tCanCfsName, tCanCfsName);
  tCanCfs->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCf1->GetXaxis()->SetRangeUser(0.,2.0);
  tCf1->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf1->Draw();
  tCf2->Draw("same");
  tCf3->Draw("same");
  //---------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf1, tDescriptor1.Data());
  tLeg->AddEntry(tCf2, tDescriptor2.Data());
  tLeg->AddEntry(tCf3, tDescriptor3.Data());

  tLeg->Draw();
  //---------------------------------------------------------------

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanCfs, tOverallDescriptor, 0.04);

  return tCanCfs;
}



//________________________________________________________________________________________________________________
TCanvas* CompareAnalyses_AllCent(TString aCfDescriptor, TString aFileNameCfs, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = CompareAnalyses(aCfDescriptor, aFileNameCfs, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_1030 = CompareAnalyses(aCfDescriptor, aFileNameCfs, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_3050 = CompareAnalyses(aCfDescriptor, aFileNameCfs, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1,3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;

}



//________________________________________________________________________________________________________________
TCanvas* CompareToAdam(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------

  TString tDescriptor1 = "Me";
  TString tDescriptor2 = "Adam";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[GetConjAnType(aAnType)]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 20;

  int tColor1 = kBlack;
  int tColor2 = kRed+1;

  //--------------------------------------------
  ThermCf* tThermCf1 = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, kMe, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf2 = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, kAdam, aRebin, aMinNorm, aMaxNorm, aUseStavCf);

  if(!aCombineImpactParams)
  {
    tThermCf1->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf2->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  }
  TH1* tCf1 = tThermCf1->GetThermCf(tMarkerStyle1, tColor1, 0.75);
  TH1* tCf2 = tThermCf2->GetThermCf(tMarkerStyle2, tColor2, 0.75);
//-------------------------------------------------------------------------------
  TString tCanCfsName;
  tCanCfsName = TString::Format("CompareToAdam_%s", cAnalysisBaseTags[aAnType]);
  if(aCombineConjugates) tCanCfsName += TString("wConj");

  if(!aCombineImpactParams) tCanCfsName += TString::Format("_b%d", aImpactParam);
  else tCanCfsName += TString(cCentralityTags[tCentType]);

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanCfsName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanCfsName += TString("_RandomEPs");
  else tCanCfsName += TString("");

  tCanCfsName += TString::Format("_%s_", aCfDescriptor.Data());

  TCanvas* tCanCfs = new TCanvas(tCanCfsName, tCanCfsName);
  tCanCfs->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCf1->GetXaxis()->SetRangeUser(0.,2.0);
  tCf1->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf1->Draw();
  tCf2->Draw("same");
  //---------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf1, tDescriptor1.Data());
  tLeg->AddEntry(tCf2, tDescriptor2.Data());

  tLeg->Draw();
  //---------------------------------------------------------------

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanCfs, tOverallDescriptor, 0.04);

  return tCanCfs;
}


//________________________________________________________________________________________________________________
TCanvas* CompareToAdam_AllCent(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, bool aCombineConjugates, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = CompareToAdam(aCfDescriptor, aFileNameCfs, aAnType, 3, aCombineConjugates, tCombineImpactParams, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_1030 = CompareToAdam(aCfDescriptor, aFileNameCfs, aAnType, 5, aCombineConjugates, tCombineImpactParams, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_3050 = CompareToAdam(aCfDescriptor, aFileNameCfs, aAnType, 8, aCombineConjugates, tCombineImpactParams, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1,3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;

}



//________________________________________________________________________________________________________________
TCanvas* DrawDataVsTherm(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  CentralityType tCentTypeData = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = kOrange;

  //--------------------------------------------
  TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates);
  ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));
  //--------------------------------------------
  ThermCf* tThermCfLong = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  if(!aCombineImpactParams) tThermCfLong->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  TH1* tCfLong = tThermCfLong->GetThermCf(tMarkerStyle, tColor, 0.75);
  //--------------------------------------------
  if(tCfLong->GetBinWidth(1) != tData->GetBinWidth(1))
  {
    cout << "tCfLong->GetBinWidth(1) != tData->GetBinWidth(1)!!!!!! CRASH" << endl;
    cout << "\t tCfLong->GetBinWidth(1) = " << tCfLong->GetBinWidth(1) << endl;
    cout << "\t tData->GetBinWidth(1) = " << tData->GetBinWidth(1) << endl;
    assert(0);
  }

  TH1D* tCf;
  if(tCfLong->GetNbinsX() != tData->GetNbinsX())  //To divide (later, for tRatio), tData and tCf need to have same number of bins, and same bin size
  {
    tCf = (TH1D*)tData->Clone(tCfLong->GetName());
    for(int i=1; i<=tCf->GetNbinsX(); i++)
    {
      tCf->SetBinContent(i, tCfLong->GetBinContent(i));
      tCf->SetBinError(i, tCfLong->GetBinError(i));
    }
    assert(tCf->GetNbinsX()==tData->GetNbinsX());
    assert(tCf->GetBinWidth(1)==tData->GetBinWidth(1));
    ThermCf::SetStyleAndColor(tCf, tMarkerStyle, tColor);
  }
  else tCf = (TH1D*)tCfLong->Clone();
//-------------------------------------------------------------------------------
  TString tCanDataVsThemName = "DataVsTherm";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanDataVsThemName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanDataVsThemName += TString("_RandomEPs");
  else tCanDataVsThemName += TString("");

  tCanDataVsThemName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanDataVsThemName += TString(cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanDataVsThemName += TString("wConj");
  if(!aCombineImpactParams) tCanDataVsThemName += TString::Format("_b%d", aImpactParam);
  else tCanDataVsThemName += TString(cCentralityTags[tCentType]);

//-------------------------------------------------------------------------------

  TCanvas* tCanDataVsThem = new TCanvas(tCanDataVsThemName, tCanDataVsThemName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tData->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tData->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tData->GetXaxis()->SetRangeUser(0.,3.0);
  tData->GetYaxis()->SetRangeUser(0.86, 1.07);

  //---------------------------------------------------------------

  TH1D* tRatio;

  tRatio = (TH1D*)tData->Clone();
  tRatio->Divide(tCf);
  ThermCf::SetStyleAndColor(tRatio, 24, kMagenta);

  //---------------------------------------------------------------

  tData->Draw();
  tCf->Draw("same");
  tRatio->Draw("same");
  //---------------------------------------------------------------
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmega=true;
  else gRejectOmega=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmega=false;  //In this case, Omega peak not present

  cout << "**************************************************" << endl;
  cout << "Fitting call from: DrawDataVsTherm" << endl;
  TF1 *tRatioFit, *tRatioFitDraw;
  int tPower = 6;
  tRatioFit = FitBackground(tRatio, tPower, 0., 3.);
  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmega=false;
    tRatioFitDraw = new TF1(tRatioFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tRatioFit->GetNpar(); i++) tRatioFitDraw->SetParameter(i, tRatioFit->GetParameter(i));
  }
  else tRatioFitDraw = tRatioFit;

  tRatioFitDraw->SetLineColor(tRatio->GetLineColor());
  tRatioFitDraw->Draw("lsame");



  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  TString tThermDescriptor = "Therm";
  if(!aCombineImpactParams) tThermDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tCanDataVsThemName += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  tLeg->AddEntry(tData, TString::Format("Data (%s)", cPrettyCentralityTags[tCentTypeData]));
  tLeg->AddEntry(tCf, tThermDescriptor);
  tLeg->AddEntry(tRatio, "Ratio (Data/Therm)");

  tLeg->Draw();
  //---------------------------------------------------------------

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanDataVsThem, tOverallDescriptor, 0.04);

  //---------------------------------------------------------------

  TH1D* tData2 = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180505");
  ThermCf::SetStyleAndColor(tData2, 24, GetColor(aAnType));
  tData2->Draw("same");
  //---------------------------------------------------------------


  return tCanDataVsThem;
}



//________________________________________________________________________________________________________________
TCanvas* DrawDataVsTherm_AllCent(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = DrawDataVsTherm(aCfDescriptor, aFileNameCfs, aAnType, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_1030 = DrawDataVsTherm(aCfDescriptor, aFileNameCfs, aAnType, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_3050 = DrawDataVsTherm(aCfDescriptor, aFileNameCfs, aAnType, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1,3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;
}


//________________________________________________________________________________________________________________
TCanvas* CompareBackgroundReductionMethods(TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aNumWeight1, bool aDrawData, bool bVerbose)
{
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  //--------------------------------------------
  TString tFileName_AlignEPs, tFileName_RandomEPs;
  if(aNumWeight1)
  {
    tFileName_AlignEPs = "CorrelationFunctions_NumWeight1.root";
    tFileName_RandomEPs = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  }
  else
  {
    tFileName_AlignEPs = "CorrelationFunctions.root";
    tFileName_RandomEPs = "CorrelationFunctions_RandomEPs.root";
  }
  //--------------------------------------------

  int tMarkerStyle_AlignEPs                = 20;
  int tColor_AlignEPs                      = kBlack;

  int tMarkerStyle_AlignEPs_UseStavCf  = 20;
  int tColor_AlignEPs_UseStavCf        = kCyan;

  int tMarkerStyle_RandomEPs               = 20;
  int tColor_RandomEPs                     = kGreen+1;

  int tMarkerStyle_RandomEPs_UseStavCf = 24;
  int tColor_RandomEPs_UseStavCf       = kMagenta;
  //--------------------------------------------

  ThermCf* tThermCf_AlignEPs                = new ThermCf(tFileName_AlignEPs, 
                                                          aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, false);
  ThermCf* tThermCf_AlignEPs_UseStavCf  = new ThermCf(tFileName_AlignEPs, 
                                                          aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, true);

  ThermCf* tThermCf_RandomEPs               = new ThermCf(tFileName_RandomEPs, 
                                                          aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, false);
  ThermCf* tThermCf_RandomEPs_UseStavCf = new ThermCf(tFileName_RandomEPs, 
                                                          aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, true);

  if(!aCombineImpactParams)
  {
    tThermCf_AlignEPs->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf_AlignEPs_UseStavCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf_RandomEPs->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf_RandomEPs_UseStavCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  }

  TH1* tCf_AlignEPs = tThermCf_AlignEPs->GetThermCf(tMarkerStyle_AlignEPs, tColor_AlignEPs, 0.75);
  TH1* tCf_AlignEPs_UseStavCf = tThermCf_AlignEPs_UseStavCf->GetThermCf(tMarkerStyle_AlignEPs_UseStavCf, tColor_AlignEPs_UseStavCf, 0.75);
  TH1* tCf_RandomEPs = tThermCf_RandomEPs->GetThermCf(tMarkerStyle_RandomEPs, tColor_RandomEPs, 0.75);
  TH1* tCf_RandomEPs_UseStavCf = tThermCf_RandomEPs_UseStavCf->GetThermCf(tMarkerStyle_RandomEPs_UseStavCf, tColor_RandomEPs_UseStavCf, 0.75);

  if(bVerbose)
  {
    for(int i=1; i<=tCf_AlignEPs_UseStavCf->GetNbinsX(); i++)
    {
      cout << "tCf_AlignEPs_UseStavCf->GetBinContent(" << i << ") = " << tCf_AlignEPs_UseStavCf->GetBinContent(i) << endl;
      cout << "tCf_RandomEPs_UseStavCf->GetBinContent(" << i << ") = " << tCf_RandomEPs_UseStavCf->GetBinContent(i) << endl;
      cout << "difference = " << tCf_AlignEPs_UseStavCf->GetBinContent(i) - tCf_RandomEPs_UseStavCf->GetBinContent(i) << endl;
      cout << "----------" << endl;
      cout << "tCf_AlignEPs_UseStavCf->GetBinError(" << i << ") = " << tCf_AlignEPs_UseStavCf->GetBinError(i) << endl;
      cout << "tCf_RandomEPs_UseStavCf->GetBinError(" << i << ") = " << tCf_RandomEPs_UseStavCf->GetBinError(i) << endl;
      cout << "difference = " << tCf_AlignEPs_UseStavCf->GetBinError(i) - tCf_RandomEPs_UseStavCf->GetBinError(i) << endl;
      cout << endl << endl;
    }
  }

//-------------------------------------------------------------------------------
  TString tCanName = "CompareBackgroundReductionMethods";

  tCanName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanName += TString(cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanName += TString("wConj");
  if(!aCombineImpactParams) tCanName += TString::Format("_b%d", aImpactParam);
  else tCanName += TString(cCentralityTags[tCentType]);

  if(aNumWeight1) tCanName += TString("_NumWeight1");
  if(aDrawData) tCanName += TString("_wData");
//-------------------------------------------------------------------------------

  TCanvas* tCan = new TCanvas(tCanName, tCanName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf_AlignEPs->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf_AlignEPs->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

//  tCf_AlignEPs->GetXaxis()->SetRangeUser(0.,3.0);
  tCf_AlignEPs->GetXaxis()->SetRangeUser(0.,2.0);
  tCf_AlignEPs->GetYaxis()->SetRangeUser(0.86, 1.07);
//  tCf_AlignEPs->GetYaxis()->SetRangeUser(0.71, 3.31);  //when using ArtificialV3Signal-1

  tCf_AlignEPs->Draw();
  tCf_AlignEPs_UseStavCf->Draw("same");
  tCf_RandomEPs->Draw("same");
  tCf_RandomEPs_UseStavCf->Draw("same");
  //---------------------------------------------------------------

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCan, tOverallDescriptor, 0.04);

  //---------------------------------------------------------------
  TLegend* tLeg = new TLegend(0.55, 0.15, 0.90, 0.30);
//  TLegend* tLeg = new TLegend(0.55, 0.40, 0.90, 0.55);  //when using ArtificialV3Signal-1
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf_AlignEPs, "Aligns EPs");
  tLeg->AddEntry(tCf_RandomEPs, "Random EPs");
  tLeg->AddEntry(tCf_RandomEPs_UseStavCf, "Random EPs, Stav.");
  tLeg->AddEntry(tCf_AlignEPs_UseStavCf, "Aligns EPs, Stav.");

  //---------------------------------------------------------------

  if(aDrawData)
  {
    CentralityType tCentTypeData = GetCentralityType(aImpactParam);

    TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180307");
    ThermCf::SetStyleAndColor(tData, 21, GetColor(aAnType));

    TH1D* tData_UseStavCf = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180505");
    ThermCf::SetStyleAndColor(tData_UseStavCf, 22, GetColor(aAnType));

    tData->Draw("same");
    tData_UseStavCf->Draw("same");

    tLeg->AddEntry(tData, TString::Format("Data (%s)", cPrettyCentralityTags[tCentTypeData]));
    tLeg->AddEntry(tData_UseStavCf, "Data, Rotate Par2");
  }

  //---------------------------------------------------------------

  tLeg->Draw();
  return tCan;
}

//________________________________________________________________________________________________________________
TCanvas* CompareBackgroundReductionMethods_AllCent(TString aCfDescriptor, AnalysisType aAnType, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aNumWeight1, bool aDrawData, bool bVerbose)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = CompareBackgroundReductionMethods(aCfDescriptor, aAnType, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aNumWeight1, aDrawData, bVerbose);
  tCan_1030 = CompareBackgroundReductionMethods(aCfDescriptor, aAnType, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aNumWeight1, aDrawData, bVerbose);
  tCan_3050 = CompareBackgroundReductionMethods(aCfDescriptor, aAnType, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aNumWeight1, aDrawData, bVerbose);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1,3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;
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

  AnalysisType tAnType = kLamKchM;
  if(tAnType==kLamKchM || tAnType==kALamKchP) gRejectOmega=true;
  else gRejectOmega=false;

  bool bCombineConjugates = true;
  bool bCombineImpactParams = true;

  ThermEventsType tEventsType = kMeAndAdam;  //kMe, kAdam, kMeAndAdam

  bool bCompareWithAndWithoutBgd = false;
  bool bDrawBgdwFitOnly = false;
  bool bDrawLamKchPMBgdwFitOnly = false;
  bool bCompareAnalyses = false;
  bool bCompareToAdam = false;
  bool bDrawDataVsTherm = false;

  bool bCompareBackgroundReductionMethods = true;
    bool bNumWeight1 = true; 
    bool bDrawData = false;
    bool bVerbose = false;
  if(bCompareBackgroundReductionMethods) tEventsType = kMe;  //TODO for now, I haven't run over Adam's results

  bool bUseStavCf=false;

  bool bDrawAllCentralities = false;

  bool bSaveFigures = false;
  int tRebin=2;
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;
  double tMaxBgdFit = 2.0;

  int tImpactParam = 9;

  TString tCfDescriptor = "Full";
//  TString tCfDescriptor = "PrimaryOnly";
//  TString tCfDescriptor = "PrimaryAndShortDecays";

  if(tCfDescriptor.EqualTo("PrimaryOnly") || tCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmega=false;  //In this case, Omega peak not present

  TString tSingleFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";

  TString tSaveDir = "/home/jesse/Analysis/Presentations/AliFemto/20180627/Figures/";
  TString tSaveFileBase = tSaveDir + TString::Format("%s/", cAnalysisBaseTags[tAnType]);


  TCanvas *tCanCfs, *tCanBgdwFit, *tCanCompareAnalyses, *tCanCompareToAdam, *tCanLamKchPMBgdwFit, *tCanDataVsTherm, *tCanCompBgdRedMethods;

  if(bCompareWithAndWithoutBgd)
  {
    tCanCfs = CompareCfWithAndWithoutBgd(tCfDescriptor, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, bUseStavCf);
    if(bSaveFigures) tCanCfs->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanCfs->GetName()));
  }

  if(bDrawBgdwFitOnly)
  {
    if(bDrawAllCentralities) tCanBgdwFit = DrawBgdwFit_AllCent(tCfDescriptor, tSingleFileName, tAnType, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, bUseStavCf);
    else tCanBgdwFit = DrawBgdwFit(tCfDescriptor, tSingleFileName, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, bUseStavCf);

    if(bSaveFigures) tCanBgdwFit->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanBgdwFit->GetName()));
  }

  if(bCompareAnalyses)
  {
    if(bDrawAllCentralities) tCanCompareAnalyses = CompareAnalyses_AllCent(tCfDescriptor, tSingleFileName, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, bUseStavCf);
    else tCanCompareAnalyses = CompareAnalyses(tCfDescriptor, tSingleFileName, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, bUseStavCf);
    if(bSaveFigures) tCanCompareAnalyses->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanCompareAnalyses->GetName()));
  }

  if(bCompareToAdam)
  {
    if(bDrawAllCentralities) tCanCompareToAdam = CompareToAdam_AllCent(tCfDescriptor, tSingleFileName, tAnType, bCombineConjugates, tRebin, tMinNorm, tMaxNorm, bUseStavCf);
    else tCanCompareToAdam = CompareToAdam(tCfDescriptor, tSingleFileName, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tRebin, tMinNorm, tMaxNorm, bUseStavCf);
    if(bSaveFigures) tCanCompareToAdam->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanCompareToAdam->GetName()));
  }


  if(bDrawLamKchPMBgdwFitOnly)
  {
    if(bDrawAllCentralities) tCanLamKchPMBgdwFit = DrawLamKchPMBgdwFit_AllCent(tCfDescriptor, tSingleFileName, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, bUseStavCf);
    else tCanLamKchPMBgdwFit = DrawLamKchPMBgdwFit(tCfDescriptor, tSingleFileName, tImpactParam, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, bUseStavCf);
    if(bSaveFigures) tCanLamKchPMBgdwFit->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanLamKchPMBgdwFit->GetName()));
  }

  if(bDrawDataVsTherm)
  {
    if(bDrawAllCentralities) tCanDataVsTherm = DrawDataVsTherm_AllCent(tCfDescriptor, tSingleFileName, tAnType, bCombineConjugates, tEventsType, 1, tMinNorm, tMaxNorm, bUseStavCf);
    else tCanDataVsTherm = DrawDataVsTherm(tCfDescriptor, tSingleFileName, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, 1, tMinNorm, tMaxNorm, bUseStavCf);
    if(bSaveFigures) tCanDataVsTherm->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanDataVsTherm->GetName()));
  }

  if(bCompareBackgroundReductionMethods)
  {
    if(bDrawAllCentralities) tCanCompBgdRedMethods = CompareBackgroundReductionMethods_AllCent(tCfDescriptor, tAnType, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, bNumWeight1, bDrawData, bVerbose);
    else tCanCompBgdRedMethods = CompareBackgroundReductionMethods(tCfDescriptor, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, bNumWeight1, bDrawData, bVerbose);
    if(bSaveFigures) tCanCompBgdRedMethods->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanCompBgdRedMethods->GetName()));

  }

//-------------------------------------------------------------------------------




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
