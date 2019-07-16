//TODO based off file with same name in LednickyFitter directory

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

#include "ThermCf.h"
#include "FitPartialAnalysis.h"
#include "NumIntLednickyCf.h"
#include "CanvasPartition.h"

bool gRejectPoints=false;
double tRejectOmegaLow = 0.19;
double tRejectOmegaHigh = 0.24;

//________________________________________________________________________________________________________________
void SetStandardPadMargins(TPad* aPad)
{
  aPad->SetRightMargin(0.025);
  aPad->SetTopMargin(0.0750);
  aPad->SetLeftMargin(0.15);
  aPad->SetBottomMargin(0.16);
}

//________________________________________________________________________________________________________________
void SetStandardAxesAttributes(TH1* aHist)
{
  aHist->GetXaxis()->SetTitleOffset(1.1);
  aHist->GetXaxis()->SetTitleSize(0.065);
  aHist->GetXaxis()->SetLabelSize(0.05);
  aHist->GetXaxis()->SetLabelOffset(0.015);

  aHist->GetYaxis()->SetTitleOffset(1.05);
  aHist->GetYaxis()->SetTitleSize(0.065);
  aHist->GetYaxis()->SetLabelSize(0.05);
  aHist->GetYaxis()->SetLabelOffset(0.0075);
}

//________________________________________________________________________________________________________________
TH1* GetThermHist1d(TString aFileLocation, TString aHistName)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TH1 *tReturnHist = (TH1*)tFile->Get(aHistName);
  TH1 *tReturnHistClone = (TH1*)tReturnHist->Clone();
  tReturnHistClone->SetDirectory(0);

  tFile->Close();
  delete tFile;

  return tReturnHistClone;
}

//________________________________________________________________________________________________________________
TH3* AddTwoTH3s(TH3* aHist1, TH3* aHist2)
{
  TH3* tReturnHist = (TH3*)aHist1->Clone(TString::Format("%sAnd%s", aHist1->GetName(), aHist2->GetName()));
  tReturnHist->Add(aHist2);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH3* GetThermHist3d(TString aFileLocation, TString aHistName)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TH3 *tReturnHist = (TH3*)tFile->Get(aHistName);
  TH3 *tReturnHistClone = (TH3*)tReturnHist->Clone();
  tReturnHistClone->SetDirectory(0);

  tFile->Close();
  delete tFile;

  return tReturnHistClone;
}

//________________________________________________________________________________________________________________
TH3* GetThermHist3d_CombConj(TString aFileLocation, TString aHistName1, TString aHistName2)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TH3* tHist1 = (TH3*)tFile->Get(aHistName1)->Clone();
  TH3* tHist2 = (TH3*)tFile->Get(aHistName2)->Clone();

  TH3 *tReturnHist = AddTwoTH3s(tHist1, tHist2);
  tReturnHist->SetDirectory(0);

  tFile->Close();
  delete tFile;

  return tReturnHist;
}

//________________________________________________________________________________________________________________
double LednickyEqWithNorm(double *x, double *par)
{
  if(gRejectPoints && x[0]>tRejectOmegaLow && x[0]<tRejectOmegaHigh)
  {
    TF1::RejectPoint();
    return 0;
  }

  double tUnNormCf = FitPartialAnalysis::LednickyEq(x, par);
  double tNormCf = par[5]*tUnNormCf;
  return tNormCf;
}

//________________________________________________________________________________________________________________
double FitFunctionGaussian(double *x, double *par)
{
  //4 parameters
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/(sqrt(2)*par[2]),2.0))) + par[3];
}

//________________________________________________________________________________________________________________
TF1* FitwGauss(TH1* aHist, double aMinFit=0., double aMaxFit=50.)
{
  TString tFitName = TString::Format("%s_FitGauss", aHist->GetName());
//  TF1* tReturnFunction = new TF1(tFitName, BackgroundFitter::FitFunctionGaussian, aMinFit, aMaxFit, 4);  //No sqrt(2) with sigma
  TF1* tReturnFunction = new TF1(tFitName, FitFunctionGaussian, -50., 50., 4);

  double tMaxVal = aHist->GetMaximum();
  double tMaxPos = aHist->GetBinCenter(aHist->GetMaximumBin());
  int tApproxSigBin = aHist->FindLastBinAbove(tMaxVal/2.);
  double tApproxSig = aHist->GetBinCenter(tApproxSigBin);

  tReturnFunction->SetParameter(0, tMaxVal);
  tReturnFunction->SetParLimits(0, 0., 1.5*tMaxVal);

  tReturnFunction->SetParameter(1, tMaxPos);
//  tReturnFunction->SetParLimits(1, 0., 50.);
//  tReturnFunction->FixParameter(1, 0.);

  tReturnFunction->SetParameter(2, tApproxSig);
  tReturnFunction->SetParLimits(2, 0., 50.);

  tReturnFunction->FixParameter(3, 0.);

  aHist->Fit(tFitName, "0", "", aMinFit, aMaxFit);
  return tReturnFunction;
}


//________________________________________________________________________________________________________________
void DrawHistwGaussFit(TPad* aPad, TH1* aHist, double aGaussFitMin, double aGaussFitMax, TString aMuName="#it{#mu}_{out}", TString aSigmaName="#it{R}_{out}", bool aDrawTextOnRight=false)
{
  TF1* tGaussFit = FitwGauss(aHist, aGaussFitMin, aGaussFitMax);

  aPad->cd();
  aHist->DrawCopy();
  tGaussFit->DrawCopy("same");

  //----- Draw lines to show fit range -----
  TLine* tLineMin = new TLine(aGaussFitMin, 0., aGaussFitMin, 0.25*aHist->GetMaximum());
  TLine* tLineMax = new TLine(aGaussFitMax, 0., aGaussFitMax, 0.25*aHist->GetMaximum());

  tLineMin->SetLineColor(TColor::GetColorTransparent(kRed,0.75));
  tLineMin->SetLineStyle(2);
  tLineMin->Draw();

  tLineMax->SetLineColor(TColor::GetColorTransparent(kRed,0.75));
  tLineMax->SetLineStyle(2);
  tLineMax->Draw();
  //----------------------------------------

  TPaveText* tText;
  if(aDrawTextOnRight) tText = new TPaveText(0.55, 0.50, 0.85, 0.80, "NDC");
  else                 tText = new TPaveText(0.20, 0.50, 0.45, 0.75, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(kRed);
    tText->SetTextFont(42);

    tText->AddText(TString::Format("N*exp#left(- #frac{#left(x-%s#right)^{2}}{4%s^{2}}#right)", aMuName.Data(), aSigmaName.Data()));
    tText->AddText("");
    tText->AddText(TString::Format("%s = %0.3e", aMuName.Data(), tGaussFit->GetParameter(1)));
    tText->AddText(TString::Format("%s = %0.3e",   aSigmaName.Data(), tGaussFit->GetParameter(2)));
    tText->Draw();

}

//________________________________________________________________________________________________________________
void Draw1DSourceProjwFit(TPad* aPad, TH3* a3DoslHist, TString aComponent, double aGaussFitMin=-20., double aGaussFitMax=20., double aProjLow=-100, double aProjHigh=-100)
{
  assert(aComponent.EqualTo("out") || aComponent.EqualTo("side") || aComponent.EqualTo("long"));

  int tHistType=-1;
  TString tAxisBaseNameOut, tAxisBaseNameSide, tAxisBaseNameLong;
  bool bDrawTextOnRight = false;
  if     (TString(a3DoslHist->GetName()).Contains("PairSource3d_osl")) 
  {
    tHistType=0;

    tAxisBaseNameOut  = "#it{r}*_{out}";
    tAxisBaseNameSide = "#it{r}*_{side}";
    tAxisBaseNameLong = "#it{r}*_{long}";
  }
  else if(TString(a3DoslHist->GetName()).Contains("TrueRosl")) 
  {
    tHistType=1;
    aGaussFitMin=0.;
    bDrawTextOnRight = true;

    tAxisBaseNameOut  = "#sqrt{#LT(#tilde{r}_{Out}-#beta_{T}#tilde{t})^{2}#GT}";
    tAxisBaseNameSide = "#sqrt{#LT#tilde{r}_{Side}^{2}#GT}";
    tAxisBaseNameLong = "#sqrt{#LT(#tilde{r}_{Long}-#beta_{l}#tilde{t})^{2}#GT}";
  }
  else if(TString(a3DoslHist->GetName()).Contains("SimpleRosl")) 
  {
    tHistType=2;

    tAxisBaseNameOut  = "#LTr*_{Out}#GT";
    tAxisBaseNameSide = "#LTr*_{Side}#GT";
    tAxisBaseNameLong = "#LTr*_{Long}#GT";
  }
  else;

  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  SetStandardPadMargins(aPad);

  int tBinProjLow, tBinProjHigh;
  if(aProjLow==-100. && aProjHigh==-100.)
  {
    tBinProjLow=-1;
    tBinProjHigh=-1;
  }
  else
  {
    tBinProjLow = a3DoslHist->GetXaxis()->FindBin(aProjLow);
    tBinProjHigh = a3DoslHist->GetXaxis()->FindBin(aProjHigh);
  }

  //-----------------------------------------------------------

  TH1D* t1DSource;
  if(aComponent.EqualTo("out"))
  {
    t1DSource = a3DoslHist->ProjectionX("out", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);
      t1DSource->SetTitle("PairSource_Out");
      t1DSource->GetXaxis()->SetTitle(TString::Format("%s (fm)", tAxisBaseNameOut.Data()));
      t1DSource->GetYaxis()->SetTitle(TString::Format("dN/d%s", tAxisBaseNameOut.Data()));
  }
  else if(aComponent.EqualTo("side"))
  {
    t1DSource = a3DoslHist->ProjectionY("side", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);
      t1DSource->SetTitle("PairSource_Side");
      t1DSource->GetXaxis()->SetTitle(TString::Format("%s(fm)", tAxisBaseNameSide.Data()));
      t1DSource->GetYaxis()->SetTitle(TString::Format("dN/d%s", tAxisBaseNameSide.Data()));
  }
  else if(aComponent.EqualTo("long"))
  {
    t1DSource = a3DoslHist->ProjectionZ("long", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);
    t1DSource->SetTitle("PairSource_Long");
    t1DSource->GetXaxis()->SetTitle(TString::Format("%s(fm)", tAxisBaseNameLong.Data()));
    t1DSource->GetYaxis()->SetTitle(TString::Format("dN/d%s", tAxisBaseNameLong.Data()));
  }
  else assert(0);

  t1DSource->SetMarkerStyle(20);
  t1DSource->SetMarkerSize(0.75);
  t1DSource->SetMarkerColor(kBlack);

  SetStandardAxesAttributes(t1DSource);
  if(tHistType==1)
  {
    t1DSource->GetXaxis()->SetTitleOffset(1.35);
    t1DSource->GetXaxis()->SetTitleSize(0.03);

    t1DSource->GetYaxis()->SetTitleOffset(1.25);
    t1DSource->GetYaxis()->SetTitleSize(0.0375);

    t1DSource->GetXaxis()->SetRangeUser(0., 50.);
  }

  //-----------------------------------------------------------
  TString tMuName = TString::Format("#it{#mu}_{%s}", aComponent.Data());
  TString tSigmaName = TString::Format("#it{R}_{%s}", aComponent.Data());

  if(tHistType!=1)
  {
    if(fabs(t1DSource->GetBinCenter(t1DSource->GetMaximumBin())) > 0.5)   //If peak off center, shift limits
    {
      aGaussFitMax += t1DSource->GetBinCenter(t1DSource->GetMaximumBin());
      aGaussFitMin += t1DSource->GetBinCenter(t1DSource->GetMaximumBin());
    }
  }
  DrawHistwGaussFit(aPad, t1DSource, aGaussFitMin, aGaussFitMax, tMuName, tSigmaName, bDrawTextOnRight);
}




//________________________________________________________________________________________________________________
void DrawDeltaT(TPad* aPad, TH1* aDeltaTHist, double aGaussFitMin=-20., double aGaussFitMax=20.)
{
  aPad->cd();
  SetStandardPadMargins(aPad);

  aDeltaTHist->SetMarkerStyle(20);
  aDeltaTHist->SetMarkerSize(0.75);
  aDeltaTHist->SetMarkerColor(kBlack);

  aDeltaTHist->GetXaxis()->SetTitle("#Deltat* (fm/#it{c})");
  aDeltaTHist->GetYaxis()->SetTitle("dN/#Deltat*");

  SetStandardAxesAttributes(aDeltaTHist);

  TString tMuName = "#it{#mu}_{#Delta#it{t}}";
  TString tSigmaName = "#Delta#it{t}";

  double tGaussFitMin = -20.;
  double tGaussFitMax = 20.;
  if(fabs(aDeltaTHist->GetBinCenter(aDeltaTHist->GetMaximumBin())) > 0.5)
  {
    tGaussFitMax += aDeltaTHist->GetBinCenter(aDeltaTHist->GetMaximumBin());
    tGaussFitMin += aDeltaTHist->GetBinCenter(aDeltaTHist->GetMaximumBin());
  }
  
  DrawHistwGaussFit(aPad, aDeltaTHist, tGaussFitMin, tGaussFitMax, tMuName, tSigmaName);

  //Draw line at delta_t*=0
  TLine* tLine = new TLine(0., 0., 0., aDeltaTHist->GetMaximum());
    tLine->SetLineColor(kBlack);
    tLine->SetLineStyle(1);
    tLine->SetLineWidth(2);
    tLine->Draw();

}




//________________________________________________________________________________________________________________
void Draw1DCfwFit(TPad* aPad, ThermCf* aThermCfObj, double aFitMax=0.3, bool aFixLambda=false)
{
  TH1* tThermCf = (TH1*)aThermCfObj->GetThermCf()->Clone();
  AnalysisType tAnType = aThermCfObj->GetAnalysisType();
  bool tCombConj = aThermCfObj->GetCombineConjugates();

  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  SetStandardPadMargins(aPad);


  double tRef0, tImf0, td0;
  if(tAnType == kLamKchP || tAnType == kKchPKchP || tAnType == kK0K0 || tAnType == kLamLam)
  {
    tRef0 = -0.60;
    tImf0 = 0.51;
    td0 = 0.83;
  }
  else if(tAnType == kLamKchM)
  {
    tRef0 = 0.27;
    tImf0 = 0.40;
    td0 = -5.23;
  }
  else if(tAnType == kLamK0)
  {
    tRef0 = 0.10;
    tImf0 = 0.58;
    td0 = -1.85;
  }
  else assert(0);

  int tNFitParams = 5;
  TString tFitName = TString::Format("tFitFcn_%s", cAnalysisBaseTags[tAnType]);
  TF1* tFitFcn = new TF1(tFitName, LednickyEqWithNorm,0.,0.5,tNFitParams+1);
    if(aFixLambda) tFitFcn->FixParameter(0, 1.);
    else
    {
      tFitFcn->SetParameter(0, 1.);
      tFitFcn->SetParLimits(0, 0., 1.);
    }

    tFitFcn->SetParameter(1, 5.);

    tFitFcn->FixParameter(2, tRef0);
    tFitFcn->FixParameter(3, tImf0);
    tFitFcn->FixParameter(4, td0);

    tFitFcn->SetParameter(5, 1.);

  tThermCf->Fit(tFitName, "0", "", 0.0, aFitMax);

  //-----------------------------------------------------------
  tThermCf->GetXaxis()->SetRangeUser(0., 0.329);
  tThermCf->GetYaxis()->SetRangeUser(0.86, 1.05);

  tThermCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tThermCf->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  SetStandardAxesAttributes(tThermCf);

  aPad->cd();

  tThermCf->DrawCopy();
  tFitFcn->Draw("same");

  TPaveText* tText1 = new TPaveText(0.60, 0.20, 0.70, 0.55, "NDC");
    tText1->SetFillColor(0);
    tText1->SetBorderSize(0);
    tText1->SetTextColor(kBlack);
    tText1->SetTextFont(42);
    tText1->SetTextAlign(21);
    tText1->AddText("#lambda");
    tText1->AddText("#it{R}_{inv}");
    tText1->AddText("#Rgothic#it{f}_{0}");
    tText1->AddText("#Jgothic#it{f}_{0}");
    tText1->AddText("#it{d}_{0}");

    tText1->Draw();


  TPaveText* tText2 = new TPaveText(0.70, 0.20, 0.95, 0.55, "NDC");
    tText2->SetFillColor(0);
    tText2->SetBorderSize(0);
    tText2->SetTextColor(kBlack);
    tText2->SetTextFont(42);
    tText2->SetTextAlign(11);
    tText2->AddText(TString::Format(" = % 0.2f", tFitFcn->GetParameter(0)));
    tText2->AddText(TString::Format(" = % 0.2f fm", tFitFcn->GetParameter(1)));
    tText2->AddText(TString::Format(" = % 0.2f fm", tFitFcn->GetParameter(2)));
    tText2->AddText(TString::Format(" = % 0.2f fm", tFitFcn->GetParameter(3)));
    tText2->AddText(TString::Format(" = % 0.2f fm", tFitFcn->GetParameter(4)));

    tText2->Draw();

  TPaveText* tText3;
  if(!tCombConj) tText3 = new TPaveText(0.20, 0.80, 0.40, 0.95, "NDC");
  else           tText3 = new TPaveText(0.20, 0.80, 0.70, 0.95, "NDC");
    tText3->SetFillColor(0);
    tText3->SetFillStyle(0);
    tText3->SetBorderSize(0);
    tText3->SetTextColor(kBlack);
    tText3->SetTextFont(63);
    tText3->SetTextSize(25);
    tText3->SetTextAlign(13);
  if(!tCombConj) tText3->AddText(cAnalysisRootTags[tAnType]);
  else           tText3->AddText(TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[tAnType], cAnalysisRootTags[tAnType+1]));

  tText3->Draw();

/*
  TPaveText* tText4 = new TPaveText(0.60, 0.80, 0.90, 0.95, "NDC");
    tText4->SetFillColor(0);
    tText4->SetFillStyle(0);
    tText4->SetBorderSize(0);
    tText4->SetTextColor(kBlack);
    tText4->SetTextFont(63);
    tText4->SetTextSize(25);
    tText4->SetTextAlign(13);
  tText4->AddText(TString::Format("#it{#mu}_{out} = %d fm", aMuOut));

  tText4->Draw();
*/
}

//________________________________________________________________________________________________________________
void Add1DCfwFitToCanPart(CanvasPartition* aCanPart, int aNx, int aNy, ThermCf* aThermCfObj, double aFitMax=0.3, bool aFixLambda=false, int aMuOut=0)
{
  TH1* tThermCf = (TH1*)aThermCfObj->GetThermCf()->Clone();
  AnalysisType tAnType = aThermCfObj->GetAnalysisType();
  bool tCombConj = aThermCfObj->GetCombineConjugates();

  double tRef0, tImf0, td0;
  if(tAnType == kLamKchP || tAnType == kKchPKchP || tAnType == kK0K0 || tAnType == kLamLam)
  {
    tRef0 = -0.60;
    tImf0 = 0.51;
    td0 = 0.83;
  }
  else if(tAnType == kLamKchM)
  {
    tRef0 = 0.27;
    tImf0 = 0.40;
    td0 = -5.23;
  }
  else if(tAnType == kLamK0)
  {
    tRef0 = 0.10;
    tImf0 = 0.58;
    td0 = -1.85;
  }
  else assert(0);

  int tNFitParams = 5;
  TString tFitName = TString::Format("tFitFcn_%s", cAnalysisBaseTags[tAnType]);
  TF1* tFitFcn = new TF1(tFitName, LednickyEqWithNorm,0.,0.5,tNFitParams+1);
    if(aFixLambda) tFitFcn->FixParameter(0, 1.);
    else
    {
      tFitFcn->SetParameter(0, 1.);
      tFitFcn->SetParLimits(0, 0., 1.);
    }

    tFitFcn->SetParameter(1, 5.);

    tFitFcn->FixParameter(2, tRef0);
    tFitFcn->FixParameter(3, tImf0);
    tFitFcn->FixParameter(4, td0);

    tFitFcn->SetParameter(5, 1.);

  tThermCf->Fit(tFitName, "0", "", 0.0, aFitMax);

  //-----------------------------------------------------------


  aCanPart->AddGraph(aNx, aNy, tThermCf, "", 20, kBlack, 1.0, "ex0");
  aCanPart->AddGraph(aNx, aNy, tFitFcn, "", 20, kMagenta+1, 1.0, "HIST samel");

  TPaveText* tText1 = aCanPart->SetupTPaveText("", aNx, aNy, 0.50, 0.05, 0.10, 0.55, 43, 17);
    tText1->SetFillColor(0);
    tText1->SetBorderSize(0);
    tText1->SetTextColor(kBlack);
    tText1->SetTextAlign(21);
    tText1->AddText("#lambda");
    tText1->AddText("#it{R}_{inv}");
    tText1->AddText("#Rgothic#it{f}_{0}");
    tText1->AddText("#Jgothic#it{f}_{0}");
    tText1->AddText("#it{d}_{0}");

  TPaveText* tText2 = aCanPart->SetupTPaveText("", aNx, aNy, 0.60, 0.05, 0.25, 0.55, 43, 17);
    tText2->SetFillColor(0);
    tText2->SetBorderSize(0);
    tText2->SetTextColor(kBlack);
    tText2->SetTextAlign(11);
    tText2->AddText(TString::Format(" = % 0.2f", tFitFcn->GetParameter(0)));
    tText2->AddText(TString::Format(" = % 0.2f fm", tFitFcn->GetParameter(1)));
    tText2->AddText(TString::Format(" = % 0.2f fm", tFitFcn->GetParameter(2)));
    tText2->AddText(TString::Format(" = % 0.2f fm", tFitFcn->GetParameter(3)));
    tText2->AddText(TString::Format(" = % 0.2f fm", tFitFcn->GetParameter(4)));

  TPaveText* tText3;
  if(!tCombConj) tText3 = aCanPart->SetupTPaveText(cAnalysisRootTags[tAnType], aNx, aNy, 0.05, 0.85, 0.20, 0.15, 63, 25, 13, true);
  else           tText3 = aCanPart->SetupTPaveText(TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[tAnType], cAnalysisRootTags[tAnType+1]), 
                                                   aNx, aNy, 0.05, 0.85, 0.50, 0.15, 63, 25, 13, true);
    tText3->SetTextColor(kBlack);

  TPaveText* tText4 = aCanPart->SetupTPaveText(TString::Format("#it{#mu}_{out} = %d fm", aMuOut), aNx, aNy, 0.60, 0.85, 0.30, 0.15, 43, 25, 13, true);
    tText4->SetTextColor(kBlack);


  aCanPart->AddPadPaveText(tText1, aNx, aNy);
  aCanPart->AddPadPaveText(tText2, aNx, aNy);
  aCanPart->AddPadPaveText(tText3, aNx, aNy);
  aCanPart->AddPadPaveText(tText4, aNx, aNy);
}


//________________________________________________________________________________________________________________
TCanvas* DrawCfwFitAndSources(TString tCanName, ThermCf* aThermCfObj, TH3* aSource3d, double aKStarFitMax=0.3, bool aFixLambdaInFit=true,
                              double aGaussFitMin = -20., double aGaussFitMax = 20., double aProjLow = -100., double aProjHigh = -100.)
{
  TCanvas* tCanCfwSource = new TCanvas(tCanName, tCanName);
  tCanCfwSource->Divide(2,2);

  Draw1DCfwFit((TPad*)tCanCfwSource->cd(1), aThermCfObj, aKStarFitMax, aFixLambdaInFit);
  Draw1DSourceProjwFit((TPad*)tCanCfwSource->cd(2), aSource3d, "out", aGaussFitMin, aGaussFitMax, aProjLow, aProjHigh);
  Draw1DSourceProjwFit((TPad*)tCanCfwSource->cd(3), aSource3d, "side", aGaussFitMin, aGaussFitMax, aProjLow, aProjHigh);
  Draw1DSourceProjwFit((TPad*)tCanCfwSource->cd(4), aSource3d, "long", aGaussFitMin, aGaussFitMax, aProjLow, aProjHigh);

  return tCanCfwSource;
}

//________________________________________________________________________________________________________________
TCanvas* DrawCfwFitAndSourceswDeltaT(TString tCanName, ThermCf* aThermCfObj, TH3* aSource3d, TH1* aDeltaTHist, double aKStarFitMax=0.3, bool aFixLambdaInFit=true,
                                     double aGaussFitMin = -20., double aGaussFitMax = 20., double aProjLow = -100., double aProjHigh = -100.)
{
  TCanvas* tCanCfwSource = new TCanvas(tCanName, tCanName);
  //tCanCfwSource->Divide(2,2);
  tCanCfwSource->SetCanvasSize(700, 750);
  tCanCfwSource->cd();
  TPad* tPadCfwFit = new TPad("tPadCfwFit", "tPadCfwFit", 0.2, 0.67, 0.8, 1.0);
    tPadCfwFit->Draw();
  TPad* tPadRout = new TPad("tPadRout", "tPadRout", 0.0, 0.33, 0.5, 0.67);
    tPadRout->Draw();
  TPad* tPadRside = new TPad("tPadRside", "tPadRside", 0.5, 0.33, 1.0, 0.67);
    tPadRside->Draw();
  TPad* tPadRlong = new TPad("tPadRlong", "tPadRlong", 0.0, 0.0, 0.5, 0.33);
    tPadRlong->Draw();
  TPad* tPadDeltaT = new TPad("tPadDeltaT", "tPadDeltaT", 0.5, 0.0, 1.0, 0.33);
    tPadDeltaT->Draw();



  Draw1DCfwFit(tPadCfwFit, aThermCfObj, aKStarFitMax, aFixLambdaInFit);
  Draw1DSourceProjwFit(tPadRout, aSource3d, "out", aGaussFitMin, aGaussFitMax, aProjLow, aProjHigh);
  Draw1DSourceProjwFit(tPadRside, aSource3d, "side", aGaussFitMin, aGaussFitMax, aProjLow, aProjHigh);
  Draw1DSourceProjwFit(tPadRlong, aSource3d, "long", aGaussFitMin, aGaussFitMax, aProjLow, aProjHigh);


  double tGaussFitMin = aGaussFitMin;
  double tGaussFitMax = aGaussFitMax;
  if(fabs(aDeltaTHist->GetBinCenter(aDeltaTHist->GetMaximumBin())) > 0.5)  //If peak off center, shift limits
  {
    tGaussFitMax += aDeltaTHist->GetBinCenter(aDeltaTHist->GetMaximumBin());
    tGaussFitMin += aDeltaTHist->GetBinCenter(aDeltaTHist->GetMaximumBin());
  } 
  DrawDeltaT(tPadDeltaT, aDeltaTHist, tGaussFitMin, tGaussFitMax);

  return tCanCfwSource;
}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
//TODO based off file with same name in LednickyFitter directory

  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  AnalysisType tAnType = kLamKchP;
  if(tAnType==kLamKchM) gRejectPoints=true;

  bool bCombineConjugates = true;
  bool bDrawDeltaT = true;
  bool bDrawCompareMuOuts = true;
  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";
//  TString tSaveDir = "/home/jesse/Analysis/Presentations/GroupMeetings/20190117/Figures/";
  TString tSaveDir = TString::Format("/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/7_ResultsAndDiscussion/7.1_ResultsLamK/7.1.5_ResultsLamK_DiscussionOfmTScaling/ThermPlots/%s/", cAnalysisBaseTags[tAnType]);
  if(bSaveFigures) gSystem->mkdir(tSaveDir, true);

  int tRebin=1;
  double tMinNorm = /*0.80*//*0.80*/0.32;
  double tMaxNorm = /*0.99*//*0.99*/0.40;

  int tImpactParam = 2;
  TString aCfDescriptor = "Full";
//  TString aCfDescriptor = "PrimaryOnly";


  TString tFileNameBase = "CorrelationFunctions_BuildCfYlm";
  TString tFileNameModifier = "";

  if(tFileNameBase.Contains("DrawRStarFromGaussian")) bDrawDeltaT=false;

  //--------------------------------------------

  TString tFileName = TString::Format("%s%s.root", tFileNameBase.Data(), tFileNameModifier.Data());

  TString tFileDir = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
  TString tFileLocation = TString::Format("%s%s", tFileDir.Data(), tFileName.Data());

  TString t3dDescriptor;
  if(aCfDescriptor.EqualTo("Full")) t3dDescriptor = TString("");
  else t3dDescriptor=aCfDescriptor;

  TString tHistName3d = TString::Format("PairSource3d_osl%s%s", t3dDescriptor.Data(), cAnalysisBaseTags[tAnType]);
  TString tHistName3dConj = TString::Format("PairSource3d_osl%s%s", t3dDescriptor.Data(), cAnalysisBaseTags[tAnType+1]);

  if(tAnType==kKchPKchP || tAnType==kK0K0 || tAnType==kLamLam) bCombineConjugates = false;
  if(tFileNameBase.Contains("PairOnly")) bCombineConjugates = false;

  //--------------------------------------------

  //ThermCf already knows the default directory, so it only needs the name of the file, not the complete path
  ThermCf* tThermCfObj = new ThermCf(tFileName, aCfDescriptor, tAnType, kMB, bCombineConjugates, kMe, tRebin, tMinNorm, tMaxNorm, false, false, tImpactParam);
  TH3* tTest3d;
  if(!bCombineConjugates) tTest3d = GetThermHist3d(tFileLocation, tHistName3d);
  else tTest3d = GetThermHist3d_CombConj(tFileLocation, tHistName3d, tHistName3dConj);

  TH1* tDeltaTHist;
  if(bDrawDeltaT)
  {
    TString tHistNameDeltaT;
    if(aCfDescriptor.EqualTo("Full")) tHistNameDeltaT = TString::Format("PairDeltaT_inPRF%s", cAnalysisBaseTags[tAnType]);
    else                              tHistNameDeltaT = TString::Format("PairDeltaT_inPRF%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
    tDeltaTHist = GetThermHist1d(tFileLocation, tHistNameDeltaT);
  }

  double tGaussFitMin = -20.;
  double tGaussFitMax = 20.;

  double tProjLow = -100.;
  double tProjHigh = -100.;

  bool tFixLambdaInFit = true;
  double tKStarFitMax = 0.3;

  TCanvas* tCanCfwSource;
  TString tCanCfwSourceName;
  if(!bDrawDeltaT) 
  {
    tCanCfwSourceName = TString::Format("CanCfwSource_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
    tCanCfwSource = DrawCfwFitAndSources(tCanCfwSourceName, tThermCfObj, tTest3d, tKStarFitMax, tFixLambdaInFit,
                                         tGaussFitMin, tGaussFitMax, tProjLow, tProjHigh);
  }
  else 
  {
    tCanCfwSourceName = TString::Format("CanCfwSourceAndDeltaT_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
    tCanCfwSource = DrawCfwFitAndSourceswDeltaT(tCanCfwSourceName, tThermCfObj, tTest3d, tDeltaTHist, tKStarFitMax, tFixLambdaInFit,
                                                tGaussFitMin, tGaussFitMax, tProjLow, tProjHigh);
  }



//  if(bSaveFigures) tCanCfwSource->SaveAs(TString::Format("%s%s_%s_%s.%s", tSaveDir.Data(), tFileNameBase.Data(), aCfDescriptor.Data(), cAnalysisBaseTags[tAnType], tSaveFileType.Data()));
  if(bSaveFigures) tCanCfwSource->SaveAs(TString::Format("%s%s_3dHist%s_FromFile%s.%s", tSaveDir.Data(), tCanCfwSourceName.Data(), tHistName3d.Data(), tFileNameBase.Data(), tSaveFileType.Data()));
  //-------------------------------------------------------------------------------

  if(bDrawCompareMuOuts)
  {
    TString tFileNameBase_Mu0 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_0.0_0.0_0.0_BuildCfYlm";
    TString tFileNameBase_Mu1 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_1.0_0.0_0.0_BuildCfYlm";
    TString tFileNameBase_Mu3 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_3.0_0.0_0.0_BuildCfYlm";
    TString tFileNameBase_Mu6 = "CorrelationFunctions_DrawRStarFromGaussian_5.0_5.0_5.0_6.0_0.0_0.0_BuildCfYlm";

    //-----
    TString tFileName_Mu0 = TString::Format("%s%s.root", tFileNameBase_Mu0.Data(), tFileNameModifier.Data());
    TString tFileName_Mu1 = TString::Format("%s%s.root", tFileNameBase_Mu1.Data(), tFileNameModifier.Data());
    TString tFileName_Mu3 = TString::Format("%s%s.root", tFileNameBase_Mu3.Data(), tFileNameModifier.Data());
    TString tFileName_Mu6 = TString::Format("%s%s.root", tFileNameBase_Mu6.Data(), tFileNameModifier.Data());

    //-----
    TString tFileLocation_Mu0 = TString::Format("%s%s", tFileDir.Data(), tFileName_Mu0.Data());
    TString tFileLocation_Mu1 = TString::Format("%s%s", tFileDir.Data(), tFileName_Mu1.Data());
    TString tFileLocation_Mu3 = TString::Format("%s%s", tFileDir.Data(), tFileName_Mu3.Data());
    TString tFileLocation_Mu6 = TString::Format("%s%s", tFileDir.Data(), tFileName_Mu6.Data());

    //--------------------------------------------
    //ThermCf already knows the default directory, so it only needs the name of the file, not the complete path
    ThermCf* tThermCfObj_Mu0 = new ThermCf(tFileName_Mu0, aCfDescriptor, tAnType, kMB, bCombineConjugates, kMe, tRebin, tMinNorm, tMaxNorm, false, false, tImpactParam);
    TH3* tTest3d_Mu0;
    if(!bCombineConjugates) tTest3d_Mu0 = GetThermHist3d(tFileLocation_Mu0, tHistName3d);
    else tTest3d_Mu0 = GetThermHist3d_CombConj(tFileLocation_Mu0, tHistName3d, tHistName3dConj);

    ThermCf* tThermCfObj_Mu1 = new ThermCf(tFileName_Mu1, aCfDescriptor, tAnType, kMB, bCombineConjugates, kMe, tRebin, tMinNorm, tMaxNorm, false, false, tImpactParam);
    TH3* tTest3d_Mu1;
    if(!bCombineConjugates) tTest3d_Mu1 = GetThermHist3d(tFileLocation_Mu1, tHistName3d);
    else tTest3d_Mu0 = GetThermHist3d_CombConj(tFileLocation_Mu0, tHistName3d, tHistName3dConj);

    ThermCf* tThermCfObj_Mu3 = new ThermCf(tFileName_Mu3, aCfDescriptor, tAnType, kMB, bCombineConjugates, kMe, tRebin, tMinNorm, tMaxNorm, false, false, tImpactParam);
    TH3* tTest3d_Mu3;
    if(!bCombineConjugates) tTest3d_Mu3 = GetThermHist3d(tFileLocation_Mu3, tHistName3d);
    else tTest3d_Mu0 = GetThermHist3d_CombConj(tFileLocation_Mu0, tHistName3d, tHistName3dConj);

    ThermCf* tThermCfObj_Mu6 = new ThermCf(tFileName_Mu6, aCfDescriptor, tAnType, kMB, bCombineConjugates, kMe, tRebin, tMinNorm, tMaxNorm, false, false, tImpactParam);
    TH3* tTest3d_Mu6;
    if(!bCombineConjugates) tTest3d_Mu6 = GetThermHist3d(tFileLocation_Mu6, tHistName3d);
    else tTest3d_Mu0 = GetThermHist3d_CombConj(tFileLocation_Mu0, tHistName3d, tHistName3dConj);


    //--------------------------------------------

    TString tCanCompMusName = TString::Format("CanCompMus_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
    TCanvas* tCanCompMus = new TCanvas(tCanCompMusName, tCanCompMusName);

    tCanCompMus->Divide(2,2);

    Draw1DCfwFit((TPad*)tCanCompMus->cd(1), tThermCfObj_Mu0, tKStarFitMax, tFixLambdaInFit);
    Draw1DCfwFit((TPad*)tCanCompMus->cd(2), tThermCfObj_Mu1, tKStarFitMax, tFixLambdaInFit);
    Draw1DCfwFit((TPad*)tCanCompMus->cd(3), tThermCfObj_Mu3, tKStarFitMax, tFixLambdaInFit);
    Draw1DCfwFit((TPad*)tCanCompMus->cd(4), tThermCfObj_Mu6, tKStarFitMax, tFixLambdaInFit);


    if(bSaveFigures) tCanCompMus->SaveAs(TString::Format("%s%s_3dHist%s.%s", tSaveDir.Data(), tCanCompMusName.Data(), tHistName3d.Data(), tSaveFileType.Data()));
    //--------------------------------------------
    int tNx=2, tNy=2;
    double tXLow = -0.02;
    double tXHigh = 0.329;
    double tYLow = 0.82;
    double tYHigh = 1.07;

    TString tCanPartCompMusName = TString::Format("CanPartCompMus_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
    CanvasPartition* tCanPart = new CanvasPartition(tCanPartCompMusName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.13,0.0025);
    tCanPart->SetDrawOptStat(false);

    Add1DCfwFitToCanPart(tCanPart, 0, 0, tThermCfObj_Mu0, tKStarFitMax, tFixLambdaInFit, 0);
    Add1DCfwFitToCanPart(tCanPart, 1, 0, tThermCfObj_Mu1, tKStarFitMax, tFixLambdaInFit, 1);
    Add1DCfwFitToCanPart(tCanPart, 0, 1, tThermCfObj_Mu3, tKStarFitMax, tFixLambdaInFit, 3);
    Add1DCfwFitToCanPart(tCanPart, 1, 1, tThermCfObj_Mu6, tKStarFitMax, tFixLambdaInFit, 6);

    tCanPart->DrawAll();
    tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
    tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

    //--------------------------------------------
    TString tCanCompMusNoFitsName = TString::Format("CanCompMusNoFits_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
    TCanvas* tCanCompMusNoFits = new TCanvas(tCanCompMusNoFitsName, tCanCompMusNoFitsName);
    tCanCompMusNoFits->cd();

    TH1* tThermCf_Mu0 = (TH1*)tThermCfObj_Mu0->GetThermCf()->Clone();
    TH1* tThermCf_Mu1 = (TH1*)tThermCfObj_Mu1->GetThermCf()->Clone();
    TH1* tThermCf_Mu3 = (TH1*)tThermCfObj_Mu3->GetThermCf()->Clone();
    TH1* tThermCf_Mu6 = (TH1*)tThermCfObj_Mu6->GetThermCf()->Clone();

    tThermCf_Mu0->GetXaxis()->SetRangeUser(tXLow, tXHigh);
    tThermCf_Mu0->GetYaxis()->SetRangeUser(tYLow, tYHigh);


    tThermCf_Mu0->SetMarkerColor(kBlack);
    tThermCf_Mu0->SetLineColor(kBlack);
    tThermCf_Mu0->SetMarkerStyle(20);

    tThermCf_Mu1->SetMarkerColor(kBlue);
    tThermCf_Mu1->SetLineColor(kBlue);
    tThermCf_Mu1->SetMarkerStyle(20);

    tThermCf_Mu3->SetMarkerColor(kGreen);
    tThermCf_Mu3->SetLineColor(kGreen);
    tThermCf_Mu3->SetMarkerStyle(20);

    tThermCf_Mu6->SetMarkerColor(kRed);
    tThermCf_Mu6->SetLineColor(kRed);
    tThermCf_Mu6->SetMarkerStyle(20);


    tThermCf_Mu0->Draw();
    tThermCf_Mu1->Draw("same");
    tThermCf_Mu3->Draw("same");
    tThermCf_Mu6->Draw("same");
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
