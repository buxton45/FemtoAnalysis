#include "UsefulMacros.C"

Bool_t reject;
Double_t ffBgFitLow[2], ffBgFitHigh[2];
//_________________________________________________________________________________________________________________
Double_t PurityBgFitFunction(Double_t *x, Double_t *par)
{
  if( reject && !(x[0]>ffBgFitLow[0] && x[0]<ffBgFitLow[1]) && !(x[0]>ffBgFitHigh[0] && x[0]<ffBgFitHigh[1]) )
    {
      TF1::RejectPoint();
      return 0;
    }
  
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
}
//_________________________________________________________________________________________________________________
TList* BuildPurityBgFit(char* name, TH1F* PurityHisto, Double_t BgFitLow[2], Double_t BgFitHigh[2], Double_t ROI[2], Double_t *info)
{
  char buffer[50];
  sprintf(buffer, "fitBgd_%s",name);

  reject = kTRUE;
  ffBgFitLow[0] = BgFitLow[0];
  ffBgFitLow[1] = BgFitLow[1];
  ffBgFitHigh[0] = BgFitHigh[0];
  ffBgFitHigh[1] = BgFitHigh[1];
  TF1 *fitBgd = new TF1("fitBgd",PurityBgFitFunction,PurityHisto->GetBinLowEdge(1),PurityHisto->GetBinLowEdge(PurityHisto->GetNbinsX()+1),5);
  PurityHisto->Fit("fitBgd","0");

  reject = kFALSE;
  TF1 *fitBgd2 = new TF1(buffer,PurityBgFitFunction,PurityHisto->GetBinLowEdge(1),PurityHisto->GetBinLowEdge(PurityHisto->GetNbinsX()+1),5);
  fitBgd2->SetParameters(fitBgd->GetParameters());

  //--------------------------------------------------------------------------------------------
  Double_t bgd = fitBgd2->Integral(ROI[0],ROI[1]);
  bgd /= PurityHisto->GetBinWidth(0);  //divide by bin size
  cout << name << ": " << "bgd = " << bgd << endl;
  //-----
  Double_t sigpbgd = PurityHisto->Integral(PurityHisto->FindBin(ROI[0]),PurityHisto->FindBin(ROI[1]));
  cout << name << ": " << "sig+bgd = " << sigpbgd << endl;
  //-----
  Double_t sig = sigpbgd-bgd;
  cout << name << ": " << "sig = " << sig << endl;
  //-----
  Double_t pur = sig/sigpbgd;
  cout << name << ": " << "Pur = " << pur << endl << endl;

  info[0] = bgd;
  info[1] = sigpbgd;
  info[2] = sig;
  info[3] = pur;
  //--------------------------------------------------------------------------------------------
  Double_t HistoMaxValue = PurityHisto->GetMaximum();
  TLine *lROImin = new TLine(ROI[0],0,ROI[0],HistoMaxValue);
  TLine *lROImax = new TLine(ROI[1],0,ROI[1],HistoMaxValue);
  //-----
  TLine *lBgFitLowMin = new TLine(BgFitLow[0],0,BgFitLow[0],HistoMaxValue);
  TLine *lBgFitLowMax = new TLine(BgFitLow[1],0,BgFitLow[1],HistoMaxValue);
  TLine *lBgFitHighMin = new TLine(BgFitHigh[0],0,BgFitHigh[0],HistoMaxValue);
  TLine *lBgFitHighMax = new TLine(BgFitHigh[1],0,BgFitHigh[1],HistoMaxValue);
  //-----
  TList* temp = new TList();
  temp->Add(fitBgd2);
  temp->Add(lROImin);
  temp->Add(lROImax);
  temp->Add(lBgFitLowMin);
  temp->Add(lBgFitLowMax);
  temp->Add(lBgFitHighMin);
  temp->Add(lBgFitHighMax);

  return temp;

}
//_________________________________________________________________________________________________________________
void PurityDrawAll(TH1F* PurityHisto, TList* FitList, Double_t info[4])
{
  PurityHisto->SetLineColor(1);
  PurityHisto->SetLineWidth(3);

  TIter iter(FitList);
  //-----
  TF1* fitBgd = *(iter.Begin());
    fitBgd->SetLineColor(4);
  //-----
  TLine* lROImin = (TLine*)iter.Next();
  TLine* lROImax = (TLine*)iter.Next();
    lROImin->SetLineColor(3);
    lROImax->SetLineColor(3);
  //-----
  TLine* lBgFitLowMin = (TLine*)iter.Next();
  TLine* lBgFitLowMax = (TLine*)iter.Next();
  TLine* lBgFitHighMin = (TLine*)iter.Next();
  TLine* lBgFitHighMax = (TLine*)iter.Next();
    lBgFitLowMin->SetLineColor(2);
    lBgFitLowMax->SetLineColor(2);
    lBgFitHighMin->SetLineColor(2);
    lBgFitHighMax->SetLineColor(2);
  //-----
  TAxis *xax = PurityHisto->GetXaxis();
    xax->SetTitle("k* (GeV/c)");
    xax->SetTitleSize(0.05);
    xax->SetTitleOffset(0.7);
  TAxis *yax = PurityHisto->GetYaxis();
    yax->SetTitle("dN/dM_{p#pi-}");
    yax->SetTitleSize(0.06);
    yax->SetTitleOffset(0.8);
  //-----
  PurityHisto->DrawCopy("Ehist");
  fitBgd->Draw("same");
  lROImin->Draw();
  lROImax->Draw();
  lBgFitLowMin->Draw();
  lBgFitLowMax->Draw();
  lBgFitHighMin->Draw();
  lBgFitHighMax->Draw();

  TPaveText *myText = new TPaveText(0.12,0.65,0.42,0.85,"NDC");
  char buffer[50];
  Double_t purity = info[3];
  char title[20] = PurityHisto->GetName();
  sprintf(buffer, "%s = %.2f\%",title, 100.*purity);
  myText->SetFillColor(0);
  myText->SetBorderSize(1);
  myText->SetTextSize(0.05);
  myText->AddText(buffer);
  myText->Draw();

  TLegend *myLegend = new TLegend(0.65,0.50,0.89,0.89);
  myLegend->SetFillColor(0);
  myLegend->AddEntry(PurityHisto, "Data", "lp");
  myLegend->AddEntry(fitBgd, "Fit to background", "l");
  myLegend->AddEntry(lROImin, "Accepted window", "l");
  myLegend->AddEntry(lBgFitLowMin, "Bkg fit window", "l");
  myLegend->Draw();
}
//_________________________________________________________________________________________________________________
void PurityDrawBg(TH1F* PurityHisto, TList* FitList, Double_t BgFitLow[2], Double_t BgFitHigh[2])
{
  PurityHisto->GetXaxis()->SetRange(PurityHisto->FindBin(BgFitLow[0]),PurityHisto->FindBin(BgFitLow[1]));
    Double_t maxLow = PurityHisto->GetMaximum();
    Double_t minLow = PurityHisto->GetMinimum();
  PurityHisto->GetXaxis()->SetRange(PurityHisto->FindBin(BgFitHigh[0]),PurityHisto->FindBin(BgFitHigh[1])-1);
    Double_t maxHigh = PurityHisto->GetMaximum();
    Double_t minHigh = PurityHisto->GetMinimum();
  Double_t maxBg;
    if(maxLow>maxHigh) maxBg = maxLow;
    else maxBg = maxHigh;
//    cout << "Background max = " << maxBg << endl;
  Double_t minBg;
    if(minLow<minHigh) minBg = minLow;
    else minBg = minHigh;
//    cout << "Background min = " << minBg << endl;
  //--Extend the y-range that I plot

  Double_t rangeBg = maxBg-minBg;
  maxBg+=rangeBg/10.;
  minBg-=rangeBg/10.;

  PurityHisto->GetXaxis()->SetTitle("");
  PurityHisto->GetXaxis()->SetLabelSize(0.08);
  PurityHisto->GetYaxis()->SetTitle("");
  PurityHisto->GetXaxis()->SetRange(1,PurityHisto->GetNbinsX());
  PurityHisto->GetYaxis()->SetRangeUser(minBg,maxBg);
  PurityHisto->SetLineColor(1);
  PurityHisto->SetLineWidth(3);

  TIter iter(FitList);
  //-----
  TF1* fitBgd = *(iter.Begin());
    fitBgd->SetLineColor(4);
  //-----
  TLine* lROImin = (TLine*)iter.Next();
  TLine* lROImax = (TLine*)iter.Next();
    lROImin->SetLineColor(3);
    lROImax->SetLineColor(3);
  //-----
  TLine* lBgFitLowMin = (TLine*)iter.Next();
  TLine* lBgFitLowMax = (TLine*)iter.Next();
  TLine* lBgFitHighMin = (TLine*)iter.Next();
  TLine* lBgFitHighMax = (TLine*)iter.Next();
    lBgFitLowMin->SetLineColor(2);
    lBgFitLowMax->SetLineColor(2);
    lBgFitHighMin->SetLineColor(2);
    lBgFitHighMax->SetLineColor(2);
  //-----
  PurityHisto->Draw("Ehist");
  fitBgd->Draw("same");
  lROImin->Draw();
  lROImax->Draw();
  lBgFitLowMin->Draw();
  lBgFitLowMax->Draw();
  lBgFitHighMin->Draw();
  lBgFitHighMax->Draw();
}
//_________________________________________________________________________________________________________________
//___***___***___***___***___***___***___***___***___***___***___***___***___***___***___***___***___***___***___***
//_________________________________________________________________________________________________________________
void calcPurInd()
{
  const Double_t LambdaMass = 1.115683, KaonMass = 0.493677;

  TString FileName = "Resultsgrid_cLamK0_Bp1.root";

  //------------------------------Lambdas---------------------------------------------------
  TH1F *LambdaPurity = GetHisto(FileName,"LamK0","LambdaPurity");

  Double_t LamInfo[4];  //will hold Bgd, SigpBgd, Sig, and Pur
  Double_t LamBgFitLow[2];
    //LamBgFitLow[0] = LambdaPurity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    LamBgFitLow[0] = 1.09;
    LamBgFitLow[1] = 1.102;
  Double_t LamBgFitHigh[2];
    LamBgFitHigh[0] = 1.130;
    LamBgFitHigh[1] = LambdaPurity->GetBinLowEdge(LambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  Double_t LamROI[2];
    LamROI[0] = LambdaMass-0.0038;
    LamROI[1] = LambdaMass+0.0038;
  TList* LamList = BuildPurityBgFit("Lambda",LambdaPurity,LamBgFitLow,LamBgFitHigh,LamROI,LamInfo);
  TF1* fitBgd_Lambda = LamList->FindObject("fitBgd_Lambda");

  //------------------------------K0Shorts(1)------------------------------------------------
  TH1F *K0ShortPurity1 = GetHisto(FileName,"LamK0","K0ShortPurity1");

  Double_t K01Info[4];  //will hold Bgd, SigpBgd, Sig, and Pur
  Double_t K01BgFitLow[2];
    K01BgFitLow[0] = K0ShortPurity1->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    K01BgFitLow[1] = 0.452;
  Double_t K01BgFitHigh[2];
    K01BgFitHigh[0] = 0.536;
    K01BgFitHigh[1] = K0ShortPurity1->GetBinLowEdge(K0ShortPurity1->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  Double_t K01ROI[2];
    K01ROI[0] = KaonMass-0.013677;
    K01ROI[1] = KaonMass+0.020323;
  TList* K01List = BuildPurityBgFit("K0Short1",K0ShortPurity1,K01BgFitLow,K01BgFitHigh,K01ROI,K01Info);
  TF1* fitBgd_K0Short1 = K01List->FindObject("fitBgd_K0Short1");

  //------------------------------AntiLambdas------------------------------------------------
  TH1F *AntiLambdaPurity = GetHisto(FileName,"ALamK0","AntiLambdaPurity");

  Double_t ALamInfo[4];  //will hold Bgd, SigpBgd, Sig, and Pur
  Double_t ALamBgFitLow[2];
    //ALamBgFitLow[0] = AntiLambdaPurity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    ALamBgFitLow[0] = 1.090;
    ALamBgFitLow[1] = 1.102;
  Double_t ALamBgFitHigh[2];
    ALamBgFitHigh[0] = 1.130;
    ALamBgFitHigh[1] = AntiLambdaPurity->GetBinLowEdge(AntiLambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  Double_t ALamROI[2];
    ALamROI[0] = LambdaMass-0.0038;
    ALamROI[1] = LambdaMass+0.0038;
  TList* ALamList = BuildPurityBgFit("AntiLambda",AntiLambdaPurity,ALamBgFitLow,ALamBgFitHigh,ALamROI,ALamInfo);
  TF1* fitBgd_AntiLambda = ALamList->FindObject("fitBgd_AntiLambda");

  //------------------------------K0Shorts(2)------------------------------------------------
  TH1F *K0ShortPurity2 = GetHisto(FileName,"ALamK0","K0ShortPurity1");

  Double_t K02Info[4];  //will hold Bgd, SigpBgd, Sig, and Pur
  Double_t K02BgFitLow[2];
    K02BgFitLow[0] = K0ShortPurity2->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    K02BgFitLow[1] = 0.452;
  Double_t K02BgFitHigh[2];
    K02BgFitHigh[0] = 0.536;
    K02BgFitHigh[1] = K0ShortPurity2->GetBinLowEdge(K0ShortPurity2->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  Double_t K02ROI[2];
    K02ROI[0] = KaonMass-0.013677;
    K02ROI[1] = KaonMass+0.020323;
  TList* K02List = BuildPurityBgFit("K0Short2",K0ShortPurity2,K02BgFitLow,K02BgFitHigh,K02ROI,K02Info);
  TF1* fitBgd_K0Short2 = K02List->FindObject("fitBgd_K0Short2");

//-----------------------------------------------------------------------------
  TString OutputName = "calcPurInd.Lambda_Bp1.pdf";

  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
//  c1->Divide(2,4);
  gStyle->SetOptStat(0);

  TPad *p1 = new TPad("p1","p1",0.0,0.3,1.0,1.0);
  p1->Draw();
  TPad *p2 = new TPad("p2","p2",0.0,0.0,1.0,0.3);
  p2->Draw();


  //-----
  p1->cd();
  LambdaPurity->SetTitle("");
  PurityDrawAll(LambdaPurity,LamList,LamInfo);
  p2->cd();
  p2->SetTopMargin(0.0);
  p2->SetBottomMargin(0.07);
  PurityDrawBg(LambdaPurity,LamList,LamBgFitLow,LamBgFitHigh);
  //-----

/*
  //-----
  p1->cd();
  K0ShortPurity1->SetTitle("");
  PurityDrawAll(K0ShortPurity1,K01List,K01Info);
  p2->cd();
  p2->SetTopMargin(0.0);
  p2->SetBottomMargin(0.07);
  PurityDrawBg(K0ShortPurity1,K01List,K01BgFitLow,K01BgFitHigh);
  //-----
*/
/*
  //-----
  p1->cd();
  AntiLambdaPurity->SetTitle("");
  PurityDrawAll(AntiLambdaPurity,ALamList,ALamInfo);
  p2->cd();
  p2->SetTopMargin(0.0);
  p2->SetBottomMargin(0.07);
  PurityDrawBg(AntiLambdaPurity,ALamList,ALamBgFitLow,ALamBgFitHigh);
  //-----
*/
  c1->SaveAs(OutputName);

}
