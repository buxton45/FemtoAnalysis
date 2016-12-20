#include "UsefulMacros.C"

Bool_t reject;
double ffBgFitLow[2], ffBgFitHigh[2];
//_________________________________________________________________________________________________________________
double PurityBgFitFunction(double *x, double *par)
{
  if( reject && !(x[0]>ffBgFitLow[0] && x[0]<ffBgFitLow[1]) && !(x[0]>ffBgFitHigh[0] && x[0]<ffBgFitHigh[1]) )
    {
      TF1::RejectPoint();
      return 0;
    }
  
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
}
//_________________________________________________________________________________________________________________
TList* BuildPurityBgFit(char* name, TH1F* PurityHisto, double BgFitLow[2], double BgFitHigh[2], double ROI[2])
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
  double bgd = fitBgd2->Integral(ROI[0],ROI[1]);
  bgd /= PurityHisto->GetBinWidth(0);  //divide by bin size
  cout << name << ": " << "bgd = " << bgd << endl;
  //-----
  double sigpbgd = PurityHisto->Integral(PurityHisto->FindBin(ROI[0]),PurityHisto->FindBin(ROI[1]));
  cout << name << ": " << "sig+bgd = " << sigpbgd << endl;
  //-----
  double sig = sigpbgd-bgd;
  cout << name << ": " << "sig = " << sig << endl;
  //-----
  double pur = sig/sigpbgd;
  cout << name << ": " << "Pur = " << pur << endl << endl;

  TVectorD *vInfo = new TVectorD(4);
    vInfo(0) = bgd;
    vInfo(1) = sigpbgd;
    vInfo(2) = sig;
    vInfo(3) = pur;

  TVectorD *vROI = new TVectorD(2);
    vROI(0) = ROI[0];
    vROI(1) = ROI[1];

  TVectorD *vBgFitLow = new TVectorD(2);
    vBgFitLow(0) = BgFitLow[0];
    vBgFitLow(1) = BgFitLow[1];

  TVectorD *vBgFitHigh = new TVectorD(2);
    vBgFitHigh(0) = BgFitHigh[0];
    vBgFitHigh(1) = BgFitHigh[1];
  //--------------------------------------------------------------------------------------------
  TList* temp = new TList();
  temp->Add(fitBgd2);
  temp->Add(vInfo);
  temp->Add(vROI);
  temp->Add(vBgFitLow);
  temp->Add(vBgFitHigh);
  return temp;

}
//_________________________________________________________________________________________________________________
void PurityDrawAll(TH1F* PurityHisto, TList* FitList)
{
  PurityHisto->SetLineColor(1);
  PurityHisto->SetLineWidth(3);

  TIter iter(FitList);
  //-----
  TF1* fitBgd = *(iter.Begin());
    fitBgd->SetLineColor(4);
  //-----
  TVectorD* vInfo = (TVectorD*)iter.Next();
  //-----
  TVectorD* vROI = (TVectorD*)iter.Next();
  //-----
  TVectorD* vBgFitLow = (TVectorD*)iter.Next();
  //-----
  TVectorD* vBgFitHigh = (TVectorD*)iter.Next();
  //--------------------------------------------------------------------------------------------
  double HistoMaxValue = PurityHisto->GetMaximum();
  TLine *lROImin = new TLine(vROI(0),0,vROI(0),HistoMaxValue);
  TLine *lROImax = new TLine(vROI(1),0,vROI(1),HistoMaxValue);
    lROImin->SetLineColor(3);
    lROImax->SetLineColor(3);
  //-----
  TLine *lBgFitLowMin = new TLine(vBgFitLow(0),0,vBgFitLow(0),HistoMaxValue);
  TLine *lBgFitLowMax = new TLine(vBgFitLow(1),0,vBgFitLow(1),HistoMaxValue);
    lBgFitLowMin->SetLineColor(2);
    lBgFitLowMax->SetLineColor(2);
  //-----
  TLine *lBgFitHighMin = new TLine(vBgFitHigh(0),0,vBgFitHigh(0),HistoMaxValue);
  TLine *lBgFitHighMax = new TLine(vBgFitHigh(1),0,vBgFitHigh(1),HistoMaxValue);
    lBgFitHighMin->SetLineColor(2);
    lBgFitHighMax->SetLineColor(2);
  //--------------------------------------------------------------------------------------------
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
  double purity = vInfo(3);
  char title[20] = PurityHisto->GetName();
  sprintf(buffer, "%s = %.2f\%",title, 100.*purity);
  myText->AddText(buffer);
  myText->Draw();
}
//_________________________________________________________________________________________________________________
void PurityDrawBg(TH1F* PurityHisto, TList* FitList)
{
  TIter iter(FitList);
  //-----
  TF1* fitBgd = *(iter.Begin());
    fitBgd->SetLineColor(4);
  //-----
  TVectorD* vInfo = (TVectorD*)iter.Next();
  //-----
  TVectorD* vROI = (TVectorD*)iter.Next();
  //-----
  TVectorD* vBgFitLow = (TVectorD*)iter.Next();
  //-----
  TVectorD* vBgFitHigh = (TVectorD*)iter.Next();
  //--------------------------------------------------------------------------------------------
  PurityHisto->GetXaxis()->SetRange(PurityHisto->FindBin(vBgFitLow(0)),PurityHisto->FindBin(vBgFitLow(1)));
    double maxLow = PurityHisto->GetMaximum();
    double minLow = PurityHisto->GetMinimum();
  PurityHisto->GetXaxis()->SetRange(PurityHisto->FindBin(vBgFitHigh(0)),PurityHisto->FindBin(vBgFitHigh(1))-1);
    double maxHigh = PurityHisto->GetMaximum();
    double minHigh = PurityHisto->GetMinimum();
  double maxBg;
    if(maxLow>maxHigh) maxBg = maxLow;
    else maxBg = maxHigh;
    //cout << "Background max = " << maxBg << endl;
  double minBg;
    if(minLow<minHigh) minBg = minLow;
    else minBg = minHigh;
    //cout << "Background min = " << minBg << endl;
  //--Extend the y-range that I plot

  double rangeBg = maxBg-minBg;
  maxBg+=rangeBg/10.;
  minBg-=rangeBg/10.;

  PurityHisto->GetXaxis()->SetRange(1,PurityHisto->GetNbinsX());
  PurityHisto->GetYaxis()->SetRangeUser(minBg,maxBg);
  PurityHisto->SetLineColor(1);
  PurityHisto->SetLineWidth(3);
  //--------------------------------------------------------------------------------------------
  TLine *lROImin = new TLine(vROI(0),minBg,vROI(0),maxBg);
  TLine *lROImax = new TLine(vROI(1),minBg,vROI(1),maxBg);
    lROImin->SetLineColor(3);
    lROImax->SetLineColor(3);
  //-----
  TLine *lBgFitLowMin = new TLine(vBgFitLow(0),minBg,vBgFitLow(0),maxBg);
  TLine *lBgFitLowMax = new TLine(vBgFitLow(1),minBg,vBgFitLow(1),maxBg);
    lBgFitLowMin->SetLineColor(2);
    lBgFitLowMax->SetLineColor(2);
  //-----
  TLine *lBgFitHighMin = new TLine(vBgFitHigh(0),minBg,vBgFitHigh(0),maxBg);
  TLine *lBgFitHighMax = new TLine(vBgFitHigh(1),minBg,vBgFitHigh(1),maxBg);
    lBgFitHighMin->SetLineColor(2);
    lBgFitHighMax->SetLineColor(2);
  //--------------------------------------------------------------------------------------------
  PurityHisto->DrawCopy("Ehist");
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
void calcPur2()
{
  const double LambdaMass = 1.115683, KaonMass = 0.493677;
  TString OutputName = "calcPur.cLamK0_Bp1.pdf";

  TString FileName = "Resultsgrid_cLamK0_Bp1.root";

  //------------------------------Lambdas---------------------------------------------------
  TH1F *LambdaPurity = GetHisto(FileName,"LamK0","LambdaPurity");

  double LamBgFitLow[2];
    //LamBgFitLow[0] = LambdaPurity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    LamBgFitLow[0] = 1.09;
    LamBgFitLow[1] = 1.102;
  double LamBgFitHigh[2];
    LamBgFitHigh[0] = 1.130;
    LamBgFitHigh[1] = LambdaPurity->GetBinLowEdge(LambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  double LamROI[2];
    LamROI[0] = LambdaMass-0.0038;
    LamROI[1] = LambdaMass+0.0038;
  TList* LamList = BuildPurityBgFit("LambdaPurity",LambdaPurity,LamBgFitLow,LamBgFitHigh,LamROI);
  TF1* fitBgd_Lambda = LamList->FindObject("fitBgd_LambdaPurity");

  //------------------------------K0Shorts(1)------------------------------------------------
  TH1F *K0Short1Purity = GetHisto(FileName,"LamK0","K0ShortPurity1");

  double K0Short1BgFitLow[2];
    K0Short1BgFitLow[0] = K0Short1Purity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    K0Short1BgFitLow[1] = 0.452;
  double K0Short1BgFitHigh[2];
    K0Short1BgFitHigh[0] = 0.536;
    K0Short1BgFitHigh[1] = K0Short1Purity->GetBinLowEdge(K0Short1Purity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  double K0Short1ROI[2];
    K0Short1ROI[0] = KaonMass-0.013677;
    K0Short1ROI[1] = KaonMass+0.020323;
  TList* K0Short1List = BuildPurityBgFit("K0Short1Purity",K0Short1Purity,K0Short1BgFitLow,K0Short1BgFitHigh,K0Short1ROI);
  TF1* fitBgd_K0Short1 = K0Short1List->FindObject("fitBgd_K0Short1Purity");

  //------------------------------AntiLambdas------------------------------------------------
  TH1F *AntiLambdaPurity = GetHisto(FileName,"ALamK0","AntiLambdaPurity");

  double ALamBgFitLow[2];
    //ALamBgFitLow[0] = AntiLambdaPurity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    ALamBgFitLow[0] = 1.090;
    ALamBgFitLow[1] = 1.102;
  double ALamBgFitHigh[2];
    ALamBgFitHigh[0] = 1.130;
    ALamBgFitHigh[1] = AntiLambdaPurity->GetBinLowEdge(AntiLambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  double ALamROI[2];
    ALamROI[0] = LambdaMass-0.0038;
    ALamROI[1] = LambdaMass+0.0038;
  TList* ALamList = BuildPurityBgFit("AntiLambdaPurity",AntiLambdaPurity,ALamBgFitLow,ALamBgFitHigh,ALamROI);
  TF1* fitBgd_AntiLambda = ALamList->FindObject("fitBgd_AntiLambdaPurity");

  //------------------------------K0Shorts(2)------------------------------------------------
  TH1F *K0Short2Purity = GetHisto(FileName,"ALamK0","K0ShortPurity1");

  double K0Short2BgFitLow[2];
    K0Short2BgFitLow[0] = K0Short2Purity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    K0Short2BgFitLow[1] = 0.452;
  double K0Short2BgFitHigh[2];
    K0Short2BgFitHigh[0] = 0.536;
    K0Short2BgFitHigh[1] = K0Short2Purity->GetBinLowEdge(K0Short2Purity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  double K0Short2ROI[2];
    K0Short2ROI[0] = KaonMass-0.013677;
    K0Short2ROI[1] = KaonMass+0.020323;
  TList* K0Short2List = BuildPurityBgFit("K0Short2Purity",K0Short2Purity,K0Short2BgFitLow,K0Short2BgFitHigh,K0Short2ROI);
  TF1* fitBgd_K0Short2 = K0Short2List->FindObject("fitBgd_K0Short2Purity");

//-----------------------------------------------------------------------------
  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(2,4);
  //-----
  c1->cd(1);
  PurityDrawAll(LambdaPurity,LamList);
  c1->cd(2);
  PurityDrawBg(LambdaPurity,LamList);
  //-----
  c1->cd(3);
  PurityDrawAll(K0Short1Purity,K0Short1List);
  c1->cd(4);
  PurityDrawBg(K0Short1Purity,K0Short1List);
  //-----
  c1->cd(5);
  PurityDrawAll(AntiLambdaPurity,ALamList);
  c1->cd(6);
  PurityDrawBg(AntiLambdaPurity,ALamList);
  //-----
  c1->cd(7);
  PurityDrawAll(K0Short2Purity,K0Short2List);
  c1->cd(8);
  PurityDrawBg(K0Short2Purity,K0Short2List);

  //c1->SaveAs(OutputName);

}
