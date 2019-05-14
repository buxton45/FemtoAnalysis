///////////////////////////////////////////////////////////////////////////
// Purity:                                                               //
///////////////////////////////////////////////////////////////////////////


#include "Purity.h"

#ifdef __ROOT__
ClassImp(Purity)
#endif


//GLOBAL!

const double LambdaMass = 1.115683, KaonMass = 0.497614, XiMass = 1.32171;

//______________________________________________________________________________________________________________
bool reject;
double ffBgFitLow[2];
double ffBgFitHigh[2];

double PurityBgFitFunction(double *x, double *par)
{
  if( reject && !(x[0]>ffBgFitLow[0] && x[0]<ffBgFitLow[1]) && !(x[0]>ffBgFitHigh[0] && x[0]<ffBgFitHigh[1]) )
    {
      TF1::RejectPoint();
      return 0;
    }
  
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
}




//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
Purity::Purity(TString aCombinedPurityName, ParticleType aParticleType, vector<TH1*> aPurityHistos) :
  fCombinedPurityName(aCombinedPurityName),
  fPurityHistos(aPurityHistos),
  fParticleType(aParticleType),

  fBgFitLow(2),
  fBgFitHigh(2),
  fROI(2),

  fCombinedPurity(0),
  fPurityFitInfo(0),

  fOutputPurityFitInfo(false),
  fPurityValue(0),
  fSignal(0),
  fSignalPlusBgd(0),
  fBgd(0)



{
  
  if( (fParticleType == kLam) || (fParticleType == kALam) )
  {
    fBgFitLow[0] = 1.09;
    fBgFitLow[1] = 1.102;

    fBgFitHigh[0] = 1.130;
//    fBgFitHigh[1] = fPurityHistos[0]->GetBinLowEdge(fPurityHistos[0]->GetNbinsX()+1);
    fBgFitHigh[1] = fPurityHistos[0]->GetBinLowEdge(fPurityHistos[0]->GetNbinsX()+-1);

    fROI[0] = LambdaMass-0.0038;
    fROI[1] = LambdaMass+0.0038;
  }

  if( fParticleType == kK0 )
  {
//    fBgFitLow[0] = fPurityHistos[0]->GetBinLowEdge(1);
    fBgFitLow[0] = fPurityHistos[0]->GetBinLowEdge(3);
    fBgFitLow[1] = 0.452;

    fBgFitHigh[0] = 0.536;
//    fBgFitHigh[1] = fPurityHistos[0]->GetBinLowEdge(fPurityHistos[0]->GetNbinsX()+1);
    fBgFitHigh[1] = fPurityHistos[0]->GetBinLowEdge(fPurityHistos[0]->GetNbinsX()-1);

    fROI[0] = KaonMass-0.017614;
    fROI[1] = KaonMass+0.017386;
  }

  if( (fParticleType == kXi) || (fParticleType == kAXi) )
  {
    fBgFitLow[0] = 1.29;
    fBgFitLow[1] = 1.31;

    fBgFitHigh[0] = 1.336;
    fBgFitHigh[1] = fPurityHistos[0]->GetBinLowEdge(fPurityHistos[0]->GetNbinsX()+1);

    fROI[0] = XiMass-0.003;
    fROI[1] = XiMass+0.003;
  }

  CombineHistos();
  CalculatePurity();
}




//________________________________________________________________________________________________________________
Purity::~Purity()
{
  cout << "Purity object is being deleted!!!" << endl;
}


//________________________________________________________________________________________________________________
void Purity::CombineHistos()
{
  TH1* tCombinedPurity = (TH1*)fPurityHistos[0]->Clone(fCombinedPurityName);
  if(!tCombinedPurity->GetSumw2N()) {tCombinedPurity->Sumw2();}

  for(unsigned int i=1; i<fPurityHistos.size(); i++)
  {
    tCombinedPurity->Add((TH1*)fPurityHistos[i]);
  }

  fCombinedPurity = tCombinedPurity;
  fCombinedPurity->SetName(fCombinedPurityName);
  fCombinedPurity->SetTitle(fCombinedPurityName);
}

//________________________________________________________________________________________________________________
double Purity::FitFunctionGaussian(double *x, double *par)
{
  //4 parameters
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3];
}
//________________________________________________________________________________________________________________
double Purity::FitFunctionGaussianPlusPoly(double *x, double *par)
{
  //9 parameters
  double tBgd =    par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
  double tGauss1 =  par[5]*exp(-0.5*(pow((x[0]-par[6])/par[7],2.0))) + par[8];

  return tBgd+tGauss1;
}

//________________________________________________________________________________________________________________
double Purity::FitFunctionTwoGaussianPlusLinear(double *x, double *par)
{
  //9 parameters
  double tBgd =    par[0] + par[1]*x[0];
  double tGauss1 =  par[2]*exp(-0.5*(pow((x[0]-par[3])/par[4],2.0))) + par[5];
  double tGauss2 =  par[6]*exp(-0.5*(pow((x[0]-par[3])/par[7],2.0))) + par[8];

  return tBgd+tGauss1+tGauss2;
}
//________________________________________________________________________________________________________________
double Purity::FitFunctionTwoGaussianPlusPoly(double *x, double *par)
{
  //12 parameters
  double tBgd =    par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
  double tGauss1 =  par[5]*exp(-0.5*(pow((x[0]-par[6])/par[7],2.0))) + par[8];
  double tGauss2 =  par[9]*exp(-0.5*(pow((x[0]-par[6])/par[10],2.0))) + par[11];

  return tBgd+tGauss1+tGauss2;
}


//________________________________________________________________________________________________________________
void Purity::CalculatePurity()
{

  //-------------------------------
  reject = true;
  
  ffBgFitLow[0] = fBgFitLow[0];
  ffBgFitLow[1] = fBgFitLow[1];
  ffBgFitHigh[0] = fBgFitHigh[0];
  ffBgFitHigh[1] = fBgFitHigh[1];

  TF1 *fitBgd = new TF1("fitBgd",PurityBgFitFunction,fCombinedPurity->GetBinLowEdge(1),fCombinedPurity->GetBinLowEdge(fCombinedPurity->GetNbinsX()+1),5);  //fit over entire range

  if(fOutputPurityFitInfo){fCombinedPurity->Fit("fitBgd","0");}  //option 0 = Do not plot the result of the fit
  else{fCombinedPurity->Fit("fitBgd","0q");}  //option q = quiet mode = minimum printing


  //-------------------------------
  reject = false;

  char buffer[50];
  sprintf(buffer, "fitBgd_%s",fCombinedPurityName.Data());

  TF1 *fitBgd2 = new TF1(buffer,PurityBgFitFunction,fCombinedPurity->GetBinLowEdge(1),fCombinedPurity->GetBinLowEdge(fCombinedPurity->GetNbinsX()+1),5);
  fitBgd2->SetParameters(fitBgd->GetParameters());

  //--------------------------------------------------------------------------------------------
  int tBinROILow = fCombinedPurity->FindBin(fROI[0]);
  int tBinROIHigh = fCombinedPurity->FindBin(fROI[1]);
  //Using tROILow and tROIHigh ensures fitBgd2 and fCombinedPurity are integrating over exactly the same region
  //because likely fROI[0] and fROI[1] fall in the middle of a fCombinedPurity bin
  double tROILow = fCombinedPurity->GetBinLowEdge(tBinROILow);  //lower edge of low bin
  double tROIHigh = fCombinedPurity->GetBinLowEdge(tBinROIHigh+1);  //upper edge of high bin

  double tBgd = fitBgd2->Integral(tROILow,tROIHigh);
  tBgd /= fCombinedPurity->GetBinWidth(0);  //divide by bin size
  cout << fCombinedPurityName << ": " << "Bgd = " << tBgd << endl;
  //-----
  double tSigpbgd = fCombinedPurity->Integral(tBinROILow,tBinROIHigh);
  cout << fCombinedPurityName << ": " << "Sig+Bgd = " << tSigpbgd << endl;
  //-----
  double tSig = tSigpbgd-tBgd;
  cout << fCombinedPurityName << ": " << "Sig = " << tSig << endl;
  //-----
  double tPur = tSig/tSigpbgd;
  cout << fCombinedPurityName << ": " << "Pur = " << tPur << endl << endl;

  TVectorD *vInfo = new TVectorD(4);
    (*vInfo)(0) = tBgd;
    (*vInfo)(1) = tSigpbgd;
    (*vInfo)(2) = tSig;
    (*vInfo)(3) = tPur;

  fPurityValue = tPur;
  fSignal = tSig;
  fSignalPlusBgd = tSigpbgd;
  fBgd = tBgd;

  TVectorD *vROI = new TVectorD(2);
    (*vROI)(0) = fROI[0];
    (*vROI)(1) = fROI[1];

  TVectorD *vBgFitLow = new TVectorD(2);
    (*vBgFitLow)(0) = fBgFitLow[0];
    (*vBgFitLow)(1) = fBgFitLow[1];

  TVectorD *vBgFitHigh = new TVectorD(2);
    (*vBgFitHigh)(0) = fBgFitHigh[0];
    (*vBgFitHigh)(1) = fBgFitHigh[1];
  //--------------------------------------------------------------------------------------------
  TObjArray* temp = new TObjArray();
  temp->Add(fitBgd2);
  temp->Add(vInfo);
  temp->Add(vROI);
  temp->Add(vBgFitLow);
  temp->Add(vBgFitHigh);

  fPurityFitInfo = temp;

}


//________________________________________________________________________________________________________________
void Purity::DrawPurity(TPad *aPad, bool aZoomBg, bool aPrintPurity, double aPadScaleX, double aPadScaleY)
{
  TGaxis::SetMaxDigits(3);

  TF1* fitBgd = (TF1*)fPurityFitInfo->At(0);
    fitBgd->SetLineColor(4);
    fitBgd->SetLineStyle(5);
    fitBgd->SetLineWidth(2);
  //-----
  TVectorD* vInfo = (TVectorD*)fPurityFitInfo->At(1);
  //-----
  TVectorD* vROI = (TVectorD*)fPurityFitInfo->At(2);
  //-----
  TVectorD* vBgFitLow = (TVectorD*)fPurityFitInfo->At(3);
  //-----
  TVectorD* vBgFitHigh = (TVectorD*)fPurityFitInfo->At(4);
  //--------------------------------------------------------------------------------------------


  TLine *lROImin, *lROImax, *lBgFitLowMin, *lBgFitLowMax, *lBgFitHighMin, *lBgFitHighMax;

  TH1* tCombinedPurity = (TH1*)fCombinedPurity->Clone("tCombinedPuirty");
  tCombinedPurity->SetMarkerStyle(20);

  tCombinedPurity->GetXaxis()->SetTitle("#it{m}_{inv} (GeV/#it{c}^{2})");
  tCombinedPurity->SetLabelSize(0.06*(aPadScaleY/aPadScaleX), "x");
  tCombinedPurity->SetTitleSize(0.075*(aPadScaleY/aPadScaleX), "x");
  tCombinedPurity->GetXaxis()->SetTitleOffset(0.375*(aPadScaleY/aPadScaleX));
  tCombinedPurity->GetXaxis()->SetTickLength(0.06*(aPadScaleY/aPadScaleX));
  tCombinedPurity->GetXaxis()->SetNdivisions(510);

  tCombinedPurity->GetYaxis()->SetTitle("dN/d#it{m}_{inv}");
  tCombinedPurity->SetLabelSize(0.06*aPadScaleY, "y");
  tCombinedPurity->SetTitleSize(0.085*aPadScaleY, "y");
  tCombinedPurity->GetYaxis()->SetTitleOffset(0.50*aPadScaleY);
  tCombinedPurity->GetYaxis()->SetTickLength(0.04*aPadScaleY);
  if(aZoomBg) tCombinedPurity->GetYaxis()->SetTickLength(0.03*aPadScaleY);
  tCombinedPurity->GetYaxis()->SetNdivisions(505);

  if(!aZoomBg)
  {
    tCombinedPurity->GetXaxis()->SetTitle("");
    tCombinedPurity->SetLabelSize(0.0, "x");

    double tHistoMaxValue = tCombinedPurity->GetMaximum();
    double tMaxVertLines;
    tMaxVertLines = tHistoMaxValue;
    if( (fParticleType == kLam) || (fParticleType == kALam) ) tMaxVertLines = 5000000;
    if( fParticleType == kK0 )                                tMaxVertLines = 9000000;
    lROImin = new TLine((*vROI)(0),0,(*vROI)(0),tMaxVertLines);
    lROImax = new TLine((*vROI)(1),0,(*vROI)(1),tMaxVertLines);
    //-----
    lBgFitLowMin = new TLine((*vBgFitLow)(0),0,(*vBgFitLow)(0),tMaxVertLines);
    lBgFitLowMax = new TLine((*vBgFitLow)(1),0,(*vBgFitLow)(1),tMaxVertLines);
    //-----
    lBgFitHighMin = new TLine((*vBgFitHigh)(0),0,(*vBgFitHigh)(0),tMaxVertLines);
    lBgFitHighMax = new TLine((*vBgFitHigh)(1),0,(*vBgFitHigh)(1),tMaxVertLines);
  }

  if(aZoomBg)
  {
    tCombinedPurity->GetYaxis()->SetNdivisions(303);

    tCombinedPurity->GetYaxis()->SetTitle("");

    tCombinedPurity->GetXaxis()->SetRange(tCombinedPurity->FindBin((*vBgFitLow)(0)),tCombinedPurity->FindBin((*vBgFitLow)(1)));
      double tMaxLow = tCombinedPurity->GetMaximum();
      double tMinLow = tCombinedPurity->GetMinimum();
    tCombinedPurity->GetXaxis()->SetRange(tCombinedPurity->FindBin((*vBgFitHigh)(0)),tCombinedPurity->FindBin((*vBgFitHigh)(1))-1);
      double tMaxHigh = tCombinedPurity->GetMaximum();
      double tMinHigh = tCombinedPurity->GetMinimum();
    double tMaxBg;
      if(tMaxLow>tMaxHigh) tMaxBg = tMaxLow;
      else tMaxBg = tMaxHigh;
      //cout << "Background max = " << tMaxBg << endl;
    double tMinBg;
      if(tMinLow<tMinHigh) tMinBg = tMinLow;
      else tMinBg = tMinHigh;
      //cout << "Background min = " << tMinBg << endl;
    //--Extend the y-range that I plot

    double tRangeBg = tMaxBg-tMinBg;
    //tMaxBg+=tRangeBg/10.;
    //tMinBg-=tRangeBg/10.;

    double tMaxYAx = tMaxBg + tRangeBg/10.;
    double tMinYAx = tMinBg - tRangeBg/10.;

    tCombinedPurity->GetXaxis()->SetRange(1,tCombinedPurity->GetNbinsX());
    tCombinedPurity->GetYaxis()->SetRangeUser(tMinBg,tMaxYAx);
    //--------------------------------------------------------------------------------------------
    lROImin = new TLine((*vROI)(0),tMinBg,(*vROI)(0),tMaxBg);
    lROImax = new TLine((*vROI)(1),tMinBg,(*vROI)(1),tMaxBg);
    //-----
    lBgFitLowMin = new TLine((*vBgFitLow)(0),tMinBg,(*vBgFitLow)(0),tMaxBg);
    lBgFitLowMax = new TLine((*vBgFitLow)(1),tMinBg,(*vBgFitLow)(1),tMaxBg);
    //-----
    lBgFitHighMin = new TLine((*vBgFitHigh)(0),tMinBg,(*vBgFitHigh)(0),tMaxBg);
    lBgFitHighMax = new TLine((*vBgFitHigh)(1),tMinBg,(*vBgFitHigh)(1),tMaxBg);
  }


  //--------------------------------------------------------------------------------------------
  tCombinedPurity->SetLineColor(1);
  tCombinedPurity->SetLineWidth(2);

  lROImin->SetLineColor(3);
  lROImax->SetLineColor(3);
    lROImin->SetLineStyle(7);
    lROImax->SetLineStyle(7);
    lROImin->SetLineWidth(2);
    lROImax->SetLineWidth(2);

  lBgFitLowMin->SetLineColor(2);
  lBgFitLowMax->SetLineColor(2);
    lBgFitLowMin->SetLineStyle(3);
    lBgFitLowMax->SetLineStyle(3);
    lBgFitLowMin->SetLineWidth(2);
    lBgFitLowMax->SetLineWidth(2);

  lBgFitHighMin->SetLineColor(2);
  lBgFitHighMax->SetLineColor(2);
    lBgFitHighMin->SetLineStyle(3);
    lBgFitHighMax->SetLineStyle(3);
    lBgFitHighMin->SetLineWidth(2);
    lBgFitHighMax->SetLineWidth(2);


  //--------------------------------------------------------------------------------------------
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(0);

  tCombinedPurity->DrawCopy("Ephist");
  fitBgd->Draw("same");
  lROImin->Draw();
  lROImax->Draw();
  lBgFitLowMin->Draw();
  lBgFitLowMax->Draw();
  lBgFitHighMin->Draw();
  lBgFitHighMax->Draw();

  if(!aZoomBg)
  {
    double purity = (*vInfo)(3);
    double tSig = (*vInfo)(2);
    double tSigpBgd = (*vInfo)(1);
    if(aPrintPurity)
    {
      TPaveText *myText = new TPaveText(0.15,0.65,0.42,0.875,"NDC");
      myText->SetFillColor(0);
      myText->SetBorderSize(1);
      myText->AddText(TString::Format("%s Purity = %0.2f%%",cRootParticleTags[fParticleType],100*purity));
      myText->Draw();

      TPaveText *myText2 = new TPaveText(0.15,0.49,0.42,0.64,"NDC");
      myText2->SetFillColor(0);
      myText2->SetBorderSize(0);
      myText2->SetTextAlign(33);
      myText2->AddText(TString::Format("Sig     = %0.3e",tSig));
      myText2->AddText(TString::Format("Sig+Bgd = %0.3e",tSigpBgd));
      myText2->Draw();
    }
    else
    {
      TPaveText *myText2 = new TPaveText(0.15,0.35,0.40,0.65,"NDC");
      myText2->SetFillColor(0);
      myText2->SetBorderSize(0);
      myText2->SetTextAlign(33);
      myText2->SetTextFont(42);
      myText2->AddText(TString::Format("Sig     = %0.3e",tSig));
      myText2->AddText(TString::Format("Sig+Bgd = %0.3e",tSigpBgd));
      myText2->AddText("");
      myText2->Draw();
    }

      TPaveText *myText3 = new TPaveText(0.125,0.675,0.50,0.825,"NDC");
      myText3->SetFillColor(0);
      myText3->SetBorderSize(0);
      myText3->SetTextAlign(22);
      myText3->SetTextFont(42);
      myText3->AddText(TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV"));
      myText3->AddText("");
      myText3->Draw();

    TLegend *tLeg = new TLegend(0.65,0.50,0.925,0.875); 
      tLeg->SetFillColor(0);
      tLeg->AddEntry(tCombinedPurity, "Data", "lp");
      tLeg->AddEntry(fitBgd, "Fit to background", "l");
      tLeg->AddEntry(lROImin, "Accepted window", "l");
      tLeg->AddEntry(lBgFitLowMin, "Bkg fit window", "l");
      tLeg->Draw();
  }
}



//________________________________________________________________________________________________________________
void Purity::DrawPurityAndBgd(TPad* aPad, bool aPrintPurity)
{
  aPad->cd();
  //aPad->Divide(1,2);

  TPad *tPad1 = new TPad("tPad1","tPad1",0.0,0.3,1.0,1.0);
    tPad1->SetTopMargin(0.10);
    tPad1->SetBottomMargin(0.025);
    tPad1->SetRightMargin(0.05);
  tPad1->Draw();
  TPad *tPad2 = new TPad("tPad2","tPad2",0.0,0.0,1.0,0.3);
    tPad2->SetTopMargin(0.20);
    tPad2->SetBottomMargin(0.35);
    tPad2->SetRightMargin(0.05);
  tPad2->Draw();

  double tPad2XScale = tPad1->GetAbsWNDC()/tPad2->GetAbsWNDC();
  double tPad2YScale = tPad1->GetAbsHNDC()/tPad2->GetAbsHNDC();

  DrawPurity(tPad1, false, aPrintPurity);
  DrawPurity(tPad2, true, aPrintPurity, tPad2XScale, tPad2YScale);  
}


//________________________________________________________________________________________________________________
TF1* Purity::GetFullFit(int aType)
{
  TF1* fitBgd = (TF1*)fPurityFitInfo->At(0);
  TVectorD* vBgFitLow = (TVectorD*)fPurityFitInfo->At(3);
  TVectorD* vBgFitHigh = (TVectorD*)fPurityFitInfo->At(4);
  //--------------------------------------------------------------------------------------------

  TH1* tCombinedPurity = (TH1*)fCombinedPurity->Clone("tCombinedPurity");
  TF1 *fitBgdPlusSignal;

  if(aType==0)
  {
    fitBgdPlusSignal = new TF1("fitBgdPlusSignal", FitFunctionGaussian, tCombinedPurity->GetBinLowEdge(1), tCombinedPurity->GetBinLowEdge(tCombinedPurity->GetNbinsX()+1), 4);

    fitBgdPlusSignal->SetParameter(0, tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(1, tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParameter(2, 0.01);
    fitBgdPlusSignal->SetParLimits(2, 0., 0.5);

    fitBgdPlusSignal->SetParameter(3, 0.0);

    tCombinedPurity->Fit("fitBgdPlusSignal","0q", "", (*vBgFitLow)(1), (*vBgFitHigh)(0));
  }

  else if(aType==1)
  {
    fitBgdPlusSignal = new TF1("fitBgdPlusSignal", FitFunctionGaussianPlusPoly, tCombinedPurity->GetBinLowEdge(1), tCombinedPurity->GetBinLowEdge(tCombinedPurity->GetNbinsX()+1), 9);

    fitBgdPlusSignal->FixParameter(0, fitBgd->GetParameter(0));
    fitBgdPlusSignal->FixParameter(1, fitBgd->GetParameter(1));
    fitBgdPlusSignal->FixParameter(2, fitBgd->GetParameter(2));
    fitBgdPlusSignal->FixParameter(3, fitBgd->GetParameter(3));
    fitBgdPlusSignal->FixParameter(4, fitBgd->GetParameter(4));

    fitBgdPlusSignal->SetParameter(5, tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(6, tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParameter(7, 0.01);
    fitBgdPlusSignal->SetParLimits(7, 0., 0.5);

    fitBgdPlusSignal->FixParameter(8, 0.);

    fitBgdPlusSignal->SetParLimits(6, 0.9*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()), 1.1*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));

    tCombinedPurity->Fit("fitBgdPlusSignal","0q", "", (*vBgFitLow)(1), (*vBgFitHigh)(0));
  }

  else if(aType==2)
  {
    fitBgdPlusSignal = new TF1("fitBgdPlusSignal", FitFunctionTwoGaussianPlusLinear, tCombinedPurity->GetBinLowEdge(1), tCombinedPurity->GetBinLowEdge(tCombinedPurity->GetNbinsX()+1), 9);

    fitBgdPlusSignal->SetParameter(0, fitBgd->GetParameter(0));
    fitBgdPlusSignal->SetParameter(1, fitBgd->GetParameter(1));

    fitBgdPlusSignal->SetParameter(2, tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(3, tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParameter(4, 0.01);
    fitBgdPlusSignal->SetParLimits(4, 0., 0.5);
    fitBgdPlusSignal->FixParameter(5, 0.0);

    fitBgdPlusSignal->SetParLimits(3, 0.9*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()), 1.1*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));

    fitBgdPlusSignal->SetParameter(6, 0.1*tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(7, 0.025);
    fitBgdPlusSignal->FixParameter(8, 0.);

    tCombinedPurity->Fit("fitBgdPlusSignal","0q", "", (*vBgFitLow)(1), (*vBgFitHigh)(0));
  }

  else if(aType==3) 
  {
    fitBgdPlusSignal = new TF1("fitBgdPlusSignal", FitFunctionTwoGaussianPlusPoly, tCombinedPurity->GetBinLowEdge(1), tCombinedPurity->GetBinLowEdge(tCombinedPurity->GetNbinsX()+1), 12);

    fitBgdPlusSignal->FixParameter(0, fitBgd->GetParameter(0));
    fitBgdPlusSignal->FixParameter(1, fitBgd->GetParameter(1));
    fitBgdPlusSignal->FixParameter(2, fitBgd->GetParameter(2));
    fitBgdPlusSignal->FixParameter(3, fitBgd->GetParameter(3));
    fitBgdPlusSignal->FixParameter(4, fitBgd->GetParameter(4));

    fitBgdPlusSignal->SetParameter(5, tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(6, tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParameter(7, 0.01);
    fitBgdPlusSignal->SetParLimits(7, 0., 0.5);

    fitBgdPlusSignal->FixParameter(8, 0.);

    fitBgdPlusSignal->SetParLimits(6, 0.9*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()), 1.1*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));

    fitBgdPlusSignal->SetParameter(9, 0.1*tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(10, 0.025);
    fitBgdPlusSignal->FixParameter(11, 0.);

    tCombinedPurity->Fit("fitBgdPlusSignal","0", "", (*vBgFitLow)(1), (*vBgFitHigh)(0));
  }

  else assert(0);

  return fitBgdPlusSignal;
}


//________________________________________________________________________________________________________________
void Purity::CalculateResolution(int aTypeOfFit)
{
  TF1 *fitBgdPlusSignal = GetFullFit(aTypeOfFit);

  cout << "(Norm) fitBgdPlusSignal->GetParameter(5)  = " << fitBgdPlusSignal->GetParameter(5) << endl;
  cout << "(mu) fitBgdPlusSignal->GetParameter(6)    = " << fitBgdPlusSignal->GetParameter(6) << endl;
  cout << "(sigma) fitBgdPlusSignal->GetParameter(7) = " << fitBgdPlusSignal->GetParameter(7) << endl;
  cout << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void Purity::DrawResolution(TPad *aPad, int aTypeOfFit, bool aZoomBg, double aPadScaleX, double aPadScaleY)
{
  TGaxis::SetMaxDigits(3);

  TF1* fitFull = GetFullFit(aTypeOfFit);
    fitFull->SetLineColor(4);
    fitFull->SetLineStyle(5);
    fitFull->SetLineWidth(2);
  //-----
  TVectorD* vBgFitLow = (TVectorD*)fPurityFitInfo->At(3);
  //-----
  TVectorD* vBgFitHigh = (TVectorD*)fPurityFitInfo->At(4);
  //--------------------------------------------------------------------------------------------

  TH1* tCombinedPurity = (TH1*)fCombinedPurity->Clone("tCombinedPuirty");
  tCombinedPurity->SetMarkerStyle(20);
  tCombinedPurity->SetLineColor(1);
  tCombinedPurity->SetLineWidth(2);

  tCombinedPurity->GetXaxis()->SetTitle("#it{m}_{inv} (GeV/#it{c}^{2})");
  tCombinedPurity->SetLabelSize(0.06*(aPadScaleY/aPadScaleX), "x");
  tCombinedPurity->SetTitleSize(0.075*(aPadScaleY/aPadScaleX), "x");
  tCombinedPurity->GetXaxis()->SetTitleOffset(0.375*(aPadScaleY/aPadScaleX));
  tCombinedPurity->GetXaxis()->SetTickLength(0.06*(aPadScaleY/aPadScaleX));
  tCombinedPurity->GetXaxis()->SetNdivisions(510);

  tCombinedPurity->GetYaxis()->SetTitle("dN/d#it{m}_{inv}");
  tCombinedPurity->SetLabelSize(0.06*aPadScaleY, "y");
  tCombinedPurity->SetTitleSize(0.085*aPadScaleY, "y");
  tCombinedPurity->GetYaxis()->SetTitleOffset(0.50*aPadScaleY);
  tCombinedPurity->GetYaxis()->SetTickLength(0.04*aPadScaleY);
  if(aZoomBg) tCombinedPurity->GetYaxis()->SetTickLength(0.03*aPadScaleY);
  tCombinedPurity->GetYaxis()->SetNdivisions(505);

  if(!aZoomBg)
  {
    tCombinedPurity->GetXaxis()->SetTitle("");
    tCombinedPurity->SetLabelSize(0.0, "x");
  }

  if(aZoomBg)
  {
    tCombinedPurity->GetYaxis()->SetNdivisions(303);
    tCombinedPurity->GetYaxis()->SetTitle("");

    tCombinedPurity->GetXaxis()->SetRange(tCombinedPurity->FindBin((*vBgFitLow)(0)),tCombinedPurity->FindBin((*vBgFitLow)(1)));
      double tMaxLow = tCombinedPurity->GetMaximum();
      double tMinLow = tCombinedPurity->GetMinimum();
    tCombinedPurity->GetXaxis()->SetRange(tCombinedPurity->FindBin((*vBgFitHigh)(0)),tCombinedPurity->FindBin((*vBgFitHigh)(1))-1);
      double tMaxHigh = tCombinedPurity->GetMaximum();
      double tMinHigh = tCombinedPurity->GetMinimum();
    double tMaxBg;
      if(tMaxLow>tMaxHigh) tMaxBg = tMaxLow;
      else tMaxBg = tMaxHigh;
      //cout << "Background max = " << tMaxBg << endl;
    double tMinBg;
      if(tMinLow<tMinHigh) tMinBg = tMinLow;
      else tMinBg = tMinHigh;
      //cout << "Background min = " << tMinBg << endl;
    //--Extend the y-range that I plot

    double tRangeBg = tMaxBg-tMinBg;
    //tMaxBg+=tRangeBg/10.;
    //tMinBg-=tRangeBg/10.;

    double tMaxYAx = tMaxBg + tRangeBg/10.;
    double tMinYAx = tMinBg - tRangeBg/10.;

    tCombinedPurity->GetXaxis()->SetRange(1,tCombinedPurity->GetNbinsX());
    tCombinedPurity->GetYaxis()->SetRangeUser(tMinBg,tMaxYAx);
    //--------------------------------------------------------------------------------------------
  }

  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(0);

  tCombinedPurity->DrawCopy("Ephist");

  fitFull->SetNpx(10000); 
  fitFull->Draw("same");

  if(!aZoomBg)
  {
    TPaveText *myText2 = new TPaveText(0.15,0.35,0.40,0.65,"NDC");
    myText2->SetFillColor(0);
    myText2->SetBorderSize(0);
    myText2->SetTextAlign(33);
    myText2->SetTextFont(42);

    double tMu, tSigma;
    if(aTypeOfFit==0)
    {
      tMu = fitFull->GetParameter(1);
      tSigma = fitFull->GetParameter(2);
    }
    else if(aTypeOfFit==1)
    {
      tMu = fitFull->GetParameter(6);
      tSigma = fitFull->GetParameter(7);
    }
    else if(aTypeOfFit==2)
    {
      tMu = fitFull->GetParameter(3);
      tSigma = fitFull->GetParameter(4);
    }
    else if(aTypeOfFit==3)
    {
      tMu = fitFull->GetParameter(6);
      tSigma = fitFull->GetParameter(7);
    }
    else assert(0);
    tMu*=1000;
    tSigma*=1000;

    myText2->AddText(TString::Format("#LT M #GT = %0.3f MeV/#it{c}^{2}",tMu));
    myText2->AddText(TString::Format("#sigma = %0.3f MeV/#it{c}^{2}",tSigma));

    myText2->AddText("");
    myText2->Draw();


      TPaveText *myText3 = new TPaveText(0.125,0.675,0.50,0.825,"NDC");
      myText3->SetFillColor(0);
      myText3->SetBorderSize(0);
      myText3->SetTextAlign(22);
      myText3->SetTextFont(42);
      myText3->AddText(TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV"));
      myText3->AddText("");
      myText3->Draw();

    TLegend *tLeg = new TLegend(0.65,0.50,0.925,0.875); 
      tLeg->SetFillColor(0);
      tLeg->AddEntry(tCombinedPurity, "Data", "lp");
      tLeg->AddEntry(fitFull, "Full fit", "l");
      tLeg->Draw();
  }
}


//________________________________________________________________________________________________________________
void Purity::DrawResolutionAndBgd(TPad* aPad, int aTypeOfFit)
{
  aPad->cd();
  //aPad->Divide(1,2);

  TPad *tPad1 = new TPad("tPad1","tPad1",0.0,0.3,1.0,1.0);
    tPad1->SetTopMargin(0.10);
    tPad1->SetBottomMargin(0.025);
    tPad1->SetRightMargin(0.05);
  tPad1->Draw();
  TPad *tPad2 = new TPad("tPad2","tPad2",0.0,0.0,1.0,0.3);
    tPad2->SetTopMargin(0.20);
    tPad2->SetBottomMargin(0.35);
    tPad2->SetRightMargin(0.05);
  tPad2->Draw();

  double tPad2XScale = tPad1->GetAbsWNDC()/tPad2->GetAbsWNDC();
  double tPad2YScale = tPad1->GetAbsHNDC()/tPad2->GetAbsHNDC();

  DrawResolution(tPad1, aTypeOfFit, false);
  DrawResolution(tPad2, aTypeOfFit, true, tPad2XScale, tPad2YScale);  
}



//________________________________________________________________________________________________________________
void Purity::AddHisto(TH1* aHisto)
{
  fPurityHistos.push_back(aHisto);
  CombineHistos();
  CalculatePurity();
}


//________________________________________________________________________________________________________________
void Purity::SetBgFitLow(double aMinLow, double aMaxLow)
{
  fBgFitLow[0] = aMinLow;
  fBgFitLow[1] = aMaxLow;

  CalculatePurity();
}


//________________________________________________________________________________________________________________
void Purity::SetBgFitHigh(double aMinHigh, double aMaxHigh)
{
  fBgFitHigh[0] = aMinHigh;
  fBgFitHigh[1] = aMaxHigh;

  CalculatePurity();
}

//________________________________________________________________________________________________________________
void Purity::SetROI(double aMinROI, double aMaxROI)
{
  fROI[0] = aMinROI;
  fROI[1] = aMaxROI;

  CalculatePurity();
}


