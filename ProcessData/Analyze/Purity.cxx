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
    fBgFitHigh[1] = fPurityHistos[0]->GetBinLowEdge(fPurityHistos[0]->GetNbinsX()+1);

    fROI[0] = LambdaMass-0.0038;
    fROI[1] = LambdaMass+0.0038;
  }

  if( fParticleType == kK0 )
  {
    fBgFitLow[0] = fPurityHistos[0]->GetBinLowEdge(1);
    fBgFitLow[1] = 0.452;

    fBgFitHigh[0] = 0.536;
    fBgFitHigh[1] = fPurityHistos[0]->GetBinLowEdge(fPurityHistos[0]->GetNbinsX()+1);

    fROI[0] = KaonMass-0.013677;
    fROI[1] = KaonMass+0.020323;
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
void Purity::DrawPurity(TPad *aPad, bool aZoomBg)
{
  TF1* fitBgd = (TF1*)fPurityFitInfo->At(0);
    fitBgd->SetLineColor(4);
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

  if(!aZoomBg)
  {
    fCombinedPurity->GetXaxis()->SetTitle("M_{inv} (GeV/c^{2})");
    fCombinedPurity->GetYaxis()->SetTitle("dN/dM_{inv}");
    fCombinedPurity->SetLabelSize(0.04, "xy");
    fCombinedPurity->SetTitleSize(0.04, "xy");

    double tHistoMaxValue = fCombinedPurity->GetMaximum();
    lROImin = new TLine((*vROI)(0),0,(*vROI)(0),tHistoMaxValue);
    lROImax = new TLine((*vROI)(1),0,(*vROI)(1),tHistoMaxValue);
    //-----
    lBgFitLowMin = new TLine((*vBgFitLow)(0),0,(*vBgFitLow)(0),tHistoMaxValue);
    lBgFitLowMax = new TLine((*vBgFitLow)(1),0,(*vBgFitLow)(1),tHistoMaxValue);
    //-----
    lBgFitHighMin = new TLine((*vBgFitHigh)(0),0,(*vBgFitHigh)(0),tHistoMaxValue);
    lBgFitHighMax = new TLine((*vBgFitHigh)(1),0,(*vBgFitHigh)(1),tHistoMaxValue);
  }

  if(aZoomBg)
  {
    fCombinedPurity->GetXaxis()->SetTitle("");
    fCombinedPurity->GetYaxis()->SetTitle("");
    fCombinedPurity->SetLabelSize(0.08, "xy");

    fCombinedPurity->GetXaxis()->SetRange(fCombinedPurity->FindBin((*vBgFitLow)(0)),fCombinedPurity->FindBin((*vBgFitLow)(1)));
      double tMaxLow = fCombinedPurity->GetMaximum();
      double tMinLow = fCombinedPurity->GetMinimum();
    fCombinedPurity->GetXaxis()->SetRange(fCombinedPurity->FindBin((*vBgFitHigh)(0)),fCombinedPurity->FindBin((*vBgFitHigh)(1))-1);
      double tMaxHigh = fCombinedPurity->GetMaximum();
      double tMinHigh = fCombinedPurity->GetMinimum();
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
    tMaxBg+=tRangeBg/10.;
    tMinBg-=tRangeBg/10.;

    fCombinedPurity->GetXaxis()->SetRange(1,fCombinedPurity->GetNbinsX());
    fCombinedPurity->GetYaxis()->SetRangeUser(tMinBg,tMaxBg);
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
  fCombinedPurity->SetLineColor(1);
  fCombinedPurity->SetLineWidth(2);

  lROImin->SetLineColor(3);
  lROImax->SetLineColor(3);

  lBgFitLowMin->SetLineColor(2);
  lBgFitLowMax->SetLineColor(2);

  lBgFitHighMin->SetLineColor(2);
  lBgFitHighMax->SetLineColor(2);


  //--------------------------------------------------------------------------------------------
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(0);

  fCombinedPurity->DrawCopy("Ehist");
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
    TPaveText *myText = new TPaveText(0.12,0.65,0.42,0.85,"NDC");
    myText->SetFillColor(0);
    myText->SetBorderSize(1);
    myText->AddText(TString::Format("%s Purity = %0.2f%%",cRootParticleTags[fParticleType],100*purity));
    myText->Draw();

    double tSig = (*vInfo)(2);
    double tSigpBgd = (*vInfo)(1);
    TPaveText *myText2 = new TPaveText(0.15,0.54,0.35,0.64,"NDC");
    myText2->SetFillColor(0);
    myText2->SetBorderSize(0);
    myText2->SetTextAlign(33);
    myText2->AddText(TString::Format("Sig     = %0.3e",tSig));
    myText2->AddText(TString::Format("Sig+Bgd = %0.3e",tSigpBgd));
    myText2->Draw();

    TLegend *tLeg = new TLegend(0.65,0.50,0.89,0.89);
      tLeg->SetFillColor(0);
      tLeg->AddEntry(fCombinedPurity, "Data", "lp");
      tLeg->AddEntry(fitBgd, "Fit to background", "l");
      tLeg->AddEntry(lROImin, "Accepted window", "l");
      tLeg->AddEntry(lBgFitLowMin, "Bkg fit window", "l");
      tLeg->Draw();
  }
}



//________________________________________________________________________________________________________________
void Purity::DrawPurityAndBgd(TPad* aPad)
{
  aPad->cd();
  //aPad->Divide(1,2);

  TPad *tPad1 = new TPad("tPad1","tPad1",0.0,0.3,1.0,1.0);
  tPad1->Draw();
  TPad *tPad2 = new TPad("tPad2","tPad2",0.0,0.0,1.0,0.3);
  tPad2->Draw();

  DrawPurity(tPad1,false);
  DrawPurity(tPad2,true);  
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


