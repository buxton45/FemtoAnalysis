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

double ScaledPurityBgFitFunction(double *x, double *par)
{  
  return par[5]*PurityBgFitFunction(x, par);
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
  //10 parameters
  double tBgd =    par[0] + par[1]*x[0];
  double tGauss1 =  par[2]*exp(-0.5*(pow((x[0]-par[3])/par[4],2.0))) + par[5];
  double tGauss2 =  par[6]*exp(-0.5*(pow((x[0]-par[7])/par[8],2.0))) + par[9];

  return tBgd+tGauss1+tGauss2;
}
//________________________________________________________________________________________________________________
double Purity::FitFunctionTwoGaussianPlusPoly(double *x, double *par)
{
  //13 parameters
  double tBgd =    par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
  double tGauss1 =  par[5]*exp(-0.5*(pow((x[0]-par[6])/par[7],2.0))) + par[8];
  double tGauss2 =  par[9]*exp(-0.5*(pow((x[0]-par[10])/par[11],2.0))) + par[12];

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
void Purity::DrawPurity(TPad *aPad, bool aZoomBg, bool aPrintPurity, bool aPutYExponentInLabel, TString aExponentToPrint, bool aSuppressYAxTitle, double aPadScaleX, double aPadScaleY, bool aScaleByBinWidth)
{
  if(aPutYExponentInLabel==false) aExponentToPrint="";
  else
  {
    if(!aExponentToPrint.EndsWith(" ")) aExponentToPrint += TString(" ");
    //Remove x10^6 from above the plot by moving it way out of the figure...
    TGaxis::SetExponentOffset(-10.0, 10.0, "y");
  }

  TGaxis::SetMaxDigits(3);
  int tLineWidth = 4;
  aPad->SetTicks(1,1);

  TF1* fitBgd = (TF1*)fPurityFitInfo->At(0);
    fitBgd->SetLineColor(4);
    fitBgd->SetLineStyle(5);
    fitBgd->SetLineWidth(tLineWidth);
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

  //tCombinedPurity->GetXaxis()->SetTitle("#it{m}_{inv} (GeV/#it{c}^{2})");
    TString tXaxTitle = "#it{m}";
  if( (fParticleType == kLam) 
   || (fParticleType == kALam) ) tXaxTitle.Append("_{p#pi}");
  else if(fParticleType == kK0)  tXaxTitle.Append("_{#pi#pi}");
  else assert(0);
  tXaxTitle.Append(" (GeV/#it{c}^{2})");
  tCombinedPurity->GetXaxis()->SetTitle(tXaxTitle);
  tCombinedPurity->SetLabelSize(0.075*(aPadScaleY/aPadScaleX), "x");
  tCombinedPurity->SetTitleSize(0.115*(aPadScaleY/aPadScaleX), "x");
  tCombinedPurity->GetXaxis()->SetTitleOffset(0.405*(aPadScaleY/aPadScaleX));
  tCombinedPurity->GetXaxis()->SetLabelOffset(0.02*(aPadScaleY/aPadScaleX));  
  tCombinedPurity->GetXaxis()->SetTickLength(0.06*(aPadScaleY/aPadScaleX));
  tCombinedPurity->GetXaxis()->SetNdivisions(510);

  if(aScaleByBinWidth)
  {
    TF1* scaledFitBgd = new TF1(fitBgd->GetName(), ScaledPurityBgFitFunction, tCombinedPurity->GetBinLowEdge(1),tCombinedPurity->GetBinLowEdge(tCombinedPurity->GetNbinsX()+1),6);
    for(unsigned int i=0; i<fitBgd->GetNpar(); i++) scaledFitBgd->SetParameter(i, fitBgd->GetParameter(i));
    scaledFitBgd->SetParameter(5, 1.0/(1000*tCombinedPurity->GetBinWidth(1)));
    fitBgd = scaledFitBgd;
    fitBgd->SetLineColor(4);
    fitBgd->SetLineStyle(5);
    fitBgd->SetLineWidth(tLineWidth);
  
    tCombinedPurity->Scale(1.0/(1000*tCombinedPurity->GetBinWidth(1)));
    //tCombinedPurity->GetYaxis()->SetTitle("dN/d#it{m}_{inv}");
    TString tYaxTitle = TString("dN/d#it{m}");
    if( (fParticleType == kLam) 
     || (fParticleType == kALam) ) tYaxTitle.Append("_{p#pi}");
    else if(fParticleType == kK0)  tYaxTitle.Append("_{#pi#pi}");
    else assert(0);
    tYaxTitle.Append(TString::Format(" (%sMeV^{-1}#it{c}^{2}#scale[0.5]{ })", aExponentToPrint.Data()));
    tCombinedPurity->GetYaxis()->SetTitle(tYaxTitle);
    if(aSuppressYAxTitle) tCombinedPurity->SetTitleSize(0.0, "y");
    else tCombinedPurity->SetTitleSize(0.0975*aPadScaleY, "y");
    tCombinedPurity->GetYaxis()->SetTitleOffset(0.575*aPadScaleY);
  }
  else
  {
    TString tYaxTitle = TString::Format("%sCounts/(%.1f MeV/#it{c}^{2}#scale[0.5]{ })", aExponentToPrint.Data(), 1000*tCombinedPurity->GetBinWidth(1));
    tCombinedPurity->GetYaxis()->SetTitle(tYaxTitle);
    if(aSuppressYAxTitle) tCombinedPurity->SetTitleSize(0.0, "y");
    else tCombinedPurity->SetTitleSize(0.0895*aPadScaleY, "y");
    tCombinedPurity->GetYaxis()->SetTitleOffset(0.65*aPadScaleY);
  }
  tCombinedPurity->SetLabelSize(0.075*aPadScaleY, "y");
  tCombinedPurity->GetYaxis()->SetLabelOffset(0.0075*aPadScaleY);    
  tCombinedPurity->GetYaxis()->SetTickLength(0.03*aPadScaleY);
  //if(aZoomBg) tCombinedPurity->GetYaxis()->SetTickLength(0.03*aPadScaleY);
  tCombinedPurity->GetYaxis()->SetNdivisions(505);

  if(!aZoomBg)
  {
    tCombinedPurity->GetXaxis()->SetTitle("");
    tCombinedPurity->SetLabelSize(0.0, "x");

    double tHistoMaxValue = tCombinedPurity->GetMaximum();    
    double tMaxVertLines;
    tMaxVertLines = tHistoMaxValue;
    if( (fParticleType == kLam) || (fParticleType == kALam) ) tMaxVertLines = 3000000;
    if( fParticleType == kK0 )                                tMaxVertLines = 5400000;
    if(aScaleByBinWidth) tMaxVertLines /= (1000*tCombinedPurity->GetBinWidth(1));
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

    //double tMaxYAx = tMaxBg + tRangeBg/10.;
    //double tMinYAx = tMinBg - tRangeBg/10.;
    
    double tMaxYAx = tMaxBg + tRangeBg/5.;
    double tMinYAx = tMinBg - tRangeBg/5.;

    // CHANGES FOR PHYS REV C
    if( (fParticleType == kLam) || (fParticleType == kALam) )
    {
      //Using automation above gives 140346, 260793
      tMinYAx = 120000;
      tMaxYAx = 300000;
      tCombinedPurity->GetYaxis()->SetNdivisions(202);
    }
    if(fParticleType == kK0)
    {
      //Using automation above gives 73677.8, 171546
      tMinYAx = 72500;
      tMaxYAx = 200000;
      tCombinedPurity->GetYaxis()->SetNdivisions(202);
    }
    if(aScaleByBinWidth)
    {
      tMinYAx /= (1000*tCombinedPurity->GetBinWidth(1));
      tMaxYAx /= (1000*tCombinedPurity->GetBinWidth(1));
      if(fParticleType == kK0)
      {
        tMinYAx = 50000;
        tMaxYAx = 150000;     
      }
    }


    tCombinedPurity->GetXaxis()->SetRange(1,tCombinedPurity->GetNbinsX());
    tCombinedPurity->GetYaxis()->SetRangeUser(tMinYAx,tMaxYAx);
    //--------------------------------------------------------------------------------------------
    lROImin = new TLine((*vROI)(0),tMinYAx,(*vROI)(0),tMaxYAx);
    lROImax = new TLine((*vROI)(1),tMinYAx,(*vROI)(1),tMaxYAx);
    //-----
    lBgFitLowMin = new TLine((*vBgFitLow)(0),tMinYAx,(*vBgFitLow)(0),tMaxYAx);
    lBgFitLowMax = new TLine((*vBgFitLow)(1),tMinYAx,(*vBgFitLow)(1),tMaxYAx);
    //-----
    lBgFitHighMin = new TLine((*vBgFitHigh)(0),tMinYAx,(*vBgFitHigh)(0),tMaxYAx);
    lBgFitHighMax = new TLine((*vBgFitHigh)(1),tMinYAx,(*vBgFitHigh)(1),tMaxYAx);
  }


  //--------------------------------------------------------------------------------------------
  tCombinedPurity->SetLineColor(1);
  tCombinedPurity->SetLineWidth(tLineWidth/2);

  lROImin->SetLineColor(3);
  lROImax->SetLineColor(3);
    lROImin->SetLineStyle(7);
    lROImax->SetLineStyle(7);
    lROImin->SetLineWidth(tLineWidth);
    lROImax->SetLineWidth(tLineWidth);

  lBgFitLowMin->SetLineColor(2);
  lBgFitLowMax->SetLineColor(2);
    lBgFitLowMin->SetLineStyle(3);
    lBgFitLowMax->SetLineStyle(3);
    lBgFitLowMin->SetLineWidth(tLineWidth);
    lBgFitLowMax->SetLineWidth(tLineWidth);

  lBgFitHighMin->SetLineColor(2);
  lBgFitHighMax->SetLineColor(2);
    lBgFitHighMin->SetLineStyle(3);
    lBgFitHighMax->SetLineStyle(3);
    lBgFitHighMin->SetLineWidth(tLineWidth);
    lBgFitHighMax->SetLineWidth(tLineWidth);


  //--------------------------------------------------------------------------------------------
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(0);

  //tCombinedPurity->DrawCopy("Ephist");
  tCombinedPurity->DrawCopy("hist");
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

    if(!aPrintPurity)
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
    
    TPaveText *myText3;
      if(aPrintPurity) myText3 = new TPaveText(0.135,0.273,0.415,0.73,"NDC");
      else             myText3 = new TPaveText(0.15,0.65,0.475,0.85,"NDC");
      myText3->SetFillColor(0);
      myText3->SetBorderSize(0);
      myText3->SetTextAlign(12);
      myText3->SetTextFont(42);
      myText3->SetFillStyle(0);
      //myText3->AddText(TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV"));
      //myText3->AddText("");
      /*
      if(fCombinedPurityName.Contains("_0010")) myText3->AddText(TString("ALICE  0-10%"));
      else                                      myText3->AddText(TString("ALICE"));
      myText3->AddText(TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV"));
      */
      myText3->AddText(TString("ALICE"));
      if(fCombinedPurityName.Contains("_0010")) myText3->AddText(TString("Pb-Pb  0-10%"));
      else                                      myText3->AddText(TString("ALICE"));
      myText3->AddText(TString("#sqrt{#it{s}_{NN}} = 2.76 TeV"));
      //if(aPrintPurity) myText3->AddText(TString::Format("%s Purity = %0.1f%%",cRootParticleTags[fParticleType],100*purity));
      if(aPrintPurity) myText3->AddText(TString::Format("Purity = %0.1f%%",100*purity));
      myText3->Draw();

    TLegend *tLeg = new TLegend(0.575,0.35,0.920,0.805); 
      tLeg->SetFillColor(0);
      tLeg->SetFillStyle(0);
      tLeg->SetBorderSize(0);
      tLeg->SetTextSize(0.0775);
      //tLeg->AddEntry(tCombinedPurity, TString::Format("%s candidates", cRootParticleTags[fParticleType]), "p");
      tLeg->AddEntry(tCombinedPurity, TString::Format("%s candidates", cRootParticleTags[fParticleType]), "l");   
      tLeg->AddEntry(lBgFitLowMin, "Bgd. fit window", "l");   
      tLeg->AddEntry(fitBgd, "Fit to bgd.", "l");
      tLeg->AddEntry(lROImin, "Accepted", "l");
      tLeg->Draw();
      
      if((fParticleType == kLam) || (fParticleType == kK0))
      {
        // Add (a) to Lam and (b) to K0 for Phys. Rev. C Publication
        TLatex* tPanelLetters = new TLatex();
        tPanelLetters->SetTextAlign(11);
        tPanelLetters->SetLineWidth(2);
        tPanelLetters->SetTextFont(62);
        tPanelLetters->SetTextSize(0.090);

        //double tXLett=0.855;
        //double tYLett=0.765;
        double tXLett=0.125;
        double tYLett=0.765;
        if(fParticleType == kLam)
        {
          tPanelLetters->DrawLatexNDC(tXLett, tYLett, "(a)");
        }
        else if(fParticleType == kK0)
        {
          tPanelLetters->DrawLatexNDC(tXLett, tYLett, "(b)");
        }
        else assert(0);
      }
  }
}



//________________________________________________________________________________________________________________
void Purity::DrawPurityAndBgd(TPad* aPad, bool aPrintPurity, bool aPutYExponentInLabel, TString aExponentToPrint, bool aScaleByBinWidth)
{
  aPad->cd();
  //aPad->Divide(1,2);

  TPad *tPad1 = new TPad("tPad1","tPad1",0.075,0.3,1.0,1.0);
    tPad1->SetTopMargin(0.10);
    tPad1->SetBottomMargin(0.050);
    tPad1->SetLeftMargin(0.085);    
    tPad1->SetRightMargin(0.05);
  tPad1->Draw();
  TPad *tPad2 = new TPad("tPad2","tPad2",0.075,0.0,1.0,0.3);
    tPad2->SetTopMargin(0.075);
    tPad2->SetBottomMargin(0.55);
    tPad2->SetLeftMargin(0.085);        
    tPad2->SetRightMargin(0.05);
  tPad2->Draw();

  double tPad2XScale = tPad1->GetAbsWNDC()/tPad2->GetAbsWNDC();
  double tPad2YScale = tPad1->GetAbsHNDC()/tPad2->GetAbsHNDC();

  DrawPurity(tPad1, false, aPrintPurity, aPutYExponentInLabel, aExponentToPrint, true, 1, 1, aScaleByBinWidth);
  DrawPurity(tPad2, true, aPrintPurity, aPutYExponentInLabel, aExponentToPrint, true, tPad2XScale, tPad2YScale, aScaleByBinWidth);  
  //------------- Y axis title ---------------------------------------
  TString tYaxTitle;
  if(!aScaleByBinWidth) tYaxTitle = TString::Format("%s#scale[0.5]{ }Counts/(%.1f MeV/#it{c}^{2}#scale[0.5]{ })", aExponentToPrint.Data(), 1000*fCombinedPurity->GetBinWidth(1));
  else
  {
    tYaxTitle = TString("dN/d#it{m}");
    if( (fParticleType == kLam) 
     || (fParticleType == kALam) ) tYaxTitle.Append("_{p#pi}");
    else if(fParticleType == kK0)  tYaxTitle.Append("_{#pi#pi}");
    else assert(0);
    tYaxTitle.Append(TString::Format(" (%s#scale[0.5]{ }MeV^{-1}#it{c}^{2}#scale[0.5]{ })", aExponentToPrint.Data()));
  }
  
  TLatex* tLaText;
  if(fParticleType == kK0) tLaText = new TLatex(0.065, 0.205, tYaxTitle);
  else                     tLaText = new TLatex(0.075, 0.205, tYaxTitle);
  tLaText->SetTextAlign(11);
  tLaText->SetLineWidth(2);
  tLaText->SetTextFont(42);
  if(aScaleByBinWidth) tLaText->SetTextSize(0.08);
  else tLaText->SetTextSize(0.075);
  tLaText->SetNDC(true);
  tLaText->SetTextAngle(90);
  tLaText->SetTextAlign(10);
  aPad->cd();
  tLaText->Draw();
}


//________________________________________________________________________________________________________________
TF1* Purity::GetFullFit(int aType)
{
  cout << "GetFullFit(" << aType << ") for fCombinedPurityName = " << fCombinedPurityName << endl;

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

    tCombinedPurity->Fit("fitBgdPlusSignal","0ML", "", (*vBgFitLow)(1), (*vBgFitHigh)(0));
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
    fitBgdPlusSignal->SetParameter(7, 0.001);
    fitBgdPlusSignal->SetParLimits(7, 0., 0.5);

    fitBgdPlusSignal->FixParameter(8, 0.);

    fitBgdPlusSignal->SetParLimits(6, 0.9*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()), 1.1*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));

    tCombinedPurity->Fit("fitBgdPlusSignal","0ML", "", (*vBgFitLow)(0), (*vBgFitHigh)(1));
  }

  else if(aType==2)
  {
    fitBgdPlusSignal = new TF1("fitBgdPlusSignal", FitFunctionTwoGaussianPlusLinear, tCombinedPurity->GetBinLowEdge(1), tCombinedPurity->GetBinLowEdge(tCombinedPurity->GetNbinsX()+1), 10);

    fitBgdPlusSignal->SetParameter(0, fitBgd->GetParameter(0));
    fitBgdPlusSignal->SetParameter(1, fitBgd->GetParameter(1));

    fitBgdPlusSignal->SetParameter(2, tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(3, tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParameter(4, 0.001);
    fitBgdPlusSignal->SetParLimits(4, 0., 0.5);
    fitBgdPlusSignal->FixParameter(5, 0.0);

    fitBgdPlusSignal->SetParameter(6, 0.1*tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(7, tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParameter(8, 0.005);
    fitBgdPlusSignal->SetParLimits(8, 0., 0.5);
    fitBgdPlusSignal->FixParameter(9, 0.);

    fitBgdPlusSignal->SetParLimits(3, 0.99*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()), 1.01*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParLimits(7, 0.99*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()), 1.01*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));

    tCombinedPurity->Fit("fitBgdPlusSignal","0ML", "", (*vBgFitLow)(1), (*vBgFitHigh)(0));
  }

  else if(aType==3) 
  {
    fitBgdPlusSignal = new TF1("fitBgdPlusSignal", FitFunctionTwoGaussianPlusPoly, tCombinedPurity->GetBinLowEdge(1), tCombinedPurity->GetBinLowEdge(tCombinedPurity->GetNbinsX()+1), 13);

    fitBgdPlusSignal->FixParameter(0, fitBgd->GetParameter(0));
    fitBgdPlusSignal->FixParameter(1, fitBgd->GetParameter(1));
    fitBgdPlusSignal->FixParameter(2, fitBgd->GetParameter(2));
    fitBgdPlusSignal->FixParameter(3, fitBgd->GetParameter(3));
    fitBgdPlusSignal->FixParameter(4, fitBgd->GetParameter(4));

    fitBgdPlusSignal->SetParameter(5, tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(6, tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParameter(7, 0.001);
    fitBgdPlusSignal->SetParLimits(7, 0., 0.5);

    fitBgdPlusSignal->FixParameter(8, 0.);


    fitBgdPlusSignal->SetParameter(9, 0.1*tCombinedPurity->GetMaximum());
    fitBgdPlusSignal->SetParameter(10, tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParameter(11, 0.005);
    fitBgdPlusSignal->SetParLimits(11, 0., 0.5);
    fitBgdPlusSignal->FixParameter(12, 0.);


    fitBgdPlusSignal->SetParLimits(10, 0.99*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()), 1.01*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));
    fitBgdPlusSignal->SetParLimits(6, 0.99*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()), 1.01*tCombinedPurity->GetBinCenter(tCombinedPurity->GetMaximumBin()));

    tCombinedPurity->Fit("fitBgdPlusSignal","0ML", "", (*vBgFitLow)(0), (*vBgFitHigh)(1));
  }

  else assert(0);

  return fitBgdPlusSignal;
}
/*
//________________________________________________________________________________________________________________
double Purity::FitFunctionTwoGaussianPlusPoly(double *x, double *par)
{
  //12 parameters
  double tBgd =    par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
  double tGauss1 =  par[5]*exp(-0.5*(pow((x[0]-par[6])/par[7],2.0))) + par[8];
  double tGauss2 =  par[9]*exp(-0.5*(pow((x[0]-par[10])/par[11],2.0))) + par[12];

  return tBgd+tGauss1+tGauss2;
}
*/

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

  //tCombinedPurity->GetXaxis()->SetTitle("#it{m}_{inv} (GeV/#it{c}^{2})");
  TString tXaxTitle = "#it{m}";
  if( (fParticleType == kLam) 
   || (fParticleType == kALam) ) tXaxTitle.Append("_{p#pi}");
  else if(fParticleType == kK0)  tXaxTitle.Append("_{#pi#pi}");
  else assert(0);
  tXaxTitle.Append(" (GeV/#it{c}^{2})");
  tCombinedPurity->GetXaxis()->SetTitle(tXaxTitle);
  tCombinedPurity->SetLabelSize(0.06*(aPadScaleY/aPadScaleX), "x");
  tCombinedPurity->SetTitleSize(0.075*(aPadScaleY/aPadScaleX), "x");
  tCombinedPurity->GetXaxis()->SetTitleOffset(0.375*(aPadScaleY/aPadScaleX));
  tCombinedPurity->GetXaxis()->SetTickLength(0.06*(aPadScaleY/aPadScaleX));
  tCombinedPurity->GetXaxis()->SetNdivisions(510);

  //tCombinedPurity->GetYaxis()->SetTitle("dN/d#it{m}_{inv}");
  TString tYaxTitle = "dN/d#it{m}";
  if( (fParticleType == kLam) 
   || (fParticleType == kALam) ) tYaxTitle.Append("_{p#pi}");
  else if(fParticleType == kK0)  tYaxTitle.Append("_{#pi#pi}");
  else assert(0);
  tCombinedPurity->GetYaxis()->SetTitle(tYaxTitle);
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

//    double tMaxYAx = tMaxBg + tRangeBg/10.;
    double tMaxYAx = tMaxBg + tRangeBg;
    double tMinYAx = tMinBg - tRangeBg/10.;

    tCombinedPurity->GetXaxis()->SetRange(1,tCombinedPurity->GetNbinsX());
    tCombinedPurity->GetYaxis()->SetRangeUser(tMinYAx,tMaxYAx);
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
      tMu = 0.5*(fitFull->GetParameter(3) + fitFull->GetParameter(7));
      tSigma = 0.5*(fitFull->GetParameter(4) + fitFull->GetParameter(8));
    }
    else if(aTypeOfFit==3)
    {
      tMu = 0.5*(fitFull->GetParameter(6) + fitFull->GetParameter(10));
      tSigma = 0.5*(fitFull->GetParameter(7) + fitFull->GetParameter(11));
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


