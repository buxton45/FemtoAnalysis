/* SuperpositionFitBgd.cxx */


#include "SuperpositionFitBgd.h"

#ifdef __ROOT__
ClassImp(SuperpositionFitBgd)
#endif




//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
SuperpositionFitBgd::SuperpositionFitBgd(TH1D* aData, TH1D* aCf1, TH1D* aCf2, double aMinBgdFit, double aMaxBgdFit):
  fData(aData),
  fCf1(aCf1),
  fCf2(aCf2),
  fSupCf12(nullptr),

  fMinBgdFit(aMinBgdFit),
  fMaxBgdFit(aMaxBgdFit),

  fMinBgdFitBin(-1),
  fMaxBgdFitBin(-1),

  fN1(-1.),
  fN2(-1.),

  fMinuit(nullptr)
{

  fMinuit = new TMinuit(50);
//  fMinuit->SetPrintLevel(-1); //Same as fMinuit->Command("SET PRINT -1");
  int tErrFlg = 0;
  fMinuit->mnparm(0, "Par0", 0.5, 0.001, 0., 1., tErrFlg);

  assert(fData->GetBinWidth(1)==fCf1->GetBinWidth(1));
  assert(fData->GetBinWidth(1)==fCf2->GetBinWidth(1));
  
  fMinBgdFitBin = fData->FindBin(aMinBgdFit);
  fMaxBgdFitBin = fData->FindBin(aMaxBgdFit);

}



//________________________________________________________________________________________________________________
SuperpositionFitBgd::~SuperpositionFitBgd()
{
  /* no-op */
}

//________________________________________________________________________________________________________________
void SuperpositionFitBgd::CalculateBgdFitFunction(int &npar, double &chi2, double *par)
{
  double tChi=0., tChi2=0.;
  double tSuperpos = 0.;
  for(int i=fMinBgdFitBin; i<fMaxBgdFitBin; i++)
  {
    tSuperpos = par[0]*fCf1->GetBinContent(i) + (1.-par[0])*fCf2->GetBinContent(i);
    tChi = (fData->GetBinContent(i)-tSuperpos)/fData->GetBinError(i);
    tChi2 += tChi*tChi;
  }
  chi2 = tChi2;
}



//________________________________________________________________________________________________________________
void SuperpositionFitBgd::DoFit()
{
  double arglist[10];
  int tErrFlg = 0;

  // for max likelihood = 0.5, for chisq = 1.0
  arglist[0] = 1.;
  fMinuit->mnexcm("SET ERR", arglist ,1,tErrFlg);

  // Set strategy to be used
  // 0 = economize (many params and/or fcn takes a long time to calculate and/or not interested in precise param erros)
  // 1 = default
  // 2 = Minuit allowed to waste calls in order to ensure precision (fcn evaluated in short time and/or param erros must be reliable)
  arglist[0] = 2;
  fMinuit->mnexcm("SET STR", arglist ,1,tErrFlg);

  fMinuit->SetPrintLevel(0);
  // Now ready for minimization step
  arglist[0] = 50000;
  arglist[1] = 0.001;
//  arglist[1] = 0.1;
  fMinuit->mnexcm("MIGRAD", arglist ,2,tErrFlg);

  if(tErrFlg != 0)
  {
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    cout << "WARNING!!!!!" << endl << "tErrFlg != 0 for the Cf: " << endl;
    cout << "tErrFlg = " << tErrFlg << endl;
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  }
  assert(tErrFlg==0);

  double tParVal, tParErr;
  fMinuit->GetParameter(0, tParVal, tParErr);

  fN1 = tParVal;
  fN2 = 1.0-tParVal;

  fSupCf12 = (TH1D*)fCf1->Clone("fSupCf12");
  fSupCf12->Scale(fN1);
  fSupCf12->Add(fCf2, fN2);

  fSupCf12->SetMarkerStyle(20);
}













