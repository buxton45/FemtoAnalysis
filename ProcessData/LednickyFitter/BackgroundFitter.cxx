/* BackgroundFitter.cxx */


#include "BackgroundFitter.h"

#ifdef __ROOT__
ClassImp(BackgroundFitter)
#endif




//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
BackgroundFitter::BackgroundFitter(TH1* aNum, TH1* aDen, TH1* aCf, NonFlatBgdFitType aBgdFitType, FitType aFitType, 
                                   double aMinBgdFit, double aMaxBgdFit, double aKStarMinNorm, double aKStarMaxNorm):
  fNum(aNum),
  fDen(aDen),
  fCf(aCf),
  fNonFlatBgdFitType(aBgdFitType),
  fFitType(aFitType),

  fMinBgdFit(aMinBgdFit),
  fMaxBgdFit(aMaxBgdFit),
  fKStarMinNorm(aKStarMinNorm),
  fKStarMaxNorm(aKStarMaxNorm),

  fMinuit(nullptr)
{
  fMinuit = new TMinuit(50);

  if(aFitType==kChi2PML)
  {
    int tErrFlg = 0;
    if(fNonFlatBgdFitType==kLinear)
    {
      fMinuit->mnparm(0, "Par0", 0., 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(1, "Par1", 1., 0.001, 0., 0., tErrFlg);
    }
    else if(fNonFlatBgdFitType == kQuadratic)
    {
      fMinuit->mnparm(0, "Par0", 0., 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(1, "Par1", 0., 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(2, "Par2", 1., 0.01, 0., 0., tErrFlg);

    }
    else if(fNonFlatBgdFitType == kGaussian)
    {
      fMinuit->mnparm(0, "Par0", 0.1, 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(1, "Par1", 0., 0.01, 0., 0., tErrFlg);
//      fMinuit->mnparm(1, "Par1", 0., 0.01, -0.05, 0.05, tErrFlg);
      fMinuit->mnparm(2, "Par2", 0.5, 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(3, "Par3", 0.96, 0.01, 0., 0., tErrFlg);
    }
    else assert(0);
  }

}



//________________________________________________________________________________________________________________
BackgroundFitter::~BackgroundFitter()
{
  /* no-op */
}

//________________________________________________________________________________________________________________
double BackgroundFitter::FitFunctionLinear(double *x, double *par)
{
  return par[0]*x[0] + par[1];
}

//________________________________________________________________________________________________________________
double BackgroundFitter::FitFunctionQuadratic(double *x, double *par)
{
  return par[0]*x[0]*x[0] + par[1]*x[0] + par[2];
}
//________________________________________________________________________________________________________________
double BackgroundFitter::FitFunctionGaussian(double *x, double *par)
{
//  return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3];
  return (1./(par[2]*sqrt(TMath::TwoPi())))*par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3];
}


//________________________________________________________________________________________________________________
double BackgroundFitter::GetPmlValue(double aNumContent, double aDenContent, double aCfContent)
{
  double tTerm1 = aNumContent*log(  (aCfContent*(aNumContent+aDenContent)) / (aNumContent*(aCfContent+1))  );
  double tTerm2 = aDenContent*log(  (aNumContent+aDenContent) / (aDenContent*(aCfContent+1))  );
  double tChi2PML = -2.0*(tTerm1+tTerm2);
  return tChi2PML;
}


//________________________________________________________________________________________________________________
void BackgroundFitter::CalculateBgdFitFunction(int &npar, double &chi2, double *par)
{
  int tMinBgdFitBin = fNum->FindBin(fMinBgdFit);
  int tMaxBgdFitBin = fNum->FindBin(fMaxBgdFit);

  double tChi2 = 0.;
  double x[1];
  double tFitVal=0., tNumContent=0., tDenContent=0.;
  double tScale = fNum->Integral(fNum->FindBin(0.32), fNum->FindBin(0.40))/fDen->Integral(fDen->FindBin(0.32), fDen->FindBin(0.40));
  for(int iBin=tMinBgdFitBin; iBin<=tMaxBgdFitBin; iBin++)
  {
    tNumContent = fNum->GetBinContent(iBin);
    tDenContent = tScale*fDen->GetBinContent(iBin);
    x[0] = fNum->GetBinCenter(iBin);

    if(fNonFlatBgdFitType==kLinear) tFitVal = FitFunctionLinear(x, par);
    else if(fNonFlatBgdFitType == kQuadratic) tFitVal = FitFunctionQuadratic(x, par);
    else if(fNonFlatBgdFitType == kGaussian) tFitVal = FitFunctionGaussian(x, par);
    else assert(0);

    tChi2 += GetPmlValue(tNumContent, tDenContent, tFitVal);
  }

  chi2 = tChi2;
}

//________________________________________________________________________________________________________________
TF1* BackgroundFitter::FitNonFlatBackgroundPML()
{
  cout << "*****************************************************************************" << endl;
  cout << "Starting to fit background " << endl;
  cout << "*****************************************************************************" << endl;

  fMinuit->SetPrintLevel(0);  // -1=quiet, 0=normal, 1=verbose (more options using mnexcm("SET PRI", ...)

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

  // Now ready for minimization step
  arglist[0] = 50000;
  arglist[1] = 0.1;
  fMinuit->mnexcm("MIGRAD", arglist ,2,tErrFlg);

  if(tErrFlg != 0)
  {
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    //cout << "WARNING!!!!!" << endl << "tErrFlg != 0 for the Cf: " << fCfName << endl;
    cout << "WARNING!!!!!" << endl << "tErrFlg != 0 for the Cf: " << endl;
    cout << "tErrFlg = " << tErrFlg << endl;
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  }
  assert(tErrFlg==0);

  // Print results
//  fMinuit->mnstat(fChi2,fEdm,fErrDef,fNvpar,fNparx,fIcstat);
//  fMinuit->mnprin(3,fChi2);

  //---------------------------------

  TF1* tNonFlatBackground;
  double tPar=0., tParErr=0.;
  int tNPars=0;

  if(fNonFlatBgdFitType==kLinear)
  {
    tNPars=2;
    TString tFitName = TString("NonFlatBackgroundFitLinear_") + TString(fCf->GetTitle());
    tNonFlatBackground = new TF1(tFitName,FitFunctionLinear,0.,1.,2);
  }
  else if(fNonFlatBgdFitType == kQuadratic)
  {
    tNPars=3;
    TString tFitName = TString("NonFlatBackgroundFitQuadratic_") + TString(fCf->GetTitle());
    tNonFlatBackground = new TF1(tFitName,FitFunctionQuadratic,0.,1.,3);
  }
  else if(fNonFlatBgdFitType == kGaussian)
  {
    tNPars=4;
    TString tFitName = TString("NonFlatBackgroundFitGaussian_") + TString(fCf->GetTitle());
    tNonFlatBackground = new TF1(tFitName,FitFunctionGaussian,0.,1.,4);
  }
  else assert(0);


  for(int i=0; i<tNPars; i++)
  {
    fMinuit->GetParameter(i, tPar, tParErr);
    tNonFlatBackground->SetParameter(i, tPar);
  }

  return tNonFlatBackground;
}


//________________________________________________________________________________________________________________
TF1* BackgroundFitter::FitNonFlatBackgroundSimple()
{
  TF1* tNonFlatBackground;

  if(fNonFlatBgdFitType==kLinear)
  {
    TString tFitName = TString("NonFlatBackgroundFitLinear_") + TString(fCf->GetTitle());
    tNonFlatBackground = new TF1(tFitName,FitFunctionLinear,0.,1.,2);
      tNonFlatBackground->SetParameter(0,0.);
      tNonFlatBackground->SetParameter(1,1.);
    fCf->Fit(tFitName,"0q","",fMinBgdFit,fMaxBgdFit);
  }
  else if(fNonFlatBgdFitType == kQuadratic)
  {
    TString tFitName = TString("NonFlatBackgroundFitQuadratic_") + TString(fCf->GetTitle());
    tNonFlatBackground = new TF1(tFitName,FitFunctionQuadratic,0.,1.,3);
      tNonFlatBackground->SetParameter(0,0.);
      tNonFlatBackground->SetParameter(1,0.);
      tNonFlatBackground->SetParameter(2,1.);
    fCf->Fit(tFitName,"0q","",fMinBgdFit,fMaxBgdFit);
  }
  else if(fNonFlatBgdFitType == kGaussian)
  {
    TString tFitName = TString("NonFlatBackgroundFitGaussian_") + TString(fCf->GetTitle());
    tNonFlatBackground = new TF1(tFitName,FitFunctionGaussian,0.,1.,4);
      tNonFlatBackground->SetParameter(0,0.1);
      tNonFlatBackground->SetParameter(1,0.);
      tNonFlatBackground->SetParameter(2,0.5);
      tNonFlatBackground->SetParameter(3,0.96);

      tNonFlatBackground->SetParLimits(1,-0.05,0.05);

    fCf->Fit(tFitName,"0q","",fMinBgdFit,fMaxBgdFit);
  }
  else assert(0);

  return tNonFlatBackground;
}


//________________________________________________________________________________________________________________
TF1* BackgroundFitter::FitNonFlatBackground()
{
  TF1* tReturnBgdFit;

  if(fFitType==kChi2PML) tReturnBgdFit = FitNonFlatBackgroundPML();
  else if(fFitType==kChi2) tReturnBgdFit = FitNonFlatBackgroundSimple();
  else assert(0);

  return tReturnBgdFit;
}





