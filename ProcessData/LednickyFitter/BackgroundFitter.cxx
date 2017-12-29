/* BackgroundFitter.cxx */


#include "BackgroundFitter.h"

#ifdef __ROOT__
ClassImp(BackgroundFitter)
#endif




//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
BackgroundFitter::BackgroundFitter(TH1* aNum, TH1* aDen, TH1* aCf, NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf, 
                                   double aMinBgdFit, double aMaxBgdFit, double aKStarMinNorm, double aKStarMaxNorm):
  fNum(aNum),
  fDen(aDen),
  fCf(aCf),
  fNonFlatBgdFitType(aBgdFitType),
  fFitType(aFitType),

  fNormalizeFitToCf(aNormalizeFitToCf),
  fScale(1.),

  fMinBgdFit(aMinBgdFit),
  fMaxBgdFit(aMaxBgdFit),
  fKStarMinNorm(aKStarMinNorm),
  fKStarMaxNorm(aKStarMaxNorm),

  fMinuit(nullptr)
{
  fMinuit = new TMinuit(50);
  fMinuit->SetPrintLevel(-1); //Same as fMinuit->Command("SET PRINT -1");

  if(aFitType==kChi2) assert(!fNormalizeFitToCf); //In this case, the Cf is fit, which is already normalized to unity, 
                                                        // so no need to normalize again.
  
  if(aFitType==kChi2PML)
  {
    fScale = fNum->Integral(fNum->FindBin(fKStarMinNorm), fNum->FindBin(fKStarMaxNorm))/fDen->Integral(fDen->FindBin(fKStarMinNorm), fDen->FindBin(fKStarMaxNorm));
    int tErrFlg = 0;
    if(fNonFlatBgdFitType==kLinear)
    {
      //par[0]*x[0] + par[1]
      fMinuit->mnparm(0, "Par0", 0., 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(1, "Par1", fScale, 0.001, 0., 0., tErrFlg);
    }
    else if(fNonFlatBgdFitType == kQuadratic)
    {
      //par[0]*x[0]*x[0] + par[1]*x[0] + par[2]
      fMinuit->mnparm(0, "Par0", 0., 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(1, "Par1", 0., 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(2, "Par2", 1., 0.01, 0., 0., tErrFlg);

    }
    else if(fNonFlatBgdFitType == kGaussian)
    {
      //par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3]
      fMinuit->mnparm(0, "Par0", 0.1, 0.1, 0., 0., tErrFlg);
      fMinuit->mnparm(1, "Par1", 0., 0.1, 0., 0., tErrFlg);
//      fMinuit->mnparm(1, "Par1", 0., 0.01, -0.05, 0.05, tErrFlg);
      fMinuit->mnparm(2, "Par2", 0.5, 0.01, 0., 0., tErrFlg);
      fMinuit->mnparm(3, "Par3", 0.9, 0.1, 0., 0., tErrFlg);

      fMinuit->FixParameter(1);
    }
    else assert(0);
  }

  assert(fMinBgdFit > fKStarMaxNorm);

}



//________________________________________________________________________________________________________________
BackgroundFitter::~BackgroundFitter()
{
  /* no-op */
}

//________________________________________________________________________________________________________________
void BackgroundFitter::PrintFitFunctionInfo()
{
  if(!fNormalizeFitToCf)
  {
    if(fNonFlatBgdFitType==kLinear)
    {
      cout << "Using fNonFlatBgdFitType=kLinear" << endl;
      cout << "==> \t Form: par[0]*x[0] + par[1]" << endl;
    }
    else if(fNonFlatBgdFitType == kQuadratic)
    {
      cout << "Using fNonFlatBgdFitType=kQuadratic" << endl;
      cout << "==> \t Form: par[0]*x[0]*x[0] + par[1]*x[0] + par[2]" << endl;
    }
    else if(fNonFlatBgdFitType == kGaussian)
    {
      cout << "Using fNonFlatBgdFitType=kGaussian" << endl;
      cout << "==> \t Form: par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3]" << endl;
    }
    else assert(0);
  }
  else
  {
    if(fNonFlatBgdFitType==kLinear)
    {
      cout << "Using fNonFlatBgdFitType=kLinear" << endl;
      cout << "==> \t Form: par[2]*(par[0]*x[0] + par[1])" << endl;
    }
    else if(fNonFlatBgdFitType == kQuadratic)
    {
      cout << "Using fNonFlatBgdFitType=kQuadratic" << endl;
      cout << "==> \t Form: par[3]*(par[0]*x[0]*x[0] + par[1]*x[0] + par[2])" << endl;
    }
    else if(fNonFlatBgdFitType == kGaussian)
    {
      cout << "Using fNonFlatBgdFitType=kGaussian" << endl;
      cout << "==> \t Form: par[4]*(par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3])" << endl;
    }
    else assert(0);
  }


  cout << "\t\t BgdFit Region = [" << fMinBgdFit << ", " << fMaxBgdFit << "]" << endl;
  cout << "\t\t CfNorm Region = [" << fKStarMinNorm << ", " << fKStarMaxNorm << "]" << endl;
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
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3];
}



//________________________________________________________________________________________________________________
double BackgroundFitter::NormalizedFitFunctionLinear(double *x, double *par)
{
  return par[2]*FitFunctionLinear(x, par);
}

//________________________________________________________________________________________________________________
double BackgroundFitter::NormalizedFitFunctionQuadratic(double *x, double *par)
{
  return par[3]*FitFunctionQuadratic(x, par);
}
//________________________________________________________________________________________________________________
double BackgroundFitter::NormalizedFitFunctionGaussian(double *x, double *par)
{
  return par[4]*FitFunctionGaussian(x, par);
}



//________________________________________________________________________________________________________________
double BackgroundFitter::AddTwoFitFunctionsLinear(double *x, double *par)
{
  //Num counts are par[2] and par[5]!
  td1dVec tParsLin1{par[0], par[1]};
  double tNumCounts1 = par[2];

  td1dVec tParsLin2{par[3], par[4]};
  double tNumCounts2 = par[5];

  double tLin1 = FitFunctionLinear(x, tParsLin1.data());
  double tLin2 = FitFunctionLinear(x, tParsLin2.data());

  return (tNumCounts1*tLin1 + tNumCounts2*tLin2)/(tNumCounts1+tNumCounts2);
}

//________________________________________________________________________________________________________________
double BackgroundFitter::AddTwoFitFunctionsQuadratic(double *x, double *par)
{
  //Num counts are par[3] and par[7]!
  td1dVec tParsQuad1{par[0], par[1], par[2]};
  double tNumCounts1 = par[3];

  td1dVec tParsQuad2{par[4], par[5], par[6]};
  double tNumCounts2 = par[7];

  double tQuad1 = FitFunctionQuadratic(x, tParsQuad1.data());
  double tQuad2 = FitFunctionQuadratic(x, tParsQuad2.data());

  return (tNumCounts1*tQuad1 + tNumCounts2*tQuad2)/(tNumCounts1+tNumCounts2);
}
//________________________________________________________________________________________________________________
double BackgroundFitter::AddTwoFitFunctionsGaussian(double *x, double *par)
{
  //Num counts are par[4] and par[9]
  td1dVec tParsGauss1{par[0], par[1], par[2], par[3]};
  double tNumCounts1 = par[4];

  td1dVec tParsGauss2{par[5], par[6], par[7], par[8]};
  double tNumCounts2 = par[9];

  double tGauss1 = FitFunctionGaussian(x, tParsGauss1.data());
  double tGauss2 = FitFunctionGaussian(x, tParsGauss2.data());

  return (tNumCounts1*tGauss1 + tNumCounts2*tGauss2)/(tNumCounts1+tNumCounts2);
}

//________________________________________________________________________________________________________________
double BackgroundFitter::AddTwoNormalizedFitFunctionsLinear(double *x, double *par)
{
  //Num counts are par[3] and par[7]!
  td1dVec tParsNormLin1{par[0], par[1], par[2]};
  double tNumCounts1 = par[3];

  td1dVec tParsNormLin2{par[4], par[5], par[6]};
  double tNumCounts2 = par[7];

  double tNormLin1 = NormalizedFitFunctionLinear(x, tParsNormLin1.data());
  double tNormLin2 = NormalizedFitFunctionLinear(x, tParsNormLin2.data());

  return (tNumCounts1*tNormLin1 + tNumCounts2*tNormLin2)/(tNumCounts1+tNumCounts2);
}

//________________________________________________________________________________________________________________
double BackgroundFitter::AddTwoNormalizedFitFunctionsQuadratic(double *x, double *par)
{
  //Num counts are par[4] and par[9]!
  td1dVec tParsNormQuad1{par[0], par[1], par[2], par[3]};
  double tNumCounts1 = par[4];

  td1dVec tParsNormQuad2{par[5], par[6], par[7], par[8]};
  double tNumCounts2 = par[9];

  double tNormQuad1 = NormalizedFitFunctionQuadratic(x, tParsNormQuad1.data());
  double tNormQuad2 = NormalizedFitFunctionQuadratic(x, tParsNormQuad2.data());

  return (tNumCounts1*tNormQuad1 + tNumCounts2*tNormQuad2)/(tNumCounts1+tNumCounts2);
}
//________________________________________________________________________________________________________________
double BackgroundFitter::AddTwoNormalizedFitFunctionsGaussian(double *x, double *par)
{
  //Num counts are par[5] and par[11]
  td1dVec tParsNormGauss1{par[0], par[1], par[2], par[3], par[4]};
  double tNumCounts1 = par[5];

  td1dVec tParsNormGauss2{par[6], par[7], par[8], par[9], par[10]};
  double tNumCounts2 = par[11];

  double tNormGauss1 = NormalizedFitFunctionGaussian(x, tParsNormGauss1.data());
  double tNormGauss2 = NormalizedFitFunctionGaussian(x, tParsNormGauss2.data());

  return (tNumCounts1*tNormGauss1 + tNumCounts2*tNormGauss2)/(tNumCounts1+tNumCounts2);
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
  for(int iBin=tMinBgdFitBin; iBin<=tMaxBgdFitBin; iBin++)
  {
    tNumContent = fNum->GetBinContent(iBin);
    tDenContent = fDen->GetBinContent(iBin);

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
  cout << endl << endl;
  cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
  cout << "Starting to fit background" << endl;
  cout << "\tUsing PML with Minuit using num & den separately" << endl;
  cout << "\tfor " << fCf->GetName() << endl;
  cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
  PrintFitFunctionInfo();

//  fMinuit->SetPrintLevel(0);  // -1=quiet, 0=normal, 1=verbose (more options using mnexcm("SET PRI", ...)

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

  //---------------------------------

  TF1* tNonFlatBackground;
  double tPar=0., tParErr=0.;
  int tNPars=0;
  TString tFitName = TString::Format("NonFlatBackgroundFit%s_%s", cNonFlatBgdFitTypeTags[fNonFlatBgdFitType], fCf->GetTitle());

  if(!fNormalizeFitToCf)
  {
    if(fNonFlatBgdFitType==kLinear)
    {
      tNPars=2;
      tNonFlatBackground = new TF1(tFitName,FitFunctionLinear,0.,1.,tNPars);
    }
    else if(fNonFlatBgdFitType == kQuadratic)
    {
      tNPars=3;
      tNonFlatBackground = new TF1(tFitName,FitFunctionQuadratic,0.,1.,tNPars);
    }
    else if(fNonFlatBgdFitType == kGaussian)
    {
      tNPars=4;
      tNonFlatBackground = new TF1(tFitName,FitFunctionGaussian,0.,1.,tNPars);
    }
    else assert(0);
  }
  else
  {
    tFitName = TString("Normalized") + tFitName;
    if(fNonFlatBgdFitType==kLinear)
    {
      tNPars=2;
      tNonFlatBackground = new TF1(tFitName,NormalizedFitFunctionLinear,0.,1.,tNPars+1);
    }
    else if(fNonFlatBgdFitType == kQuadratic)
    {
      tNPars=3;
      tNonFlatBackground = new TF1(tFitName,NormalizedFitFunctionQuadratic,0.,1.,tNPars+1);
    }
    else if(fNonFlatBgdFitType == kGaussian)
    {
      tNPars=4;
      tNonFlatBackground = new TF1(tFitName,NormalizedFitFunctionGaussian,0.,1.,tNPars+1);
    }
    else assert(0);
  }

  //---------------------------------

  for(int i=0; i<tNPars; i++)
  {
    fMinuit->GetParameter(i, tPar, tParErr);
    tNonFlatBackground->SetParameter(i, tPar);
  }
  if(fNormalizeFitToCf) tNonFlatBackground->SetParameter(tNPars, 1./fScale);

  cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
  return tNonFlatBackground;
}


//________________________________________________________________________________________________________________
TF1* BackgroundFitter::FitNonFlatBackgroundSimple()
{
  cout << endl << endl;
  cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
  cout << "Starting to fit background" << endl;
  cout << "\tUsing simple TH1::Fit method with Cf" << endl;
  cout << "\tfor " << fCf->GetName() << endl;
  cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
  PrintFitFunctionInfo();

  TF1* tNonFlatBackground;
  TString tFitName = TString::Format("NonFlatBackgroundFit%s_%s", cNonFlatBgdFitTypeTags[fNonFlatBgdFitType], fCf->GetTitle());

  if(fNonFlatBgdFitType==kLinear)
  {
    tNonFlatBackground = new TF1(tFitName,FitFunctionLinear,0.,1.,2);
      tNonFlatBackground->SetParameter(0,0.);
      tNonFlatBackground->SetParameter(1,1.);
    fCf->Fit(tFitName,"0q","",fMinBgdFit,fMaxBgdFit);
  }
  else if(fNonFlatBgdFitType == kQuadratic)
  {
    tNonFlatBackground = new TF1(tFitName,FitFunctionQuadratic,0.,1.,3);
      tNonFlatBackground->SetParameter(0,0.);
      tNonFlatBackground->SetParameter(1,0.);
      tNonFlatBackground->SetParameter(2,1.);
    fCf->Fit(tFitName,"0q","",fMinBgdFit,fMaxBgdFit);
  }
  else if(fNonFlatBgdFitType == kGaussian)
  {
    tNonFlatBackground = new TF1(tFitName,FitFunctionGaussian,0.,1.,4);
      tNonFlatBackground->SetParameter(0,0.1);
      tNonFlatBackground->SetParameter(1,0.);
      tNonFlatBackground->SetParameter(2,0.5);
      tNonFlatBackground->SetParameter(3,0.96);

//      tNonFlatBackground->SetParLimits(1,-0.05,0.05);
      tNonFlatBackground->FixParameter(1,0.);

    fCf->Fit(tFitName,"0q","",fMinBgdFit,fMaxBgdFit);
  }
  else assert(0);

  for(int i=0; i<tNonFlatBackground->GetNpar(); i++) cout << "Par[" << i << "] = " << tNonFlatBackground->GetParameter(i) << endl;

  cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
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




