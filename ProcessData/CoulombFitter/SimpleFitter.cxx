///////////////////////////////////////////////////////////////////////////
// SimpleFitter:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "SimpleFitter.h"

#ifdef __ROOT__
ClassImp(SimpleFitter)
#endif


double GetGaussian(double *x, double *par)
{
  //par[0] = mean
  //par[1] = sigma

  double tRoot2Pi = sqrt(2.0*M_PI);
  double tReturn = (1.0/(par[1]*tRoot2Pi))*exp(-(x[0]-par[0])*(x[0]-par[0])/(2*par[1]*par[1]));
  return tReturn;

}

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

//________________________________________________________________________________________________________________
SimpleFitter::SimpleFitter(TString aName, double aMean, double aSigma, int aNBins, double aMin, double aMax) :
  fMinuit(0),
  fIdealGaussian(0),
  fIdealGaussianHistogram(0),
  fIdealMean(aMean),
  fIdealSigma(aSigma),
  fNBinsIdeal(aNBins),
  fIdealMin(aMin),
  fIdealMax(aMax),

  fChi2(0),
  fErrFlg(0),
  fNpFits(0),
  fNDF(0)

{
  int tErrFlg = 0;
  fMinuit = new TMinuit(50);
    fMinuit->mnparm(0,"Mean",0.,1.0,-100.,100.,tErrFlg);
    fMinuit->mnparm(1,"Sigma",1.0,1.0,0.,100.,tErrFlg);

  fIdealGaussian = new TF1(aName,GetGaussian,fIdealMin,fIdealMax,2);
    fIdealGaussian->SetParameter(0,fIdealMean);
    fIdealGaussian->SetParameter(1,fIdealSigma);
  
  TString tHistogramName = aName + TString("Histogram");
  fIdealGaussianHistogram = new TH1D(tHistogramName,tHistogramName,aNBins,fIdealMin,fIdealMax);
/*
  int tNFills = 100000;
  fIdealGaussianHistogram->FillRandom(aName,tNFills);
  fIdealGaussianHistogram->Scale(1./tNFills);
*/
  double tValue;
  double tX[1];
  double tPar[2];
    tPar[0] = fIdealMean;
    tPar[1] = fIdealSigma;
  for(int i=1; i<=fIdealGaussianHistogram->GetNbinsX(); i++)
  {
    tX[0] = fIdealGaussianHistogram->GetXaxis()->GetBinCenter(i);
    tValue = GetGaussian(tX,tPar);
    fIdealGaussianHistogram->SetBinContent(i,tValue);
    fIdealGaussianHistogram->SetBinError(i,1.);
  }


}



//________________________________________________________________________________________________________________
SimpleFitter::~SimpleFitter()
{

}

//________________________________________________________________________________________________________________
double SimpleFitter::GetFitCfContentNumerically(double aMin, double aMax, double *par)
{
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tDistribution(par[0],par[1]);

  int tMaxCalls = 100;
  int tCounter = 0;
  double tValue = 0.;
  double tTotalValue = 0.;

  while(tCounter < tMaxCalls)
  {
//cout << "tCounter = " << tCounter << endl;
    tValue = tDistribution(generator);
    if(tValue >= aMin && tValue < aMax)
    {
      tTotalValue += tValue;
      tCounter ++;
    }
  }

  tTotalValue /= tCounter;
  return tTotalValue;
}

//________________________________________________________________________________________________________________
TH1D* SimpleFitter::GetEntireFitCfContentNumerically(double *par)
{
  TH1D* tReturnHist = new TH1D("tFit","tFit",fNBinsIdeal,fIdealMin,fIdealMax);

  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tDistribution(par[0],par[1]);

  int tMaxCalls = 100000;
  int tCounter = 0;
  double tValue = 0.;
  double tTotalValue = 0.;

  while(tCounter < tMaxCalls)
  {
    tValue = tDistribution(generator);
    tReturnHist->Fill(tValue);
    tCounter ++;
  }

  double tIntegral = tReturnHist->Integral("width");
  tReturnHist->Scale(1./tIntegral);

  return tReturnHist;
}


//________________________________________________________________________________________________________________
void SimpleFitter::CalculateChi2Numerically(int &npar, double &chi2, double *par)
{
  cout << "par[0] = " << par[0] << endl;
  cout << "par[1] = " << par[1] << endl << endl;

  int tNBinsXToFit = fIdealGaussianHistogram->GetNbinsX();
  TAxis* tXaxis = fIdealGaussianHistogram->GetXaxis();

  double tmp;
  double tXmin, tXmax;
  double tFitContent;
  fChi2 = 0.;
  fNpFits = 0;

  TH1D* tFitHistogram = GetEntireFitCfContentNumerically(par);

  for(int ix=1; ix<=tNBinsXToFit; ix++)
  {
//    tXmin = tXaxis->GetBinLowEdge(ix);
//    tXmax = tXaxis->GetBinLowEdge(ix+1);

//    tFitContent = GetFitCfContentNumerically(tXmin,tXmax,par);
    tFitContent = tFitHistogram->GetBinContent(ix);
cout << "tFitContent                                = " << tFitContent << endl;
cout << "fIdealGaussianHistogram->GetBinContent(ix) = " << fIdealGaussianHistogram->GetBinContent(ix) << endl <<  endl;
    tmp = (fIdealGaussianHistogram->GetBinContent(ix) - tFitContent)/fIdealGaussianHistogram->GetBinError(ix);
    fChi2 += tmp*tmp;
  }

  if(std::isnan(fChi2) || std::isinf(fChi2))
  {
    cout << "WARNING: fChi2 = nan, setting it equal to 10^9-----------------------------------------" << endl << endl;
    fChi2 = pow(10,9);
  }

  chi2 = fChi2;

  cout << "fChi2 = " << fChi2 << endl << endl;

}


//________________________________________________________________________________________________________________
void SimpleFitter::CalculateChi2(int &npar, double &chi2, double *par)
{
  cout << "par[0] = " << par[0] << endl;
  cout << "par[1] = " << par[1] << endl << endl;

  int tNBinsXToFit = fIdealGaussianHistogram->GetNbinsX();
  TAxis* tXaxis = fIdealGaussianHistogram->GetXaxis();

  double tmp;
  double tXmin, tXmax;
  double tFitContent;
  fChi2 = 0.;
  fNpFits = 0;
  double x[1];

  for(int ix=1; ix<=tNBinsXToFit; ix++)
  {
    x[0] = fIdealGaussianHistogram->GetXaxis()->GetBinCenter(ix);
    tFitContent = GetGaussian(x,par);
cout << "tFitContent                                = " << tFitContent << endl;
cout << "fIdealGaussianHistogram->GetBinContent(ix) = " << fIdealGaussianHistogram->GetBinContent(ix) << endl <<  endl;
    tmp = (fIdealGaussianHistogram->GetBinContent(ix) - tFitContent)/fIdealGaussianHistogram->GetBinError(ix);
    fChi2 += tmp*tmp;
  }

  if(std::isnan(fChi2) || std::isinf(fChi2))
  {
    cout << "WARNING: fChi2 = nan, setting it equal to 10^9-----------------------------------------" << endl << endl;
    fChi2 = pow(10,9);
  }

  chi2 = fChi2;

  cout << "fChi2 = " << fChi2 << endl << endl;

}

//________________________________________________________________________________________________________________
void SimpleFitter::DoFit()
{
  cout << "*****************************************************************************" << endl;
  //cout << "Starting to fit " << fCfName << endl;
  cout << "Starting to fit " << endl;
  cout << "*****************************************************************************" << endl;

  fMinuit->SetPrintLevel(0);  // -1=quiet, 0=normal, 1=verbose (more options using mnexcm("SET PRI", ...)

  double arglist[10];
  fErrFlg = 0;

  // for max likelihood = 0.5, for chisq = 1.0
  arglist[0] = 1.;
  fMinuit->mnexcm("SET ERR", arglist ,1,fErrFlg);

  arglist[0] = 0.0000001;
  fMinuit->mnexcm("SET EPS", arglist, 1, fErrFlg);

  // Set strategy to be used
  // 0 = economize (many params and/or fcn takes a long time to calculate and/or not interested in precise param erros)
  // 1 = default
  // 2 = Minuit allowed to waste calls in order to ensure precision (fcn evaluated in short time and/or param erros must be reliable)
  arglist[0] = 1;
  fMinuit->mnexcm("SET STR", arglist ,1,fErrFlg);

  // Now ready for minimization step
  arglist[0] = 5000;
  arglist[1] = 1.;
//  fMinuit->mnexcm("MIGRAD", arglist ,2,fErrFlg);  //I do not think MIGRAD will work here because depends on derivates, etc
  fMinuit->mnexcm("MINI", arglist ,2,fErrFlg);  //MINI also uses MIGRAD, so probably won't work
//  fMinuit->mnexcm("SIM", arglist ,2,fErrFlg);  //MINI also uses MIGRAD, so probably won't work
//  fMinuit->mnscan();

  if(fErrFlg != 0)
  {
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    //cout << "WARNING!!!!!" << endl << "fErrFlg != 0 for the Cf: " << fCfName << endl;
    cout << "WARNING!!!!!" << endl << "fErrFlg != 0 for the Cf: " << endl;
    cout << "fErrFlg = " << fErrFlg << endl;
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  }

  // Print results
  fMinuit->mnstat(fChi2,fEdm,fErrDef,fNvpar,fNparx,fIcstat);
  fMinuit->mnprin(3,fChi2);

  fNDF = fNpFits-fNvpar;


}

