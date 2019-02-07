///////////////////////////////////////////////////////////////////////////
// StrippedSimpleFitter:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "StrippedSimpleFitter.h"

#ifdef __ROOT__
ClassImp(StrippedSimpleFitter)
#endif

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
StrippedSimpleFitter::StrippedSimpleFitter(TH1* aNum, TH1* aDen, double aMaxFitKStar, double aMinNormKStar, double aMaxNormKStar):
  fNum(aNum),
  fDen(aDen),
  fCf(nullptr),

  fFitType(kChi2PML),
  fVerbose(false),
  fMinuit(nullptr),

  fMaxFitKStar(aMaxFitKStar),
  fMinNormKStar(aMinNormKStar),
  fMaxNormKStar(aMaxNormKStar),

  fNbinsXToBuild(0),
  fNbinsXToFit(0),
  fKStarBinWidth(0.),
  fKStarBinCenters(0),
  fRejectOmega(false),

  fChi2(0),
  fChi2GlobalMin(1000000000),
  fNpFits(0),
  fNDF(0),
  fErrFlg(0),
  fMinParams(0),
  fParErrors(0)

{
  fMinuit = new TMinuit(50);
  BuildCf();
}


//________________________________________________________________________________________________________________
StrippedSimpleFitter::~StrippedSimpleFitter()
{
  cout << "StrippedSimpleFitter object is being deleted!!!" << endl;
}


//________________________________________________________________________
double StrippedSimpleFitter::GetLednickyF1(double z)
{
  double result = (1./z)*Faddeeva::Dawson(z);
  return result;
}

//________________________________________________________________________
double StrippedSimpleFitter::GetLednickyF2(double z)
{
  double result = (1./z)*(1.-exp(-z*z));
  return result;
}

//________________________________________________________________________
double StrippedSimpleFitter::LednickyEq(double *x, double *par)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  //should probably do x[0] /= hbarc, but let me test first

  std::complex<double> f0 (par[2],par[3]);
  double Alpha = 0.; // alpha = 0 for non-identical
  double z = 2.*(x[0]/hbarc)*par[1];  //z = 2k*R, to be fed to GetLednickyF1(2)

  double C_QuantumStat = Alpha*exp(-z*z);  // will be zero for my analysis

  std::complex<double> ScattAmp = pow( (1./f0) + 0.5*par[4]*(x[0]/hbarc)*(x[0]/hbarc) - ImI*(x[0]/hbarc),-1);

  double C_FSI = (1+Alpha)*( 0.5*norm(ScattAmp)/(par[1]*par[1])*(1.-1./(2*sqrt(TMath::Pi()))*(par[4]/par[1])) + 2.*real(ScattAmp)/(par[1]*sqrt(TMath::Pi()))*GetLednickyF1(z) - (imag(ScattAmp)/par[1])*GetLednickyF2(z));

  double Cf = 1. + par[0]*(C_QuantumStat + C_FSI);
  //Cf *= par[5];

  return Cf;
}

//________________________________________________________________________
double StrippedSimpleFitter::LednickyEqWithNorm(double *x, double *par)
{
  double tUnNormCf = LednickyEq(x, par);
  double tNormCf = par[5]*tUnNormCf;
  return tNormCf;
}


//________________________________________________________________________________________________________________
double StrippedSimpleFitter::GetNumScale()
{
  return fNum->Integral(fNum->FindBin(fMinNormKStar), fNum->FindBin(fMaxNormKStar));
}

//________________________________________________________________________________________________________________
double StrippedSimpleFitter::GetDenScale()
{
  return fDen->Integral(fDen->FindBin(fMinNormKStar), fDen->FindBin(fMaxNormKStar));
}

//________________________________________________________________________________________________________________
void StrippedSimpleFitter::BuildCf()
{
  fCf = (TH1*)fNum->Clone("fCf");
  if(!fCf->GetSumw2N()) fCf->Sumw2();
  fCf->Divide(fDen);
  fCf->Scale(GetDenScale()/GetNumScale());
}


//________________________________________________________________________________________________________________
void StrippedSimpleFitter::PrintCurrentParamValues(int aNpar, double* aPar)
{
  for(int i=0; i<aNpar; i++) cout << "par[" << i << "] = " << aPar[i] << endl;
  cout << endl;
}

//________________________________________________________________________________________________________________
double StrippedSimpleFitter::GetChi2Value(int aKStarBin, TH1* aCfToFit, double aFitCfContent)
{
  double tChi = (aCfToFit->GetBinContent(aKStarBin) - aFitCfContent)/aCfToFit->GetBinError(aKStarBin);
  return tChi*tChi;
}


//________________________________________________________________________________________________________________
double StrippedSimpleFitter::GetPmlValue(double aNumContent, double aDenContent, double aCfContent)
{
  double tTerm1=0, tTerm2=0.;

  if(aNumContent==0.) tTerm1 = 0.;
  else tTerm1 = aNumContent*log(  (aCfContent*(aNumContent+aDenContent)) / (aNumContent*(aCfContent+1))  );

  if(aDenContent==0.) tTerm2 = 0.;
  else tTerm2 = aDenContent*log(  (aNumContent+aDenContent) / (aDenContent*(aCfContent+1))  );

  double tChi2PML = -2.0*(tTerm1+tTerm2);
  return tChi2PML;
}


//________________________________________________________________________________________________________________
void StrippedSimpleFitter::ApplyNormalization(double aNorm, td1dVec &aCf)
{
  for(unsigned int i=0; i<aCf.size(); i++) aCf[i] *= aNorm;
}




//________________________________________________________________________________________________________________
void StrippedSimpleFitter::CreateMinuitParameters()
{
  int tErrFlg = 0;

  double tStartVal_Lambda = 1.0;
  double tStepSize_Lambda = 0.001;
  double tLowerBound_Lambda = 0.;
  double tUpperBound_Lambda = 0.;
  fMinuit->mnparm(0, TString("Lambda"), tStartVal_Lambda, tStepSize_Lambda, tLowerBound_Lambda, tUpperBound_Lambda, tErrFlg);
//  fMinuit->FixParameter(0);

  double tStartVal_Radius = 5.0;
  double tStepSize_Radius = 0.001;
  double tLowerBound_Radius = 0.;
  double tUpperBound_Radius = 0.;
  fMinuit->mnparm(1, TString("Radius"), tStartVal_Radius, tStepSize_Radius, tLowerBound_Radius, tUpperBound_Radius, tErrFlg);
//  fMinuit->FixParameter(1);

  double tStartVal_ReF0 = -0.5;
  double tStepSize_ReF0 = 0.001;
  double tLowerBound_ReF0 = 0.;
  double tUpperBound_ReF0 = 0.;
  fMinuit->mnparm(2, TString("ReF0"), tStartVal_ReF0, tStepSize_ReF0, tLowerBound_ReF0, tUpperBound_ReF0, tErrFlg);
//  fMinuit->FixParameter(2);

  double tStartVal_ImF0 = 0.5;
  double tStepSize_ImF0 = 0.001;
  double tLowerBound_ImF0 = 0.;
  double tUpperBound_ImF0 = 0.;
  fMinuit->mnparm(3, TString("ImF0"), tStartVal_ImF0, tStepSize_ImF0, tLowerBound_ImF0, tUpperBound_ImF0, tErrFlg);
//  fMinuit->FixParameter(3);

  double tStartVal_D0 = 0.0;
  double tStepSize_D0 = 0.001;
  double tLowerBound_D0 = 0.;
  double tUpperBound_D0 = 0.;
  fMinuit->mnparm(4, TString("D0"), tStartVal_D0, tStepSize_D0, tLowerBound_D0, tUpperBound_D0, tErrFlg);
//  fMinuit->FixParameter(4);

  double tStartVal_Norm = 1.;
  if(fFitType==kChi2PML) tStartVal_Norm = GetNumScale()/GetDenScale();
  double tStepSize_Norm = 0.001;
  double tLowerBound_Norm = 0.;
  double tUpperBound_Norm = 0.;
  fMinuit->mnparm(5, TString("Norm"), tStartVal_Norm, tStepSize_Norm, tLowerBound_Norm, tUpperBound_Norm, tErrFlg);
//  fMinuit->FixParameter(5);
}


//________________________________________________________________________________________________________________
void StrippedSimpleFitter::CalculateFitFunction(int &npar, double &chi2, double *par)
{
  if(fVerbose) PrintCurrentParamValues(6,par);
  //---------------------------------------------------------
  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;
  bool tRejectOmega = false;

  int tNFitParams = 6;

  vector<double> tPrimaryFitCfContent(fNbinsXToBuild,0.);
  vector<double> tNumContent(fNbinsXToBuild,0.);
  vector<double> tDenContent(fNbinsXToBuild,0.);

  fChi2 = 0.;

  fNpFits = 0.;

  int tLambdaMinuitParamNumber = 0;
  int tRadiusMinuitParamNumber = 1;
  int tRef0MinuitParamNumber = 2;
  int tImf0MinuitParamNumber = 3;
  int td0MinuitParamNumber = 4;
  int tNormMinuitParamNumber = 5;

  double *tParPrim = new double[tNFitParams];

  tParPrim[0] = par[tLambdaMinuitParamNumber];
  tParPrim[1] = par[tRadiusMinuitParamNumber];
  tParPrim[2] = par[tRef0MinuitParamNumber];
  tParPrim[3] = par[tImf0MinuitParamNumber];
  tParPrim[4] = par[td0MinuitParamNumber];
  tParPrim[5] = par[tNormMinuitParamNumber];

  for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
  {
    if(std::isnan(tParPrim[i])) {cout <<"CRASH:  In CalculateFitFunction, a tParPrim elemement " << i << " DNE!!!!!" << endl;}
    assert(!std::isnan(tParPrim[i]));
  }

  double x[1];


  for(int ix=1; ix <= fNbinsXToBuild; ix++)
  {
    x[0] = fKStarBinCenters[ix-1];

    tNumContent[ix-1] = fNum->GetBinContent(ix);
    tDenContent[ix-1] = fDen->GetBinContent(ix);

    tPrimaryFitCfContent[ix-1] = LednickyEq(x,tParPrim);
  }


  ApplyNormalization(tParPrim[5], tPrimaryFitCfContent);

  for(int ix=0; ix < fNbinsXToFit; ix++)
  {
    if(tRejectOmega && (fKStarBinCenters[ix] > tRejectOmegaLow) && (fKStarBinCenters[ix] < tRejectOmegaHigh)) {fChi2+=0;}
    else
    {
      if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tPrimaryFitCfContent[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
      {
        double tChi2 = 0.;
        if(fFitType == kChi2PML) tChi2 = GetPmlValue(tNumContent[ix],tDenContent[ix],tPrimaryFitCfContent[ix]);
        else if(fFitType == kChi2) tChi2 = GetChi2Value(ix+1,fCf,tPrimaryFitCfContent[ix]);
        else tChi2 = 0.;

        fChi2 += tChi2;
        fNpFits++;
      }
    }
  }
  delete[] tParPrim;


  if(std::isnan(fChi2) || std::isinf(fChi2))
  {
    cout << "WARNING: fChi2 = nan, setting it equal to 10^9-----------------------------------------" << endl << endl;
    fChi2 = pow(10,9);
  }

  chi2 = fChi2;
  if(fChi2 < fChi2GlobalMin) fChi2GlobalMin = fChi2;

  if(fVerbose)
  {
    cout << "fChi2 = " << fChi2 << endl;
    cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;
  }

}

//________________________________________________________________________________________________________________
TF1* StrippedSimpleFitter::CreateFitFunction(TString aName)
{
  int tNFitParams = 5;
  TF1* ReturnFunction = new TF1(aName,LednickyEq,0.,0.5,tNFitParams+1);
  double tParamValue, tParamError;
  for(int iPar=0; iPar<tNFitParams; iPar++)
  {
    tParamValue = fMinParams[iPar];
    tParamError = fParErrors[iPar];

    ReturnFunction->SetParameter(iPar,tParamValue);
    ReturnFunction->SetParError(iPar,tParamError);
  }

  ReturnFunction->SetParameter(5,1.);
  ReturnFunction->SetParError(5,0.);

  ReturnFunction->SetChisquare(fChi2);
  ReturnFunction->SetNDF(fNDF);

  ReturnFunction->SetParNames("Lambda","Radius","Ref0","Imf0","d0","Norm");

  return ReturnFunction;
}



//________________________________________________________________________________________________________________
void StrippedSimpleFitter::InitializeFitter()
{
  cout << "----- Initializing fitter -----" << endl;
  CreateMinuitParameters();

  fNbinsXToBuild = 0;
  fNbinsXToFit = 0;
  fKStarBinWidth = 0.;
  fKStarBinCenters.clear();
  //-------------------------

  fNbinsXToFit = fNum->FindBin(fMaxFitKStar);
  if(fNum->GetBinLowEdge(fNbinsXToFit) == fMaxFitKStar) fNbinsXToFit--;

  fNbinsXToBuild = fNbinsXToFit;
  fKStarBinWidth = fNum->GetXaxis()->GetBinWidth(1);
  //-------------------------------------------------------------------------------------------

  assert(fNum->GetXaxis()->GetBinWidth(1) == fDen->GetXaxis()->GetBinWidth(1));
  assert(fNum->GetXaxis()->GetBinWidth(1) == fCf->GetXaxis()->GetBinWidth(1));
  assert(fNum->GetXaxis()->GetBinWidth(1) == fKStarBinWidth);

  //make sure fNum and fDen have same number of bins
  assert(fNum->GetNbinsX() == fDen->GetNbinsX());
  assert(fNum->GetNbinsX() == fCf->GetNbinsX());

  fKStarBinCenters.resize(fNbinsXToBuild,0.);
  for(int ix=1; ix <= fNbinsXToBuild; ix++)
  {
    fKStarBinCenters[ix-1] = fNum->GetXaxis()->GetBinCenter(ix);
  }
}

//________________________________________________________________________________________________________________
void StrippedSimpleFitter::DoFit()
{
  InitializeFitter();

  cout << "*****************************************************************************" << endl;
  cout << "Starting to fit " << endl;
  cout << "*****************************************************************************" << endl;

  fMinuit->SetPrintLevel(0);  // -1=quiet, 0=normal, 1=verbose (more options using mnexcm("SET PRI", ...)

  double arglist[10];
  fErrFlg = 0;

  // for max likelihood = 0.5, for chisq = 1.0
  arglist[0] = 1.;
  fMinuit->mnexcm("SET ERR", arglist ,1,fErrFlg);

  // Set strategy to be used
  // 0 = economize (many params and/or fcn takes a long time to calculate and/or not interested in precise param erros)
  // 1 = default
  // 2 = Minuit allowed to waste calls in order to ensure precision (fcn evaluated in short time and/or param erros must be reliable)
  arglist[0] = 2;
  fMinuit->mnexcm("SET STR", arglist ,1,fErrFlg);

  // Now ready for minimization step
  arglist[0] = 50000;
  arglist[1] = 0.1;
  fMinuit->mnexcm("MIGRAD", arglist ,2,fErrFlg);

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

  //---------------------------------
  Finalize();
}


//________________________________________________________________________________________________________________
void StrippedSimpleFitter::Finalize()
{
  int tNParams = 6;
  fNDF = fNpFits-fNvpar;

  //get result
  for(int i=0; i<tNParams; i++)
  {
    double tempMinParam;
    double tempParError;
    fMinuit->GetParameter(i,tempMinParam,tempParError);
    
    fMinParams.push_back(tempMinParam);
    fParErrors.push_back(tempParError);
  }
}




//________________________________________________________________________________________________________________
TPaveText* StrippedSimpleFitter::CreateParamFinalValuesText(TF1* aFit, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight)
{
  double tLambda, tRadius, tReF0, tImF0, tD0;
  double tLambdaErr, tRadiusErr, tReF0Err, tImF0Err, tD0Err;

  tLambda = aFit->GetParameter(0);
  tRadius = aFit->GetParameter(1);
  tReF0 = aFit->GetParameter(2);
  tImF0 = aFit->GetParameter(3);
  tD0 = aFit->GetParameter(4);

  tLambdaErr = aFit->GetParError(0);
  tRadiusErr = aFit->GetParError(1);
  tReF0Err = aFit->GetParError(2);
  tImF0Err = aFit->GetParError(3);
  tD0Err = aFit->GetParError(4);


  TPaveText *tText = new TPaveText(aTextXmin, aTextYmin, aTextXmin+aTextWidth, aTextYmin+aTextHeight, "NDC");
  tText->AddText(TString::Format("#lambda = %0.2f #pm %0.2f",tLambda,tLambdaErr));
  tText->AddText(TString::Format("R = %0.2f #pm %0.2f",tRadius,tRadiusErr));
  tText->AddText(TString::Format("Re[f0] = %0.2f #pm %0.2f",tReF0,tReF0Err));
  tText->AddText(TString::Format("Im[f0] = %0.2f #pm %0.2f",tImF0,tImF0Err));
  tText->AddText(TString::Format("d0 = %0.2f #pm %0.2f",tD0,tD0Err));

  return tText;
}


//________________________________________________________________________________________________________________
void StrippedSimpleFitter::DrawCfWithFit(TPad *aPad, TString aDrawOption)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TF1* tFit = CreateFitFunction(TString("Fit"));


  fCf->Draw(aDrawOption);
  if(aDrawOption.EqualTo("same")) tFit->Draw(aDrawOption);
  else tFit->Draw(aDrawOption+TString("same"));

  TPaveText* tText = CreateParamFinalValuesText(tFit, 0.5, 0.5, 0.25, 0.25);
  tText->Draw();

}

