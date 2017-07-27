///////////////////////////////////////////////////////////////////////////
// SimulatedCoulombCf:                                                   //
///////////////////////////////////////////////////////////////////////////


#include "SimulatedCoulombCf.h"

#ifdef __ROOT__
ClassImp(SimulatedCoulombCf)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________

SimulatedCoulombCf::SimulatedCoulombCf(vector<tmpAnalysisInfo> &aAnalysesInfo, TString aInterpHistFileBaseName, TString aLednickyHFunctionFileBaseName):
  fNAnalyses((int)aAnalysesInfo.size()),
  fAnalysesInfo(aAnalysesInfo),
  fTurnOffCoulomb(false),
  fIncludeSingletAndTriplet(false),
  fUseRandomKStarVectors(true),
  fCoulombCf(0),
  fCurrentFitParams(new double[8]),

  fCoulombType(kRepulsive),
  fWaveFunction(0),
  fBohrRadius(1000000000),

  fSimPairCollection(nullptr),

  fInterpHistFile(0), fInterpHistFileLednickyHFunction(0),

  fLednickyHFunctionHist(0),

  fGTildeRealHist(0),
  fGTildeImagHist(0),

  fHyperGeo1F1RealHist(0),
  fHyperGeo1F1ImagHist(0),

  fMinInterpKStar(0), fMinInterpRStar(0), fMinInterpTheta(0),
  fMaxInterpKStar(0), fMaxInterpRStar(0), fMaxInterpTheta(0)

{
  gRandom->SetSeed();
  fWaveFunction = new WaveFunction();
  omp_set_num_threads(3);

  LoadInterpHistFile(aInterpHistFileBaseName);
  LoadLednickyHFunctionFile(aLednickyHFunctionFileBaseName);

//  SetBohrRadius();

 // BuildPairSample3dVec();  //TODO if !fUseRandomKStarVectors, then I can only build out to k* = 0.3 GeV/c

  //TODO figure out better way to achieve this
  for(int i=0; i<8; i++) fCurrentFitParams[i] = 0.;
}





//________________________________________________________________________________________________________________
SimulatedCoulombCf::~SimulatedCoulombCf()
{
  cout << "SimulatedCoulombCf object is being deleted!!!" << endl;

  delete fLednickyHFunctionHist;

  fInterpHistFileLednickyHFunction->Close();
  delete fInterpHistFileLednickyHFunction;

  delete fHyperGeo1F1RealHist;
  delete fHyperGeo1F1ImagHist;

  delete fGTildeRealHist;
  delete fGTildeImagHist;

  fInterpHistFile->Close();
  delete fInterpHistFile;

}


//________________________________________________________________________________________________________________
double SimulatedCoulombCf::GetBohrRadius(AnalysisType aAnalysisType)
{
  double tReturnRadius;

  switch(aAnalysisType) {
  case kXiKchP:
  case kAXiKchM:
  case kResXiCKchP:
  case kResAXiCKchM:
    tReturnRadius = -gBohrRadiusXiK;
    break;

  case kXiKchM:
  case kAXiKchP:
  case kResXiCKchM:
  case kResAXiCKchP:
    tReturnRadius = gBohrRadiusXiK;
    break;


  case kResOmegaKchP:
  case kResAOmegaKchM:
    tReturnRadius = -gBohrRadiusOmegaK;
    break;


  case kResOmegaKchM:
  case kResAOmegaKchP:
    tReturnRadius = gBohrRadiusOmegaK;
    break;


  case kResSigStPKchM:
  case kResASigStPKchM:
    tReturnRadius = -gBohrRadiusSigStPK;
    break;

  case kResASigStMKchP:
  case kResSigStMKchP:
    tReturnRadius = -gBohrRadiusSigStMK;
    break;


  case kResSigStPKchP:
  case kResASigStPKchP:
    tReturnRadius = gBohrRadiusSigStPK;
    break;

  case kResASigStMKchM:
  case kResSigStMKchM:
    tReturnRadius = gBohrRadiusSigStMK;
    break;

  default:
    tReturnRadius = 1000000000;
  }

  return tReturnRadius;
}



//________________________________________________________________________________________________________________
CoulombType SimulatedCoulombCf::GetCoulombType(AnalysisType aAnalysisType)
{
  if(fBohrRadius < 0) return kAttractive;
  else if(fBohrRadius > 0 && fBohrRadius < 1000000000) return kRepulsive;
  else if(fBohrRadius==1000000000) return kNeutral;
  else assert(0);

  return kNeutral;
}


//________________________________________________________________________________________________________________
double SimulatedCoulombCf::GetBohrRadius()
{
  return fBohrRadius;
}




//________________________________________________________________________________________________________________
void SimulatedCoulombCf::LoadLednickyHFunctionFile(TString aFileBaseName)
{
  TString tFileName = aFileBaseName+".root";
  fInterpHistFileLednickyHFunction = TFile::Open(tFileName);

  fLednickyHFunctionHist = (TH1D*)fInterpHistFileLednickyHFunction->Get("LednickyHFunction");
    fLednickyHFunctionHist->SetDirectory(0);

  fInterpHistFileLednickyHFunction->Close();

  if(fMinInterpKStar==0 && fMaxInterpKStar==0)
  {
    fMinInterpKStar = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(1);
    fMaxInterpKStar = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(fLednickyHFunctionHist->GetNbinsX());
  }
  else
  {
    assert(fMinInterpKStar == fLednickyHFunctionHist->GetXaxis()->GetBinCenter(1));
    assert(fMaxInterpKStar == fLednickyHFunctionHist->GetXaxis()->GetBinCenter(fLednickyHFunctionHist->GetNbinsX()));
  }
}

//________________________________________________________________________________________________________________
void SimulatedCoulombCf::LoadInterpHistFile(TString aFileBaseName)
{
ChronoTimer tTimer(kSec);
tTimer.Start();
cout << "Starting LoadInterpHistFile" << endl;

//--------------------------------------------------------------

  TString aFileName = aFileBaseName+".root";
  fInterpHistFile = TFile::Open(aFileName);
//--------------------------------------------------------------
  fHyperGeo1F1RealHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Real");
  fHyperGeo1F1ImagHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Imag");
    fHyperGeo1F1RealHist->SetDirectory(0);
    fHyperGeo1F1ImagHist->SetDirectory(0);

  fGTildeRealHist = (TH2D*)fInterpHistFile->Get("GTildeReal");
  fGTildeImagHist = (TH2D*)fInterpHistFile->Get("GTildeImag");
    fGTildeRealHist->SetDirectory(0);
    fGTildeImagHist->SetDirectory(0);

  fInterpHistFile->Close();

  if(fMinInterpKStar==0 && fMaxInterpKStar==0)
  {
    fMinInterpKStar = fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(1);
    fMaxInterpKStar = fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsX());
  }
  else
  {
    assert(fMinInterpKStar == fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(1));
    assert(fMaxInterpKStar == fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsX()));
  }

  fMinInterpRStar = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(1);
  fMaxInterpRStar = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsY());

  fMinInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(1);
  fMaxInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsZ());

//--------------------------------------------------------------
  cout << "Interpolation histograms LOADED!" << endl;

tTimer.Stop();
cout << "LoadInterpHistFile: ";
tTimer.PrintInterval();
}

//________________________________________________________________________________________________________________
bool SimulatedCoulombCf::AreParamsSameExcludingLambda(double *aCurrent, double *aNew, int aNEntries)
{
//TODO NOTE, want to exclude Lambda parameter here, becuase it's always changing
  bool tAreSame = true;
  for(int i=1; i<aNEntries; i++)
  {
    if(abs(aCurrent[i]-aNew[i]) > std::numeric_limits< double >::min()) tAreSame = false;
  }

  if(!tAreSame)
  {
    for(int i=0; i<aNEntries; i++) aCurrent[i] = aNew[i];
  }

  return tAreSame;
}

//________________________________________________________________________________________________________________
void SimulatedCoulombCf::AdjustLambdaParam(td1dVec &aCoulombCf, double aOldLambda, double aNewLambda)
{
  double tRawValue, tCurrentValue;
  for(unsigned int i=0; i<aCoulombCf.size(); i++)
  {
    tCurrentValue = aCoulombCf[i];
    tRawValue = (1.0/aOldLambda)*(tCurrentValue-(1.0-aOldLambda));

    aCoulombCf[i] = aNewLambda*tRawValue + (1.0-aNewLambda);
  }
}

//________________________________________________________________________________________________________________
double SimulatedCoulombCf::GetEta(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else return pow(((aKStar/hbarc)*fBohrRadius),-1);
}

//________________________________________________________________________________________________________________
double SimulatedCoulombCf::GetGamowFactor(double aKStar)
{
  if(fTurnOffCoulomb) return 1.;
  else
  {
    double tEta = GetEta(aKStar);
    tEta *= TMath::TwoPi();  //eta always comes with 2Pi here
    double tGamow = tEta*pow((TMath::Exp(tEta)-1),-1);
    return tGamow;
  }
}


//________________________________________________________________________________________________________________
complex<double> SimulatedCoulombCf::GetExpTerm(double aKStar, double aRStar, double aTheta)
{
  complex<double> tReturnValue = exp(-ImI*(aKStar/hbarc)*aRStar*cos(aTheta));
  return tReturnValue;
}

//________________________________________________________________________________________________________________
complex<double> SimulatedCoulombCf::BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0)
{
  complex<double> tF0 (aReF0,aImF0);
  double tKStar = aKStar/hbarc;
  complex<double> tScattAmp;

  if(fTurnOffCoulomb)
  {
    complex<double> tLastTerm (0.,tKStar);
    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - tLastTerm,-1);
  }
  else
  {
    double tLednickyHFunction = Interpolator::LinearInterpolate(fLednickyHFunctionHist,aKStar);
    double tImagChi = GetGamowFactor(aKStar)/(2.*GetEta(aKStar));
    complex<double> tLednickyChi (tLednickyHFunction,tImagChi);

    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - 2.*tLednickyChi/fBohrRadius,-1);
  }

  return tScattAmp;
}



//________________________________________________________________________________________________________________
double SimulatedCoulombCf::InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
{

  bool tDebug = true; //debug means use personal interpolation methods, instead of standard root ones

//  assert(fInterpHistFile->IsOpen());

  double tGamow = GetGamowFactor(aKStarMag);
  complex<double> tExpTermComplex = GetExpTerm(aKStarMag,aRStarMag,aTheta);

  complex<double> tScattLenComplexConj;

  complex<double> tHyperGeo1F1Complex;
  complex<double> tGTildeComplexConj;

  if(!fTurnOffCoulomb)
  {
    double tHyperGeo1F1Real, tHyperGeo1F1Imag, tGTildeReal, tGTildeImag;
    //-------------------------------------
    if(tDebug)
    {
      tHyperGeo1F1Real = Interpolator::TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = Interpolator::TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

      tGTildeReal = Interpolator::BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
      tGTildeImag = Interpolator::BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);
    }

    else
    {
      tHyperGeo1F1Real = fHyperGeo1F1RealHist->Interpolate(aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = fHyperGeo1F1ImagHist->Interpolate(aKStarMag,aRStarMag,aTheta);

      tGTildeReal = fGTildeRealHist->Interpolate(aKStarMag,aRStarMag);
      tGTildeImag = fGTildeImagHist->Interpolate(aKStarMag,aRStarMag);
    }

    tHyperGeo1F1Complex = complex<double> (tHyperGeo1F1Real,tHyperGeo1F1Imag);
    tGTildeComplexConj = complex<double> (tGTildeReal,-tGTildeImag);
    //-------------------------------------

    complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
    tScattLenComplexConj = std::conj(tScattLenComplex);
  }

  else
  {
    tHyperGeo1F1Complex = complex<double> (1.,0.);
    tGTildeComplexConj = exp(-ImI*(aKStarMag/hbarc)*aRStarMag);

    complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
    tScattLenComplexConj = std::conj(tScattLenComplex);
  }

  //-------------------------------------------

  complex<double> tResultComplex = tGamow*( norm(tHyperGeo1F1Complex) + norm(tScattLenComplexConj)*norm(tGTildeComplexConj)/(aRStarMag*aRStarMag) + 2.*real(tExpTermComplex*tHyperGeo1F1Complex*tScattLenComplexConj*tGTildeComplexConj/aRStarMag) );

  if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in SimulatedCoulombCf::InterpolateWfSquared !!!!!" << endl;
  assert(imag(tResultComplex) < std::numeric_limits< double >::min());

  return real(tResultComplex);

}



//________________________________________________________________________________________________________________
bool SimulatedCoulombCf::CanInterpKStar(double aKStar)
{
  if(aKStar < fMinInterpKStar) return false;
  if(aKStar > fMaxInterpKStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool SimulatedCoulombCf::CanInterpRStar(double aRStar)
{
  if(aRStar < fMinInterpRStar || aRStar > fMaxInterpRStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool SimulatedCoulombCf::CanInterpTheta(double aTheta)
{
  if(aTheta < fMinInterpTheta || aTheta > fMaxInterpTheta) return false;
  return true;
}



//________________________________________________________________________________________________________________
bool SimulatedCoulombCf::CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0)
{
  if(CanInterpKStar(aKStar) && CanInterpRStar(aRStar) && CanInterpTheta(aTheta)) return true;
  return false;
}

//________________________________________________________________________________________________________________
void SimulatedCoulombCf::SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax)
{
  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::uniform_real_distribution<double> tKStarMagDistribution(aKStarMagMin,aKStarMagMax);
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);

  double tKStarMag = tKStarMagDistribution(tGenerator);
  double tU = tUnityDistribution(tGenerator);
  double tV = tUnityDistribution(tGenerator);

  double tTheta = acos(2.*tV-1.); //polar angle
  double tPhi = 2.*M_PI*tU; //azimuthal angle

  aKStar3Vec->SetMagThetaPhi(tKStarMag,tTheta,tPhi);
}





//________________________________________________________________________________________________________________
double SimulatedCoulombCf::GetFitCfContentCompletewStaticPairs(int aAnalysisNumber, double aKStarMagMin, double aKStarMagMax, double *par)
{
  omp_set_num_threads(6);

  // if fIncludeSingletAndTriplet == false
  //    par[0] = Lambda 
  //    par[1] = Radius
  //    par[2] = Ref0
  //    par[3] = Imf0
  //    par[4] = d0
  //    par[5] = Norm

  // if fIncludeSingletAndTriplet == true
  //    par[0] = kLambda
  //    par[1] = kRadius
  //    par[2] = kRef0
  //    par[3] = kImf0
  //    par[4] = kd0
  //    par[5] = kRef02
  //    par[6] = kImf02
  //    par[7] = kd02
  //    par[8] = kNorm

  fSimPairCollection->UpdatePairRadiusParameter(par[1], aAnalysisNumber);

  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  double tBinSize = 0.01;
  int tBin = std::round(aKStarMagMin/tBinSize);  //IMPORTANT!!!!! w/o using std::round, some bins were being calculated incorrectly

  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;
  tMaxKStarCalls = fSimPairCollection->GetNumberOfPairsInBin(aAnalysisNumber, tBin);

  bool tCanInterp;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq;
  complex<double> tWaveFunctionSinglet, tWaveFunctionTriplet;

  vector<vector<double> > tMathematicaPairs;
  vector<double> tTempPair(3);

  int tNInterpolate = 0;
  int tNMathematica = 0;

//ChronoTimer tIntTimer;
//tIntTimer.Start();


  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tKStarMag = fSimPairCollection->GetPairKStarMag(aAnalysisNumber, tBin, i);
    tRStarMag = fSimPairCollection->GetPairRStarMag(aAnalysisNumber, tBin, i);
    tTheta = fSimPairCollection->GetPairTheta(aAnalysisNumber, tBin, i);

    tCanInterp = CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
    if(fTurnOffCoulomb || tCanInterp)
    {
      if(fIncludeSingletAndTriplet)
      {
        tWaveFunctionSqSinglet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
        tWaveFunctionSqTriplet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[5],par[6],par[7]);

        tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
      }
      else tWaveFunctionSq = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);

      tReturnCfContent += tWaveFunctionSq;
      tCounter ++;
    }


    if(!tCanInterp)
    {
      tTempPair[0] = tKStarMag;
      tTempPair[1] = tRStarMag;
      tTempPair[2] = tTheta;
      tMathematicaPairs.push_back(tTempPair);
    }
  }

//tIntTimer.Stop();
//cout << "Interpolation in GetFitCfContentComplete ";
//tIntTimer.PrintInterval();

  tNInterpolate = tCounter;
  tNMathematica = tMathematicaPairs.size();

//ChronoTimer tMathTimer;
//tMathTimer.Start();

  for(int i=0; i<(int)tMathematicaPairs.size(); i++)
  {
    if(fIncludeSingletAndTriplet)
    {
      tWaveFunctionSinglet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[2],par[3],par[4]);
      tWaveFunctionSqSinglet = norm(tWaveFunctionSinglet);

      tWaveFunctionTriplet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[5],par[6],par[7]);
      tWaveFunctionSqTriplet = norm(tWaveFunctionTriplet);

      tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
    }
    else
    {
      tWaveFunctionSinglet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[2],par[3],par[4]);
      tWaveFunctionSqSinglet = norm(tWaveFunctionSinglet);

      tWaveFunctionSq = tWaveFunctionSqSinglet;
    }

    tReturnCfContent += tWaveFunctionSq;
    tCounter ++;
  }
//tMathTimer.Stop();
//cout << "Mathematica calls in GetFitCfContentComplete ";
//tMathTimer.PrintInterval();

  tReturnCfContent /= tCounter;

//  tReturnCfContent = par[8]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));


  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
  }

  return tReturnCfContent;
}


//________________________________________________________________________________________________________________
td1dVec SimulatedCoulombCf::GetCoulombParentCorrelation(int aAnalysisNumber, double *aParentCfParams, vector<double> &aKStarBinCenters, CentralityType aCentType)
{
  double tKStarBinWidth = aKStarBinCenters[1]-aKStarBinCenters[0];

  double tKStarMin, tKStarMax;
  tKStarMax = aKStarBinCenters[aKStarBinCenters.size()-1]+tKStarBinWidth/2.;

  if(AreParamsSameExcludingLambda(fCurrentFitParams,aParentCfParams,8))
  {
    AdjustLambdaParam(fCoulombCf, fCurrentFitParams[0], aParentCfParams[0]);
  }
  else
  {
    vector<double> tParentCf(aKStarBinCenters.size(),0.);

    for(unsigned int i=0; i<aKStarBinCenters.size(); i++)
    {
      tKStarMin = aKStarBinCenters[i]-tKStarBinWidth/2.;
      tKStarMax = aKStarBinCenters[i]+tKStarBinWidth/2.;

      if(i==0) tKStarMin = 0.; //TODO this is small, but nonzero

      tParentCf[i] = GetFitCfContentCompletewStaticPairs(aAnalysisNumber,tKStarMin,tKStarMax,fCurrentFitParams);
    }
    fCoulombCf = tParentCf;
  }
  return fCoulombCf;
}





