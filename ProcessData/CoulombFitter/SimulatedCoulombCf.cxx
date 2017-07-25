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

SimulatedCoulombCf::SimulatedCoulombCf(ResidualType aResidualType, TString aInterpHistFileBaseName, TString aLednickyHFunctionFileBaseName):
  fResidualType(aResidualType),
  fTurnOffCoulomb(false),
  fIncludeSingletAndTriplet(false),
  fUseRandomKStarVectors(true),
  fCoulombCf(0),
  fCurrentFitParams(new double[8]),

  fCoulombType(kRepulsive),
  fWaveFunction(0),
  fBohrRadius(1000000000),

  fNPairsPerKStarBin(1000),
  fCurrentRadiusParameter(1.),
  fPairSample3dVec(0),
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

  SetBohrRadius();

  BuildPairSample3dVec();  //TODO if !fUseRandomKStarVectors, then I can only build out to k* = 0.3 GeV/c

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
AnalysisType SimulatedCoulombCf::GetDaughterAnalysisType()
{
  AnalysisType tReturnType;

  switch(fResidualType) {
  case kXiCKchP:
  case kOmegaKchP:
    tReturnType = kLamKchP;
    break;

  case kAXiCKchP:
  case kAOmegaKchP:
    tReturnType = kALamKchP;
    break;

  case kXiCKchM:
  case kOmegaKchM:
    tReturnType = kLamKchM;
    break;

  case kAXiCKchM:
  case kAOmegaKchM:
    tReturnType = kALamKchM;
    break;

  default:
    cout << "ERROR: SimulatedCoulombCf::GetDaughterAnalysisType:  fResidualType = " << fResidualType << " is not apropriate" << endl << endl;
    assert(0);
  }

  return tReturnType;
}


//________________________________________________________________________________________________________________
void SimulatedCoulombCf::SetBohrRadius()
{
  switch(fResidualType) {
  case kXiCKchP:
  case kAXiCKchM:
    fBohrRadius = -gBohrRadiusXiK;
    break;

  case kXiCKchM:
  case kAXiCKchP:
    fBohrRadius = gBohrRadiusXiK;
    break;

  case kOmegaKchP:
  case kAOmegaKchM:
    fBohrRadius = -gBohrRadiusOmegaK;
    break;

  case kOmegaKchM:
  case kAOmegaKchP:
    fBohrRadius = gBohrRadiusOmegaK;
    break;

  default:
    cout << "ERROR: SimulatedCoulombCf::SetBohrRadius:  fResidualType = " << fResidualType << " is not apropriate" << endl << endl;
    assert(0);
  }

  fWaveFunction->SetCurrentBohrRadius(fResidualType);

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
void SimulatedCoulombCf::AdjustLambdaParam(td1dVec &aCoulombResidualCf, double aOldLambda, double aNewLambda)
{
  double tRawValue, tCurrentValue;
  for(unsigned int i=0; i<aCoulombResidualCf.size(); i++)
  {
    tCurrentValue = aCoulombResidualCf[i];
    tRawValue = (1.0/aOldLambda)*(tCurrentValue-(1.0-aOldLambda));

    aCoulombResidualCf[i] = aNewLambda*tRawValue + (1.0-aNewLambda);
  }
}

//________________________________________________________________________________________________________________
td3dVec SimulatedCoulombCf::BuildPairKStar3dVecFromTxt(double aMaxFitKStar, TString aFileBaseName)
{
ChronoTimer tTimer;
tTimer.Start();

  TString aFileName = aFileBaseName;

  //----------------------------------------------
  switch(fResidualType) {
  case kXiCKchP:
  case kOmegaKchP:
    aFileName += TString("XiKchP_0010.txt");
    break;

  case kXiCKchM:
  case kOmegaKchM:
    aFileName += TString("XiKchM_0010.txt");
    break;

  case kAXiCKchP:
  case kAOmegaKchP:
    aFileName += TString("AXiKchP_0010.txt");
    break;

  case kAXiCKchM:
  case kAOmegaKchM:
    aFileName += TString("AXiKchM_0010.txt");
    break;

  default:
    cout << "ERROR: SimulatedCoulombCf::BuildPairKStar3dVecFromTxt:  fResidualType = " << fResidualType << " is not apropriate" << endl << endl;
    assert(0);
  }
  //----------------------------------------------

  ifstream tFileIn(aFileName);

  td3dVec tPairKStar3dVec;
    tPairKStar3dVec.clear();

  vector<vector<double> > tTempBin2dVec;
  vector<double> tTempPair1dVec;

  int aNbinsKStar;
  double aKStarMin, aKStarMax, aBinWidth;

  int tNbinsKStarNeeded = 200; //This will be set to correct value below

  string tString;
  int tCount = 0;
  while(getline(tFileIn, tString) && tCount <= tNbinsKStarNeeded)
  {
    tTempPair1dVec.clear();
    istringstream tStream(tString);
    string tElement;
    while(tStream >> tElement)
    {
      stringstream ss (tElement);
      double dbl;
      ss >> dbl;
      tTempPair1dVec.push_back(dbl);
    }

    if(tTempPair1dVec.size() == 2)  //bin header
    {           
      tCount++;
      if(tCount==1) continue;
      else
      {
        tPairKStar3dVec.push_back(tTempBin2dVec);
        tTempBin2dVec.clear();
      }
    }
    else if(tTempPair1dVec.size() == 4)  //pair
    {
      tTempBin2dVec.push_back(tTempPair1dVec);
    }
    else if(tTempPair1dVec.size() == 3) //File header
    {
      aNbinsKStar = tTempPair1dVec[0];
      aKStarMin = tTempPair1dVec[1];
      aKStarMax = tTempPair1dVec[2];
      aBinWidth = (aKStarMax-aKStarMin)/aNbinsKStar;
      tNbinsKStarNeeded = aMaxFitKStar/aBinWidth;
      tNbinsKStarNeeded ++;  //include one more bin than really needed, to be safe
      assert(tNbinsKStarNeeded <= aNbinsKStar); //cannot need more than we have
    }
    else
    {
      cout << "ERROR: Incorrect row size in BuildPairKStar3dVecFromTxt" << endl;
      assert(0);
    }
  }
  if(tNbinsKStarNeeded == aNbinsKStar) tPairKStar3dVec.push_back(tTempBin2dVec);  //if reading the entire file, need this final push_back
  tTempBin2dVec.clear();

tTimer.Stop();
cout << "BuildPairKStar3dVecFromTxt finished: ";
tTimer.PrintInterval();

  assert(tNbinsKStarNeeded == (int)tPairKStar3dVec.size());
  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<(int)tPairKStar3dVec.size(); i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

  //Add the binning information for use by CoulombFitterParallel::BuildPairKStar3dVecFromTxt
  tTempPair1dVec.clear();
  tTempPair1dVec.push_back(aNbinsKStar);
  tTempPair1dVec.push_back(aKStarMin);
  tTempPair1dVec.push_back(aKStarMax);
  tTempBin2dVec.push_back(tTempPair1dVec);
  tPairKStar3dVec.push_back(tTempBin2dVec);

  tTempPair1dVec.clear();
  tTempBin2dVec.clear();

  return tPairKStar3dVec;
}



//________________________________________________________________________________________________________________
void SimulatedCoulombCf::BuildPairSample3dVec(double aMaxFitKStar, int aNPairsPerKStarBin)
{
ChronoTimer tTimer(kSec);
tTimer.Start();

  fNPairsPerKStarBin = aNPairsPerKStarBin;

  double tBinSize = 0.01;  //TODO make this automated
  int tNBinsKStar = std::round(aMaxFitKStar/tBinSize);  //TODO make this general, ie subtract 1 if aMaxFitKStar is on bin edge (maybe, maybe not bx of iKStarBin<tNBinsKStar)

  fPairSample3dVec.resize(tNBinsKStar, td2dVec(0, td1dVec(0)));

  //Create the source Gaussians
  double tRoot2 = sqrt(2.);  //need this scale to get 4 on denominator of exp in normal dist instead of 2
  double tRadius = 1.0;
  fCurrentRadiusParameter = tRadius;
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,tRoot2*tRadius);
  std::normal_distribution<double> tRSideSource(0.,tRoot2*tRadius);
  std::normal_distribution<double> tRLongSource(0.,tRoot2*tRadius);

  std::uniform_int_distribution<int> tRandomKStarElement;

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);

  int tI;
  double tTheta, tKStarMag, tRStarMag;
  double tKStarMagMin, tKStarMagMax;
  td1dVec tTempPair(3);
  td2dVec tTemp2dVec;

  //------------------------------------
  td3dVec tPairKStar3dVec (0);
  double tMaxBuildKStar = 0.3;  //TODO for now, only possible to use real pairs out to 0.3 GeV/c
                                //Generating file to go beyond this would be difficult, probably need to
                                //generate multiple files, but probably not necessary because uniform dist should
                                //be alright far out in k* space (really only seems to matter for lowest k* bin)
  int tNbinKStar_3dVecFromTxt=0;
  double tKStarMin_3dVecFromTxt=0., tKStarMax_3dVecFromTxt=0., tBinWidth_3dVecFromTxt=0.;
  if(aMaxFitKStar < 0.3) tMaxBuildKStar = aMaxFitKStar;
  if(!fUseRandomKStarVectors)
  {
    tPairKStar3dVec = BuildPairKStar3dVecFromTxt(tMaxBuildKStar);

    tNbinKStar_3dVecFromTxt = tPairKStar3dVec[tPairKStar3dVec.size()-1][0][0];
    tKStarMin_3dVecFromTxt = tPairKStar3dVec[tPairKStar3dVec.size()-1][0][1];
    tKStarMax_3dVecFromTxt = tPairKStar3dVec[tPairKStar3dVec.size()-1][0][2];

    tBinWidth_3dVecFromTxt = (tKStarMax_3dVecFromTxt-tKStarMin_3dVecFromTxt)/tNbinKStar_3dVecFromTxt;
    assert(tBinWidth_3dVecFromTxt==tBinSize);

    tPairKStar3dVec.pop_back();  //strip off the binning information
  }
//TODO Check randomization
//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iKStarBin=0; iKStarBin<tNBinsKStar; iKStarBin++)
  {
    tKStarMagMin = iKStarBin*tBinSize;
    if(iKStarBin==0) tKStarMagMin=0.004;  //TODO here and in CoulombFitter
    tKStarMagMax = (iKStarBin+1)*tBinSize;

    if(!fUseRandomKStarVectors && tKStarMagMin<0.3) tRandomKStarElement = std::uniform_int_distribution<int>(0, tPairKStar3dVec[iKStarBin].size()-1);

    tTemp2dVec.clear();
    for(int iPair=0; iPair<fNPairsPerKStarBin; iPair++)
    {
      if(!fUseRandomKStarVectors && tKStarMagMin<0.3)
      {
        tI = tRandomKStarElement(generator);
        tKStar3Vec->SetXYZ(tPairKStar3dVec[iKStarBin][tI][1],tPairKStar3dVec[iKStarBin][tI][2],tPairKStar3dVec[iKStarBin][tI][3]);
      }
      else SetRandomKStar3Vec(tKStar3Vec,tKStarMagMin,tKStarMagMax);

      tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric

      tTheta = tKStar3Vec->Angle(*tSource3Vec);
      tKStarMag = tKStar3Vec->Mag();
      tRStarMag = tSource3Vec->Mag();

      tTempPair[0] = tKStarMag;
      tTempPair[1] = tRStarMag;
      tTempPair[2] = tTheta;

      tTemp2dVec.push_back(tTempPair);
    }
    fPairSample3dVec[iKStarBin] = tTemp2dVec;
  }

  delete tKStar3Vec;
  delete tSource3Vec;


tTimer.Stop();
cout << "BuildPairSample3dVec finished: ";
tTimer.PrintInterval();

}

//________________________________________________________________________________________________________________
void SimulatedCoulombCf::UpdatePairRadiusParameters(double aNewRadius)
{
  //TODO allow for change of MU also, not just SIGMA!
  double tScaleFactor = aNewRadius/fCurrentRadiusParameter;
  fCurrentRadiusParameter = aNewRadius;

//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iKStarBin=0; iKStarBin<(int)fPairSample3dVec.size(); iKStarBin++)
  {
    for(int iPair=0; iPair<(int)fPairSample3dVec[iKStarBin].size(); iPair++)
    {
      fPairSample3dVec[iKStarBin][iPair][1] *= tScaleFactor;
    }
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
double SimulatedCoulombCf::GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par)
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

  UpdatePairRadiusParameters(par[1]);

  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  double tBinSize = 0.01;
  int tBin = std::round(aKStarMagMin/tBinSize);  //IMPORTANT!!!!! w/o using std::round, some bins were being calculated incorrectly

  //KStarMag = fPairSample3dVec[tBin][i][0]
  //RStarMag = fPairSample3dVec[tBin][i][1]
  //Theta    = fPairSample3dVec[tBin][i][2]


  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;
  tMaxKStarCalls = fPairSample3dVec[tBin].size();

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
    tKStarMag = fPairSample3dVec[tBin][i][0];
    tRStarMag = fPairSample3dVec[tBin][i][1];
    tTheta = fPairSample3dVec[tBin][i][2];

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
td1dVec SimulatedCoulombCf::GetCoulombParentCorrelation(double *aParentCfParams, vector<double> &aKStarBinCenters, CentralityType aCentType)
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

      tParentCf[i] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,fCurrentFitParams);
    }
    fCoulombCf = tParentCf;
  }
  return fCoulombCf;
}





