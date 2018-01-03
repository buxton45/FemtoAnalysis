///////////////////////////////////////////////////////////////////////////
// ChargedResidualCf:                                                    //
///////////////////////////////////////////////////////////////////////////


#include "ChargedResidualCf.h"

#ifdef __ROOT__
ClassImp(ChargedResidualCf)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________

ChargedResidualCf::ChargedResidualCf(AnalysisType aResidualType, TString aInterpHistFileBaseName, TString aLednickyHFunctionFileBaseName, int aTransRebin, TString aTransformMatricesLocation):
  fResidualType(aResidualType),
  fTransformMatrix(),
  fTurnOffCoulomb(false),
  fIncludeSingletAndTriplet(false),
  fUseRandomKStarVectors(true),
  fUseExpXiData(false),
  fExpXiData(0),
  fCoulombCf(0),
  fCoulombResidualCf(0),
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

  LoadTransformMatrix(aTransRebin,aTransformMatricesLocation);
  SetBohrRadius();

  BuildPairSample3dVec();  //TODO if !fUseRandomKStarVectors, then I can only build out to k* = 0.3 GeV/c

  //TODO figure out better way to achieve this
  for(int i=0; i<8; i++) fCurrentFitParams[i] = 0.;
}





//________________________________________________________________________________________________________________
ChargedResidualCf::~ChargedResidualCf()
{
  cout << "ChargedResidualCf object is being deleted!!!" << endl;

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
AnalysisType ChargedResidualCf::GetDaughterAnalysisType()
{
  AnalysisType tReturnType;

  switch(fResidualType) {
  case kResXiCKchP:
  case kResOmegaKchP:
    tReturnType = kLamKchP;
    break;

  case kResAXiCKchP:
  case kResAOmegaKchP:
    tReturnType = kALamKchP;
    break;

  case kResXiCKchM:
  case kResOmegaKchM:
    tReturnType = kLamKchM;
    break;

  case kResAXiCKchM:
  case kResAOmegaKchM:
    tReturnType = kALamKchM;
    break;

  default:
    cout << "ERROR: ChargedResidualCf::GetDaughterAnalysisType:  fResidualType = " << fResidualType << " is not apropriate" << endl << endl;
    assert(0);
  }

  return tReturnType;
}


//________________________________________________________________________________________________________________
void ChargedResidualCf::LoadTransformMatrix(int aRebin, TString aFileLocation)
{
  if(aFileLocation.IsNull()) aFileLocation = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/TransformMatrices_Mix5.root";
  TFile *tFile = TFile::Open(aFileLocation);

  AnalysisType tDaughterAnalysisType = GetDaughterAnalysisType();

  TString tName2 = cAnalysisBaseTags[tDaughterAnalysisType] + TString("Transform");
  TString tName1, tFullName;

  switch(fResidualType) {
  case kResXiCKchP:
  case kResXiCKchM:
    tName1 = TString("fXiCTo");
    break;

  case kResAXiCKchP:
  case kResAXiCKchM:
    tName1 = TString("fAXiCTo");
    break;

  case kResOmegaKchP:
  case kResOmegaKchM:
    tName1 = TString("fOmegaTo");
    break;

  case kResAOmegaKchP:
  case kResAOmegaKchM:
    tName1 = TString("fAOmegaTo");
    break;

  default:
    cout << "ERROR: ChargedResidualCf::LoadTransformMatrix:  fResidualType = " << fResidualType << " is not apropriate" << endl << endl;
    assert(0);
  }

  tFullName = tName1+tName2;

  fTransformMatrix = (TH2D*)tFile->Get(tFullName);
    fTransformMatrix->SetDirectory(0);
    fTransformMatrix->Rebin2D(aRebin);

  tFile->Close();
}

//________________________________________________________________________________________________________________
void ChargedResidualCf::SetBohrRadius()
{
  switch(fResidualType) {
  case kResXiCKchP:
  case kResAXiCKchM:
    fBohrRadius = -gBohrRadiusXiK;
    break;

  case kResXiCKchM:
  case kResAXiCKchP:
    fBohrRadius = gBohrRadiusXiK;
    break;

  case kResOmegaKchP:
  case kResAOmegaKchM:
    fBohrRadius = -gBohrRadiusOmegaK;
    break;

  case kResOmegaKchM:
  case kResAOmegaKchP:
    fBohrRadius = gBohrRadiusOmegaK;
    break;

  default:
    cout << "ERROR: ChargedResidualCf::SetBohrRadius:  fResidualType = " << fResidualType << " is not apropriate" << endl << endl;
    assert(0);
  }

  fWaveFunction->SetCurrentBohrRadius(fResidualType);

}



//________________________________________________________________________________________________________________
CoulombType ChargedResidualCf::GetCoulombType(AnalysisType aAnalysisType)
{
  if(fBohrRadius < 0) return kAttractive;
  else if(fBohrRadius > 0 && fBohrRadius < 1000000000) return kRepulsive;
  else if(fBohrRadius==1000000000) return kNeutral;
  else assert(0);

  return kNeutral;
}


//________________________________________________________________________________________________________________
double ChargedResidualCf::GetBohrRadius()
{
  return fBohrRadius;
}




//________________________________________________________________________________________________________________
void ChargedResidualCf::LoadLednickyHFunctionFile(TString aFileBaseName)
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
void ChargedResidualCf::LoadInterpHistFile(TString aFileBaseName)
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
int ChargedResidualCf::GetBinNumber(double aBinSize, int aNbins, double aValue)
{
//TODO check the accuracy of this
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<aNbins; i++)
  {
    tBinKStarMin = i*aBinSize;
    tBinKStarMax = (i+1)*aBinSize;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
int ChargedResidualCf::GetBinNumber(int aNbins, double aMin, double aMax, double aValue)
{
//TODO check the accuracy of this
  double tBinSize = (aMax-aMin)/aNbins;
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<aNbins; i++)
  {
    tBinKStarMin = i*tBinSize + aMin;
    tBinKStarMax = (i+1)*tBinSize + aMin;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
int ChargedResidualCf::GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue)
{
//TODO check the accuracy of this
  int tNbins = (aMax-aMin)/aBinWidth;
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<tNbins; i++)
  {
    tBinKStarMin = i*aBinWidth + aMin;
    tBinKStarMax = (i+1)*aBinWidth + aMin;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
bool ChargedResidualCf::AreParamsSameExcludingLambda(double *aCurrent, double *aNew, int aNEntries)
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
void ChargedResidualCf::AdjustLambdaParam(td1dVec &aCoulombResidualCf, double aOldLambda, double aNewLambda)
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
td3dVec ChargedResidualCf::BuildPairKStar3dVecFromTxt(double aMaxFitKStar, TString aFileBaseName)
{
ChronoTimer tTimer;
tTimer.Start();

  TString aFileName = aFileBaseName;

  //----------------------------------------------
  switch(fResidualType) {
  case kResXiCKchP:
  case kResOmegaKchP:
    aFileName += TString("XiKchP_0010.txt");
    break;

  case kResXiCKchM:
  case kResOmegaKchM:
    aFileName += TString("XiKchM_0010.txt");
    break;

  case kResAXiCKchP:
  case kResAOmegaKchP:
    aFileName += TString("AXiKchP_0010.txt");
    break;

  case kResAXiCKchM:
  case kResAOmegaKchM:
    aFileName += TString("AXiKchM_0010.txt");
    break;

  default:
    cout << "ERROR: ChargedResidualCf::BuildPairKStar3dVecFromTxt:  fResidualType = " << fResidualType << " is not apropriate" << endl << endl;
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
void ChargedResidualCf::BuildPairSample3dVec(double aMaxFitKStar, int aNPairsPerKStarBin)
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
void ChargedResidualCf::UpdatePairRadiusParameters(double aNewRadius)
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
double ChargedResidualCf::LinearInterpolate(TH1* a1dHisto, double aX)
{
  if(a1dHisto->GetBuffer()) a1dHisto->BufferEmpty();  //not sure what this is all about

  int tXbin = a1dHisto->FindBin(aX);
  double tX0, tX1, tY0, tY1;

  //TODO: These allows evaluation in underflow and overflow bins, not sure I like this
  if(aX <= a1dHisto->GetBinCenter(1)) return a1dHisto->GetBinContent(1);
  else if( aX >= a1dHisto->GetBinCenter(a1dHisto->GetNbinsX()) ) return a1dHisto->GetBinContent(a1dHisto->GetNbinsX()); 

  else
  {
    if(aX <= a1dHisto->GetBinCenter(tXbin))
    {
      tY0 = a1dHisto->GetBinContent(tXbin-1);
      tX0 = a1dHisto->GetBinCenter(tXbin-1);
      tY1 = a1dHisto->GetBinContent(tXbin);
      tX1 = a1dHisto->GetBinCenter(tXbin);
    }
    else
    {
      tY0 = a1dHisto->GetBinContent(tXbin);
      tX0 = a1dHisto->GetBinCenter(tXbin);
      tY1 = a1dHisto->GetBinContent(tXbin+1);
      tX1 = a1dHisto->GetBinCenter(tXbin+1);
    }
    return tY0 + (aX-tX0)*((tY1-tY0)/(tX1-tX0));
  }
}

//________________________________________________________________________________________________________________
double ChargedResidualCf::BilinearInterpolate(TH2* a2dHisto, double aX, double aY)
{
  double tF = 0.;
  double tX1=0., tX2=0., tY1=0., tY2=0.;
  double tdX, tdY;

  TAxis* tXaxis = a2dHisto->GetXaxis();
  TAxis* tYaxis = a2dHisto->GetYaxis();

  int tXbin = tXaxis->FindBin(aX);
  int tYbin = tYaxis->FindBin(aY);

  //---------------------------------
  if(tXbin<1 || tXbin>a2dHisto->GetNbinsX() || tYbin<1 || tYbin>a2dHisto->GetNbinsY()) 
  {
    cout << "Error in ChargedResidualCf::BilinearInterpolate, cannot interpolate outside histogram domain" << endl;
  }
  assert(tXbin>0);
  assert(tXbin<=a2dHisto->GetNbinsX());
  assert(tYbin>0);
  assert(tYbin<=a2dHisto->GetNbinsY());
  //---------------------------------

  int tQuadrant = 0; //CCW from UR 1,2,3,4
  // which quadrant of the bin (bin_P) are we in?
  tdX = tXaxis->GetBinUpEdge(tXbin) - aX;
  tdY = tYaxis->GetBinUpEdge(tYbin) - aY;

  if(tdX<=tXaxis->GetBinWidth(tXbin)/2 && tdY<=tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 1; //upper right
  if(tdX>tXaxis->GetBinWidth(tXbin)/2 && tdY<=tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 2; //upper left
  if(tdX>tXaxis->GetBinWidth(tXbin)/2 && tdY>tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 3; //lower left
  if(tdX<=tXaxis->GetBinWidth(tXbin)/2 && tdY>tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 4; //lower right

  switch(tQuadrant)
  {
    case 1:
      tX1 = tXaxis->GetBinCenter(tXbin);
      tY1 = tYaxis->GetBinCenter(tYbin);
      tX2 = tXaxis->GetBinCenter(tXbin+1);
      tY2 = tYaxis->GetBinCenter(tYbin+1);
      break;
    case 2:
      tX1 = tXaxis->GetBinCenter(tXbin-1);
      tY1 = tYaxis->GetBinCenter(tYbin);
      tX2 = tXaxis->GetBinCenter(tXbin);
      tY2 = tYaxis->GetBinCenter(tYbin+1);
      break;
    case 3:
      tX1 = tXaxis->GetBinCenter(tXbin-1);
      tY1 = tYaxis->GetBinCenter(tYbin-1);
      tX2 = tXaxis->GetBinCenter(tXbin);
      tY2 = tYaxis->GetBinCenter(tYbin);
      break;
    case 4:
      tX1 = tXaxis->GetBinCenter(tXbin);
      tY1 = tYaxis->GetBinCenter(tYbin-1);
      tX2 = tXaxis->GetBinCenter(tXbin+1);
      tY2 = tYaxis->GetBinCenter(tYbin);
      break;
  }

  int tBinX1 = tXaxis->FindBin(tX1);
  if(tBinX1<1) tBinX1 = 1;

  int tBinX2 = tXaxis->FindBin(tX2);
  if(tBinX2>a2dHisto->GetNbinsX()) tBinX2=a2dHisto->GetNbinsX();

  int tBinY1 = tYaxis->FindBin(tY1);
  if(tBinY1<1) tBinY1 = 1;

  int tBinY2 = tYaxis->FindBin(tY2);
  if(tBinY2>a2dHisto->GetNbinsY()) tBinY2=a2dHisto->GetNbinsY();

  int tBinQ22 = a2dHisto->GetBin(tBinX2,tBinY2);
  int tBinQ12 = a2dHisto->GetBin(tBinX1,tBinY2);
  int tBinQ11 = a2dHisto->GetBin(tBinX1,tBinY1);
  int tBinQ21 = a2dHisto->GetBin(tBinX2,tBinY1);

  double tQ11 = a2dHisto->GetBinContent(tBinQ11);
  double tQ12 = a2dHisto->GetBinContent(tBinQ12);
  double tQ21 = a2dHisto->GetBinContent(tBinQ21);
  double tQ22 = a2dHisto->GetBinContent(tBinQ22);

  double tD = 1.0*(tX2-tX1)*(tY2-tY1);

  tF = (1.0/tD)*(tQ11*(tX2-aX)*(tY2-aY) + tQ21*(aX-tX1)*(tY2-aY) + tQ12*(tX2-aX)*(aY-tY1) + tQ22*(aX-tX1)*(aY-tY1));

  return tF;
}


//________________________________________________________________________________________________________________
double ChargedResidualCf::BilinearInterpolateVector(vector<vector<double> > &a2dVec, double aX, int aNbinsX, double aMinX, double aMaxX, double aY, int aNbinsY, double aMinY, double aMaxY)
{
  //NOTE: THIS IS SLOWER THAN BilinearInterpolate, but a method like this may be necessary for parallelization

  double tF = 0.;
  double tX1=0., tX2=0., tY1=0., tY2=0.;
  double tdX, tdY;

  int tXbin = GetBinNumber(aNbinsX,aMinX,aMaxX,aX);
  int tYbin = GetBinNumber(aNbinsY,aMinY,aMaxY,aY);

  double tBinWidthX = (aMaxX-aMinX)/aNbinsX;
  double tBinMinX = aMinX + tXbin*tBinWidthX;
  double tBinMaxX = aMinX + (tXbin+1)*tBinWidthX;

  double tBinWidthY = (aMaxY-aMinY)/aNbinsY;
  double tBinMinY = aMinY + tYbin*tBinWidthY;
  double tBinMaxY = aMinY + (tYbin+1)*tBinWidthY;

  //---------------------------------
  if(tXbin<0 || tYbin<0) 
  {
    cout << "Error in ChargedResidualCf::BilinearInterpolateVector, cannot interpolate outside histogram domain" << endl;
  }
  assert(tXbin >= 0);
  assert(tYbin >= 0);

  //---------------------------------

  int tQuadrant = 0; //CCW from UR 1,2,3,4
  // which quadrant of the bin (bin_P) are we in?
  tdX = tBinMaxX - aX;
  tdY = tBinMaxY - aY;

  int tBinX1, tBinX2, tBinY1, tBinY2;

  if(tdX<=tBinWidthX/2 && tdY<=tBinWidthY/2) tQuadrant = 1; //upper right
  else if(tdX>tBinWidthX/2 && tdY<=tBinWidthY/2) tQuadrant = 2; //upper left
  else if(tdX>tBinWidthX/2 && tdY>tBinWidthY/2) tQuadrant = 3; //lower left
  else if(tdX<=tBinWidthX/2 && tdY>tBinWidthY/2) tQuadrant = 4; //lower right
  else cout << "ERROR IN BilinearInterpolateVector!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;


  switch(tQuadrant)
  {
    case 1:
      tX1 = tBinMinX + tBinWidthX/2;
      tY1 = tBinMinY + tBinWidthY/2;
      tX2 = tBinMaxX + tBinWidthX/2;
      tY2 = tBinMaxY + tBinWidthY/2;

      tBinX1 = tXbin;
      tBinX2 = tXbin+1;
      tBinY1 = tYbin;
      tBinY2 = tYbin+1;

      break;
    case 2:
      tX1 = tBinMinX - tBinWidthX/2;
      tY1 = tBinMinY + tBinWidthY/2;
      tX2 = tBinMinX + tBinWidthX/2;
      tY2 = tBinMaxY + tBinWidthY/2;

      tBinX1 = tXbin-1;
      tBinX2 = tXbin;
      tBinY1 = tYbin;
      tBinY2 = tYbin+1;

      break;
    case 3:
      tX1 = tBinMinX - tBinWidthX/2;
      tY1 = tBinMinY - tBinWidthY/2;
      tX2 = tBinMinX + tBinWidthX/2;
      tY2 = tBinMinY + tBinWidthY/2;

      tBinX1 = tXbin-1;
      tBinX2 = tXbin;
      tBinY1 = tYbin-1;
      tBinY2 = tYbin;

      break;
    case 4:
      tX1 = tBinMinX + tBinWidthX/2;
      tY1 = tBinMinY - tBinWidthY/2;
      tX2 = tBinMaxX + tBinWidthX/2;
      tY2 = tBinMinY + tBinWidthY/2;

      tBinX1 = tXbin;
      tBinX2 = tXbin+1;
      tBinY1 = tYbin-1;
      tBinY2 = tYbin;

      break;
  }

  if(tBinX1<1) tBinX1 = 1;
  if(tBinX2>aNbinsX) tBinX2=aNbinsX;
  if(tBinY1<1) tBinY1 = 1;
  if(tBinY2>aNbinsY) tBinY2=aNbinsY;

  double tQ11 = a2dVec[tBinX1][tBinY1];
  double tQ12 = a2dVec[tBinX1][tBinY2];
  double tQ21 = a2dVec[tBinX2][tBinY1];
  double tQ22 = a2dVec[tBinX2][tBinY2];

  double tD = 1.0*(tX2-tX1)*(tY2-tY1);

  tF = (1.0/tD)*(tQ11*(tX2-aX)*(tY2-aY) + tQ21*(aX-tX1)*(tY2-aY) + tQ12*(tX2-aX)*(aY-tY1) + tQ22*(aX-tX1)*(aY-tY1));

  return tF;
}

//________________________________________________________________________________________________________________
double ChargedResidualCf::TrilinearInterpolate(TH3* a3dHisto, double aX, double aY, double aZ)
{
  TAxis* tXaxis = a3dHisto->GetXaxis();
  TAxis* tYaxis = a3dHisto->GetYaxis();
  TAxis* tZaxis = a3dHisto->GetZaxis();

  //--------------------------------
  int ubx = tXaxis->FindBin(aX);
  if( aX < tXaxis->GetBinCenter(ubx) ) ubx -= 1;
  int obx = ubx + 1;

  int uby = tYaxis->FindBin(aY);
  if( aY < tYaxis->GetBinCenter(uby) ) uby -= 1;
  int oby = uby + 1;

  int ubz = tZaxis->FindBin(aZ);
  if( aZ < tZaxis->GetBinCenter(ubz) ) ubz -= 1;
  int obz = ubz + 1;

  //--------------------------------
  //make sure (aX,aY,aZ) is within the limits, so I can interpolate
  if(ubx<=0 || uby<=0 || ubz<=0 || obx>tXaxis->GetNbins() || oby>tYaxis->GetNbins() || obz>tZaxis->GetNbins())
  {
    cout << "Error in ChargedResidualCf::TrilinearInterpolate, cannot interpolate outside histogram domain" << endl;

    cout << "aX = " << aX << "\taY = " << aY << "\taZ = " << aZ << endl;
    cout << "ubx = " << ubx << "\tuby = " << uby << "\tubz = " << ubz << endl;
    cout << "obx = " << obx << "\toby = " << oby << "\tobz = " << obz << endl;
  }
  assert(ubx>0);
  assert(uby>0);
  assert(ubz>0);
  assert(obx<=tXaxis->GetNbins());
  assert(oby<=tYaxis->GetNbins());
  assert(obz<=tZaxis->GetNbins());
  //--------------------------------

  double xw = tXaxis->GetBinCenter(obx) - tXaxis->GetBinCenter(ubx);
  double yw = tYaxis->GetBinCenter(oby) - tYaxis->GetBinCenter(uby);
  double zw = tZaxis->GetBinCenter(obz) - tZaxis->GetBinCenter(ubz);

  double xd = (aX - tXaxis->GetBinCenter(ubx))/xw;
  double yd = (aY - tYaxis->GetBinCenter(uby))/yw;
  double zd = (aZ - tZaxis->GetBinCenter(ubz))/zw;

  double v[] = { a3dHisto->GetBinContent(ubx, uby, ubz), a3dHisto->GetBinContent(ubx, uby, obz),
                 a3dHisto->GetBinContent(ubx, oby, ubz), a3dHisto->GetBinContent(ubx, oby, obz),
                 a3dHisto->GetBinContent(obx, uby, ubz), a3dHisto->GetBinContent(obx, uby, obz),
                 a3dHisto->GetBinContent(obx, oby, ubz), a3dHisto->GetBinContent(obx, oby, obz) };

  double i1 = v[0]*(1.-zd) + v[1]*zd;
  double i2 = v[2]*(1.-zd) + v[3]*zd;
  double j1 = v[4]*(1.-zd) + v[5]*zd;
  double j2 = v[6]*(1.-zd) + v[7]*zd;

  double w1 = i1*(1.-yd) + i2*yd;
  double w2 = j1*(1.-yd) + j2*yd;

  double tResult = w1*(1.-xd) + w2*xd;

  return tResult;
}



//________________________________________________________________________________________________________________
double ChargedResidualCf::QuadrilinearInterpolate(THn* a4dHisto, double aT, double aX, double aY, double aZ)
{
  TAxis* tTaxis = a4dHisto->GetAxis(0);
  TAxis* tXaxis = a4dHisto->GetAxis(1);
  TAxis* tYaxis = a4dHisto->GetAxis(2);
  TAxis* tZaxis = a4dHisto->GetAxis(3);

  //--------------------------------

  int ubt = tTaxis->FindBin(aT);
  if( aT < tTaxis->GetBinCenter(ubt) ) ubt -= 1;
  int obt = ubt + 1;

  int ubx = tXaxis->FindBin(aX);
  if( aX < tXaxis->GetBinCenter(ubx) ) ubx -= 1;
  int obx = ubx + 1;

  int uby = tYaxis->FindBin(aY);
  if( aY < tYaxis->GetBinCenter(uby) ) uby -= 1;
  int oby = uby + 1;

  int ubz = tZaxis->FindBin(aZ);
  if( aZ < tZaxis->GetBinCenter(ubz) ) ubz -= 1;
  int obz = ubz + 1;

  //--------------------------------
  //make sure (aT,aX,aY,aZ) is within the limits, so I can interpolate
  if(ubt<=0 || ubx<=0 || uby<=0 || ubz<=0 || obt>tTaxis->GetNbins() || obx>tXaxis->GetNbins() || oby>tYaxis->GetNbins() || obz>tZaxis->GetNbins())
  {
    cout << "Error in ChargedResidualCf::QuadrilinearInterpolate, cannot interpolate outside histogram domain" << endl;
  }
  assert(ubt>0);
  assert(ubx>0);
  assert(uby>0);
  assert(ubz>0);
  assert(obt<=tTaxis->GetNbins());
  assert(obx<=tXaxis->GetNbins());
  assert(oby<=tYaxis->GetNbins());
  assert(obz<=tZaxis->GetNbins());
  //--------------------------------

  double tw = tTaxis->GetBinCenter(obt) - tTaxis->GetBinCenter(ubt);
  double xw = tXaxis->GetBinCenter(obx) - tXaxis->GetBinCenter(ubx);
  double yw = tYaxis->GetBinCenter(oby) - tYaxis->GetBinCenter(uby);
  double zw = tZaxis->GetBinCenter(obz) - tZaxis->GetBinCenter(ubz);

  double td = (aT - tTaxis->GetBinCenter(ubt))/tw;
  double xd = (aX - tXaxis->GetBinCenter(ubx))/xw;
  double yd = (aY - tYaxis->GetBinCenter(uby))/yw;
  double zd = (aZ - tZaxis->GetBinCenter(ubz))/zw;

  //--------------------------------
  //TODO these probably don't need to all be made at the same time
  // i.e., I can have a general int tBin, and put the appropriate numbers in there we needed
  int tBin0000[4] = {ubt,ubx,uby,ubz};
  int tBin1000[4] = {obt,ubx,uby,ubz};

  int tBin0100[4] = {ubt,obx,uby,ubz};
  int tBin1100[4] = {obt,obx,uby,ubz};

  int tBin0010[4] = {ubt,ubx,oby,ubz};
  int tBin1010[4] = {obt,ubx,oby,ubz};

  int tBin0110[4] = {ubt,obx,oby,ubz};
  int tBin1110[4] = {obt,obx,oby,ubz};

  int tBin0001[4] = {ubt,ubx,uby,obz};
  int tBin1001[4] = {obt,ubx,uby,obz};

  int tBin0101[4] = {ubt,obx,uby,obz};
  int tBin1101[4] = {obt,obx,uby,obz};

  int tBin0011[4] = {ubt,ubx,oby,obz};
  int tBin1011[4] = {obt,ubx,oby,obz};

  int tBin0111[4] = {ubt,obx,oby,obz};
  int tBin1111[4] = {obt,obx,oby,obz};

  //--------------------------------

  //Interpolate along t
  double tC000 = a4dHisto->GetBinContent(tBin0000)*(1.-td) + a4dHisto->GetBinContent(tBin1000)*td;
  double tC100 = a4dHisto->GetBinContent(tBin0100)*(1.-td) + a4dHisto->GetBinContent(tBin1100)*td;

  double tC010 = a4dHisto->GetBinContent(tBin0010)*(1.-td) + a4dHisto->GetBinContent(tBin1010)*td;
  double tC110 = a4dHisto->GetBinContent(tBin0110)*(1.-td) + a4dHisto->GetBinContent(tBin1110)*td;

  double tC001 = a4dHisto->GetBinContent(tBin0001)*(1.-td) + a4dHisto->GetBinContent(tBin1001)*td;
  double tC101 = a4dHisto->GetBinContent(tBin0101)*(1.-td) + a4dHisto->GetBinContent(tBin1101)*td;

  double tC011 = a4dHisto->GetBinContent(tBin0011)*(1.-td) + a4dHisto->GetBinContent(tBin1011)*td;
  double tC111 = a4dHisto->GetBinContent(tBin0111)*(1.-td) + a4dHisto->GetBinContent(tBin1111)*td;

  //--------------------------------

  //Interpolate along x
  double tC00 = tC000*(1.-xd) + tC100*xd;
  double tC10 = tC010*(1.-xd) + tC110*xd;
  double tC01 = tC001*(1.-xd) + tC101*xd;
  double tC11 = tC011*(1.-xd) + tC111*xd;

  //--------------------------------

  //Interpolate along y
  double tC0 = tC00*(1.-yd) + tC10*yd;
  double tC1 = tC01*(1.-yd) + tC11*yd;

  //--------------------------------

  //Interpolate along z
  double tC = tC0*(1.-zd) + tC1*zd;

  return tC;
}


//________________________________________________________________________________________________________________
double ChargedResidualCf::GetEta(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else return pow(((aKStar/hbarc)*fBohrRadius),-1);
}

//________________________________________________________________________________________________________________
double ChargedResidualCf::GetGamowFactor(double aKStar)
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
complex<double> ChargedResidualCf::GetExpTerm(double aKStar, double aRStar, double aTheta)
{
  complex<double> tReturnValue = exp(-ImI*(aKStar/hbarc)*aRStar*cos(aTheta));
  return tReturnValue;
}

//________________________________________________________________________________________________________________
complex<double> ChargedResidualCf::BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0)
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
    double tLednickyHFunction = LinearInterpolate(fLednickyHFunctionHist,aKStar);
    double tImagChi = GetGamowFactor(aKStar)/(2.*GetEta(aKStar));
    complex<double> tLednickyChi (tLednickyHFunction,tImagChi);

    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - 2.*tLednickyChi/fBohrRadius,-1);
  }

  return tScattAmp;
}



//________________________________________________________________________________________________________________
double ChargedResidualCf::InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
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
      tHyperGeo1F1Real = TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

      tGTildeReal = BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
      tGTildeImag = BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);
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

  if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in ChargedResidualCf::InterpolateWfSquared !!!!!" << endl;
  assert(imag(tResultComplex) < std::numeric_limits< double >::min());

  return real(tResultComplex);

}



//________________________________________________________________________________________________________________
bool ChargedResidualCf::CanInterpKStar(double aKStar)
{
  if(aKStar < fMinInterpKStar) return false;
  if(aKStar > fMaxInterpKStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool ChargedResidualCf::CanInterpRStar(double aRStar)
{
  if(aRStar < fMinInterpRStar || aRStar > fMaxInterpRStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool ChargedResidualCf::CanInterpTheta(double aTheta)
{
  if(aTheta < fMinInterpTheta || aTheta > fMaxInterpTheta) return false;
  return true;
}



//________________________________________________________________________________________________________________
bool ChargedResidualCf::CanInterpAll(double aKStar, double aRStar, double aTheta)
{
  if(CanInterpKStar(aKStar) && CanInterpRStar(aRStar) && CanInterpTheta(aTheta)) return true;
  return false;
}

//________________________________________________________________________________________________________________
void ChargedResidualCf::SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax)
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
td1dVec ChargedResidualCf::GetExpXiData(double aMaxKStar, CentralityType aCentType)
{
  if(fExpXiData.size()==0)
  {
//    TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170406/Results_cXicKch_20170406";
    TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170505_ignoreOnFlyStatus/Results_cXicKch_20170505_ignoreOnFlyStatus";
    AnalysisType tAnType;
    AnalysisRunType tRunType=kTrain;
    int tNFitPartialAnalysis=2;

    switch(fResidualType) {
    case kResXiCKchP:
    case kResOmegaKchP:
      tAnType = kXiKchP;
      break;

    case kResAXiCKchP:
    case kResAOmegaKchP:
      tAnType = kAXiKchP;
      break;

    case kResXiCKchM:
    case kResOmegaKchM:
      tAnType = kXiKchM;
      break;

    case kResAXiCKchM:
    case kResAOmegaKchM:
      tAnType = kAXiKchM;
      break;

    default:
      cout << "ERROR: ChargedResidualCf::GetExpXiData():  fResidualType = " << fResidualType << " is not apropriate" << endl << endl;
      assert(0);
    }

    FitPairAnalysis* tPairAn = new FitPairAnalysis(tFileLocationBase,tAnType,aCentType,tRunType,tNFitPartialAnalysis);
    tPairAn->RebinKStarCfHeavy(2,0.32,0.4);
    TH1D* tExpHist = (TH1D*)tPairAn->GetKStarCfHeavy()->GetHeavyCfClone();
    assert(tExpHist->GetXaxis()->GetBinWidth(1)==0.01);  //TODO make general
    assert(tExpHist->GetNbinsX()==100);

    int tNbins = std::round(aMaxKStar/tExpHist->GetXaxis()->GetBinWidth(1));

    td1dVec tReturnVec(tNbins,0.);
    for(int i=0; i<tNbins; i++) tReturnVec[i] = tExpHist->GetBinContent(i+1);

    fExpXiData = tReturnVec;
    delete tPairAn;
  }

  return fExpXiData;
}



//________________________________________________________________________________________________________________
double ChargedResidualCf::GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par)
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

    tCanInterp = CanInterpAll(tKStarMag,tRStarMag,tTheta);
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
td1dVec ChargedResidualCf::GetCoulombParentCorrelation(double *aParentCfParams, vector<double> &aKStarBinCenters, bool aUseExpXiData, CentralityType aCentType)
{
  fUseExpXiData = aUseExpXiData;

  double tKStarBinWidth = aKStarBinCenters[1]-aKStarBinCenters[0];

  double tKStarMin, tKStarMax;
  tKStarMax = aKStarBinCenters[aKStarBinCenters.size()-1]+tKStarBinWidth/2.;

  if(fUseExpXiData) fCoulombCf = GetExpXiData(tKStarMax,aCentType);
  else
  {
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
  }
  return fCoulombCf;
}



//________________________________________________________________________________________________________________
td1dVec ChargedResidualCf::GetCoulombResidualCorrelation(double *aParentCfParams, vector<double> &aKStarBinCenters, bool aUseExpXiData, CentralityType aCentType)
{
  fUseExpXiData = aUseExpXiData;

  double tKStarBinWidth = aKStarBinCenters[1]-aKStarBinCenters[0];
  double tKStarMin, tKStarMax;
  tKStarMax = aKStarBinCenters[aKStarBinCenters.size()-1]+tKStarBinWidth/2.;

  if(fUseExpXiData) fCoulombCf = GetExpXiData(tKStarMax,aCentType);
  else
  {
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
  }

  unsigned int tDaughterPairKStarBin, tParentPairKStarBin;
  double tDaughterPairKStar, tParentPairKStar;
  assert(fCoulombCf.size() == aKStarBinCenters.size());
  assert(fCoulombCf.size() == (unsigned int)fTransformMatrix->GetNbinsX());
  assert(fCoulombCf.size() == (unsigned int)fTransformMatrix->GetNbinsY());

  vector<double> tReturnResCf(fCoulombCf.size(),0.);
  vector<double> tNormVec(fCoulombCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

  for(unsigned int i=0; i<fCoulombCf.size(); i++)
  {
    tDaughterPairKStar = aKStarBinCenters[i];
    tDaughterPairKStarBin = fTransformMatrix->GetXaxis()->FindBin(tDaughterPairKStar);

    for(unsigned int j=0; j<fCoulombCf.size(); j++)
    {
      tParentPairKStar = aKStarBinCenters[j];
      tParentPairKStarBin = fTransformMatrix->GetYaxis()->FindBin(tParentPairKStar);

      tReturnResCf[i] += fCoulombCf[j]*fTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
      tNormVec[i] += fTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
    }
    if(tNormVec[i]!=0.) tReturnResCf[i] /= tNormVec[i];
  }
  fCoulombResidualCf = tReturnResCf;
  return fCoulombResidualCf;
}





