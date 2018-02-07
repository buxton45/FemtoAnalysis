///////////////////////////////////////////////////////////////////////////
// CoulombFitterParallel:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "CoulombFitterParallel.h"

#ifdef __ROOT__
ClassImp(CoulombFitterParallel)
#endif

//  Global variables needed to be seen by FCN
/*
vector<TH1F*> gCfsToFit;
int gNpFits;
vector<int> gNpfitsVec;
//vector<double> gMaxFitKStar;
double gMaxFitKStar;
*/

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

/*
//________________________________________________________________________________________________________________
CoulombFitterParallel::CoulombFitterParallel():
  CoulombFitter(),

  fGTildeReal(0),
  fGTildeImag(0),

  fHyperGeo1F1Real(0),
  fHyperGeo1F1Imag(0),

  fParallelWaveFunction(0)

{
  fParallelWaveFunction = new ParallelWaveFunction();
}
*/


//________________________________________________________________________________________________________________
CoulombFitterParallel::CoulombFitterParallel(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar):
  CoulombFitter(aFitSharedAnalyses,aMaxFitKStar),

  fGTildeReal(0),
  fGTildeImag(0),

  fHyperGeo1F1Real(0),
  fHyperGeo1F1Imag(0),

  fParallelWaveFunction(0)

{
  fParallelWaveFunction = new ParallelWaveFunction();
  SetVerbose(true);

  SetCoulombAttributes(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetAnalysisType());
}


//________________________________________________________________________________________________________________
CoulombFitterParallel::~CoulombFitterParallel()
{
/*
  cout << "CoulombFitterParallel object is being deleted!!!" << endl;

  //---Clean up
  if(fInterpHistsLoaded)
  {
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
*/
  //TODO figure out how to deallocate and delete multi dimensional vectors
/*
  delete fHyperGeo1F1Real;
  delete fHyperGeo1F1Imag;

  delete fGTildeReal;
  delete fGTildeImag;
*/
}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::SetCoulombAttributes(AnalysisType aAnalysisType)
{
  CoulombFitter::SetCoulombAttributes(aAnalysisType);
  fParallelWaveFunction->LoadBohrRadius(fBohrRadius);
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::PassHyperGeo1F1AndGTildeToParallelWaveFunction()
{
ChronoTimer tTimer(kSec);
tTimer.Start();
cout << "Starting PassHyperGeo1F1AndGTildeToParallelWaveFunction" << endl;

//--------------------------------------------------------------
  fHyperGeo1F1Info.nBinsK = fHyperGeo1F1RealHist->GetXaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthK = fHyperGeo1F1RealHist->GetXaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minK = fHyperGeo1F1RealHist->GetXaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpK = fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxK = fHyperGeo1F1RealHist->GetXaxis()->GetBinUpEdge(fHyperGeo1F1RealHist->GetNbinsX());
  fHyperGeo1F1Info.maxInterpK = fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsX());

  fHyperGeo1F1Info.nBinsR = fHyperGeo1F1RealHist->GetYaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthR = fHyperGeo1F1RealHist->GetYaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minR = fHyperGeo1F1RealHist->GetYaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpR = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxR = fHyperGeo1F1RealHist->GetYaxis()->GetBinUpEdge(fHyperGeo1F1RealHist->GetNbinsY());
  fHyperGeo1F1Info.maxInterpR = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsY());

  fHyperGeo1F1Info.nBinsTheta = fHyperGeo1F1RealHist->GetZaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinUpEdge(fHyperGeo1F1RealHist->GetNbinsZ());
  fHyperGeo1F1Info.maxInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsZ());

  //-----

  fGTildeInfo.nBinsK = fGTildeRealHist->GetXaxis()->GetNbins();
  fGTildeInfo.binWidthK = fGTildeRealHist->GetXaxis()->GetBinWidth(1);
  fGTildeInfo.minK = fGTildeRealHist->GetXaxis()->GetBinLowEdge(1);
  fGTildeInfo.minInterpK = fGTildeRealHist->GetXaxis()->GetBinCenter(1);
  fGTildeInfo.maxK = fGTildeRealHist->GetXaxis()->GetBinUpEdge(fGTildeRealHist->GetNbinsX());
  fGTildeInfo.maxInterpK = fGTildeRealHist->GetXaxis()->GetBinCenter(fGTildeRealHist->GetNbinsX());

  fGTildeInfo.nBinsR = fGTildeRealHist->GetYaxis()->GetNbins();
  fGTildeInfo.binWidthR = fGTildeRealHist->GetYaxis()->GetBinWidth(1);
  fGTildeInfo.minR = fGTildeRealHist->GetYaxis()->GetBinLowEdge(1);
  fGTildeInfo.minInterpR = fGTildeRealHist->GetYaxis()->GetBinCenter(1);
  fGTildeInfo.maxR = fGTildeRealHist->GetYaxis()->GetBinUpEdge(fGTildeRealHist->GetNbinsY());
  fGTildeInfo.maxInterpR = fGTildeRealHist->GetYaxis()->GetBinCenter(fGTildeRealHist->GetNbinsY());

//--------------------------------------------------------------

  int tNbinsK = fHyperGeo1F1RealHist->GetXaxis()->GetNbins();
  int tNbinsR = fHyperGeo1F1RealHist->GetYaxis()->GetNbins();
  int tNbinsTheta = fHyperGeo1F1RealHist->GetZaxis()->GetNbins();

  fHyperGeo1F1Real.resize(tNbinsK, td2dVec(tNbinsR, td1dVec(tNbinsTheta,0)));
  fHyperGeo1F1Imag.resize(tNbinsK, td2dVec(tNbinsR, td1dVec(tNbinsTheta,0)));

  int iK, iR, iTheta;
  #pragma omp parallel for private(iK,iR,iTheta)
  for(iK=1; iK<=tNbinsK; iK++)
  {
    for(iR=1; iR<=tNbinsR; iR++)
    {
      for(iTheta=1; iTheta<=tNbinsTheta; iTheta++)
      {
        fHyperGeo1F1Real[iK-1][iR-1][iTheta-1] = fHyperGeo1F1RealHist->GetBinContent(iK,iR,iTheta);
        fHyperGeo1F1Imag[iK-1][iR-1][iTheta-1] = fHyperGeo1F1ImagHist->GetBinContent(iK,iR,iTheta);
      }
    }
  }

  fGTildeReal.resize(tNbinsK, td1dVec(tNbinsR,0));
  fGTildeImag.resize(tNbinsK, td1dVec(tNbinsR,0));

  #pragma omp parallel for private(iK,iR)
  for(iK=1; iK<=tNbinsK; iK++)
  {
    for(iR=1; iR<=tNbinsR; iR++)
    {
      fGTildeReal[iK-1][iR-1] = fGTildeRealHist->GetBinContent(iK,iR);
      fGTildeImag[iK-1][iR-1] = fGTildeImagHist->GetBinContent(iK,iR);
    }
  }

  //--------------------------------------------------------------

  fParallelWaveFunction->LoadGTildeReal(fGTildeReal);
  fParallelWaveFunction->LoadGTildeImag(fGTildeImag);
  fParallelWaveFunction->LoadGTildeInfo(fGTildeInfo);

  fParallelWaveFunction->LoadHyperGeo1F1Real(fHyperGeo1F1Real);
  fParallelWaveFunction->LoadHyperGeo1F1Imag(fHyperGeo1F1Imag);
  fParallelWaveFunction->LoadHyperGeo1F1Info(fHyperGeo1F1Info);

//--------------------------------------------------------------
tTimer.Stop();
cout << "PassHyperGeo1F1AndGTildeToParallelWaveFunction: ";
tTimer.PrintInterval();
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::LoadInterpHistFile(TString aFileBaseName)
{
  CoulombFitter::LoadInterpHistFile(aFileBaseName);

//--------------------------------------------------------------
//Send things over to ParallWaveFunction object
//TODO after sending these vectors over, they can probably be deleted
//maybe I should not even make them class members, but only members of this function

  td1dVec tLednickyHFunction(0);
  for(int i=1; i<=fLednickyHFunctionHist->GetNbinsX(); i++) tLednickyHFunction.push_back(fLednickyHFunctionHist->GetBinContent(i));
  fParallelWaveFunction->LoadLednickyHFunction(tLednickyHFunction);
  PassHyperGeo1F1AndGTildeToParallelWaveFunction();
}

//________________________________________________________________________________________________________________
td3dVec CoulombFitterParallel::BuildPairKStar3dVecFull(TString aPairKStarNtupleBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  td3dVec tPairKStar3dVec = CoulombFitter::BuildPairKStar3dVecFull(aPairKStarNtupleBaseName,aNFiles,aAnalysisType,aCentralityType,aNbinsKStar,aKStarMin,aKStarMax);

  //----****----****----****----****----****----****
  //TODO make this automated
  //TODO add centrality selection to this
  //Also, this is basically FitPartialAnalysis::ConnectAnalysisDirectory


  //--Set the fPairKStar3dVecInfo
  fPairKStar3dVecInfo.nBinsK = aNbinsKStar;
  fPairKStar3dVecInfo.minK = aKStarMin;
  fPairKStar3dVecInfo.maxK = aKStarMax;
  fPairKStar3dVecInfo.binWidthK = ((aKStarMax-aKStarMin)/aNbinsKStar);
  //---------------------------

//TODO
/*
  assert(aNbinsKStar == (int)tPairKStar3dVec.size());
  int tOffset=0;
  for(int i=0; i<aNbinsKStar; i++)
  {
    fPairKStar3dVecInfo.nPairsPerBin[i] = tPairKStar3dVec[i].size();
    fPairKStar3dVecInfo.binOffset[i] = tOffset;
    tOffset += 4*tPairKStar3dVec[i].size();
  }

  fParallelWaveFunction->LoadPairKStar3dVec(tPairKStar3dVec,fPairKStar3dVecInfo);
*/
  //---------------------------------------------------

  return tPairKStar3dVec;
}



//________________________________________________________________________________________________________________
td3dVec CoulombFitterParallel::BuildPairKStar3dVecFromTxt(TString aFileName)
{
  td3dVec tPairKStar3dVec = CoulombFitter::BuildPairKStar3dVecFromTxt(aFileName);

  int tNbinsKStar = tPairKStar3dVec[tPairKStar3dVec.size()-1][0][0];
  double tKStarMin = tPairKStar3dVec[tPairKStar3dVec.size()-1][0][1];
  double tKStarMax = tPairKStar3dVec[tPairKStar3dVec.size()-1][0][2];

cout << "tNbinsKStar = " << tNbinsKStar << endl;
cout << "tKStarMin = " << tKStarMin << endl;
cout << "tKStarMax = " << tKStarMax << endl;
//---------------------------------------------------

  //--Set the fPairKStar3dVecInfo
  fPairKStar3dVecInfo.nBinsK = tNbinsKStar;
  fPairKStar3dVecInfo.minK = tKStarMin;
  fPairKStar3dVecInfo.maxK = tKStarMax;
  fPairKStar3dVecInfo.binWidthK = ((tKStarMax-tKStarMin)/tNbinsKStar);

  //---------------------------------------------------
//TODO
/*
  assert(tNbinsKStar == (int)tPairKStar3dVec.size());
  int tOffset=0;
  for(int i=0; i<tNbinsKStar; i++)
  {
    fPairKStar3dVecInfo.nPairsPerBin[i] = tPairKStar3dVec[i].size();
    fPairKStar3dVecInfo.binOffset[i] = tOffset;
    tOffset += 4*tPairKStar3dVec[i].size();
  }

  fParallelWaveFunction->LoadPairKStar3dVec(tPairKStar3dVec,fPairKStar3dVecInfo);
*/
  //---------------------------------------------------

  return tPairKStar3dVec;
}

//________________________________________________________________________________________________________________
td1dVec CoulombFitterParallel::BuildPairKStar4dVecFromTxt(TString aFileBaseName)
{
  td1dVec tReturnBinInfo = CoulombFitter::BuildPairKStar4dVecFromTxt(aFileBaseName);
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::BuildPairKStar4dVecOnFly(TString aPairKStarNtupleBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  CoulombFitter::BuildPairKStar4dVecOnFly(aPairKStarNtupleBaseName, aNFiles, aNbinsKStar, aKStarMin, aKStarMax);
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::Sort2dPairsByRadius(const td1dVec &aVec1, const td1dVec &aVec2)
{
  return aVec1[1] < aVec2[1];
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::Sort2dPairsByGlobalBin(const td1dVec &aVec1, const td1dVec &aVec2)
{
  return aVec1[3] < aVec2[3];
}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::FullSort2dPairVec(td2dVec &aPairs, BinInfoHyperGeo1F1 &aBinInfo)
{
  int tGlobalBin=-1;

  int tBinNumberK=-1;
  int tBinNumberR=-1;
  int tBinNumberTheta=-1;
  for(unsigned int i=0; i<aPairs.size(); i++)
  {
    tBinNumberK = Interpolator::GetBinNumber(aBinInfo.binWidthK, aBinInfo.minK, aBinInfo.maxK, aPairs[i][0]);
    tBinNumberR = Interpolator::GetBinNumber(aBinInfo.binWidthR, aBinInfo.minR, aBinInfo.maxR, aPairs[i][1]);
    tBinNumberTheta = Interpolator::GetBinNumber(aBinInfo.binWidthTheta, aBinInfo.minTheta, aBinInfo.maxTheta, aPairs[i][2]);

    tGlobalBin = tBinNumberK*aBinInfo.nBinsR*aBinInfo.nBinsTheta + tBinNumberR*aBinInfo.nBinsTheta + tBinNumberTheta;

    aPairs[i].push_back(tGlobalBin);
  }
  sort(aPairs.begin(), aPairs.end(), Sort2dPairsByGlobalBin);
  for(unsigned int i=0; i<aPairs.size(); i++) aPairs[i].pop_back();
}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::FullSort4dPairVec(td4dVec &a4dPairs, BinInfoHyperGeo1F1 &aBinInfo)
{
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    for(int iKStarBin=0; iKStarBin<fNbinsXToBuild; iKStarBin++)
    {
      FullSort2dPairVec(a4dPairs[iAnaly][iKStarBin], aBinInfo);
    }
  }
}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::BuildPairSample4dVec(int aNPairsPerKStarBin, double aBinSize, bool aSort)
{
  CoulombFitter::BuildPairSample4dVec(aNPairsPerKStarBin, aBinSize);

  if(aSort) FullSort4dPairVec(fPairSample4dVec, fHyperGeo1F1Info);

  fSamplePairsBinInfo.nAnalyses = fNAnalyses;
  fSamplePairsBinInfo.nBinsK = fNbinsXToBuild;
  fSamplePairsBinInfo.nPairsPerBin = fNPairsPerKStarBin;

  fSamplePairsBinInfo.minK = 0.; //TODO
  fSamplePairsBinInfo.maxK = fMaxFitKStar;
  fSamplePairsBinInfo.binWidthK = fKStarBinWidth;
  fSamplePairsBinInfo.nElementsPerPair = fPairSample4dVec[0][0][0].size();

  fParallelWaveFunction->LoadPairSample4dVec(fPairSample4dVec, fSamplePairsBinInfo);
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::CanInterpAllSamplePairs()
{
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    for(int iKStarBin=0; iKStarBin<(int)fPairSample4dVec[iAnaly].size(); iKStarBin++)
    {
      for(int iPair=0; iPair<(int)fPairSample4dVec[iAnaly][iKStarBin].size(); iPair++)
      {
        if(!CanInterpKStar(fPairSample4dVec[iAnaly][iKStarBin][iPair][0])) return false;
        if(!CanInterpRStar(fPairSample4dVec[iAnaly][iKStarBin][iPair][1])) return false;
        if(!CanInterpTheta(fPairSample4dVec[iAnaly][iKStarBin][iPair][2])) return false;
      }
    }
  }
  return true;
}

//________________________________________________________________________________________________________________
td3dVec CoulombFitterParallel::GetCPUSamplePairs(int aAnalysisNumber)
{
  td3dVec tReturnVec;

  td2dVec tTempKStarBinVec;
  td1dVec tTempPairVec;

  for(int iKStarBin=0; iKStarBin<(int)fPairSample4dVec[aAnalysisNumber].size(); iKStarBin++)
  {
    tTempKStarBinVec.clear();
    for(int iPair=0; iPair<(int)fPairSample4dVec[aAnalysisNumber][iKStarBin].size(); iPair++)
    {
      tTempPairVec.clear();
      if(!CanInterpKStar(fPairSample4dVec[aAnalysisNumber][iKStarBin][iPair][0]) || !CanInterpRStar(fPairSample4dVec[aAnalysisNumber][iKStarBin][iPair][1]) || !CanInterpTheta(fPairSample4dVec[aAnalysisNumber][iKStarBin][iPair][2]))
      {
        tTempPairVec.push_back(fPairSample4dVec[aAnalysisNumber][iKStarBin][iPair][0]);
        tTempPairVec.push_back(fPairSample4dVec[aAnalysisNumber][iKStarBin][iPair][1]);
        tTempPairVec.push_back(fPairSample4dVec[aAnalysisNumber][iKStarBin][iPair][2]);

        tTempKStarBinVec.push_back(tTempPairVec);
      }
    }
    tReturnVec.push_back(tTempKStarBinVec);
  }

  return tReturnVec;
}

//________________________________________________________________________________________________________________
double CoulombFitterParallel::GetFitCfContentParallel(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  //should probably do x[0] /= hbarc, but let me test first


  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  double tBinSize = 0.01;
  int tBin = aKStarMagMin/tBinSize;

  //KStarMag = fPairKStar4dVec[aAnalysisNumber][tBin][i][0]
  //KStarOut = fPairKStar4dVec[aAnalysisNumber][tBin][i][1]
  //KStarSide = fPairKStar4dVec[aAnalysisNumber][tBin][i][2]
  //KStarLong = fPairKStar4dVec[aAnalysisNumber][tBin][i][3]

  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tCounterGPU = 0;
  double tCfContentGPU = 0.;

  int tCounterCPU = 0;
  double tCfContentCPU = 0.;

//  int tMaxKStarCalls = 10000;
  int tMaxKStarCalls = 16384; //==2^14

ChronoTimer InitTimer;
InitTimer.Start();

  //Create the source Gaussians
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,par[1]);
  std::normal_distribution<double> tRSideSource(0.,par[1]);
  std::normal_distribution<double> tRLongSource(0.,par[1]);

  std::uniform_int_distribution<int> tRandomKStarElement(0.0, fPairKStar4dVec[aAnalysisNumber][tBin].size()-1);

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);

  int tI;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSq;
  complex<double> tWaveFunction;

  vector<vector<double> > tPairsGPU;
  vector<vector<double> > tPairsCPU;
  vector<double> tTempPair(3);

InitTimer.Stop();
cout << "InitTimer: ";
InitTimer.PrintInterval();

  int tNGood=0;


ChronoTimer PairGenTimer;
PairGenTimer.Start();
  while(tNGood<tMaxKStarCalls)
//  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tI = tRandomKStarElement(generator);
    tKStar3Vec->SetXYZ(fPairKStar4dVec[aAnalysisNumber][tBin][tI][1],fPairKStar4dVec[aAnalysisNumber][tBin][tI][2],fPairKStar4dVec[aAnalysisNumber][tBin][tI][3]);
      tKStarMag = tKStar3Vec->Mag();
    tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric
      tRStarMag = tSource3Vec->Mag();
    tTheta = tKStar3Vec->Angle(*tSource3Vec);

//    tTempPair.clear();  //TODO NEVER CLEAR A VECTOR IF YOU WANT IT TO MAINTAIN ITS SIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      tTempPair[0] = tKStarMag;
      tTempPair[1] = tRStarMag;
      tTempPair[2] = tTheta;


    if(CanInterpAll(tKStarMag,tRStarMag,tTheta))
    {
      tPairsGPU.push_back(tTempPair);
      tNGood++;
    }
    else tPairsCPU.push_back(tTempPair);
  }
  delete tKStar3Vec;
  delete tSource3Vec;
PairGenTimer.Stop();
cout << "PairGenTimer: ";
PairGenTimer.PrintInterval();


ChronoTimer SerialTimer;
SerialTimer.Start();
  //---------Do serial calculations------------
  for(int i=0; i<(int)tPairsCPU.size(); i++)
  {
    tWaveFunction = fWaveFunction->GetWaveFunction(tPairsCPU[i][0],tPairsCPU[i][1],tPairsCPU[i][2],par[2],par[3],par[4]);
    tWaveFunctionSq = norm(tWaveFunction);

    tCfContentCPU += tWaveFunctionSq;
    tCounterCPU++;
  }
SerialTimer.Stop();
cout << "SerialTimer: ";
SerialTimer.PrintInterval();


ChronoTimer ParallelTimer;
ParallelTimer.Start();
  //--------Do parallel calculations!----------
  vector<double> tGPUResults = fParallelWaveFunction->RunInterpolateWfSquared(tPairsGPU,par[2],par[3],par[4]);

  for(int i=0; i<(int)tGPUResults.size(); i++)
  {
    tCfContentGPU += tGPUResults[i];
    tCounterGPU++;
  }
ParallelTimer.Stop();
cout << "ParallelTimer: ";
ParallelTimer.PrintInterval();
cout << endl;

  tReturnCfContent = (tCfContentCPU + tCfContentGPU)/(tCounterCPU+tCounterGPU);
//  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));

  return tReturnCfContent;
}



//________________________________________________________________________________________________________________
td4dVec CoulombFitterParallel::Get3dPairs(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber)
{
//  double tBinSize = (aKStarMagMax-aKStarMagMin)/aNbinsK;

  td3dVec tPairsGPU(aNbinsK);
  td3dVec tPairsCPU(aNbinsK);

  int tMaxKStarCalls = 16384; //==2^14

  //Create the source Gaussians
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,par[1]);
  std::normal_distribution<double> tRSideSource(0.,par[1]);
  std::normal_distribution<double> tRLongSource(0.,par[1]);

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);

  double tBinSize = (aKStarMagMax-aKStarMagMin)/aNbinsK;
  if(tBinSize != fPairKStar3dVecInfo.binWidthK) cout << "Imminent CRASH in CoulombFitterParallel::Get3dPairs due to enequal bin sizes!!!!!!!!" << endl;
  assert(tBinSize == fPairKStar3dVecInfo.binWidthK);  //Life is much easier when these are equal
                                                      //TODO in future, maybe make method to allow for unequal bin sizes

  int tI;
  double tTheta, tKStarMag, tRStarMag;
  vector<double> tTempPair(3);

  int tNGood = 0;

  #pragma omp parallel for private(tI,tKStarMag,tRStarMag,tTheta,tNGood) firstprivate(tKStar3Vec,tSource3Vec,tTempPair,generator,tROutSource,tRSideSource,tRLongSource)
  for(int iBin=0; iBin<aNbinsK; iBin++)
  {
    tNGood=0;
    std::uniform_int_distribution<int> tRandomKStarElement(0.0, fPairKStar4dVec[aAnalysisNumber][iBin].size()-1);  //TODO could possible be drawing from multiple
                                                                                                  //iBins, depending on size
    while(tNGood<tMaxKStarCalls)
    {
      tI = tRandomKStarElement(generator);
      tKStar3Vec->SetXYZ(fPairKStar4dVec[aAnalysisNumber][iBin][tI][1],fPairKStar4dVec[aAnalysisNumber][iBin][tI][2],fPairKStar4dVec[aAnalysisNumber][iBin][tI][3]);
        tKStarMag = tKStar3Vec->Mag();
      tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric
        tRStarMag = tSource3Vec->Mag();
      tTheta = tKStar3Vec->Angle(*tSource3Vec);

//    tTempPair.clear();  //TODO NEVER CLEAR A VECTOR IF YOU WANT IT TO MAINTAIN ITS SIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      tTempPair[0] = tKStarMag;
      tTempPair[1] = tRStarMag;
      tTempPair[2] = tTheta;

      if(CanInterpAll(tKStarMag,tRStarMag,tTheta))
      {
        tNGood++;
        tPairsGPU[iBin].push_back(tTempPair);
      }
      else tPairsCPU[iBin].push_back(tTempPair);
    }
  }
  delete tKStar3Vec;
  delete tSource3Vec;

  td4dVec tReturnVec(2);
    tReturnVec[0] = tPairsGPU;
    tReturnVec[1] = tPairsCPU;

  return tReturnVec;  
}



//________________________________________________________________________________________________________________
td1dVec CoulombFitterParallel::GetEntireFitCfContent(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber)
{

//ChronoTimer t4dVecTimer;
//t4dVecTimer.Start();

  td4dVec tPairs4d = Get3dPairs(aKStarMagMin,aKStarMagMax,aNbinsK,par,aAnalysisNumber);

//t4dVecTimer.Stop();
//cout << "t4dVecTimer: ";
//t4dVecTimer.PrintInterval();


//ChronoTimer CfParallelTimer;
//CfParallelTimer.Start();

  td3dVec tPairsGPU = tPairs4d[0];
  td3dVec tPairsCPU = tPairs4d[1];

  //--------Do parallel calculations!----------
  td1dVec tResultsGPU = fParallelWaveFunction->RunInterpolateEntireCf(tPairsGPU,par[2],par[3],par[4]);
  int tNPairsPerBinGPU = 16384;  //TODO make this autoatically set itself to correct value

//CfParallelTimer.Stop();
//cout << "CfParallelTimer in GetEntireFitCfContent: ";
//CfParallelTimer.PrintInterval();


//ChronoTimer CfSerialTimer;
//CfSerialTimer.Start();

  td1dVec tResultsCPU(tPairsCPU.size());
  complex<double> tWaveFunction;
  double tWaveFunctionSq;
  double tCPUContent;
  //---------Do serial calculations------------
  for(int i=0; i<(int)tPairsCPU.size(); i++)
  {
    tCPUContent = 0.0;
    for(int j=0; j<(int)tPairsCPU[i].size(); j++)
    {
      tWaveFunction = fWaveFunction->GetWaveFunction(tPairsCPU[i][j][0],tPairsCPU[i][j][1],tPairsCPU[i][j][2],par[2],par[3],par[4]);
      tWaveFunctionSq = norm(tWaveFunction);

      tCPUContent += tWaveFunctionSq;
    }
    tResultsCPU[i] = tCPUContent;
  }

//CfSerialTimer.Stop();
//cout << "CfSerialTimer in GetEntireFitCfContent: ";
//CfSerialTimer.PrintInterval();

//ChronoTimer CfCombineTimer;
//CfCombineTimer.Start();

  //------ Combine parallel and serial calculations -------
  assert(tResultsGPU.size() == tResultsCPU.size());
  td1dVec tReturnVec(tResultsGPU.size());
  for(int i=0; i<(int)tResultsGPU.size(); i++)
  {
    tReturnVec[i] = (tResultsGPU[i] + tResultsCPU[i])/(tNPairsPerBinGPU + tPairsCPU[i].size());
//    tReturnVec[i] = par[5]*(par[0]*tReturnVec[i] + (1.0-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
    tReturnVec[i] = (par[0]*tReturnVec[i] + (1.0-par[0]));  //C = (Lam*C_gen + (1-Lam));
  }

//CfCombineTimer.Stop();
//cout << "CfCombineTimer: ";
//CfCombineTimer.PrintInterval();

  return tReturnVec;

}


//________________________________________________________________________________________________________________
td1dVec CoulombFitterParallel::GetEntireFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber)
{
  //par[0] = kLambda
  //par[1] = kRadius
  //par[2] = kRef0
  //par[3] = kImf0
  //par[4] = kd0
  //par[5] = kRef02
  //par[6] = kImf02
  //par[7] = kd02
  //par[8] = kNorm



//ChronoTimer t4dVecTimer;
//t4dVecTimer.Start();

  td4dVec tPairs4d = Get3dPairs(aKStarMagMin,aKStarMagMax,aNbinsK,par,aAnalysisNumber);

//t4dVecTimer.Stop();
//cout << "t4dVecTimer: ";
//t4dVecTimer.PrintInterval();


//ChronoTimer CfParallelTimer;
//CfParallelTimer.Start();

  td3dVec tPairsGPU = tPairs4d[0];
  td3dVec tPairsCPU = tPairs4d[1];

  //--------Do parallel calculations!----------
  td1dVec tResultsGPU = fParallelWaveFunction->RunInterpolateEntireCfComplete(tPairsGPU,par[2],par[3],par[4],par[5],par[6],par[7]);
  int tNPairsPerBinGPU = 16384;  //TODO make this autoatically set itself to correct value

//CfParallelTimer.Stop();
//cout << "CfParallelTimer in GetEntireFitCfContent: ";
//CfParallelTimer.PrintInterval();


//ChronoTimer CfSerialTimer;
//CfSerialTimer.Start();

  td1dVec tResultsCPU(tPairsCPU.size());
  complex<double> tWaveFunctionSinglet, tWaveFunctionTriplet;
  double tWaveFunctionSq;
  double tCPUContent;
  //---------Do serial calculations------------
  for(int i=0; i<(int)tPairsCPU.size(); i++)
  {
    tCPUContent = 0.0;
    for(int j=0; j<(int)tPairsCPU[i].size(); j++)
    {
      tWaveFunctionSinglet = fWaveFunction->GetWaveFunction(tPairsCPU[i][j][0],tPairsCPU[i][j][1],tPairsCPU[i][j][2],par[2],par[3],par[4]);
      tWaveFunctionTriplet = fWaveFunction->GetWaveFunction(tPairsCPU[i][j][0],tPairsCPU[i][j][1],tPairsCPU[i][j][2],par[5],par[6],par[7]);

      tWaveFunctionSq = 0.25*norm(tWaveFunctionSinglet) + 0.75*norm(tWaveFunctionTriplet);

      tCPUContent += tWaveFunctionSq;
    }
    tResultsCPU[i] = tCPUContent;
  }

//CfSerialTimer.Stop();
//cout << "CfSerialTimer in GetEntireFitCfContent: ";
//CfSerialTimer.PrintInterval();

//ChronoTimer CfCombineTimer;
//CfCombineTimer.Start();

  //------ Combine parallel and serial calculations -------
  assert(tResultsGPU.size() == tResultsCPU.size());
  td1dVec tReturnVec(tResultsGPU.size());
  for(int i=0; i<(int)tResultsGPU.size(); i++)
  {
    tReturnVec[i] = (tResultsGPU[i] + tResultsCPU[i])/(tNPairsPerBinGPU + tPairsCPU[i].size());
//    tReturnVec[i] = par[8]*(par[0]*tReturnVec[i] + (1.0-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
    tReturnVec[i] = (par[0]*tReturnVec[i] + (1.0-par[0]));  //C = (Lam*C_gen + (1-Lam));
  }

//CfCombineTimer.Stop();
//cout << "CfCombineTimer: ";
//CfCombineTimer.PrintInterval();

  return tReturnVec;

}

//________________________________________________________________________________________________________________
td1dVec CoulombFitterParallel::GetEntireFitCfContentwStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  //par[0] = kLambda
  //par[1] = kRadius
  //par[2] = kRef0
  //par[3] = kImf0
  //par[4] = kd0
  //par[5] = kNorm


//ChronoTimer tUpdateTimer;
//tUpdateTimer.Start();

  if(abs(par[1]-fCurrentRadii[aAnalysisNumber]) > std::numeric_limits< double >::min()) UpdatePairRadiusParameter(par[1], aAnalysisNumber);

//tUpdateTimer.Stop();
//cout << "tUpdateTimer: ";
//tUpdateTimer.PrintInterval();


//ChronoTimer CfParallelTimer;
//CfParallelTimer.Start();

  //--------Do parallel calculations!----------
  td2dVec tResultsGPU = fParallelWaveFunction->RunInterpolateEntireCfwStaticPairs(aAnalysisNumber,par[1],par[2],par[3],par[4]);

//CfParallelTimer.Stop();
//cout << "CfParallelTimer in GetEntireFitCfContent: ";
//CfParallelTimer.PrintInterval();




//ChronoTimer CfSerialTimer;
//CfSerialTimer.Start();

  td3dVec tPairsCPU = GetCPUSamplePairs(aAnalysisNumber);
  td1dVec tResultsCPU(tPairsCPU.size());
  complex<double> tWaveFunction;
  double tWaveFunctionSq;
  double tCPUContent;
  //---------Do serial calculations------------
  assert(tResultsGPU.size() == tResultsCPU.size());
  for(int i=0; i<(int)tPairsCPU.size(); i++)
  {
    if( ((int)tPairsCPU[i].size() + (int)tResultsGPU[i][1]) != fNPairsPerKStarBin)
    {
      cout << "tPairsCPU[i].size() + (int)tResultsGPU[i][1]) != fNPairsPerKStarBin!!!!!!!!!" << endl;
      cout << "FAILURE IMMINENT!!!!!!!!!!!!!!!!!!" << endl;
      cout << "tPairsCPU[" << i << "].size() = " << tPairsCPU[i].size() << endl;
      cout << "tResultsGPU[" << i << "][1] = " << tResultsGPU[i][1] << endl;
      cout << "fNPairsPerKStarBin = " << fNPairsPerKStarBin << endl;
    }
    assert( ((int)tPairsCPU[i].size() + (int)tResultsGPU[i][1]) == fNPairsPerKStarBin);
    tCPUContent = 0.0;
    for(int j=0; j<(int)tPairsCPU[i].size(); j++)
    {
      tWaveFunction = fWaveFunction->GetWaveFunction(tPairsCPU[i][j][0],tPairsCPU[i][j][1],tPairsCPU[i][j][2],par[2],par[3],par[4]);
      tWaveFunctionSq = norm(tWaveFunction);
      tCPUContent += tWaveFunctionSq;
    }
    tResultsCPU[i] = tCPUContent;
  }

//CfSerialTimer.Stop();
//cout << "CfSerialTimer in GetEntireFitCfContent: ";
//CfSerialTimer.PrintInterval();

//ChronoTimer CfCombineTimer;
//CfCombineTimer.Start();

  td1dVec tReturnVec(tResultsGPU.size());
  for(int i=0; i<(int)tResultsGPU.size(); i++)
  {
    tReturnVec[i] = (tResultsGPU[i][0] + tResultsCPU[i])/(tResultsGPU[i][1] + tPairsCPU[i].size());
//    tReturnVec[i] = par[8]*(par[0]*tReturnVec[i] + (1.0-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
    tReturnVec[i] = (par[0]*tReturnVec[i] + (1.0-par[0]));  //C = (Lam*C_gen + (1-Lam));
  }

  return tReturnVec;

}


//________________________________________________________________________________________________________________
td1dVec CoulombFitterParallel::GetEntireFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  //par[0] = kLambda
  //par[1] = kRadius
  //par[2] = kRef0
  //par[3] = kImf0
  //par[4] = kd0
  //par[5] = kRef02
  //par[6] = kImf02
  //par[7] = kd02
  //par[8] = kNorm


//ChronoTimer tUpdateTimer;
//tUpdateTimer.Start();

  if(abs(par[1]-fCurrentRadii[aAnalysisNumber]) > std::numeric_limits< double >::min()) UpdatePairRadiusParameter(par[1], aAnalysisNumber);

//tUpdateTimer.Stop();
//cout << "tUpdateTimer: ";
//tUpdateTimer.PrintInterval();


//ChronoTimer CfParallelTimer;
//CfParallelTimer.Start();

  //--------Do parallel calculations!----------
  td2dVec tResultsGPU = fParallelWaveFunction->RunInterpolateEntireCfCompletewStaticPairs(aAnalysisNumber,par[1],par[2],par[3],par[4],par[5],par[6],par[7]);

//CfParallelTimer.Stop();
//cout << "CfParallelTimer in GetEntireFitCfContent: ";
//CfParallelTimer.PrintInterval();




//ChronoTimer CfSerialTimer;
//CfSerialTimer.Start();

  td3dVec tPairsCPU = GetCPUSamplePairs(aAnalysisNumber);
  td1dVec tResultsCPU(tPairsCPU.size());
  complex<double> tWaveFunctionSinglet, tWaveFunctionTriplet;
  double tWaveFunctionSq;
  double tCPUContent;
  //---------Do serial calculations------------
  assert(tResultsGPU.size() == tResultsCPU.size());
  for(int i=0; i<(int)tPairsCPU.size(); i++)
  {
    if( ((int)tPairsCPU[i].size() + (int)tResultsGPU[i][1]) != fNPairsPerKStarBin)
    {
      cout << "tPairsCPU[i].size() + (int)tResultsGPU[i][1]) != fNPairsPerKStarBin!!!!!!!!!" << endl;
      cout << "FAILURE IMMINENT!!!!!!!!!!!!!!!!!!" << endl;
      cout << "tPairsCPU[" << i << "].size() = " << tPairsCPU[i].size() << endl;
      cout << "tResultsGPU[" << i << "][1] = " << tResultsGPU[i][1] << endl;
      cout << "fNPairsPerKStarBin = " << fNPairsPerKStarBin << endl;
    }
    assert( ((int)tPairsCPU[i].size() + (int)tResultsGPU[i][1]) == fNPairsPerKStarBin);
    tCPUContent = 0.0;
    for(int j=0; j<(int)tPairsCPU[i].size(); j++)
    {
      tWaveFunctionSinglet = fWaveFunction->GetWaveFunction(tPairsCPU[i][j][0],tPairsCPU[i][j][1],tPairsCPU[i][j][2],par[2],par[3],par[4]);
      tWaveFunctionTriplet = fWaveFunction->GetWaveFunction(tPairsCPU[i][j][0],tPairsCPU[i][j][1],tPairsCPU[i][j][2],par[5],par[6],par[7]);

      tWaveFunctionSq = 0.25*norm(tWaveFunctionSinglet) + 0.75*norm(tWaveFunctionTriplet);

      tCPUContent += tWaveFunctionSq;
    }
    tResultsCPU[i] = tCPUContent;
  }

//CfSerialTimer.Stop();
//cout << "CfSerialTimer in GetEntireFitCfContent: ";
//CfSerialTimer.PrintInterval();

//ChronoTimer CfCombineTimer;
//CfCombineTimer.Start();

  td1dVec tReturnVec(tResultsGPU.size());
  for(int i=0; i<(int)tResultsGPU.size(); i++)
  {
    tReturnVec[i] = (tResultsGPU[i][0] + tResultsCPU[i])/(tResultsGPU[i][1] + tPairsCPU[i].size());
//    tReturnVec[i] = par[8]*(par[0]*tReturnVec[i] + (1.0-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
    tReturnVec[i] = (par[0]*tReturnVec[i] + (1.0-par[0]));  //C = (Lam*C_gen + (1-Lam));
  }

  return tReturnVec;

}



//________________________________________________________________________________________________________________
td1dVec CoulombFitterParallel::GetEntireFitCfContentComplete2(int aNSimPairsPerBin, double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par)
{
//ChronoTimer CfParallelTimer;
//CfParallelTimer.Start();

//  int tNPairsPerBinGPU = 16384;  //TODO make this autoatically set itself to correct value
  td1dVec tResultsGPU = fParallelWaveFunction->RunInterpolateEntireCfComplete2(aNSimPairsPerBin,aKStarMagMin,aKStarMagMax,aNbinsK,par[1],par[2],par[3],par[4],par[5],par[6],par[7]);

//CfParallelTimer.Stop();
//cout << "CfParallelTimer: ";
//CfParallelTimer.PrintInterval();

  for(int i=0; i<(int)tResultsGPU.size(); i++)
  {
    tResultsGPU[i] = tResultsGPU[i]/aNSimPairsPerBin;
//    tResultsGPU[i] = par[8]*(par[0]*tResultsGPU[i] + (1.0-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
    tResultsGPU[i] = (par[0]*tResultsGPU[i] + (1.0-par[0]));  //C = (Lam*C_gen + (1-Lam));
  }

  return tResultsGPU;

}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::CalculateFitFunction(int &npar, double &chi2, double *par)
{
ChronoTimer tTotalTimer;
tTotalTimer.Start();

  if(fVerbose)
  {
    cout << "\tfNCalls = " << fNCalls << endl;
    PrintCurrentParamValues(fFitSharedAnalyses->GetNMinuitParams(),par);
  }
  fNCalls++;

  //--------------------------------------------------------------
  if(!fUseStaticPairs) BuildPairSample4dVec(fNPairsPerKStarBin, fKStarBinWidth);
  //--------------------------------------------------------------

  int tNFitParPerAnalysis;
  if(fIncludeSingletAndTriplet) tNFitParPerAnalysis = 8;
  else tNFitParPerAnalysis = 5;

  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  td1dVec tPrimaryFitCfContent;

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}


  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();
    SetCoulombAttributes(tAnalysisType);  //TODO probably put this in Initialize

    TH2* tMomResMatrix = NULL;
    if(fApplyMomResCorrection)
    {
      tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
      assert(tMomResMatrix);
    }

/*
    if(!fAllOfSameCoulombType)
    {
      assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
      if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
      else fBohrRadius = gBohrRadiusXiK; //repulsive
      fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
    }
*/
    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
//cout << "iPartAn = " << iPartAn << endl;
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();
      TH1* tCf = tKStarCfLite->Cf();

      TAxis* tXaxisNum = tNum->GetXaxis();
      TAxis* tXaxisDen = tDen->GetXaxis();

      int tNFitParams = tFitPartialAnalysis->GetNFitParams() +1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams = tNFitParPerAnalysis+1);

      int tLambdaMinuitParamNumber, tRadiusMinuitParamNumber, tRef0MinuitParamNumber, tImf0MinuitParamNumber, td0MinuitParamNumber, tNormMinuitParamNumber;
      int tRef02MinuitParamNumber, tImf02MinuitParamNumber, td02MinuitParamNumber;

      tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      double *tParPrim = new double[tNFitParams];
      if(fIncludeResidualsType != kIncludeNoResiduals) tParPrim[0] = cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][tFitPairAnalysis->GetAnalysisType()]*par[tLambdaMinuitParamNumber];
      else tParPrim[0] = par[tLambdaMinuitParamNumber];
      if(!fIncludeSingletAndTriplet)
      {
        assert(tNFitParams == 6);

        tParPrim[1] = par[tRadiusMinuitParamNumber];
        tParPrim[2] = par[tRef0MinuitParamNumber];
        tParPrim[3] = par[tImf0MinuitParamNumber];
        tParPrim[4] = par[td0MinuitParamNumber];
        tParPrim[5] = par[tNormMinuitParamNumber];
      }

      else
      {
        assert(tNFitParams == 9);

        tRef02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef02)->GetMinuitParamNumber();
        tImf02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf02)->GetMinuitParamNumber();
        td02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd02)->GetMinuitParamNumber();

        tParPrim[1] = par[tRadiusMinuitParamNumber];
        tParPrim[2] = par[tRef0MinuitParamNumber];
        tParPrim[3] = par[tImf0MinuitParamNumber];
        tParPrim[4] = par[td0MinuitParamNumber];
        tParPrim[5] = par[tRef02MinuitParamNumber];
        tParPrim[6] = par[tImf02MinuitParamNumber];
        tParPrim[7] = par[td02MinuitParamNumber];
        tParPrim[8] = par[tNormMinuitParamNumber];
      }


      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tParPrim[i])) {cout <<"CRASH:  In CalculateFitFunction, a tParPrim elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tParPrim[i]));
      }


      vector<double> tFitCfContent;
      vector<double> tCorrectedFitCfContent;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tParPrim,tNFitParPerAnalysis);
      if(!tAreParamsSame) 
      {
        if(fUseStaticPairs && fIncludeSingletAndTriplet) tPrimaryFitCfContent = GetEntireFitCfContentCompletewStaticPairs(0.,fMaxFitKStar,tParPrim,iAnaly);
        else if(fIncludeSingletAndTriplet) tPrimaryFitCfContent = GetEntireFitCfContentComplete(0.,fMaxFitKStar,fNbinsXToFit,tParPrim,iAnaly);  //TODO include fMinFitKStar
        else if(fUseStaticPairs) tPrimaryFitCfContent = GetEntireFitCfContentwStaticPairs(0.,fMaxFitKStar,tParPrim,iAnaly);
        else tPrimaryFitCfContent = GetEntireFitCfContent(0.,fMaxFitKStar,fNbinsXToFit,tParPrim,iAnaly);  //TODO include fMinFitKStar
      }

//      int tNPairsPerBin = 16384;
//      if(!tAreParamsSame) tPrimaryFitCfContent = GetEntireFitCfContentComplete2(tNPairsPerBin,0.,fMaxFitKStar,fNbinsXToFit,tParPrim);

      if(fIncludeResidualsType != kIncludeNoResiduals) 
      {
/*
//TODO
        double *tParOverall = new double[tNFitParams];
        tParOverall[0] = par[tLambdaMinuitParamNumber];
        tParOverall[1] = par[tRadiusMinuitParamNumber];
        tParOverall[2] = par[tRef0MinuitParamNumber];
        tParOverall[3] = par[tImf0MinuitParamNumber];
        tParOverall[4] = par[td0MinuitParamNumber];
        tParOverall[5] = par[tNormMinuitParamNumber];
        tFitCfContent = GetFitCfIncludingResiduals(tFitPairAnalysis, tPrimaryFitCfContent, tParOverall);
        delete[] tParOverall;
*/
      }
      else tFitCfContent = tPrimaryFitCfContent;


      if(fApplyMomResCorrection) tCorrectedFitCfContent = ApplyMomResCorrection(tFitCfContent, fKStarBinCenters, tMomResMatrix);
      else tCorrectedFitCfContent = tFitCfContent;

      bool tNormalizeBgdFitToCf=false;
      if(fApplyNonFlatBackgroundCorrection)
      {
        TF1* tNonFlatBgd;
        //I thought using PairAnalysis, when BgdFitType != kLinear, would help stabilize things, but it doesn't seem to help all that much.
        //  Things have been stabilized with other tweaks.
        tNonFlatBgd = tFitPartialAnalysis->GetNonFlatBackground(fNonFlatBgdFitType, fFitSharedAnalyses->GetFitType(), tNormalizeBgdFitToCf);
        ApplyNonFlatBackgroundCorrection(tCorrectedFitCfContent, fKStarBinCenters, tNonFlatBgd);
      }

      fCorrectedFitVecs[iAnaly][iPartAn] = tCorrectedFitCfContent;
      if(!fIncludeSingletAndTriplet) ApplyNormalization(tParPrim[5], tCorrectedFitCfContent);
      else ApplyNormalization(tParPrim[8], tCorrectedFitCfContent);

      if(fApplyNonFlatBackgroundCorrection && fFitSharedAnalyses->GetFitType()==kChi2PML && !tNormalizeBgdFitToCf)
      {
        //In this case, ApplyNonFlatBackgroundCorrection essentially takes care of the normalization, since it fits raw Num and Den
        // ApplyNormalization applies a normalization that is very close to 1.  Therefore, for the plots in fCorrectedFitVecs to look pretty,
        // I must scale them back up to around unity...also need to apply normalization = tParPrim[5]
        //TODO make this more general (to be sure, inclusion of tParPrim here is almost certainly correct)
        //  Would simply putting ApplyNormalization(tParPrim[5], tCorrectedFitCfContent); before fCorrectedFitVecs[iAnaly][iPartAn] = tCorrectedFitCfContent; 
        //  solve this FOR ALL CASES?
        if(!fIncludeSingletAndTriplet) ApplyNormalization(tParPrim[5]*(tKStarCfLite->GetDenScale()/tKStarCfLite->GetNumScale()), fCorrectedFitVecs[iAnaly][iPartAn]);
        else ApplyNormalization(tParPrim[8]*(tKStarCfLite->GetDenScale()/tKStarCfLite->GetNumScale()), fCorrectedFitVecs[iAnaly][iPartAn]);
      }

      double tNumContent=0., tDenContent=0.;
      for(int ix=0; ix < fNbinsXToFit; ix++)
      {
      //TODO check to make sure correct tCf bin and tNumContent tDenContent bins are being compared

        tNumContent = tNum->GetBinContent(ix+1);
        tDenContent = tDen->GetBinContent(ix+1);
//TODO tNumContent and tDenContent are zero sometimes for XiKch 3050!
/*
if(tNumContent==0 || tDenContent==0 || tCorrectedFitCfContent[ix]==0)
{
cout << "tNumContent = " << tNumContent << endl;
cout << "tDenContent = " << tDenContent << endl;
cout << "tCorrectedFitCfContent[" << ix << "] = " << tCorrectedFitCfContent[ix] << endl;
}
assert(tNumContent!=0 && tDenContent!=0 && tCorrectedFitCfContent[ix]!=0);
*/
        if(tNumContent!=0 && tDenContent!=0 && tCorrectedFitCfContent[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          double tChi2 = 0.;
          if(fFitSharedAnalyses->GetFitType() == kChi2PML) tChi2 = GetPmlValue(tNumContent,tDenContent,tCorrectedFitCfContent[ix]);
          else if(fFitSharedAnalyses->GetFitType() == kChi2) tChi2 = GetChi2Value(ix+1,tCf,tCorrectedFitCfContent[ix]);
          else tChi2 = 0.;

          fChi2Vec[iAnaly] += tChi2;
          fChi2 += tChi2;

          fNpFitsVec[iAnaly]++;
          fNpFits++;
        }
      }

      delete[] tParPrim;
    }
  }
    delete[] tCurrentFitPar;

  chi2 = fChi2;
  if(fChi2 < fChi2GlobalMin) fChi2GlobalMin = fChi2;

  tTotalTimer.Stop();
  if(fVerbose)
  {
    cout << "tTotalTimer: ";
    tTotalTimer.PrintInterval();
    cout << endl;

    cout << "fChi2 = " << fChi2 << endl;
    cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl;

    cout << "____________________________________________" << endl;
  }
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::CalculateFitFunctionFake(int &npar, double &chi2, double *par)
{

  //--------------------------------------------------------------
  if(!fUseStaticPairs) BuildPairSample4dVec(fNPairsPerKStarBin, fKStarBinWidth);
  //--------------------------------------------------------------

  TAxis *tXaxis = fFakeCf->GetXaxis();
  double tChi2 = 0.;
  double tmp = 0.;
  int tNpfits = 0;

  double tKStarMin, tKStarMax;

  int tNbinsK = fFakeCf->GetNbinsX();
  int tAnalysisNumber = 0;  //TODO

  for(int ix=1; ix<=tNbinsK; ix++)
  {
    tKStarMin = tXaxis->GetBinLowEdge(ix);
    tKStarMax = tXaxis->GetBinLowEdge(ix+1);

    tmp = (fFakeCf->GetBinContent(ix) - par[5]*GetFitCfContentParallel(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    tChi2 += tmp*tmp;
    tNpfits++;
  }

  chi2 = tChi2;
cout << "tChi2 = " << tChi2 << endl << endl;
}


//________________________________________________________________________________________________________________
TH1* CoulombFitterParallel::CreateFitHistogramParallel(TString aName, int aAnalysisNumber)
{
  if(!fUseStaticPairs) BuildPairSample4dVec(fNPairsPerKStarBin, fKStarBinWidth);
  //--------------------------------------------------------------
  FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();
/*
  if(!fAllOfSameCoulombType)
  {
    assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
    if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else fBohrRadius = gBohrRadiusXiK; //repulsive
    fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
  }
*/
  int tNFitParams = tFitPairAnalysis->GetNFitParams(); //should be equal to 5

  TH1* tKStarCf = tFitPairAnalysis->GetKStarCf();
//  int tNbins = tKStarCf->GetNbinsX();
  int tNbins = tKStarCf->FindBin(fMaxFitKStar);
  double tBinKStarMin = tKStarCf->GetBinLowEdge(1);
  double tBinKStarMax = tKStarCf->GetBinLowEdge(tNbins+1);

  TH1D* tReturnHist = new TH1D(aName,aName,tNbins,tBinKStarMin,tBinKStarMax);

  double *tPar = new double[tNFitParams];
  double *tParError = new double[tNFitParams];

  for(int iPar=0; iPar<tNFitParams; iPar++)
  {
    ParameterType tParamType = static_cast<ParameterType>(iPar);
    tPar[iPar] = tFitPairAnalysis->GetFitParameter(tParamType)->GetFitValue();
    tParError[iPar] = tFitPairAnalysis->GetFitParameter(tParamType)->GetFitValueError();
  }

  double tKStarMin, tKStarMax, tCfContent;
  for(int ix=1; ix <= tNbins; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);

    tCfContent = tPar[5]*GetFitCfContentParallel(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
    tReturnHist->SetBinContent(ix,tCfContent);
  }

  delete[] tPar;
  delete[] tParError;

  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* CoulombFitterParallel::CreateFitHistogramSampleParallel(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm)
{
  cout << "Beginning CreateFitHistogramSampleParallel" << endl;
ChronoTimer tTimer;
tTimer.Start();

  //--------------------------------------------------------------
  InitializeFitter();
  if(!fUseStaticPairs) BuildPairSample4dVec(fNPairsPerKStarBin, fKStarBinWidth);
  //--------------------------------------------------------------
/*
  if(!fAllOfSameCoulombType)
  {
    assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
    if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else fBohrRadius = gBohrRadiusXiK; //repulsive
    fWaveFunction->SetCurrentAnalysisType(aAnalysisType);
  }
*/
  TH1D* tReturnHist = new TH1D(aName,aName,aNbinsK,aKMin,aKMax);

  double *tPar = new double[6];
  tPar[0] = aLambda;
  tPar[1] = aR;
  tPar[2] = aReF0;
  tPar[3] = aImF0;
  tPar[4] = aD0;
  tPar[5] = aNorm;

  int tAnalysisNumber = 0;  //TODO

  double tBinSize = (aKMax-aKMin)/aNbinsK;
ChronoTimer CfParallelTimer;
CfParallelTimer.Start();

//  int tNPairsPerBin = 16384;
//  td1dVec tCf = GetEntireFitCfContentComplete2(tNPairsPerBin,aKMin,aKMax,aNbinsK,tPar);
  td1dVec tCf = GetEntireFitCfContent(aKMin,aKMax,aNbinsK,tPar,tAnalysisNumber);

CfParallelTimer.Stop();
cout << "CfParallelTimer in CreateFitHistogramSampleParallel: ";
CfParallelTimer.PrintInterval();

 for(int i=0; i<aNbinsK; i++) tReturnHist->SetBinContent(i+1,tCf[i]);

/*
  double tKStarMin, tKStarMax;
  td1dVec tCf2(aNbinsK);
ChronoTimer CfSerialTimer;
CfSerialTimer.Start();
  for(int ix=1; ix <= aNbinsK; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);

    tCf2[ix-1] = GetFitCfContentSerialv2(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
  }
CfSerialTimer.Stop();
cout << "CfSerialTimer: ";
CfSerialTimer.PrintInterval();

  for(int i=0; i<aNbinsK; i++)
  {
    cout << "i = " << i << endl;
    cout << "tCfContent = " << tCf[i] << endl;
    cout << "tCfContentv2 = " << tCf2[i] << endl << endl;
  }

*/
tTimer.Stop();
  cout << "Finished CreateFitHistogramSampleParallel  ";
tTimer.PrintInterval();

  delete[] tPar;

  fFakeCf = tReturnHist;  //TODO delete this
  return tReturnHist;
}



//________________________________________________________________________________________________________________
TH1* CoulombFitterParallel::CreateFitHistogramSampleCompleteParallel(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double aNorm)
{
  cout << "Beginning CreateFitHistogramSampleCompleteParallel" << endl;
ChronoTimer tTimer;
tTimer.Start();

  //--------------------------------------------------------------
  InitializeFitter();
  if(!fUseStaticPairs) BuildPairSample4dVec(fNPairsPerKStarBin, fKStarBinWidth);
  //--------------------------------------------------------------

  SetCoulombAttributes(aAnalysisType);

/*
  if(!fAllOfSameCoulombType)
  {
    assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
    if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else fBohrRadius = gBohrRadiusXiK; //repulsive
    fWaveFunction->SetCurrentAnalysisType(aAnalysisType);
  }
*/
  TH1D* tReturnHist = new TH1D(aName,aName,aNbinsK,aKMin,aKMax);

  double *tPar = new double[9];
  tPar[0] = aLambda;
  tPar[1] = aR;
  tPar[2] = aReF0s;
  tPar[3] = aImF0s;
  tPar[4] = aD0s;
  tPar[5] = aReF0t;
  tPar[6] = aImF0t;
  tPar[7] = aD0t;
  tPar[8] = aNorm;

  int tAnalysisNumber = 0; //TODO

  double tBinSize = (aKMax-aKMin)/aNbinsK;
ChronoTimer CfParallelTimer;
CfParallelTimer.Start();

//  int tNPairsPerBin = 16384;
//  td1dVec tCfUnNorm = GetEntireFitCfContentComplete2(tNPairsPerBin,aKMin,aKMax,aNbinsK,tPar);
  td1dVec tCfUnNorm;
  if(fUseStaticPairs && fIncludeSingletAndTriplet) tCfUnNorm = GetEntireFitCfContentCompletewStaticPairs(aKMin,aKMax,tPar,tAnalysisNumber);
  else if(fIncludeSingletAndTriplet) tCfUnNorm = GetEntireFitCfContentComplete(aKMin,aKMax,aNbinsK,tPar,tAnalysisNumber);
  else if(fUseStaticPairs) tCfUnNorm = GetEntireFitCfContentwStaticPairs(aKMin,aKMax,tPar,tAnalysisNumber);
  else tCfUnNorm = GetEntireFitCfContent(aKMin,aKMax,aNbinsK,tPar,tAnalysisNumber);

CfParallelTimer.Stop();
cout << "CfParallelTimer in CreateFitHistogramSampleCompleteParallel: ";
CfParallelTimer.PrintInterval();

 for(int i=0; i<aNbinsK; i++) tReturnHist->SetBinContent(i+1,tPar[8]*tCfUnNorm[i]);

tTimer.Stop();
  cout << "Finished CreateFitHistogramSampleCompleteParallel  ";
tTimer.PrintInterval();

  delete[] tPar;

  fFakeCf = tReturnHist;  //TODO delete this

  return tReturnHist;
}





//________________________________________________________________________________________________________________
void CoulombFitterParallel::InitializeFitter()
{
  CoulombFitter::InitializeFitter();
  //------------------------------

  if(!fUseRandomKStarVectors)  //Using KStarVectors from data, so must build fPairKStar4dVec
  {
    if(fReadPairsFromTxtFiles) 
    {
      td1dVec tBinInfo = BuildPairKStar4dVecFromTxt(fPairKStar3dVecBaseName);
      assert(fNbinsXToBuild <= tBinInfo[0]);
      assert(0.0 == tBinInfo[1]);
      assert(fMaxBuildKStar <= tBinInfo[2]);
    }
    else BuildPairKStar4dVecOnFly(fPairKStarNtupleBaseName, fNFilesNtuple, fNbinsXToBuild, 0.0, fMaxBuildKStar);
  }
  
  BuildPairSample4dVec(fNPairsPerKStarBin, fKStarBinWidth);



}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::DoFit()
{
  InitializeFitter();

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

//  arglist[0] = 0.0000000001;
//  fMinuit->mnexcm("SET EPS", arglist, 1, fErrFlg);

  // Set strategy to be used
  // 0 = economize (many params and/or fcn takes a long time to calculate and/or not interested in precise param erros)
  // 1 = default
  // 2 = Minuit allowed to waste calls in order to ensure precision (fcn evaluated in short time and/or param erros must be reliable)
  arglist[0] = 1;
  fMinuit->mnexcm("SET STR", arglist ,1,fErrFlg);

  // Now ready for minimization step
  arglist[0] = 5000;
  arglist[1] = 0.1;
  fMinuit->mnexcm("MIGRAD", arglist ,2,fErrFlg);  //I do not think MIGRAD will work here because depends on derivates, etc
//  fMinuit->mnexcm("MINI", arglist ,2,fErrFlg);  //MINI also uses MIGRAD, so probably won't work
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

  int tNParams = fFitSharedAnalyses->GetNMinuitParams();

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

  fFitSharedAnalyses->SetMinuitMinParams(fMinParams);
  fFitSharedAnalyses->SetMinuitParErrors(fParErrors);
  fFitSharedAnalyses->ReturnFitParametersToAnalyses();


//TODO return the fit histogram to the analyses
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    TH1* tHistTest = CreateFitHistogram("FitHis",iAnaly);
  }
/*
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->SetFit(CreateFitFunction("fit",iAnaly));
  }
*/

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    for(int iPartAn=0; iPartAn<fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetNFitPartialAnalysis(); iPartAn++)
    {
      fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetFitPartialAnalysis(iPartAn)->SetCorrectedFitVec(fCorrectedFitVecs[iAnaly][iPartAn]);
    }
  }


}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::Finalize()
{
  LednickyFitter::Finalize();
}


