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


//________________________________________________________________________________________________________________
CoulombFitterParallel::CoulombFitterParallel():
  CoulombFitter(),
  fParallelWaveFunction(0)

{
  fParallelWaveFunction = new ParallelWaveFunction(fUseScattLenHists);
}



//________________________________________________________________________________________________________________
CoulombFitterParallel::CoulombFitterParallel(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar, bool aCreateInterpVectors, bool aUseScattLenHists):
  CoulombFitter(aFitSharedAnalyses,aMaxFitKStar,aCreateInterpVectors,aUseScattLenHists),
  fParallelWaveFunction(0)

{
  fParallelWaveFunction = new ParallelWaveFunction(fUseScattLenHists);
}


//________________________________________________________________________________________________________________
CoulombFitterParallel::~CoulombFitterParallel()
{
  cout << "CoulombFitterParallel object is being deleted!!!" << endl;

  //---Clean up
  //TODO
  delete fLednickyHFunctionHist;

  fInterpHistFileLednickyHFunction->Close();
  delete fInterpHistFileLednickyHFunction;

  //TODO figure out how to deallocate and delete multi dimensional vectors
/*
  delete fHyperGeo1F1Real;
  delete fHyperGeo1F1Imag;

  delete fGTildeReal;
  delete fGTildeImag;

  delete fCoulombScatteringLengthReal;
  delete fCoulombScatteringLengthImag;
*/


  delete fCoulombScatteringLengthRealHist1;
  delete fCoulombScatteringLengthImagHist1;

  delete fCoulombScatteringLengthRealHist2;
  delete fCoulombScatteringLengthImagHist2;

  if(fUseScattLenHists)
  {
    fInterpHistFileScatLenReal1->Close();
    delete fInterpHistFileScatLenReal1;

    fInterpHistFileScatLenImag1->Close();
    delete fInterpHistFileScatLenImag1;

    fInterpHistFileScatLenReal2->Close();
    delete fInterpHistFileScatLenReal2;

    fInterpHistFileScatLenImag2->Close();
    delete fInterpHistFileScatLenImag2;
  }
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::LoadInterpHistFile(TString aFileBaseName)
{
  CoulombFitter::LoadInterpHistFile(aFileBaseName);

//--------------------------------------------------------------
//Send things over to ParallWaveFunction object
//TODO after sending these vectors over, they can probably be deleted
//maybe I should not even make them class members, but only members of this function

  assert(fCreateInterpVectors);

  fParallelWaveFunction->LoadLednickyHFunction(fLednickyHFunction);

  fParallelWaveFunction->LoadGTildeReal(fGTildeReal);
  fParallelWaveFunction->LoadGTildeImag(fGTildeImag);
  fParallelWaveFunction->LoadGTildeInfo(fGTildeInfo);

  fParallelWaveFunction->LoadHyperGeo1F1Real(fHyperGeo1F1Real);
  fParallelWaveFunction->LoadHyperGeo1F1Imag(fHyperGeo1F1Imag);
  fParallelWaveFunction->LoadHyperGeo1F1Info(fHyperGeo1F1Info);

  fParallelWaveFunction->LoadScattLenInfo(fScattLenInfo);  //TODO maybe this should be within the following if statement

}

//________________________________________________________________________________________________________________
td3dVec CoulombFitterParallel::BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  td3dVec tPairKStar3dVec = CoulombFitter::BuildPairKStar3dVecFull(aPairKStarNtupleDirName,aFileBaseName,aNFiles,aAnalysisType,aCentralityType,aNbinsKStar,aKStarMin,aKStarMax);

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
void CoulombFitterParallel::BuildPairKStar4dVecFromTxt(TString aFileBaseName)
{
  fPairKStar4dVec.resize(fNAnalyses, td3dVec(0, td2dVec(0, td1dVec(0))));

  AnalysisType tAnalysisType;
  CentralityType tCentralityType;
  TString tFileName;
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    tAnalysisType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetAnalysisType();
    tCentralityType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetCentralityType();
    tFileName = aFileBaseName + TString(cAnalysisBaseTags[tAnalysisType])+TString(cCentralityTags[tCentralityType]) + TString(".txt");

    td3dVec tPairKStar3dVec = BuildPairKStar3dVecFromTxt(tFileName);
      tPairKStar3dVec.pop_back();  //strip off the binning information

    fPairKStar4dVec[iAnaly] = tPairKStar3dVec;
  }

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    cout << "iAnaly = " << iAnaly << endl;
    cout << "fPairKStar4dVec[iAnaly].size() = " << fPairKStar4dVec[iAnaly].size() << endl;
    for(int i=0; i<(int)fPairKStar4dVec[iAnaly].size(); i++)
    {
      cout << "\t i = " << i << endl;
      cout << "\t\tfPairKStar4dVec[iAnaly][i].size() = " << fPairKStar4dVec[iAnaly][i].size() << endl;
    }
  }
}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::UpdatePairRadiusParameters(double aNewRadius)
{
  double tScaleFactor = aNewRadius/fCurrentRadiusParameter;
  CoulombFitter::UpdatePairRadiusParameters(aNewRadius);
  fParallelWaveFunction->UpdatePairSampleRadii(tScaleFactor);
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::SetUseStaticPairs(bool aUseStaticPairs, int aNPairsPerKStarBin)
{
  CoulombFitter::SetUseStaticPairs(aUseStaticPairs,aNPairsPerKStarBin);
  fParallelWaveFunction->LoadPairSample4dVec(fPairSample4dVec,fSamplePairsBinInfo);
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::CreateScattLenSubs(double aReF0, double aImF0, double aD0)
{
  int tLowBinReF0 = GetInterpLowBin(kScattLen,kReF0axis,aReF0);
  int tLowBinImF0 = GetInterpLowBin(kScattLen,kImF0axis,aImF0);
  int tLowBinD0 = GetInterpLowBin(kScattLen,kD0axis,aD0);

  vector<int> tRelevantReF0Bins(2);
    tRelevantReF0Bins[0] = tLowBinReF0;
    tRelevantReF0Bins[1] = tLowBinReF0+1;

  vector<int> tRelevantImF0Bins(2);
    tRelevantImF0Bins[0] = tLowBinImF0;
    tRelevantImF0Bins[1] = tLowBinImF0+1;

  vector<int> tRelevantD0Bins(2);
    tRelevantD0Bins[0] = tLowBinD0;
    tRelevantD0Bins[1] = tLowBinD0+1;

  int tNbinsReF0 = 2;
  int tNbinsImF0 = 2;
  int tNbinsD0 = 2;
  int tNbinsK = fScattLenInfo.nBinsK;

  fCoulombScatteringLengthRealSub.clear();
  fCoulombScatteringLengthImagSub.clear();

  fCoulombScatteringLengthRealSub.resize(tNbinsReF0, td3dVec(tNbinsImF0, td2dVec(tNbinsD0, td1dVec(tNbinsK,0))));
  fCoulombScatteringLengthImagSub.resize(tNbinsReF0, td3dVec(tNbinsImF0, td2dVec(tNbinsD0, td1dVec(tNbinsK,0))));

  int iReF0, iImF0, iD0, iK;
  #pragma omp parallel for private(iReF0,iImF0,iD0,iK)
  for(iReF0=0; iReF0<tNbinsReF0; iReF0++)
  {
    for(iImF0=0; iImF0<tNbinsImF0; iImF0++)
    {
      for(iD0=0; iD0<tNbinsD0; iD0++)
      {
        for(iK=0; iK<tNbinsK; iK++)
        {
          fCoulombScatteringLengthRealSub[iReF0][iImF0][iD0][iK] = fCoulombScatteringLengthReal[tRelevantReF0Bins[iReF0]][tRelevantImF0Bins[iImF0]][tRelevantD0Bins[iD0]][iK];
          fCoulombScatteringLengthImagSub[iReF0][iImF0][iD0][iK] = fCoulombScatteringLengthImag[tRelevantReF0Bins[iReF0]][tRelevantImF0Bins[iImF0]][tRelevantD0Bins[iD0]][iK];
        }
      }
    }
  }

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


    if(CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]))
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
  vector<double> tGPUResults = fParallelWaveFunction->RunInterpolateWfSquared(aPairs,aReF0,aImF0,aD0);

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

      if(CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]))
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
void CoulombFitterParallel::CalculateChi2PMLParallel(int &npar, double &chi2, double *par)
{
ChronoTimer tTotalTimer;
tTotalTimer.Start();

  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;

  cout << "\t\tParameter update: " << endl;
  cout << "\t\t\tpar[0] = " << par[0] << endl;
  cout << "\t\t\tpar[1] = " << par[1] << endl;
  cout << "\t\t\tpar[2] = " << par[2] << endl;
  cout << "\t\t\tpar[3] = " << par[3] << endl;
  cout << "\t\t\tpar[4] = " << par[4] << endl;
  cout << "\t\t\tpar[5] = " << par[5] << endl;
  cout << "\t\t\tpar[6] = " << par[6] << endl;
  cout << "\t\t\tpar[7] = " << par[7] << endl;
  cout << "\t\t\tpar[8] = " << par[8] << endl;
  cout << "\t\t\tpar[9] = " << par[9] << endl;
  cout << "\t\t\tpar[10] = " << par[10] << endl;

  int tNFitParPerAnalysis = 8;
  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  int tNbinsXToFitGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar)-1;//TODO why -1?
  td1dVec tCfContentUnNorm;

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}


  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

    //-----
    if(fUseScattLenHists)
    {
//TODO if I am using GetEntireFitCfContentComplete, or any other "Complete", I will need to create separate
// ScattLenSubs for singlet and triplet
      double aReF0, aImF0, aD0;
//ChronoTimer tTimerSubs;
//tTimerSubs.Start();

      aReF0 = par[tFitPairAnalysis->GetFitPartialAnalysis(0)->GetFitParameter(kRef0)->GetMinuitParamNumber()];
      aImF0 = par[tFitPairAnalysis->GetFitPartialAnalysis(0)->GetFitParameter(kImf0)->GetMinuitParamNumber()];
      aD0 = par[tFitPairAnalysis->GetFitPartialAnalysis(0)->GetFitParameter(kd0)->GetMinuitParamNumber()];
      CreateScattLenSubs(aReF0,aImF0,aD0);  //TODO evaluate whether this can be placed here
                                                                    //Should be able to, because ReF0, ImF0, and D0 should not change
                                                                    //within this loop
      //TODO if any of the parameters cannot be interpolate, CreateScattLenSubs will fail!  Fix this!Z
      fParallelWaveFunction->LoadScattLenRealSub(fCoulombScatteringLengthRealSub);
      fParallelWaveFunction->LoadScattLenImagSub(fCoulombScatteringLengthImagSub);

//tTimerSubs.Stop();
//cout << "tTimerSubs: ";
//tTimerSubs.PrintInterval();
    }
    //-----

    if(!fAllOfSameCoulombType)
    {
      assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
      if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
      else fBohrRadius = gBohrRadiusXiK; //repulsive
      fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
//cout << "iPartAn = " << iPartAn << endl;
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();

      //make sure tNum and tDen have same number of bins
      assert(tNum->GetNbinsX() == tDen->GetNbinsX());

      TAxis* tXaxisNum = tNum->GetXaxis();
      TAxis* tXaxisDen = tDen->GetXaxis();

      //make sure tNum and tDen have to same bin width
      assert(tXaxisNum->GetBinWidth(1) == tXaxisDen->GetBinWidth(1));

      int tNbinsX = tNum->GetNbinsX();
      int tNbinsXToFit = tNum->FindBin(fMaxFitKStar)-1;  //TODO why -1?
      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tNbinsXToFit == tNbinsXToFitGlobal);

      int tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      int tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      int tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      int tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      int td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();

      int tRef02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef02)->GetMinuitParamNumber();
      int tImf02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf02)->GetMinuitParamNumber();
      int td02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd02)->GetMinuitParamNumber();

      int tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      int tNFitParams = tFitPartialAnalysis->GetNFitParams() + 1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams == 9);
      double *tPar = new double[tNFitParams];

      tPar[0] = par[tLambdaMinuitParamNumber];
      tPar[1] = par[tRadiusMinuitParamNumber];
      tPar[2] = par[tRef0MinuitParamNumber];
      tPar[3] = par[tImf0MinuitParamNumber];
      tPar[4] = par[td0MinuitParamNumber];
      tPar[5] = par[tRef02MinuitParamNumber];
      tPar[6] = par[tImf02MinuitParamNumber];
      tPar[7] = par[td02MinuitParamNumber];
      tPar[8] = par[tNormMinuitParamNumber];

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PMLParallel, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);
      if(!tAreParamsSame) tCfContentUnNorm = GetEntireFitCfContentComplete(0.,fMaxFitKStar,tNbinsXToFit,tPar,iAnaly);  //TODO include fMinFitKStar

//      int tNPairsPerBin = 16384;
//      if(!tAreParamsSame) tCfContentUnNorm = GetEntireFitCfContentComplete2(tNPairsPerBin,0.,fMaxFitKStar,tNbinsXToFit,tPar);
      assert((int)tCfContentUnNorm.size() == tNbinsXToFit);

      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
      //TODO check to make sure correct tCf bin and tNumContent tDenContent bins are being compared

        double tNumContent = tNum->GetBinContent(ix);
        double tDenContent = tDen->GetBinContent(ix);

        double tCfContent = tPar[8]*tCfContentUnNorm[ix-1];

        if(tNumContent!=0 && tDenContent!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          double tTerm1 = tNumContent*log(  (tCfContent*(tNumContent+tDenContent)) / (tNumContent*(tCfContent+1))  );
          double tTerm2 = tDenContent*log(  (tNumContent+tDenContent) / (tDenContent*(tCfContent+1))  );
          tmp = -2.0*(tTerm1+tTerm2);

          fChi2Vec[iAnaly] += tmp;
          fChi2 += tmp;

          fNpFitsVec[iAnaly]++;
          fNpFits++;
        }
      }

      delete[] tPar;
    }
    if(fUseScattLenHists)
    {
      fParallelWaveFunction->UnLoadScattLenRealSub();
      fParallelWaveFunction->UnLoadScattLenImagSub();
    }
  }
    delete[] tCurrentFitPar;

  chi2 = fChi2;
  if(fChi2 < fChi2GlobalMin) fChi2GlobalMin = fChi2;
//gObjectTable->Print();
cout << "fChi2 = " << fChi2 << endl;
cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;

tTotalTimer.Stop();
cout << "tTotalTimer: ";
tTotalTimer.PrintInterval();
}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::CalculateChi2Parallel(int &npar, double &chi2, double *par)
{
  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;

  cout << "\t\tParameter update: " << endl;
  cout << "\t\t\tpar[0] = " << par[0] << endl;
  cout << "\t\t\tpar[1] = " << par[1] << endl;
  cout << "\t\t\tpar[2] = " << par[2] << endl;
  cout << "\t\t\tpar[3] = " << par[3] << endl;
  cout << "\t\t\tpar[4] = " << par[4] << endl;
  cout << "\t\t\tpar[5] = " << par[5] << endl;
  cout << "\t\t\tpar[6] = " << par[6] << endl;
  cout << "\t\t\tpar[7] = " << par[7] << endl;
  cout << "\t\t\tpar[8] = " << par[8] << endl;
  cout << "\t\t\tpar[9] = " << par[9] << endl;
  cout << "\t\t\tpar[10] = " << par[10] << endl;

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

    //-----
    if(fUseScattLenHists)
    {
      double aReF0, aImF0, aD0;
//ChronoTimer tTimerSubs;
//tTimerSubs.Start();

      aReF0 = par[tFitPairAnalysis->GetFitPartialAnalysis(0)->GetFitParameter(kRef0)->GetMinuitParamNumber()];
      aImF0 = par[tFitPairAnalysis->GetFitPartialAnalysis(0)->GetFitParameter(kImf0)->GetMinuitParamNumber()];
      aD0 = par[tFitPairAnalysis->GetFitPartialAnalysis(0)->GetFitParameter(kd0)->GetMinuitParamNumber()];
      CreateScattLenSubs(aReF0,aImF0,aD0);  //TODO evaluate whether this can be placed here
                                                                    //Should be able to, because ReF0, ImF0, and D0 should not change
                                                                    //within this loop
      //TODO if any of the parameters cannot be interpolate, CreateScattLenSubs will fail!  Fix this!Z
      fParallelWaveFunction->LoadScattLenRealSub(fCoulombScatteringLengthRealSub);
      fParallelWaveFunction->LoadScattLenImagSub(fCoulombScatteringLengthImagSub);

//tTimerSubs.Stop();
//cout << "tTimerSubs: ";
//tTimerSubs.PrintInterval();
    }
    //-----
    if(!fAllOfSameCoulombType)
    {
      assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
      if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
      else fBohrRadius = gBohrRadiusXiK; //repulsive
      fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
//cout << "iPartAn = " << iPartAn << endl;
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();
      TH1* tCfToFit = tKStarCfLite->Cf();

      TAxis* tXaxis = tCfToFit->GetXaxis();
      int tNbinsX = tCfToFit->GetNbinsX();
      int tNbinsXToFit = tCfToFit->FindBin(fMaxFitKStar);
      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit

      int tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      int tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      int tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      int tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      int td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();

      int tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      int tNFitParams = tFitPartialAnalysis->GetNFitParams();
      assert(tNFitParams == 6);
      double *tPar = new double[tNFitParams];

      tPar[0] = par[tLambdaMinuitParamNumber];
      tPar[1] = par[tRadiusMinuitParamNumber];
      tPar[2] = par[tRef0MinuitParamNumber];
      tPar[3] = par[tImf0MinuitParamNumber];
      tPar[4] = par[td0MinuitParamNumber];
      tPar[5] = par[tNormMinuitParamNumber];

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2Parallel, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxis->GetBinLowEdge(ix);
        double tKStarMax = tXaxis->GetBinLowEdge(ix+1);

        double tCfContentUnNorm = GetFitCfContentParallel(tKStarMin,tKStarMax,tPar,iAnaly);
        double tCfContent = par[5]*tCfContentUnNorm;

        if(tCfToFit->GetBinContent(ix)!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          tmp = (tCfToFit->GetBinContent(ix) - tCfContent)/tCfToFit->GetBinError(ix);

          fChi2Vec[iAnaly] += tmp*tmp;
          fChi2 += tmp*tmp;

          fNpFitsVec[iAnaly]++;
          fNpFits++;
        }
      }

      delete[] tPar;
    }
    if(fUseScattLenHists)
    {
      fParallelWaveFunction->UnLoadScattLenRealSub();
      fParallelWaveFunction->UnLoadScattLenImagSub();
    }
  }

  chi2 = fChi2;
  if(fChi2 < fChi2GlobalMin) fChi2GlobalMin = fChi2;
//gObjectTable->Print();
cout << "fChi2 = " << fChi2 << endl;
cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;
}



//________________________________________________________________________________________________________________
void CoulombFitterParallel::CalculateFakeChi2Parallel(int &npar, double &chi2, double *par)
{

  TAxis *tXaxis = fFakeCf->GetXaxis();
  double tChi2 = 0.;
  double tmp = 0.;
  int tNpfits = 0;

  double tKStarMin, tKStarMax;

  int tNbinsK = fFakeCf->GetNbinsX();
  int tAnalysisNumber = 0;  //TODO

    //-----
  if(fUseScattLenHists)
  {
    double aReF0 = par[2];
    double aImF0 = par[3];
    double aD0 = par[4];
    CreateScattLenSubs(aReF0,aImF0,aD0);  //TODO 
    fParallelWaveFunction->LoadScattLenRealSub(fCoulombScatteringLengthRealSub);
    fParallelWaveFunction->LoadScattLenImagSub(fCoulombScatteringLengthImagSub);
  }
    //-----

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

  if(fUseScattLenHists)
  {
    fParallelWaveFunction->UnLoadScattLenRealSub();
    fParallelWaveFunction->UnLoadScattLenImagSub();
  }
}


//________________________________________________________________________________________________________________
TH1* CoulombFitterParallel::CreateFitHistogramParallel(TString aName, int aAnalysisNumber)
{
  FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

  if(!fAllOfSameCoulombType)
  {
    assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
    if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else fBohrRadius = gBohrRadiusXiK; //repulsive
    fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
  }

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

  //-----
  if(fUseScattLenHists)
  {
    double aReF0 = tFitPairAnalysis->GetFitParameter(kRef0)->GetFitValue();
    double aImF0 = tFitPairAnalysis->GetFitParameter(kImf0)->GetFitValue();
    double aD0 = tFitPairAnalysis->GetFitParameter(kd0)->GetFitValue();
    CreateScattLenSubs(aReF0,aImF0,aD0);  //TODO 
    fParallelWaveFunction->LoadScattLenRealSub(fCoulombScatteringLengthRealSub);
    fParallelWaveFunction->LoadScattLenImagSub(fCoulombScatteringLengthImagSub);
  }
  //-----


  double tKStarMin, tKStarMax, tCfContent;
  for(int ix=1; ix <= tNbins; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);

    tCfContent = par[5]*GetFitCfContentParallel(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
    tReturnHist->SetBinContent(ix,tCfContent);
  }

  delete[] tPar;
  delete[] tParError;

  if(fUseScattLenHists)
  {
    fParallelWaveFunction->UnLoadScattLenRealSub();
    fParallelWaveFunction->UnLoadScattLenImagSub();
  }

  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* CoulombFitterParallel::CreateFitHistogramSampleParallel(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm)
{
  cout << "Beginning CreateFitHistogramSampleParallel" << endl;
ChronoTimer tTimer;
tTimer.Start();

  if(!fAllOfSameCoulombType)
  {
    assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
    if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else fBohrRadius = gBohrRadiusXiK; //repulsive
    fWaveFunction->SetCurrentAnalysisType(aAnalysisType);
  }

  TH1D* tReturnHist = new TH1D(aName,aName,aNbinsK,aKMin,aKMax);

  double *tPar = new double[6];
  tPar[0] = aLambda;
  tPar[1] = aR;
  tPar[2] = aReF0;
  tPar[3] = aImF0;
  tPar[4] = aD0;
  tPar[5] = aNorm;

  int tAnalysisNumber = 0;  //TODO

  //-----
  if(fUseScattLenHists)
  {
    CreateScattLenSubs(aReF0,aImF0,aD0);  //TODO 
    fParallelWaveFunction->LoadScattLenRealSub(fCoulombScatteringLengthRealSub);
    fParallelWaveFunction->LoadScattLenImagSub(fCoulombScatteringLengthImagSub);
  }
  //-----

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

  if(fUseScattLenHists)
  {
    fParallelWaveFunction->UnLoadScattLenRealSub();
    fParallelWaveFunction->UnLoadScattLenImagSub();
  }

  return tReturnHist;
}



//________________________________________________________________________________________________________________
TH1* CoulombFitterParallel::CreateFitHistogramSampleCompleteParallel(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double aNorm)
{
  cout << "Beginning CreateFitHistogramSampleCompleteParallel" << endl;
ChronoTimer tTimer;
tTimer.Start();

  if(!fAllOfSameCoulombType)
  {
    assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
    if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else fBohrRadius = gBohrRadiusXiK; //repulsive
    fWaveFunction->SetCurrentAnalysisType(aAnalysisType);
  }

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

  assert(!fUseScattLenHists);

  double tBinSize = (aKMax-aKMin)/aNbinsK;
ChronoTimer CfParallelTimer;
CfParallelTimer.Start();

//  int tNPairsPerBin = 16384;
//  td1dVec tCfUnNorm = GetEntireFitCfContentComplete2(tNPairsPerBin,aKMin,aKMax,aNbinsK,tPar);
  td1dVec tCfUnNorm = GetEntireFitCfContentComplete(aKMin,aKMax,aNbinsK,tPar,tAnalysisNumber);

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


