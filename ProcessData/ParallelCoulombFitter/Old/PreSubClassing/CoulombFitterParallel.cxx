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
  fUseScattLenHists(false),

  fNCalls(0),
  fFakeCf(0),

  fFitSharedAnalyses(0),
  fMinuit(0),
  fNAnalyses(0),
  fAllOfSameCoulombType(false),
  fCoulombType(kRepulsive),
  fWaveFunction(0),
  fParallelWaveFunction(0),
  fBohrRadius(gBohrRadiusXiK),

  fPairKStar3dVec(0),
  fPairKStar4dVec(0),

  fLednickyHFunction(0),

  fGTildeReal(0),
  fGTildeImag(0),

  fHyperGeo1F1Real(0),
  fHyperGeo1F1Imag(0),

  fCoulombScatteringLengthReal(0),
  fCoulombScatteringLengthImag(0),

  fCoulombScatteringLengthRealSub(0),
  fCoulombScatteringLengthImagSub(0),

  //TODO delete the following after testing
  fInterpHistFile(0), fInterpHistFileScatLenReal1(0), fInterpHistFileScatLenImag1(0), fInterpHistFileScatLenReal2(0), fInterpHistFileScatLenImag2(0), fInterpHistFileLednickyHFunction(0),

  fLednickyHFunctionHist(0),

  fHyperGeo1F1RealHist(0),
  fHyperGeo1F1ImagHist(0),

  fGTildeRealHist(0),
  fGTildeImagHist(0),

  fCoulombScatteringLengthRealHist1(0),
  fCoulombScatteringLengthImagHist1(0),

  fCoulombScatteringLengthRealHist2(0),
  fCoulombScatteringLengthImagHist2(0),
  //end TODO

  fMinInterpKStar1(0), fMinInterpKStar2(0), fMinInterpRStar(0), fMinInterpTheta(0), fMinInterpReF0(0), fMinInterpImF0(0), fMinInterpD0(0),
  fMaxInterpKStar1(0), fMaxInterpKStar2(0), fMaxInterpRStar(0), fMaxInterpTheta(0), fMaxInterpReF0(0), fMaxInterpImF0(0), fMaxInterpD0(0),


  fCfsToFit(0),
  fFits(0),
  fMaxFitKStar(0),
  fRejectOmega(false),
  fChi2(0),
  fChi2GlobalMin(1000000000),
  fChi2Vec(0),
  fNpFits(0),
  fNpFitsVec(0),
  fNDF(0),
  fErrFlg(0),
  fMinParams(0),
  fParErrors(0)

{
  gRandom->SetSeed();

  fWaveFunction = new WaveFunction();
  fParallelWaveFunction = new ParallelWaveFunction(fUseScattLenHists);

  omp_set_num_threads(4);

}



//________________________________________________________________________________________________________________
CoulombFitterParallel::CoulombFitterParallel(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar, bool aUseScattLenHists):

  fUseScattLenHists(aUseScattLenHists),

  fNCalls(0),
  fFakeCf(0),

  fFitSharedAnalyses(aFitSharedAnalyses),
  fMinuit(fFitSharedAnalyses->GetMinuitObject()),
  fNAnalyses(fFitSharedAnalyses->GetNFitPairAnalysis()),
  fAllOfSameCoulombType(false),
  fCoulombType(kRepulsive),
  fWaveFunction(0),
  fParallelWaveFunction(0),
  fBohrRadius(gBohrRadiusXiK),

  fPairKStar3dVec(0),
  fPairKStar4dVec(0),

  fLednickyHFunction(0),
  fGTildeReal(0),
  fGTildeImag(0),

  fHyperGeo1F1Real(0),
  fHyperGeo1F1Imag(0),

  fCoulombScatteringLengthReal(0),
  fCoulombScatteringLengthImag(0),

  fCoulombScatteringLengthRealSub(0),
  fCoulombScatteringLengthImagSub(0),

  //TODO delete the following after testing
  fInterpHistFile(0), fInterpHistFileScatLenReal1(0), fInterpHistFileScatLenImag1(0), fInterpHistFileScatLenReal2(0), fInterpHistFileScatLenImag2(0), fInterpHistFileLednickyHFunction(0),

  fLednickyHFunctionHist(0),

  fHyperGeo1F1RealHist(0),
  fHyperGeo1F1ImagHist(0),

  fGTildeRealHist(0),
  fGTildeImagHist(0),

  fCoulombScatteringLengthRealHist1(0),
  fCoulombScatteringLengthImagHist1(0),

  fCoulombScatteringLengthRealHist2(0),
  fCoulombScatteringLengthImagHist2(0),
  //end TODO

  fMinInterpKStar1(0), fMinInterpKStar2(0), fMinInterpRStar(0), fMinInterpTheta(0), fMinInterpReF0(0), fMinInterpImF0(0), fMinInterpD0(0),
  fMaxInterpKStar1(0), fMaxInterpKStar2(0), fMaxInterpRStar(0), fMaxInterpTheta(0), fMaxInterpReF0(0), fMaxInterpImF0(0), fMaxInterpD0(0),

  fCfsToFit(fNAnalyses),
  fFits(fNAnalyses),
  fMaxFitKStar(aMaxFitKStar),
  fRejectOmega(false),
  fChi2(0),
  fChi2GlobalMin(1000000000),
  fChi2Vec(fNAnalyses),
  fNpFits(0),
  fNpFitsVec(fNAnalyses),
  fNDF(0),
  fErrFlg(0),
  fMinParams(0),
  fParErrors(0)

{
  gRandom->SetSeed();

  fWaveFunction = new WaveFunction();
  fParallelWaveFunction = new ParallelWaveFunction(fUseScattLenHists);
  CheckIfAllOfSameCoulombType();

  for(int i=0; i<fNAnalyses; i++)
  {
    //fCfsToFit[i] = fFitSharedAnalyses->GetPairAnalysis(i)->GetCf();
  }

  omp_set_num_threads(4);

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
CoulombType CoulombFitterParallel::GetCoulombType(AnalysisType aAnalysisType)
{
  assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
  if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) return kAttractive;  //attractive
  else return kRepulsive; //repulsive
}


//________________________________________________________________________________________________________________
double CoulombFitterParallel::GetBohrRadius(CoulombType aCoulombType)
{
  double tBohrRadius;
  if(aCoulombType == kAttractive) tBohrRadius = -gBohrRadiusXiK;
  else if(aCoulombType == kRepulsive) tBohrRadius = gBohrRadiusXiK;
  else
  {
    cout << "ERROR in GetBohrRadius:  Invalid fCoulombType selected" << endl;
    assert(0);
  }

  return tBohrRadius;
}


//________________________________________________________________________________________________________________
double CoulombFitterParallel::GetBohrRadius(AnalysisType aAnalysisType)
{
  CoulombType tCoulombType = GetCoulombType(aAnalysisType);
  double tBohrRadius = GetBohrRadius(tCoulombType);
  return tBohrRadius;
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::CheckIfAllOfSameCoulombType()
{
  CoulombType tCoulombType0, tCoulombType1;

  bool tAllSame = true;
  for(int iAnaly1=1; iAnaly1<fNAnalyses; iAnaly1++)
  {
    tCoulombType0 = GetCoulombType(fFitSharedAnalyses->GetFitPairAnalysis(iAnaly1-1)->GetAnalysisType());
    tCoulombType1 = GetCoulombType(fFitSharedAnalyses->GetFitPairAnalysis(iAnaly1)->GetAnalysisType());

    if(tCoulombType0 != tCoulombType1) tAllSame = false;
  }

  fAllOfSameCoulombType = tAllSame;
  if(fAllOfSameCoulombType)
  {
    fCoulombType = tCoulombType1;
    fBohrRadius = GetBohrRadius(fCoulombType);
    fWaveFunction->SetCurrentAnalysisType(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetAnalysisType());
  }
}



//________________________________________________________________________________________________________________
int CoulombFitterParallel::GetBinNumber(double aBinSize, int aNbins, double aValue)
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
int CoulombFitterParallel::GetBinNumber(int aNbins, double aMin, double aMax, double aValue)
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
int CoulombFitterParallel::GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue)
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
td3dVec CoulombFitterParallel::BuildPairKStar3dVec(TString aPairKStarNtupleLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  cout << "Beginning conversion of TNtuple to 3dVec" << endl;
  std::clock_t start = std::clock();

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

  TString tFileLocation = aPairKStarNtupleLocationBase+TString(cBFieldTags[aBFieldType])+".root";
  TFile *tFile = TFile::Open(tFileLocation);

  TList *tFemtoList = (TList*)tFile->Get("femtolist");

  TString tArrayName = TString(cAnalysisBaseTags[aAnalysisType])+TString(cCentralityTags[aCentralityType]);
  TObjArray *tArray = (TObjArray*)tFemtoList->FindObject(tArrayName)->Clone();
    tArray->SetOwner();

  TString tNtupleName = "PairKStarKStarCf_"+TString(cAnalysisBaseTags[aAnalysisType]);
  TNtuple *tPairKStarNtuple = (TNtuple*)tArray->FindObject(tNtupleName);

  //----****----****----****----****----****----****

  float tTupleKStarMag, tTupleKStarOut, tTupleKStarSide, tTupleKStarLong;

  tPairKStarNtuple->SetBranchAddress("KStarMag", &tTupleKStarMag);
  tPairKStarNtuple->SetBranchAddress("KStarOut", &tTupleKStarOut);
  tPairKStarNtuple->SetBranchAddress("KStarSide", &tTupleKStarSide);
  tPairKStarNtuple->SetBranchAddress("KStarLong", &tTupleKStarLong);

  //--------------------------------------

  double tBinSize = (aKStarMax-aKStarMin)/aNbinsKStar;
cout << "tBinSize = " << tBinSize << endl;

  td3dVec tPairKStar3dVec;
    tPairKStar3dVec.clear();
  tPairKStar3dVec.resize(aNbinsKStar, td2dVec(0, td1dVec(0)));

cout << "Pre: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;

  vector<double> tTempEntry;
  int tKStarBinNumber;

  for(int i=0; i<tPairKStarNtuple->GetEntries(); i++)
  {
    tPairKStarNtuple->GetEntry(i);

    tKStarBinNumber = GetBinNumber(tBinSize, aNbinsKStar, tTupleKStarMag);
    if(tKStarBinNumber>=0)  //i.e, the KStarMag value is within my bins of interest
    {
      tTempEntry.clear();
        tTempEntry.push_back(tTupleKStarMag);
        tTempEntry.push_back(tTupleKStarOut);
        tTempEntry.push_back(tTupleKStarSide);
        tTempEntry.push_back(tTupleKStarLong);

      tPairKStar3dVec[tKStarBinNumber].push_back(tTempEntry);
    }
  }

  //TODO!!!!!!!!!
/*
  assert(aNbinsKStar == (int)fPairKStar3dVec.size());
  int tOffset=0;
  for(int i=0; i<aNbinsKStar; i++)
  {
    fPairKStar3dVecInfo.nPairsPerBin[i] = fPairKStar3dVec[i].size();
    fPairKStar3dVecInfo.binOffset[i] = tOffset;
    tOffset += 4*fPairKStar3dVec[i].size();
  }

  fParallelWaveFunction->LoadPairKStar3dVec(fPairKStar3dVec,fPairKStar3dVecInfo);
*/

  //Clean up----------------------------------------------------
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtoList object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtoList);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  tFemtoList->Delete();
  delete tFemtoList;

  tArray->Delete();
  delete tArray;

  //NOTE:  Apparently, tPairKStarNtuple is deleted when file is closed
  //       In fact, trying to delete it manually causes errors!

  tFile->Close();
  delete tFile;
  //------------------------------------------------------------

  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Conversion complete in " << duration << " seconds" << endl;


cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
for(int i=0; i<aNbinsKStar; i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

  return tPairKStar3dVec;
}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::ExtractPairKStar3dVecFromSingleFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill)
{
  vector<double> tTempEntry;
  int tKStarBinNumber;
  //---------------------------------------------------

  TFile *tFile = TFile::Open(aFileLocation);
  TList *tFemtoList = (TList*)tFile->Get("femtolist");
  TObjArray *tArray = (TObjArray*)tFemtoList->FindObject(aArrayName)->Clone();
    tArray->SetOwner();
  TNtuple *tPairKStarNtuple = (TNtuple*)tArray->FindObject(aNtupleName);
  //---------------------------------------------------

  float tTupleKStarMag, tTupleKStarOut, tTupleKStarSide, tTupleKStarLong;

  tPairKStarNtuple->SetBranchAddress("KStarMag", &tTupleKStarMag);
  tPairKStarNtuple->SetBranchAddress("KStarOut", &tTupleKStarOut);
  tPairKStarNtuple->SetBranchAddress("KStarSide", &tTupleKStarSide);
  tPairKStarNtuple->SetBranchAddress("KStarLong", &tTupleKStarLong);

  //--------------------------------------

  assert(aVecToFill.size() == aNbinsKStar);

  for(int i=0; i<tPairKStarNtuple->GetEntries(); i++)
  {
    tPairKStarNtuple->GetEntry(i);

    tKStarBinNumber = GetBinNumber(aBinSizeKStar, aNbinsKStar, tTupleKStarMag);
    if(tKStarBinNumber>=0)  //i.e, the KStarMag value is within my bins of interest
    {
      tTempEntry.clear();
        tTempEntry.push_back(tTupleKStarMag);
        tTempEntry.push_back(tTupleKStarOut);
        tTempEntry.push_back(tTupleKStarSide);
        tTempEntry.push_back(tTupleKStarLong);

      aVecToFill[tKStarBinNumber].push_back(tTempEntry);
    }
  }


  //Clean up----------------------------------------------------
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtoList object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtoList);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  tFemtoList->Delete();
  delete tFemtoList;

  tArray->Delete();
  delete tArray;

  //NOTE:  Apparently, tPairKStarNtuple is deleted when file is closed
  //       In fact, trying to delete it manually causes errors!

  tFile->Close();
  delete tFile;
  //------------------------------------------------------------

}


//________________________________________________________________________________________________________________
td3dVec CoulombFitterParallel::BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, vector<int> &aNFilesPerSubDir, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  cout << "Beginning FULL conversion of TNtuple to 3dVec" << endl;
  std::clock_t start = std::clock();

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


  //---------------------------------------------------
  int tNSubDir = aNFilesPerSubDir.size(); 
  assert(tNSubDir=13); // BmA->BmH && BpA->BpE
  vector<TString> tSubDirTags(13);
    tSubDirTags[0] = "BmA";
    tSubDirTags[1] = "BmB";
    tSubDirTags[2] = "BmC";
    tSubDirTags[3] = "BmD";
    tSubDirTags[4] = "BmE";
    tSubDirTags[5] = "BmF";
    tSubDirTags[6] = "BmG";
    tSubDirTags[7] = "BmH";

    tSubDirTags[8] = "BpA";
    tSubDirTags[9] = "BpB";
    tSubDirTags[10] = "BpC";
    tSubDirTags[11] = "BpD";
    tSubDirTags[12] = "BpE";
  //---------------------------------------------------

  double tBinSize = (aKStarMax-aKStarMin)/aNbinsKStar;
cout << "tBinSize = " << tBinSize << endl;

  td3dVec tPairKStar3dVec;
    tPairKStar3dVec.clear();
  tPairKStar3dVec.resize(aNbinsKStar, td2dVec(0, td1dVec(0)));

cout << "Pre: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;

  //---------------------------------------------------
  TString tSubDirName;
  TString tFileName;
  TString tFileLocation;

  TString tArrayName = TString(cAnalysisBaseTags[aAnalysisType])+TString(cCentralityTags[aCentralityType]);
  TString tNtupleName = "PairKStarKStarCf_"+TString(cAnalysisBaseTags[aAnalysisType]);

  for(int iSubDir=0; iSubDir<tNSubDir; iSubDir++)
  {
    tSubDirName = aPairKStarNtupleDirName+TString("/") + tSubDirTags[iSubDir];
cout << "iSubDir = " << iSubDir << endl;
    for(int iFile=0; iFile<aNFilesPerSubDir[iSubDir]; iFile++)
    {
//      tFileName = aFileBaseName + TString("_") + tSubDirTags[iSubDir] + TString(iFile) + TString(".root");
      tFileName = aFileBaseName + TString("_") + tSubDirTags[iSubDir];
        tFileName += iFile;
        tFileName += TString(".root");

      tFileLocation = tSubDirName + TString("/") + tFileName;
cout << "iFile = " << iFile << endl;
cout << "tFileLocation = " << tFileLocation << endl << endl;

      ExtractPairKStar3dVecFromSingleFile(tFileLocation,tArrayName,tNtupleName,tBinSize,aNbinsKStar,tPairKStar3dVec);
    }
  }

  //---------------------------------------------------
//TODO
/*
  assert(aNbinsKStar == (int)fPairKStar3dVec.size());
  int tOffset=0;
  for(int i=0; i<aNbinsKStar; i++)
  {
    fPairKStar3dVecInfo.nPairsPerBin[i] = fPairKStar3dVec[i].size();
    fPairKStar3dVecInfo.binOffset[i] = tOffset;
    tOffset += 4*fPairKStar3dVec[i].size();
  }

  fParallelWaveFunction->LoadPairKStar3dVec(fPairKStar3dVec,fPairKStar3dVecInfo);
*/

  //---------------------------------------------------
  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Conversion complete in " << duration << " seconds" << endl;


  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<aNbinsKStar; i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

  return tPairKStar3dVec;
}



//________________________________________________________________________________________________________________
td3dVec CoulombFitterParallel::BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  cout << "Beginning FULL conversion of TNtuple to 3dVec" << endl;
  std::clock_t start = std::clock();

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

  //---------------------------------------------------
  assert(aNFiles==27);

  vector<TString> tRomanNumerals(17);
    tRomanNumerals[0] = "I";
    tRomanNumerals[1] = "II";
    tRomanNumerals[2] = "III";
    tRomanNumerals[3] = "IV";
    tRomanNumerals[4] = "V";
    tRomanNumerals[5] = "VI";
    tRomanNumerals[6] = "VII";
    tRomanNumerals[7] = "VIII";
    tRomanNumerals[8] = "IX";
    tRomanNumerals[9] = "X";
    tRomanNumerals[10] = "XI";
    tRomanNumerals[11] = "XII";
    tRomanNumerals[12] = "XIII";
    tRomanNumerals[13] = "XIV";
    tRomanNumerals[14] = "XV";
    tRomanNumerals[15] = "XVI";
    tRomanNumerals[16] = "XVII";
  //---------------------------------------------------

  double tBinSize = (aKStarMax-aKStarMin)/aNbinsKStar;
cout << "tBinSize = " << tBinSize << endl;

  td3dVec tPairKStar3dVec;
    tPairKStar3dVec.clear();
  tPairKStar3dVec.resize(aNbinsKStar, td2dVec(0, td1dVec(0)));

cout << "Pre: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;

  //---------------------------------------------------
  TString tFileLocation;

  TString tArrayName = TString(cAnalysisBaseTags[aAnalysisType])+TString(cCentralityTags[aCentralityType]);
  TString tNtupleName = "PairKStarKStarCf_"+TString(cAnalysisBaseTags[aAnalysisType]);

  for(int iFile=0; iFile<17; iFile++) //Bm files
  {
    cout << "\t iFile = Bm" << iFile << endl;
    tFileLocation = aPairKStarNtupleDirName + TString("/") + aFileBaseName + TString("_Bm") + tRomanNumerals[iFile] + TString(".root");
    ExtractPairKStar3dVecFromSingleFile(tFileLocation,tArrayName,tNtupleName,tBinSize,aNbinsKStar,tPairKStar3dVec);
  }

  for(int iFile=0; iFile<10; iFile++) //Bp files
  {
    cout << "\t iFile = Bp" << iFile << endl;
    tFileLocation = aPairKStarNtupleDirName + TString("/") + aFileBaseName + TString("_Bp") + tRomanNumerals[iFile] + TString(".root");
    ExtractPairKStar3dVecFromSingleFile(tFileLocation,tArrayName,tNtupleName,tBinSize,aNbinsKStar,tPairKStar3dVec);
  }

  //---------------------------------------------------
//TODO
/*
  assert(aNbinsKStar == (int)fPairKStar3dVec.size());
  int tOffset=0;
  for(int i=0; i<aNbinsKStar; i++)
  {
    fPairKStar3dVecInfo.nPairsPerBin[i] = fPairKStar3dVec[i].size();
    fPairKStar3dVecInfo.binOffset[i] = tOffset;
    tOffset += 4*fPairKStar3dVec[i].size();
  }

  fParallelWaveFunction->LoadPairKStar3dVec(fPairKStar3dVec,fPairKStar3dVecInfo);
*/
  //---------------------------------------------------
  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Conversion complete in " << duration << " seconds" << endl;


  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<aNbinsKStar; i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

  return tPairKStar3dVec;
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::WriteRow(ostream &aOutput, vector<double> &aRow)
{
  for(int i = 0; i < aRow.size(); i++)
  {
    if( i < aRow.size()-1) aOutput << aRow[i] << " ";
    else if(i == aRow.size()-1) aOutput << aRow[i] << endl;
    else cout << "SOMETHING IS WRONG!!!!!\n";
  }
}



//________________________________________________________________________________________________________________
  void CoulombFitterParallel::WritePairKStar3dVecFile(TString aOutputName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  ofstream tFileOut(aOutputName);

  cout << "Beginning FULL writing of TNtuple to 3dVec" << endl;
  std::clock_t start = std::clock();

  //----****----****----****----****----****----****
  //TODO make this automated
  //TODO add centrality selection to this
  //Also, this is basically FitPartialAnalysis::ConnectAnalysisDirectory

  td3dVec tPairKStar3dVec = BuildPairKStar3dVecFull(aPairKStarNtupleDirName,aFileBaseName,aNFiles,aAnalysisType,aCentralityType,aNbinsKStar,aKStarMin,aKStarMax);

  //---------------------------------------------------
  tFileOut << aNbinsKStar << " " << aKStarMin << " " << aKStarMax << endl;

  vector<double> tTempPair;
  for(int iBin=0; iBin<(int)tPairKStar3dVec.size(); iBin++)
  {
    tFileOut << iBin << " " << tPairKStar3dVec[iBin].size() << endl;
    for(int iPair=0; iPair<(int)tPairKStar3dVec[iBin].size(); iPair++)
    {
      WriteRow(tFileOut,tPairKStar3dVec[iBin][iPair]);
    }
  }

  tFileOut.close();
  //---------------------------------------------------
  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Writing complete in " << duration << " seconds" << endl;


  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<aNbinsKStar; i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;
}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::WriteAllPairKStar3dVecFiles(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  AnalysisType tAnalysisType;
  CentralityType tCentralityType;
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    tAnalysisType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetAnalysisType();
    tCentralityType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetCentralityType();
    WritePairKStar3dVecFile(aOutputBaseName,aPairKStarNtupleDirName,aFileBaseName,aNFiles,tAnalysisType,tCentralityType,aNbinsKStar,aKStarMin,aKStarMax);
  }
}


//________________________________________________________________________________________________________________
td3dVec CoulombFitterParallel::BuildPairKStar3dVecFromTxt(TString aFileName)
{
ChronoTimer tTimer;
tTimer.Start();

  ifstream tFileIn(aFileName);

  td3dVec tPairKStar3dVec;
    tPairKStar3dVec.clear();

  vector<vector<double> > tTempBin2dVec;
  vector<double> tTempPair1dVec;

  int aNbinsKStar;
  double aKStarMin, aKStarMax;

  string tString;
  int tCount = 0;
  while(getline(tFileIn, tString))
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
    else if(tTempPair1dVec.size() == 4) //pair
    {
      tTempBin2dVec.push_back(tTempPair1dVec);
    }
    else if(tTempPair1dVec.size() == 3) //File header
    {
      aNbinsKStar = tTempPair1dVec[0];
      aKStarMin = tTempPair1dVec[1];
      aKStarMax = tTempPair1dVec[2];
    }
    else
    {
      cout << "ERROR: Incorrect row size in BuildPairKStar3dVecFromTxt" << endl;
      assert(0);
    }
  }
  tPairKStar3dVec.push_back(tTempBin2dVec);
  tTempBin2dVec.clear();

//---------------------------------------------------

  //--Set the fPairKStar3dVecInfo
  fPairKStar3dVecInfo.nBinsK = aNbinsKStar;
  fPairKStar3dVecInfo.minK = aKStarMin;
  fPairKStar3dVecInfo.maxK = aKStarMax;
  fPairKStar3dVecInfo.binWidthK = ((aKStarMax-aKStarMin)/aNbinsKStar);

  //---------------------------------------------------
//TODO
/*
  assert(aNbinsKStar == (int)tPairKStar3dVec.size());
  int tOffset=0;
  for(int i=0; i<aNbinsKStar; i++)
  {
    fPairKStar3dVecInfo.nPairsPerBin[i] = tPairKStar3dVec[i].size();
    fPairKStar3dVecInfo.binOffset[i] = tOffset;
    tOffset += 4*fPairKStar3dVec[i].size();
  }

  fParallelWaveFunction->LoadPairKStar3dVec(fPairKStar3dVec,fPairKStar3dVecInfo);
*/
  //---------------------------------------------------


tTimer.Stop();
cout << "BuildPairKStar3dVecFromTxt finished: ";
tTimer.PrintInterval();

  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<(int)tPairKStar3dVec.size(); i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

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
void CoulombFitterParallel::BuildPairKStar4dVecOnFly(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  fPairKStar4dVec.resize(fNAnalyses, td3dVec(0, td2dVec(0, td1dVec(0))));

  AnalysisType tAnalysisType;
  CentralityType tCentralityType;
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    tAnalysisType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetAnalysisType();
    tCentralityType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetCentralityType();

    td3dVec tPairKStar3dVec = BuildPairKStar3dVecFull(aPairKStarNtupleDirName,aFileBaseName,aNFiles,tAnalysisType,tCentralityType,aNbinsKStar,aKStarMin,aKStarMax);
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
void CoulombFitterParallel::LoadLednickyHFunctionFile(TString aFileBaseName)
{

  TString aFileName = aFileBaseName+".root";

  fInterpHistFileLednickyHFunction = TFile::Open(aFileName);

  fLednickyHFunctionHist = (TH1D*)fInterpHistFileLednickyHFunction->Get("LednickyHFunction");
//    fLednickyHFunction->SetDirectory(0);

//  fInterpHistFileLednickyHFunction->Close();
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::LoadInterpHistFile(TString aFileBaseName)
{
  MakeOtherVectors(aFileBaseName);

  //--TODO delete the following after testing has completed
  TString aFileName = aFileBaseName+".root";

  fInterpHistFile = TFile::Open(aFileName);

  fHyperGeo1F1RealHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Real");
  fHyperGeo1F1ImagHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Imag");
//    fHyperGeo1F1Real->SetDirectory(0);
//    fHyperGeo1F1Imag->SetDirectory(0);

  fGTildeRealHist = (TH2D*)fInterpHistFile->Get("GTildeReal");
  fGTildeImagHist = (TH2D*)fInterpHistFile->Get("GTildeImag");
//    fGTildeReal->SetDirectory(0);
//    fGTildeImag->SetDirectory(0);

//  fInterpHistFile->Close();

  //--------------------------------------------------------------

  //No SetDirectory call for THn, so unfortunately these files must remain open 
  if(fUseScattLenHists)
  {
    MakeNiceScattLenVectors(aFileBaseName);

    TString aFileNameScatLenReal1 = aFileBaseName+"ScatLenReal1.root";  // 0. < k* < 0.2
    TString aFileNameScatLenImag1 = aFileBaseName+"ScatLenImag1.root";

    TString aFileNameScatLenReal2 = aFileBaseName+"ScatLenReal2.root";  // 0.2 < k* < 0.4
    TString aFileNameScatLenImag2 = aFileBaseName+"ScatLenImag2.root";


    fInterpHistFileScatLenReal1 = TFile::Open(aFileNameScatLenReal1);
    fCoulombScatteringLengthRealHist1 = (THnD*)fInterpHistFileScatLenReal1->Get("CoulombScatteringLengthReal");

    fInterpHistFileScatLenImag1 = TFile::Open(aFileNameScatLenImag1);
    fCoulombScatteringLengthImagHist1 = (THnD*)fInterpHistFileScatLenImag1->Get("CoulombScatteringLengthImag");

    fInterpHistFileScatLenReal2 = TFile::Open(aFileNameScatLenReal2);
    fCoulombScatteringLengthRealHist2 = (THnD*)fInterpHistFileScatLenReal2->Get("CoulombScatteringLengthReal");

    fInterpHistFileScatLenImag2 = TFile::Open(aFileNameScatLenImag2);
    fCoulombScatteringLengthImagHist2 = (THnD*)fInterpHistFileScatLenImag2->Get("CoulombScatteringLengthImag");
  }
}


//________________________________________________________________________________________________________________
void CoulombFitterParallel::MakeNiceScattLenVectors(TString aFileBaseName)
{
ChronoTimer tTimer(kSec);
tTimer.Start();
  cout << "Starting MakeNiceScattLenVectors" << endl;

  TString aFileNameScatLenReal1 = aFileBaseName+"ScatLenReal1.root";  // 0. < k* < 0.2
  TString aFileNameScatLenImag1 = aFileBaseName+"ScatLenImag1.root";

  TString aFileNameScatLenReal2 = aFileBaseName+"ScatLenReal2.root";  // 0.2 < k* < 0.4
  TString aFileNameScatLenImag2 = aFileBaseName+"ScatLenImag2.root";

//--------------------------------------------------------------
  TFile* tInterpHistFileScatLenReal1 = TFile::Open(aFileNameScatLenReal1);
  THnD* tCoulombScatteringLengthRealHist1 = (THnD*)tInterpHistFileScatLenReal1->Get("CoulombScatteringLengthReal");

  TFile* tInterpHistFileScatLenImag1 = TFile::Open(aFileNameScatLenImag1);
  THnD* tCoulombScatteringLengthImagHist1 = (THnD*)tInterpHistFileScatLenImag1->Get("CoulombScatteringLengthImag");

  TFile* tInterpHistFileScatLenReal2 = TFile::Open(aFileNameScatLenReal2);
  THnD* tCoulombScatteringLengthRealHist2 = (THnD*)tInterpHistFileScatLenReal2->Get("CoulombScatteringLengthReal");

  TFile* tInterpHistFileScatLenImag2 = TFile::Open(aFileNameScatLenImag2);
  THnD* tCoulombScatteringLengthImagHist2 = (THnD*)tInterpHistFileScatLenImag2->Get("CoulombScatteringLengthImag");

//--------------------------------------------------------------

//TODO probably remove these in favor of struct object
  fMinInterpKStar1 = tCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinCenter(1);
  fMaxInterpKStar1 = tCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinCenter(tCoulombScatteringLengthRealHist1->GetAxis(0)->GetNbins());

  fMinInterpKStar2 = tCoulombScatteringLengthRealHist2->GetAxis(0)->GetBinCenter(1);
  fMaxInterpKStar2 = tCoulombScatteringLengthRealHist2->GetAxis(0)->GetBinCenter(tCoulombScatteringLengthRealHist2->GetAxis(0)->GetNbins());

  fMinInterpReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinCenter(1);
  fMaxInterpReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinCenter(tCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins());

  fMinInterpImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinCenter(1);
  fMaxInterpImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinCenter(tCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins());

  fMinInterpD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinCenter(1);
  fMaxInterpD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinCenter(tCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins());

//--------------------------------------------------------------

  fScattLenInfo.nBinsK = tCoulombScatteringLengthRealHist1->GetAxis(0)->GetNbins();
  fScattLenInfo.binWidthK = tCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinWidth(1);
  fScattLenInfo.minK = tCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinLowEdge(1);
  fScattLenInfo.minInterpK = tCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinCenter(1);
  fScattLenInfo.maxK = tCoulombScatteringLengthRealHist2->GetAxis(0)->GetBinUpEdge(tCoulombScatteringLengthRealHist2->GetAxis(0)->GetNbins());
  fScattLenInfo.maxInterpK = tCoulombScatteringLengthRealHist2->GetAxis(0)->GetBinCenter(tCoulombScatteringLengthRealHist2->GetAxis(0)->GetNbins());

  fScattLenInfo.nBinsReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins();
  fScattLenInfo.binWidthReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinWidth(1);
  fScattLenInfo.minReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinLowEdge(1);
  fScattLenInfo.minInterpReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinCenter(1);
  fScattLenInfo.maxReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinUpEdge(tCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins());
  fScattLenInfo.maxInterpReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinCenter(tCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins());

  fScattLenInfo.nBinsImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins();
  fScattLenInfo.binWidthImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinWidth(1);
  fScattLenInfo.minImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinLowEdge(1);
  fScattLenInfo.minInterpImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinCenter(1);
  fScattLenInfo.maxImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinUpEdge(tCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins());
  fScattLenInfo.maxInterpImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinCenter(tCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins());

  fScattLenInfo.nBinsD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins();
  fScattLenInfo.binWidthD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinWidth(1);
  fScattLenInfo.minD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinLowEdge(1);
  fScattLenInfo.minInterpD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinCenter(1);
  fScattLenInfo.maxD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinUpEdge(tCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins());
  fScattLenInfo.maxInterpD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinCenter(tCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins());
//--------------------------------------------------------------


  int tNbinsReF0 = tCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins();
  int tNbinsImF0 = tCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins();
  int tNbinsD0 = tCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins();
  int tNbinsK = 2*tCoulombScatteringLengthRealHist1->GetAxis(0)->GetNbins();

  fCoulombScatteringLengthReal.resize(tNbinsReF0, td3dVec(tNbinsImF0, td2dVec(tNbinsD0, td1dVec(tNbinsK,0))));
  fCoulombScatteringLengthImag.resize(tNbinsReF0, td3dVec(tNbinsImF0, td2dVec(tNbinsD0, td1dVec(tNbinsK,0))));

cout << "fCoulombScatteringLengthReal.size() = " << fCoulombScatteringLengthReal.size() << endl;
cout << "fCoulombScatteringLengthReal[0].size() = " << fCoulombScatteringLengthReal[0].size() << endl;
cout << "fCoulombScatteringLengthReal[0][0].size() = " << fCoulombScatteringLengthReal[0][0].size() << endl;
cout << "fCoulombScatteringLengthReal[0][0][0].size() = " << fCoulombScatteringLengthReal[0][0][0].size() << endl;

  int tBin[4];
  int iK, iReF0, iImF0, iD0;
  #pragma omp parallel for private(iK,iReF0,iImF0,iD0) firstprivate(tBin)
  for(iK=1; iK<=tNbinsK; iK++)
  {
    for(iReF0=1; iReF0<=tNbinsReF0; iReF0++)
    {
      for(iImF0=1; iImF0<=tNbinsImF0; iImF0++)
      {
        for(iD0=1; iD0<=tNbinsD0; iD0++)
        {
          tBin[0] = iK;
          tBin[1] = iReF0;
          tBin[2] = iImF0;
          tBin[3] = iD0;

          if(iK <= tNbinsK/2)
          {
            fCoulombScatteringLengthReal[iReF0-1][iImF0-1][iD0-1][iK-1] = tCoulombScatteringLengthRealHist1->GetBinContent(tBin);
            fCoulombScatteringLengthImag[iReF0-1][iImF0-1][iD0-1][iK-1] = tCoulombScatteringLengthImagHist1->GetBinContent(tBin);
          }
          else
          {
	    tBin[0] = iK-tNbinsK/2;
            fCoulombScatteringLengthReal[iReF0-1][iImF0-1][iD0-1][iK-1] = tCoulombScatteringLengthRealHist2->GetBinContent(tBin);
            fCoulombScatteringLengthImag[iReF0-1][iImF0-1][iD0-1][iK-1] = tCoulombScatteringLengthImagHist2->GetBinContent(tBin);
          }
        }
      }
    }
  }

//--------------------------------------------------------------

  delete tCoulombScatteringLengthRealHist1;
  delete tCoulombScatteringLengthImagHist1;

  delete tCoulombScatteringLengthRealHist2;
  delete tCoulombScatteringLengthImagHist2;

  tInterpHistFileScatLenReal1->Close();
  delete tInterpHistFileScatLenReal1;

  tInterpHistFileScatLenImag1->Close();
  delete tInterpHistFileScatLenImag1;

  tInterpHistFileScatLenReal2->Close();
  delete tInterpHistFileScatLenReal2;

  tInterpHistFileScatLenImag2->Close();
  delete tInterpHistFileScatLenImag2;

//  delete[] tBin;  //TODO NO!!!!!!!!!!!!!!! This causes "double free or corruption"!!!!!!!!!

//--------------------------------------------------------------
//Send things over to ParallWaveFunction object
//TODO after sending these vectors over, they can probably be deleted
//maybe I should not even make them class members, but only members of this function

//  fParallelWaveFunction->LoadScattLenReal(fCoulombScatteringLengthReal);
//  fParallelWaveFunction->LoadScattLenImag(fCoulombScatteringLengthImag);
  fParallelWaveFunction->LoadScattLenInfo(fScattLenInfo);



  //------------------------------------------------
tTimer.Stop();
cout << "MakeNiceScattLenVectors: ";
tTimer.PrintInterval();

}

//________________________________________________________________________________________________________________
void CoulombFitterParallel::MakeOtherVectors(TString aFileBaseName)
{
ChronoTimer tTimer(kSec);
tTimer.Start();
  cout << "Starting MakeOtherVectors" << endl;

  LoadLednickyHFunctionFile();

  TString aFileName = aFileBaseName+".root";
  TFile* tInterpHistFile = TFile::Open(aFileName);

//--------------------------------------------------------------

  TH3D* tHyperGeo1F1RealHist = (TH3D*)tInterpHistFile->Get("HyperGeo1F1Real");
  TH3D* tHyperGeo1F1ImagHist = (TH3D*)tInterpHistFile->Get("HyperGeo1F1Imag");

  TH2D* tGTildeRealHist = (TH2D*)tInterpHistFile->Get("GTildeReal");
  TH2D* tGTildeImagHist = (TH2D*)tInterpHistFile->Get("GTildeImag");

//--------------------------------------------------------------
//TODO probably remove these in favor of struct objects
  fMinInterpKStar1 = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(1);
  fMaxInterpKStar2 = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(fLednickyHFunctionHist->GetNbinsX());

  fMinInterpRStar = tHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(1);
  fMaxInterpRStar = tHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(tHyperGeo1F1RealHist->GetNbinsY());

  fMinInterpTheta = tHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(1);
  fMaxInterpTheta = tHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(tHyperGeo1F1RealHist->GetNbinsZ());

//--------------------------------------------------------------
  fHyperGeo1F1Info.nBinsK = tHyperGeo1F1RealHist->GetXaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthK = tHyperGeo1F1RealHist->GetXaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minK = tHyperGeo1F1RealHist->GetXaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpK = tHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxK = tHyperGeo1F1RealHist->GetXaxis()->GetBinUpEdge(tHyperGeo1F1RealHist->GetNbinsX());
  fHyperGeo1F1Info.maxInterpK = tHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(tHyperGeo1F1RealHist->GetNbinsX());

  fHyperGeo1F1Info.nBinsR = tHyperGeo1F1RealHist->GetYaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthR = tHyperGeo1F1RealHist->GetYaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minR = tHyperGeo1F1RealHist->GetYaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpR = tHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxR = tHyperGeo1F1RealHist->GetYaxis()->GetBinUpEdge(tHyperGeo1F1RealHist->GetNbinsY());
  fHyperGeo1F1Info.maxInterpR = tHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(tHyperGeo1F1RealHist->GetNbinsY());

  fHyperGeo1F1Info.nBinsTheta = tHyperGeo1F1RealHist->GetZaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthTheta = tHyperGeo1F1RealHist->GetZaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minTheta = tHyperGeo1F1RealHist->GetZaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpTheta = tHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxTheta = tHyperGeo1F1RealHist->GetZaxis()->GetBinUpEdge(tHyperGeo1F1RealHist->GetNbinsZ());
  fHyperGeo1F1Info.maxInterpTheta = tHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(tHyperGeo1F1RealHist->GetNbinsZ());

  //-----

  fGTildeInfo.nBinsK = tGTildeRealHist->GetXaxis()->GetNbins();
  fGTildeInfo.binWidthK = tGTildeRealHist->GetXaxis()->GetBinWidth(1);
  fGTildeInfo.minK = tGTildeRealHist->GetXaxis()->GetBinLowEdge(1);
  fGTildeInfo.minInterpK = tGTildeRealHist->GetXaxis()->GetBinCenter(1);
  fGTildeInfo.maxK = tGTildeRealHist->GetXaxis()->GetBinUpEdge(tGTildeRealHist->GetNbinsX());
  fGTildeInfo.maxInterpK = tGTildeRealHist->GetXaxis()->GetBinCenter(tGTildeRealHist->GetNbinsX());

  fGTildeInfo.nBinsR = tGTildeRealHist->GetYaxis()->GetNbins();
  fGTildeInfo.binWidthR = tGTildeRealHist->GetYaxis()->GetBinWidth(1);
  fGTildeInfo.minR = tGTildeRealHist->GetYaxis()->GetBinLowEdge(1);
  fGTildeInfo.minInterpR = tGTildeRealHist->GetYaxis()->GetBinCenter(1);
  fGTildeInfo.maxR = tGTildeRealHist->GetYaxis()->GetBinUpEdge(tGTildeRealHist->GetNbinsY());
  fGTildeInfo.maxInterpR = tGTildeRealHist->GetYaxis()->GetBinCenter(tGTildeRealHist->GetNbinsY());


//--------------------------------------------------------------

  int tNbinsK = tHyperGeo1F1RealHist->GetXaxis()->GetNbins();
  int tNbinsR = tHyperGeo1F1RealHist->GetYaxis()->GetNbins();
  int tNbinsTheta = tHyperGeo1F1RealHist->GetZaxis()->GetNbins();

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
        fHyperGeo1F1Real[iK-1][iR-1][iTheta-1] = tHyperGeo1F1RealHist->GetBinContent(iK,iR,iTheta);
        fHyperGeo1F1Imag[iK-1][iR-1][iTheta-1] = tHyperGeo1F1ImagHist->GetBinContent(iK,iR,iTheta);
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
      fGTildeReal[iK-1][iR-1] = tGTildeRealHist->GetBinContent(iK,iR);
      fGTildeImag[iK-1][iR-1] = tGTildeImagHist->GetBinContent(iK,iR);
    }
  }

  fLednickyHFunction.resize(tNbinsK,0.);
  #pragma omp parallel for private(iK)
  for(iK=1; iK<=tNbinsK; iK++)
  {
    fLednickyHFunction[iK-1] = fLednickyHFunctionHist->GetBinContent(iK);
  }

//--------------------------------------------------------------

  delete tHyperGeo1F1RealHist;
  delete tHyperGeo1F1ImagHist;

  delete tGTildeRealHist;
  delete tGTildeImagHist;

  tInterpHistFile->Close();
  delete tInterpHistFile;

//--------------------------------------------------------------
//Send things over to ParallWaveFunction object
//TODO after sending these vectors over, they can probably be deleted
//maybe I should not even make them class members, but only members of this function

  fParallelWaveFunction->LoadLednickyHFunction(fLednickyHFunction);

  fParallelWaveFunction->LoadGTildeReal(fGTildeReal);
  fParallelWaveFunction->LoadGTildeImag(fGTildeImag);
  fParallelWaveFunction->LoadGTildeInfo(fGTildeInfo);

  fParallelWaveFunction->LoadHyperGeo1F1Real(fHyperGeo1F1Real);
  fParallelWaveFunction->LoadHyperGeo1F1Imag(fHyperGeo1F1Imag);
  fParallelWaveFunction->LoadHyperGeo1F1Info(fHyperGeo1F1Info);


  //----------------------------------------------------------------
tTimer.Stop();
cout << "MakeOtherVectors: ";
tTimer.PrintInterval();

}


//________________________________________________________________________________________________________________
int CoulombFitterParallel::GetInterpLowBin(InterpType aInterpType, InterpAxisType aAxisType, double aVal)
{
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  bool tErrorFlag = false;

  switch(aInterpType)
  {
    case kGTilde:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = fGTildeInfo.nBinsK;
          tBinWidth = fGTildeInfo.binWidthK;
          tMin = fGTildeInfo.minK;
          tMax = fGTildeInfo.maxK;
          break;

        case kRaxis:
          tNbins = fGTildeInfo.nBinsR;
          tBinWidth = fGTildeInfo.binWidthR;
          tMin = fGTildeInfo.minR;
          tMax = fGTildeInfo.maxR;
          break;

        //Invalid axis selection
        case kThetaaxis:
          tErrorFlag = true;
          break;
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kHyperGeo1F1:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = fHyperGeo1F1Info.nBinsK;
          tBinWidth = fHyperGeo1F1Info.binWidthK;
          tMin = fHyperGeo1F1Info.minK;
          tMax = fHyperGeo1F1Info.maxK;
          break;

        case kRaxis:
          tNbins = fHyperGeo1F1Info.nBinsR;
          tBinWidth = fHyperGeo1F1Info.binWidthR;
          tMin = fHyperGeo1F1Info.minR;
          tMax = fHyperGeo1F1Info.maxR;
          break;

        case kThetaaxis:
          tNbins = fHyperGeo1F1Info.nBinsTheta;
          tBinWidth = fHyperGeo1F1Info.binWidthTheta;
          tMin = fHyperGeo1F1Info.minTheta;
          tMax = fHyperGeo1F1Info.maxTheta;
          break;

        //Invalid axis selection
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kScattLen:
      switch(aAxisType)
      {
        case kReF0axis:
          tNbins = fScattLenInfo.nBinsReF0;
          tBinWidth = fScattLenInfo.binWidthReF0;
          tMin = fScattLenInfo.minReF0;
          tMax = fScattLenInfo.maxReF0;
          break;

        case kImF0axis:
          tNbins = fScattLenInfo.nBinsImF0;
          tBinWidth = fScattLenInfo.binWidthImF0;
          tMin = fScattLenInfo.minImF0;
          tMax = fScattLenInfo.maxImF0;
          break;

        case kD0axis:
          tNbins = fScattLenInfo.nBinsD0;
          tBinWidth = fScattLenInfo.binWidthD0;
          tMin = fScattLenInfo.minD0;
          tMax = fScattLenInfo.maxD0;
          break;

        case kKaxis:
          tNbins = fScattLenInfo.nBinsK;
          tBinWidth = fScattLenInfo.binWidthK;
          tMin = fScattLenInfo.minK;
          tMax = fScattLenInfo.maxK;
          break;


        //Invalid axis selection
        case kRaxis:
          tErrorFlag = true;
          break;
        case kThetaaxis:
          tErrorFlag = true;
          break;

      }
      break;
  }

  //Check error
  if(tErrorFlag) return -2;

  //---------------------------------
  tBin = GetBinNumber(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;
  else return tReturnBin;

}

//________________________________________________________________________________________________________________
double CoulombFitterParallel::GetInterpLowBinCenter(InterpType aInterpType, InterpAxisType aAxisType, double aVal)
{
  double tReturnValue;
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  bool tErrorFlag = false;

  switch(aInterpType)
  {
    case kGTilde:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = fGTildeInfo.nBinsK;
          tBinWidth = fGTildeInfo.binWidthK;
          tMin = fGTildeInfo.minK;
          tMax = fGTildeInfo.maxK;
          break;

        case kRaxis:
          tNbins = fGTildeInfo.nBinsR;
          tBinWidth = fGTildeInfo.binWidthR;
          tMin = fGTildeInfo.minR;
          tMax = fGTildeInfo.maxR;
          break;

        //Invalid axis selection
        case kThetaaxis:
          tErrorFlag = true;
          break;
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kHyperGeo1F1:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = fHyperGeo1F1Info.nBinsK;
          tBinWidth = fHyperGeo1F1Info.binWidthK;
          tMin = fHyperGeo1F1Info.minK;
          tMax = fHyperGeo1F1Info.maxK;
          break;

        case kRaxis:
          tNbins = fHyperGeo1F1Info.nBinsR;
          tBinWidth = fHyperGeo1F1Info.binWidthR;
          tMin = fHyperGeo1F1Info.minR;
          tMax = fHyperGeo1F1Info.maxR;
          break;

        case kThetaaxis:
          tNbins = fHyperGeo1F1Info.nBinsTheta;
          tBinWidth = fHyperGeo1F1Info.binWidthTheta;
          tMin = fHyperGeo1F1Info.minTheta;
          tMax = fHyperGeo1F1Info.maxTheta;
          break;

        //Invalid axis selection
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kScattLen:
      switch(aAxisType)
      {
        case kReF0axis:
          tNbins = fScattLenInfo.nBinsReF0;
          tBinWidth = fScattLenInfo.binWidthReF0;
          tMin = fScattLenInfo.minReF0;
          tMax = fScattLenInfo.maxReF0;
          break;

        case kImF0axis:
          tNbins = fScattLenInfo.nBinsImF0;
          tBinWidth = fScattLenInfo.binWidthImF0;
          tMin = fScattLenInfo.minImF0;
          tMax = fScattLenInfo.maxImF0;
          break;

        case kD0axis:
          tNbins = fScattLenInfo.nBinsD0;
          tBinWidth = fScattLenInfo.binWidthD0;
          tMin = fScattLenInfo.minD0;
          tMax = fScattLenInfo.maxD0;
          break;

        case kKaxis:
          tNbins = fScattLenInfo.nBinsK;
          tBinWidth = fScattLenInfo.binWidthK;
          tMin = fScattLenInfo.minK;
          tMax = fScattLenInfo.maxK;
          break;


        //Invalid axis selection
        case kRaxis:
          tErrorFlag = true;
          break;
        case kThetaaxis:
          tErrorFlag = true;
          break;

      }
      break;
  }

  //Check error
  if(tErrorFlag) return -2;

  //---------------------------------
  tBin = GetBinNumber(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;

  tReturnValue = tMin + (tReturnBin+0.5)*tBinWidth;
  return tReturnValue;
}

/*
//________________________________________________________________________________________________________________
//TODO fix the entire functions
vector<int> CoulombFitterParallel::GetRelevantKStarBinNumbers(double aKStarMagMin, double aKStarMagMax)
{
  int tMinBin = GetInterpLowBin(kScattLen,kKaxis,aKStarMagMin);
  int tMaxBin = GetInterpLowBin(kScattLen,kKaxis,aKStarMagMax);

  if(tMaxBin<(fScattLenInfo.nBinsK-1)) tMaxBin++;  //must include first bin over range
                                                   //Note:  GetInterpLowBin already return first underneath bin (if exists)
  //TODO fix this shit
  if(tMinBin<0) tMinBin=0;

  int tNbins = (tMaxBin-tMinBin)+1;  //+1 because inclusive, i.e. if bins 3,4,5,6,7,8 -> (8-3)+1 = 6 bins 
  vector<int> tReturnVector(tNbins);
  for(int i=0; i<tNbins; i++)
  {
    tReturnVector[i] = tMinBin+i;
  }
  return tReturnVector;
}
*/

//________________________________________________________________________________________________________________
//TODO fix the entire functions
vector<int> CoulombFitterParallel::GetRelevantKStarBinNumbers(double aKStarMagMin, double aKStarMagMax)
{
  int tMinBin = GetBinNumber(fScattLenInfo.binWidthK,fScattLenInfo.minK,fScattLenInfo.maxK,aKStarMagMin);
  int tMaxBin = GetBinNumber(fScattLenInfo.binWidthK,fScattLenInfo.minK,fScattLenInfo.maxK,aKStarMagMax);

  if(tMaxBin<(fScattLenInfo.nBinsK-1)) tMaxBin++;  //must include first bin over range
                                                   //Note:  GetInterpLowBin already return first underneath bin (if exists)
  //TODO fix this shit
  if(tMinBin>0) tMinBin--;

  int tNbins = (tMaxBin-tMinBin)+1;  //+1 because inclusive, i.e. if bins 3,4,5,6,7,8 -> (8-3)+1 = 6 bins 
  vector<int> tReturnVector(tNbins);
  for(int i=0; i<tNbins; i++)
  {
    tReturnVector[i] = tMinBin+i;
  }
  return tReturnVector;
}

//TODO delete the following functions after testing****************************************************************************





//________________________________________________________________________________________________________________
double CoulombFitterParallel::LinearInterpolate(TH1* a1dHisto, double aX)
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
double CoulombFitterParallel::BilinearInterpolate(TH2* a2dHisto, double aX, double aY)
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
    cout << "Error in CoulombFitterParallel::BilinearInterpolate, cannot interpolate outside histogram domain" << endl;
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
double CoulombFitterParallel::BilinearInterpolateVector(vector<vector<double> > &a2dVec, double aX, int aNbinsX, double aMinX, double aMaxX, double aY, int aNbinsY, double aMinY, double aMaxY)
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
    cout << "Error in CoulombFitterParallel::BilinearInterpolateVector, cannot interpolate outside histogram domain" << endl;
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
double CoulombFitterParallel::TrilinearInterpolate(TH3* a3dHisto, double aX, double aY, double aZ)
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
    cout << "Error in CoulombFitterParallel::TrilinearInterpolate, cannot interpolate outside histogram domain" << endl;

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
double CoulombFitterParallel::QuadrilinearInterpolate(THn* a4dHisto, double aT, double aX, double aY, double aZ)
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
    cout << "Error in CoulombFitterParallel::QuadrilinearInterpolate, cannot interpolate outside histogram domain" << endl;
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
double CoulombFitterParallel::ScattLenInterpolate(vector<vector<vector<vector<double> > > > &aScatLen4dSubVec, double aReF0, double aImF0, double aD0, double aKStarMin, double aKStarMax, double aKStarVal)
{
  double tBinWidthK = (fScattLenInfo.minK-fScattLenInfo.maxK)/fScattLenInfo.nBinsK;
  int tLocalBinNumberK = GetBinNumber(tBinWidthK,aKStarMin,aKStarMax,aKStarVal);
  double tBinLowCenterK = GetInterpLowBinCenter(kScattLen,kKaxis,aKStarVal);

  double tBinWidthReF0 = (fScattLenInfo.minReF0-fScattLenInfo.maxReF0)/fScattLenInfo.nBinsReF0;
  double tBinLowCenterReF0 = GetInterpLowBinCenter(kScattLen,kReF0axis,aReF0);

  double tBinWidthImF0 = (fScattLenInfo.minImF0-fScattLenInfo.maxImF0)/fScattLenInfo.nBinsImF0;
  double tBinLowCenterImF0 = GetInterpLowBinCenter(kScattLen,kImF0axis,aImF0);

  double tBinWidthD0 = (fScattLenInfo.minD0-fScattLenInfo.maxD0)/fScattLenInfo.nBinsD0;
  double tBinLowCenterD0 = GetInterpLowBinCenter(kScattLen,kD0axis,aD0);

  //--------------------------------
  if(aKStarVal < tBinLowCenterK) tLocalBinNumberK--;
  int tLocalBinNumbersK[2] = {tLocalBinNumberK, tLocalBinNumberK+1};
  

  //--------------------------------
  double tDiffReF0 = (aReF0 - tBinLowCenterReF0)/tBinWidthReF0;
  double tDiffImF0 = (aImF0 - tBinLowCenterImF0)/tBinWidthImF0;
  double tDiffD0 = (aD0 - tBinLowCenterD0)/tBinWidthD0;
  double tDiffK = (aKStarVal - tBinLowCenterK)/tBinWidthK;

  //--------------------------------

  //Interpolate along ReF0
  double tC000 = aScatLen4dSubVec[0][0][0][tLocalBinNumbersK[0]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][0][0][tLocalBinNumbersK[0]]*tDiffReF0;
  double tC100 = aScatLen4dSubVec[0][1][0][tLocalBinNumbersK[0]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][1][0][tLocalBinNumbersK[0]]*tDiffReF0;

  double tC010 = aScatLen4dSubVec[0][0][1][tLocalBinNumbersK[0]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][0][1][tLocalBinNumbersK[0]]*tDiffReF0;
  double tC110 = aScatLen4dSubVec[0][1][1][tLocalBinNumbersK[0]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][1][1][tLocalBinNumbersK[0]]*tDiffReF0;

  double tC001 = aScatLen4dSubVec[0][0][0][tLocalBinNumbersK[1]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][0][0][tLocalBinNumbersK[1]]*tDiffReF0;
  double tC101 = aScatLen4dSubVec[0][1][0][tLocalBinNumbersK[1]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][1][0][tLocalBinNumbersK[1]]*tDiffReF0;

  double tC011 = aScatLen4dSubVec[0][0][1][tLocalBinNumbersK[1]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][0][1][tLocalBinNumbersK[1]]*tDiffReF0;
  double tC111 = aScatLen4dSubVec[0][1][1][tLocalBinNumbersK[1]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][1][1][tLocalBinNumbersK[1]]*tDiffReF0;


  //--------------------------------

  //Interpolate along ImF0
  double tC00 = tC000*(1.-tDiffImF0) + tC100*tDiffImF0;
  double tC10 = tC010*(1.-tDiffImF0) + tC110*tDiffImF0;
  double tC01 = tC001*(1.-tDiffImF0) + tC101*tDiffImF0;
  double tC11 = tC011*(1.-tDiffImF0) + tC111*tDiffImF0;

  //--------------------------------

  //Interpolate along D0
  double tC0 = tC00*(1.-tDiffD0) + tC10*tDiffD0;
  double tC1 = tC01*(1.-tDiffD0) + tC11*tDiffD0;

  //--------------------------------

  //Interpolate along KStar
  double tC = tC0*(1.-tDiffK) + tC1*tDiffK;

  return tC;
}


//________________________________________________________________________________________________________________
double CoulombFitterParallel::GetEta(double aKStar)
{
  return pow(((aKStar/hbarc)*fBohrRadius),-1);
}

//________________________________________________________________________________________________________________
double CoulombFitterParallel::GetGamowFactor(double aKStar)
{
  double tEta = GetEta(aKStar);
  tEta *= TMath::TwoPi();  //eta always comes with 2Pi here
  double tGamow = tEta*pow((TMath::Exp(tEta)-1),-1);
  return tGamow;
}


//________________________________________________________________________________________________________________
complex<double> CoulombFitterParallel::GetExpTerm(double aKStar, double aRStar, double aTheta)
{
  complex<double> tImI (0.,1.);
  complex<double> tReturnValue = exp(-tImI*(aKStar/hbarc)*aRStar*cos(aTheta));
  return tReturnValue;
}

//________________________________________________________________________________________________________________
complex<double> CoulombFitterParallel::BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0)
{
  double tLednickyHFunction = LinearInterpolate(fLednickyHFunctionHist, aKStar);
  double tImag = GetGamowFactor(aKStar)/(2.*GetEta(aKStar));
  complex<double> tLednickyChi (tLednickyHFunction,tImag);

  complex<double> tF0 (aReF0,aImF0);
  double tKStar = aKStar/hbarc;

  complex<double> tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - 2.*tLednickyChi/fBohrRadius,-1);

  return tScattAmp;
}


//________________________________________________________________________________________________________________
vector<double> CoulombFitterParallel::InterpolateWfSquaredSerial(vector<vector<double> > &aPairs, double aKStarMagMin, double aKStarMagMax, double aReF0, double aImF0, double aD0)
{
  vector<double> tReturnVector;
    tReturnVector.clear();
/*
  vector<int> tRelevantKbins = GetRelevantKStarBinNumbers(aKStarMagMin,aKStarMagMax);

  int tLowBinReF0 = GetInterpLowBin(kScattLen,kReF0axis,aReF0);
  int tLowBinImF0 = GetInterpLowBin(kScattLen,kImF0axis,aImF0);
  int tLowBinD0 = GetInterpLowBin(kScattLen,kD0axis,aD0);

  int tNbinsReF0 = 2;
  int tNbinsImF0 = 2;
  int tNbinsD0 = 2;
  int tNbinsK = tRelevantKbins.size();

  vector<int> tRelevantReF0Bins(2);
    tRelevantReF0Bins[0] = tLowBinReF0;
    tRelevantReF0Bins[1] = tLowBinReF0+1;

  vector<int> tRelevantImF0Bins(2);
    tRelevantImF0Bins[0] = tLowBinImF0;
    tRelevantImF0Bins[1] = tLowBinImF0+1;

  vector<int> tRelevantD0Bins(2);
    tRelevantD0Bins[0] = tLowBinD0;
    tRelevantD0Bins[1] = tLowBinD0+1;

  vector<vector<vector<vector<double> > > > tRelevantScattLengthReal;
  vector<vector<vector<vector<double> > > > tRelevantScattLengthImag;

  tRelevantScattLengthReal.resize(tNbinsReF0, td3dVec(tNbinsImF0, td2dVec(tNbinsD0, td1dVec(tNbinsK,0))));
  tRelevantScattLengthImag.resize(tNbinsReF0, td3dVec(tNbinsImF0, td2dVec(tNbinsD0, td1dVec(tNbinsK,0))));

  for(int iReF0=0; iReF0<2; iReF0++)
  {
    for(int iImF0=0; iImF0<2; iImF0++)
    {
      for(int iD0=0; iD0<2; iD0++)
      {
        for(int iK=0; iK<(int)tRelevantKbins.size(); iK++)
        {
          tRelevantScattLengthReal[iReF0][iImF0][iD0][iK] = fCoulombScatteringLengthReal[tRelevantReF0Bins[iReF0]][tRelevantImF0Bins[iImF0]][tRelevantD0Bins[iD0]][tRelevantKbins[iK]];
          tRelevantScattLengthImag[iReF0][iImF0][iD0][iK] = fCoulombScatteringLengthImag[tRelevantReF0Bins[iReF0]][tRelevantImF0Bins[iImF0]][tRelevantD0Bins[iD0]][tRelevantKbins[iK]];
        }
      }
    }
  }
*/
  //-----------------------------------------------

  double tGamow;
  complex<double> tExpTermComplex;
  double tHyperGeo1F1Real, tHyperGeo1F1Imag, tGTildeReal, tGTildeImag, tScattLenReal, tScattLenImag;
  complex<double> tScattLenComplexConj;

  double aKStarMag, aRStarMag, aTheta;

  for(int i=0; i<(int)aPairs.size(); i++)
  {
    aKStarMag = aPairs[i][0];
    aRStarMag = aPairs[i][1];
    aTheta = aPairs[i][2];

    tGamow = GetGamowFactor(aKStarMag);
    tExpTermComplex = GetExpTerm(aKStarMag,aRStarMag,aTheta);

    tHyperGeo1F1Real = TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
    tHyperGeo1F1Imag = TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

    tGTildeReal = BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
    tGTildeImag = BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);

    if(fUseScattLenHists)
    {

//      tScattLenReal = ScattLenInterpolate(tRelevantScattLengthReal,aReF0,aImF0,aD0,aKStarMagMin,aKStarMagMax,aKStarMag);
//      tScattLenImag = ScattLenInterpolate(tRelevantScattLengthImag,aReF0,aImF0,aD0,aKStarMagMin,aKStarMagMax,aKStarMag);


      if(aKStarMag >= 0. && aKStarMag < 0.2)
      {
//cout << "In 0-0.2" << endl;
//cout << "\taKStarMag = " << aKStarMag << "\taReF0 = " << aReF0 << "\taImF0 = " << aImF0 << "\taD0 = " << aD0 << endl;
        tScattLenReal = QuadrilinearInterpolate(fCoulombScatteringLengthRealHist1,aKStarMag,aReF0,aImF0,aD0);
        tScattLenImag = QuadrilinearInterpolate(fCoulombScatteringLengthImagHist1,aKStarMag,aReF0,aImF0,aD0);
      }
      else if(aKStarMag >= 0.2 && aKStarMag < 0.4)
      {
//cout << "In 0.2-0.4" << endl;
//cout << "\taKStarMag = " << aKStarMag << "\taReF0 = " << aReF0 << "\taImF0 = " << aImF0 << "\taD0 = " << aD0 << endl;
        tScattLenReal = QuadrilinearInterpolate(fCoulombScatteringLengthRealHist2,aKStarMag,aReF0,aImF0,aD0);
        tScattLenImag = QuadrilinearInterpolate(fCoulombScatteringLengthImagHist2,aKStarMag,aReF0,aImF0,aD0);
      }
      else
      {
        cout << "In CoulombFitter::InterpolateWfSquared, aKStarMag is outside of limits!!!!!!!!!!" << endl;
        assert(0);
      }
      tScattLenComplexConj = complex<double>(tScattLenReal,-tScattLenImag);
    }

    else
    {
      complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
      tScattLenComplexConj = std::conj(tScattLenComplex);
    }

    complex<double> tHyperGeo1F1Complex (tHyperGeo1F1Real,tHyperGeo1F1Imag);
    complex<double> tGTildeComplexConj (tGTildeReal,-tGTildeImag);

    complex<double> tResultComplex = tGamow*( norm(tHyperGeo1F1Complex) + norm(tScattLenComplexConj)*norm(tGTildeComplexConj)/(aRStarMag*aRStarMag) + 2.*real(tExpTermComplex*tHyperGeo1F1Complex*tScattLenComplexConj*tGTildeComplexConj/aRStarMag) );

    if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in CoulombFitterParallel::InterpolateWfSquared !!!!!" << endl;
    assert(imag(tResultComplex) < std::numeric_limits< double >::min());

    tReturnVector.push_back(real(tResultComplex));
  }


  return tReturnVector;

}

//end TODO***********************************************************************************************************************


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
vector<double> CoulombFitterParallel::InterpolateWfSquared(vector<vector<double> > &aPairs, double aReF0, double aImF0, double aD0)
{
/*
  vector<double> tReturnVector;
    tReturnVector.clear();

  tReturnVector = fParallelWaveFunction->RunInterpolateWfSquared(aPairs,aReF0,aImF0,aD0);
  return tReturnVector;
*/
  return fParallelWaveFunction->RunInterpolateWfSquared(aPairs,aReF0,aImF0,aD0);
}


//________________________________________________________________________________________________________________
bool CoulombFitterParallel::CanInterpKStar(double aKStar)
{
  if(aKStar < fMinInterpKStar1) return false;
  if(aKStar > fMaxInterpKStar2) return false;
  if(fUseScattLenHists && aKStar > fMaxInterpKStar1 && aKStar < fMinInterpKStar2) return false;  //awkward non-overlapping scenarios  TODO fix this in InterpolationHistograms.cxx
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::CanInterpRStar(double aRStar)
{
  if(aRStar < fMinInterpRStar || aRStar > fMaxInterpRStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::CanInterpTheta(double aTheta)
{
  if(aTheta < fMinInterpTheta || aTheta > fMaxInterpTheta) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::CanInterpReF0(double aReF0)
{
  if(aReF0 < fMinInterpReF0 || aReF0 > fMaxInterpReF0) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::CanInterpImF0(double aImF0)
{
  if(aImF0 < fMinInterpImF0 || aImF0 > fMaxInterpImF0) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::CanInterpD0(double aD0)
{
  if(aD0 < fMinInterpD0 || aD0 > fMaxInterpD0) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitterParallel::CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0)
{
  if(fUseScattLenHists)
  {
    if(CanInterpKStar(aKStar) && CanInterpRStar(aRStar) && CanInterpTheta(aTheta) && CanInterpReF0(aReF0) && CanInterpImF0(aImF0) && CanInterpD0(aD0)) return true;
  }
  else
  {
    if(CanInterpKStar(aKStar) && CanInterpRStar(aRStar) && CanInterpTheta(aTheta)) return true;
  }

  return false;
}
/*
//________________________________________________________________________________________________________________
td4dVec CoulombFitterParallel::Get3dPairs(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par)
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

  int tI;
  double tTheta, tKStarMag, tRStarMag;
  vector<double> tTempPair(3);

  int tNGood = 0;

  double tBinSize = (aKStarMagMax-aKStarMagMin)/aNbinsK;
  if(tBinSize != fPairKStar3dVecInfo.binWidthK) cout << "Imminent CRASH in CoulombFitterParallel::Get3dPairs due to enequal bin sizes!!!!!!!!" << endl;
  assert(tBinSize == fPairKStar3dVecInfo.binWidthK);  //Life is much easier when these are equal
                                                      //TODO in future, maybe make method to allow for unequal bin sizes

  for(int iBin=0; iBin<aNbinsK; iBin++)
  {
    tNGood=0;

    std::uniform_int_distribution<int> tRandomKStarElement(0.0, fPairKStar3dVec[iBin].size()-1);  //TODO could possible be drawing from multiple
                                                                                                  //iBins, depending on size
    while(tNGood<tMaxKStarCalls)
    {
      tI = tRandomKStarElement(generator);
      tKStar3Vec->SetXYZ(fPairKStar3dVec[iBin][tI][1],fPairKStar3dVec[iBin][tI][2],fPairKStar3dVec[iBin][tI][3]);
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
        tPairsGPU[iBin].push_back(tTempPair);
        tNGood++;
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
*/

//________________________________________________________________________________________________________________
td4dVec CoulombFitterParallel::Get3dPairs(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par)
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
    std::uniform_int_distribution<int> tRandomKStarElement(0.0, fPairKStar3dVec[iBin].size()-1);  //TODO could possible be drawing from multiple
                                                                                                  //iBins, depending on size
    while(tNGood<tMaxKStarCalls)
    {
      tI = tRandomKStarElement(generator);
      tKStar3Vec->SetXYZ(fPairKStar3dVec[iBin][tI][1],fPairKStar3dVec[iBin][tI][2],fPairKStar3dVec[iBin][tI][3]);
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
td1dVec CoulombFitterParallel::GetEntireFitCfContent(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par)
{

//ChronoTimer t4dVecTimer;
//t4dVecTimer.Start();

  td4dVec tPairs4d = Get3dPairs(aKStarMagMin,aKStarMagMax,aNbinsK,par);

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
    tReturnVec[i] = par[5]*(par[0]*tReturnVec[i] + (1.0-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  }

//CfCombineTimer.Stop();
//cout << "CfCombineTimer: ";
//CfCombineTimer.PrintInterval();

  return tReturnVec;

}


//________________________________________________________________________________________________________________
td1dVec CoulombFitterParallel::GetEntireFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par)
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

  td4dVec tPairs4d = Get3dPairs(aKStarMagMin,aKStarMagMax,aNbinsK,par);

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
double CoulombFitterParallel::GetFitCfContent(double aKStarMagMin, double aKStarMagMax, double *par)
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

  //KStarMag = fPairKStar3dVec[tBin][i][0]
  //KStarOut = fPairKStar3dVec[tBin][i][1]
  //KStarSide = fPairKStar3dVec[tBin][i][2]
  //KStarLong = fPairKStar3dVec[tBin][i][3]

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

  std::uniform_int_distribution<int> tRandomKStarElement(0.0, fPairKStar3dVec[tBin].size()-1);

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
    tKStar3Vec->SetXYZ(fPairKStar3dVec[tBin][tI][1],fPairKStar3dVec[tBin][tI][2],fPairKStar3dVec[tBin][tI][3]);
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
  vector<double> tGPUResults = InterpolateWfSquared(tPairsGPU,par[2],par[3],par[4]);

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
  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));

  return tReturnCfContent;
}

//TODO delete GetFitCfContentSerial
//________________________________________________________________________________________________________________
double CoulombFitterParallel::GetFitCfContentSerial(double aKStarMagMin, double aKStarMagMax, double *par)
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

  //KStarMag = fPairKStar3dVec[tBin][i][0]
  //KStarOut = fPairKStar3dVec[tBin][i][1]
  //KStarSide = fPairKStar3dVec[tBin][i][2]
  //KStarLong = fPairKStar3dVec[tBin][i][3]

  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tCounterGPU = 0;
  double tCfContentGPU = 0.;

  int tCounterCPU = 0;
  double tCfContentCPU = 0.;

//  int tMaxKStarCalls = 10000;
  int tMaxKStarCalls = 16384; //==2^14

  //Create the source Gaussians
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,par[1]);
  std::normal_distribution<double> tRSideSource(0.,par[1]);
  std::normal_distribution<double> tRLongSource(0.,par[1]);

  std::uniform_int_distribution<int> tRandomKStarElement(0.0, fPairKStar3dVec[tBin].size()-1);

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);

  int tI;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSq;
  complex<double> tWaveFunction;

  vector<vector<double> > tPairsGPU;
  vector<vector<double> > tPairsCPU;
  vector<double> tTempPair(3);

  int tNGood=0;
  while(tNGood<tMaxKStarCalls)
//  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tI = tRandomKStarElement(generator);
    tKStar3Vec->SetXYZ(fPairKStar3dVec[tBin][tI][1],fPairKStar3dVec[tBin][tI][2],fPairKStar3dVec[tBin][tI][3]);
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

  //---------Do serial calculations------------
  for(int i=0; i<(int)tPairsCPU.size(); i++)
  {
    tWaveFunction = fWaveFunction->GetWaveFunction(tPairsCPU[i][0],tPairsCPU[i][1],tPairsCPU[i][2],par[2],par[3],par[4]);
    tWaveFunctionSq = norm(tWaveFunction);

    tCfContentCPU += tWaveFunctionSq;
    tCounterCPU++;
  }

  //--------Do parallel calculations!----------
ChronoTimer Intv2Timer;
Intv2Timer.Start();

  vector<double> tGPUResultsSerial = InterpolateWfSquaredSerial(tPairsGPU,aKStarMagMin,aKStarMagMax,par[2],par[3],par[4]);

Intv2Timer.Stop();
cout << "Intv2Timer: ";
Intv2Timer.PrintInterval();

  for(int i=0; i<(int)tGPUResultsSerial.size(); i++)
  {
    tCfContentGPU += tGPUResultsSerial[i];
    tCounterGPU++;
  }

  tReturnCfContent = (tCfContentCPU + tCfContentGPU)/(tCounterCPU+tCounterGPU);
  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));

  return tReturnCfContent;
}


//________________________________________________________________________________________________________________
bool CoulombFitterParallel::AreParamsSame(double *aCurrent, double *aNew, int aNEntries)
{
  bool tAreSame = true;
  for(int i=0; i<aNEntries; i++)
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
void CoulombFitterParallel::CalculateChi2PML(int &npar, double &chi2, double *par)
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
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);
      if(!tAreParamsSame) tCfContentUnNorm = GetEntireFitCfContentComplete(0.,fMaxFitKStar,tNbinsXToFit,tPar);  //TODO include fMinFitKStar

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
void CoulombFitterParallel::CalculateChi2(int &npar, double &chi2, double *par)
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
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxis->GetBinLowEdge(ix);
        double tKStarMax = tXaxis->GetBinLowEdge(ix+1);

        double tCfContent = GetFitCfContent(tKStarMin,tKStarMax,tPar);

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
void CoulombFitterParallel::CalculateFakeChi2(int &npar, double &chi2, double *par)
{

  TAxis *tXaxis = fFakeCf->GetXaxis();
  double tChi2 = 0.;
  double tmp = 0.;
  int tNpfits = 0;

  double tKStarMin, tKStarMax;

  int tNbinsK = fFakeCf->GetNbinsX();

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

    tmp = (fFakeCf->GetBinContent(ix) - GetFitCfContent(tKStarMin,tKStarMax,par))/fFakeCf->GetBinError(ix);
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
TH1* CoulombFitterParallel::CreateFitHistogram(TString aName, int aAnalysisNumber)
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

    tCfContent = GetFitCfContent(tKStarMin,tKStarMax,tPar);
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
TH1* CoulombFitterParallel::CreateFitHistogramSample(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm)
{
  cout << "Beginning CreateFitHistogramSample" << endl;
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
  td1dVec tCf = GetEntireFitCfContent(aKMin,aKMax,aNbinsK,tPar);

CfParallelTimer.Stop();
cout << "CfParallelTimer in CreateFitHistogramSample: ";
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

    tCf2[ix-1] = GetFitCfContentSerial(tKStarMin,tKStarMax,tPar);
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
  cout << "Finished CreateFitHistogramSample  ";
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
TH1* CoulombFitterParallel::CreateFitHistogramSampleComplete(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double aNorm)
{
  cout << "Beginning CreateFitHistogramSampleComplete" << endl;
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

  assert(!fUseScattLenHists);

  double tBinSize = (aKMax-aKMin)/aNbinsK;
ChronoTimer CfParallelTimer;
CfParallelTimer.Start();

//  int tNPairsPerBin = 16384;
//  td1dVec tCfUnNorm = GetEntireFitCfContentComplete2(tNPairsPerBin,aKMin,aKMax,aNbinsK,tPar);
  td1dVec tCfUnNorm = GetEntireFitCfContentComplete(aKMin,aKMax,aNbinsK,tPar);

CfParallelTimer.Stop();
cout << "CfParallelTimer in CreateFitHistogramSampleComplete: ";
CfParallelTimer.PrintInterval();

 for(int i=0; i<aNbinsK; i++) tReturnHist->SetBinContent(i+1,tPar[8]*tCfUnNorm[i]);

tTimer.Stop();
  cout << "Finished CreateFitHistogramSampleComplete  ";
tTimer.PrintInterval();

  delete[] tPar;

  fFakeCf = tReturnHist;  //TODO delete this

  return tReturnHist;
}



//________________________________________________________________________________________________________________
void CoulombFitterParallel::DoFit()
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
//  fMinuit->mnexcm("MINI", arglist ,2,fErrFlg);  //MINI also uses MIGRAD, so probably won't work
  fMinuit->mnexcm("SIM", arglist ,2,fErrFlg);  //MINI also uses MIGRAD, so probably won't work
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
}


//________________________________________________________________________________________________________________


