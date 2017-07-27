///////////////////////////////////////////////////////////////////////////
// GeneralFitter:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "GeneralFitter.h"

#ifdef __ROOT__
ClassImp(GeneralFitter)
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
/*
GeneralFitter::GeneralFitter():
  fGeneralFitterType(kLedEq),
  fTurnOffCoulomb(false),
  fInterpHistsLoaded(false),
  fIncludeSingletAndTriplet(true),
  fUseRandomKStarVectors(false),
  fUseStaticPairs(false),

  fNCalls(0),
  fFakeCf(0),

  fAllOfSameCoulombType(false),
  fCoulombType(kRepulsive),
  fWaveFunction(0),
  fBohrRadius(gBohrRadiusXiK),

  fPairKStar4dVec(0),
  fNPairsPerKStarBin(16384),
  fCurrentRadiusParameter(1.),
  fPairSample4dVec(0),
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

  fNAnalyses=1;
  fUseRandomKStarVectors = true;
  SetUseStaticPairs(true,16384);

}
*/


//________________________________________________________________________________________________________________
GeneralFitter::GeneralFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar):
  LednickyFitter(aFitSharedAnalyses, aMaxFitKStar),
  fGeneralFitterType(kLedEq),
  fTurnOffCoulomb(false),
  fInterpHistsLoaded(false),
  fIncludeSingletAndTriplet(false),
  fUseRandomKStarVectors(false),
  fUseStaticPairs(false),

  fNCalls(0),
  fFakeCf(0),

  fAllOfSameCoulombType(false),
  fCoulombType(kRepulsive),
  fWaveFunction(0),
  fBohrRadius(gBohrRadiusXiK),

  fPairKStar4dVec(0),
  fNPairsPerKStarBin(16384),
  fCurrentRadiusParameter(1.),
  fPairSample4dVec(0),
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
  CheckIfAllOfSameCoulombType();

  omp_set_num_threads(3);

}


//________________________________________________________________________________________________________________
GeneralFitter::~GeneralFitter()
{
  cout << "GeneralFitter object is being deleted!!!" << endl;

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

}



//________________________________________________________________________________________________________________
CoulombType GeneralFitter::GetCoulombType(AnalysisType aAnalysisType)
{
//  assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
  if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) return kAttractive; //attractive
  else if(aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP) return kRepulsive; //repulsive
  else return kNeutral;
}


//________________________________________________________________________________________________________________
double GeneralFitter::GetBohrRadius(CoulombType aCoulombType)
{
  double tBohrRadius;
  if(aCoulombType == kAttractive) tBohrRadius = -gBohrRadiusXiK;
  else if(aCoulombType == kRepulsive) tBohrRadius = gBohrRadiusXiK;
  else if(aCoulombType == kNeutral) tBohrRadius = 1000000000;
  else
  {
    cout << "ERROR in GetBohrRadius:  Invalid fCoulombType selected" << endl;
    assert(0);
  }

  return tBohrRadius;
}


//________________________________________________________________________________________________________________
double GeneralFitter::GetBohrRadius(AnalysisType aAnalysisType)
{
  CoulombType tCoulombType = GetCoulombType(aAnalysisType);
  double tBohrRadius = GetBohrRadius(tCoulombType);
  return tBohrRadius;
}

//________________________________________________________________________________________________________________
void GeneralFitter::CheckIfAllOfSameCoulombType()
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
void GeneralFitter::LoadLednickyHFunctionFile(TString aFileBaseName)
{
  TString tFileName = aFileBaseName+".root";
  fInterpHistFileLednickyHFunction = TFile::Open(tFileName);

  fLednickyHFunctionHist = (TH1D*)fInterpHistFileLednickyHFunction->Get("LednickyHFunction");
//    fLednickyHFunctionHist->SetDirectory(0);

//  fInterpHistFileLednickyHFunction->Close();
}

//________________________________________________________________________________________________________________
void GeneralFitter::LoadInterpHistFile(TString aFileBaseName)
{
ChronoTimer tTimer(kSec);
tTimer.Start();
cout << "Starting LoadInterpHistFile" << endl;

//--------------------------------------------------------------

  LoadLednickyHFunctionFile();

  TString aFileName = aFileBaseName+".root";
  fInterpHistFile = TFile::Open(aFileName);
//--------------------------------------------------------------
  fHyperGeo1F1RealHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Real");
  fHyperGeo1F1ImagHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Imag");
//    fHyperGeo1F1RealHist->SetDirectory(0);
//    fHyperGeo1F1ImagHist->SetDirectory(0);

  fGTildeRealHist = (TH2D*)fInterpHistFile->Get("GTildeReal");
  fGTildeImagHist = (TH2D*)fInterpHistFile->Get("GTildeImag");
//    fGTildeRealHist->SetDirectory(0);
//    fGTildeImagHist->SetDirectory(0);

//  fInterpHistFile->Close();

  fMinInterpKStar = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(1);
  fMaxInterpKStar = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(fLednickyHFunctionHist->GetNbinsX());

  fMinInterpRStar = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(1);
  fMaxInterpRStar = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsY());

  fMinInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(1);
  fMaxInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsZ());

//--------------------------------------------------------------
  cout << "Interpolation histograms LOADED!" << endl;

tTimer.Stop();
cout << "LoadInterpHistFile: ";
tTimer.PrintInterval();

  fInterpHistsLoaded = true;
}


//________________________________________________________________________________________________________________
int GeneralFitter::GetBinNumber(double aBinSize, int aNbins, double aValue)
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
int GeneralFitter::GetBinNumber(int aNbins, double aMin, double aMax, double aValue)
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
int GeneralFitter::GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue)
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
void GeneralFitter::ExtractPairKStar3dVecFromSingleFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill)
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
td3dVec GeneralFitter::BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  cout << "Beginning FULL conversion of TNtuple to 3dVec" << endl;
  std::clock_t start = std::clock();

  //----****----****----****----****----****----****
  //TODO make this automated
  //TODO add centrality selection to this
  //Also, this is basically FitPartialAnalysis::ConnectAnalysisDirectory

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
  tPairKStar3dVec.resize(aNbinsKStar, td2dVec(0,td1dVec(0)));

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

  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Conversion complete in " << duration << " seconds" << endl;


  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<aNbinsKStar; i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

  return tPairKStar3dVec;
}

//________________________________________________________________________________________________________________
void GeneralFitter::WriteRow(ostream &aOutput, vector<double> &aRow)
{
  for(int i = 0; i < (int)aRow.size(); i++)
  {
    if( i < (int)aRow.size()-1) aOutput << aRow[i] << " ";
    else if(i == (int)aRow.size()-1) aOutput << aRow[i] << endl;
    else cout << "SOMETHING IS WRONG!!!!!\n";
  }
}



//________________________________________________________________________________________________________________
void GeneralFitter::WritePairKStar3dVecFile(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  TString tOutputName = aOutputBaseName + TString(cAnalysisBaseTags[aAnalysisType])+TString(cCentralityTags[aCentralityType]) + TString(".txt");
  ofstream tFileOut(tOutputName);

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
void GeneralFitter::WriteAllPairKStar3dVecFiles(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
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
td3dVec GeneralFitter::BuildPairKStar3dVecFromTxt(TString aFileName)
{
ChronoTimer tTimer;
tTimer.Start();

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
      tNbinsKStarNeeded = fMaxFitKStar/aBinWidth;
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

  //Add the binning information for use by GeneralFitterParallel::BuildPairKStar3dVecFromTxt
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
void GeneralFitter::BuildPairKStar4dVecFromTxt(TString aFileBaseName)
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
void GeneralFitter::BuildPairKStar4dVecOnFly(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
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
void GeneralFitter::BuildPairSample4dVec(int aNPairsPerKStarBin)
{
ChronoTimer tTimer(kSec);
tTimer.Start();

  fNPairsPerKStarBin = aNPairsPerKStarBin;
  fPairSample4dVec.resize(fNAnalyses, td3dVec(0, td2dVec(0, td1dVec(0))));

  double tBinSize = 0.01;  //TODO make this automated
  int tNBinsKStar = fMaxFitKStar/tBinSize;  //TODO make this general, ie subtract 1 if fMaxFitKStar is on bin edge (maybe, maybe not bx of iKStarBin<tNBinsKStar)

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
  td3dVec tTemp3dVec;

  //------------------------------------
//TODO Check randomization
//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    tTemp3dVec.clear();
    for(int iKStarBin=0; iKStarBin<tNBinsKStar; iKStarBin++)
    {
      if(!fUseRandomKStarVectors) tRandomKStarElement = std::uniform_int_distribution<int>(0.0, fPairKStar4dVec[iAnaly][iKStarBin].size()-1);
      tKStarMagMin = iKStarBin*tBinSize;
      tKStarMagMax = (iKStarBin+1)*tBinSize;
      tTemp2dVec.clear();
      for(int iPair=0; iPair<fNPairsPerKStarBin; iPair++)
      {
        if(!fUseRandomKStarVectors)
        {
          tI = tRandomKStarElement(generator);
          tKStar3Vec->SetXYZ(fPairKStar4dVec[iAnaly][iKStarBin][tI][1],fPairKStar4dVec[iAnaly][iKStarBin][tI][2],fPairKStar4dVec[iAnaly][iKStarBin][tI][3]);
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
      tTemp3dVec.push_back(tTemp2dVec);
    }
    fPairSample4dVec[iAnaly] = tTemp3dVec;
  }

  delete tKStar3Vec;
  delete tSource3Vec;


tTimer.Stop();
cout << "BuildPairSample4dVec finished: ";
tTimer.PrintInterval();

}

//________________________________________________________________________________________________________________
void GeneralFitter::UpdatePairRadiusParameters(double aNewRadius)
{
  //TODO allow for change of MU also, not just SIGMA!
  double tScaleFactor = aNewRadius/fCurrentRadiusParameter;
  fCurrentRadiusParameter = aNewRadius;

//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    for(int iKStarBin=0; iKStarBin<(int)fPairSample4dVec[iAnaly].size(); iKStarBin++)
    {
      for(int iPair=0; iPair<(int)fPairSample4dVec[iAnaly][iKStarBin].size(); iPair++)
      {
        fPairSample4dVec[iAnaly][iKStarBin][iPair][1] *= tScaleFactor;
      }
    }
  }

}

//________________________________________________________________________________________________________________
void GeneralFitter::SetUseStaticPairs(bool aUseStaticPairs, int aNPairsPerKStarBin)
{
  fUseStaticPairs = aUseStaticPairs;
  BuildPairSample4dVec(aNPairsPerKStarBin);
}

//________________________________________________________________________________________________________________
void GeneralFitter::SetIncludeResidualCorrelations(bool aInclude)
{
  fIncludeResidualCorrelations = aInclude;
  SetUseStaticPairs(true,16384);
}


//________________________________________________________________________________________________________________
double GeneralFitter::LinearInterpolate(TH1* a1dHisto, double aX)
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
double GeneralFitter::BilinearInterpolate(TH2* a2dHisto, double aX, double aY)
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
    cout << "Error in GeneralFitter::BilinearInterpolate, cannot interpolate outside histogram domain" << endl;
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
double GeneralFitter::BilinearInterpolateVector(vector<vector<double> > &a2dVec, double aX, int aNbinsX, double aMinX, double aMaxX, double aY, int aNbinsY, double aMinY, double aMaxY)
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
    cout << "Error in GeneralFitter::BilinearInterpolateVector, cannot interpolate outside histogram domain" << endl;
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
double GeneralFitter::TrilinearInterpolate(TH3* a3dHisto, double aX, double aY, double aZ)
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
    cout << "Error in GeneralFitter::TrilinearInterpolate, cannot interpolate outside histogram domain" << endl;

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
double GeneralFitter::QuadrilinearInterpolate(THn* a4dHisto, double aT, double aX, double aY, double aZ)
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
    cout << "Error in GeneralFitter::QuadrilinearInterpolate, cannot interpolate outside histogram domain" << endl;
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
double GeneralFitter::GetEta(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else return pow(((aKStar/hbarc)*fBohrRadius),-1);
}

//________________________________________________________________________________________________________________
double GeneralFitter::GetGamowFactor(double aKStar)
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
complex<double> GeneralFitter::GetExpTerm(double aKStar, double aRStar, double aTheta)
{
  complex<double> tReturnValue = exp(-ImI*(aKStar/hbarc)*aRStar*cos(aTheta));
  return tReturnValue;
}

//________________________________________________________________________________________________________________
complex<double> GeneralFitter::BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0)
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
double GeneralFitter::InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
{

  bool tDebug = true; //debug means use personal interpolation methods, instead of standard root ones

  //TODO put check to make sure file is open, not sure if assert(fInterpHistFile->IsOpen works);
  assert(fInterpHistsLoaded);
  assert(fInterpHistFile->IsOpen());

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

  if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in GeneralFitter::InterpolateWfSquared !!!!!" << endl;
  assert(imag(tResultComplex) < std::numeric_limits< double >::min());

  return real(tResultComplex);

}

//________________________________________________________________________________________________________________
vector<double> GeneralFitter::InterpolateWfSquaredSerialv2(vector<vector<double> > &aPairs, double aKStarMagMin, double aKStarMagMax, double aReF0, double aImF0, double aD0)
{
  vector<double> tReturnVector;
    tReturnVector.clear();

  vector<vector<vector<vector<double> > > > tRelevantScattLengthReal;
  vector<vector<vector<vector<double> > > > tRelevantScattLengthImag;

  //-----------------------------------------------

  double tGamow;
  complex<double> tExpTermComplex;
  
  complex<double> tScattLenComplexConj;

  complex<double> tHyperGeo1F1Complex;
  complex<double> tGTildeComplexConj;

  double aKStarMag, aRStarMag, aTheta;

  for(int i=0; i<(int)aPairs.size(); i++)
  {
    aKStarMag = aPairs[i][0];
    aRStarMag = aPairs[i][1];
    aTheta = aPairs[i][2];

    tGamow = GetGamowFactor(aKStarMag);
    tExpTermComplex = GetExpTerm(aKStarMag,aRStarMag,aTheta);

    if(!fTurnOffCoulomb)
    {
      double tHyperGeo1F1Real, tHyperGeo1F1Imag, tGTildeReal, tGTildeImag;

      tHyperGeo1F1Real = TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

      tGTildeReal = BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
      tGTildeImag = BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);

      tHyperGeo1F1Complex = complex<double> (tHyperGeo1F1Real,tHyperGeo1F1Imag);
      tGTildeComplexConj = complex<double> (tGTildeReal,-tGTildeImag);

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

    complex<double> tResultComplex = tGamow*( norm(tHyperGeo1F1Complex) + norm(tScattLenComplexConj)*norm(tGTildeComplexConj)/(aRStarMag*aRStarMag) + 2.*real(tExpTermComplex*tHyperGeo1F1Complex*tScattLenComplexConj*tGTildeComplexConj/aRStarMag) );

    if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in GeneralFitter::InterpolateWfSquaredSerialv2 !!!!!" << endl;
    assert(imag(tResultComplex) < std::numeric_limits< double >::min());

    tReturnVector.push_back(real(tResultComplex));
  }

  return tReturnVector;
}


//________________________________________________________________________________________________________________
bool GeneralFitter::CanInterpKStar(double aKStar)
{
  if(aKStar < fMinInterpKStar) return false;
  if(aKStar > fMaxInterpKStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool GeneralFitter::CanInterpRStar(double aRStar)
{
  if(aRStar < fMinInterpRStar || aRStar > fMaxInterpRStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool GeneralFitter::CanInterpTheta(double aTheta)
{
  if(aTheta < fMinInterpTheta || aTheta > fMaxInterpTheta) return false;
  return true;
}



//________________________________________________________________________________________________________________
bool GeneralFitter::CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0)
{
  if(CanInterpKStar(aKStar) && CanInterpRStar(aRStar) && CanInterpTheta(aTheta)) return true;
  return false;
}

//________________________________________________________________________________________________________________
void GeneralFitter::SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax)
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
double GeneralFitter::GetFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
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
//cout << "fPairKStar4dVec[aAnalysisNumber][tBin].size() = " << fPairKStar4dVec[aAnalysisNumber][tBin].size() << endl << endl;

  int tMaxKStarCalls;

//  if(fPairKStar4dVec[aAnalysisNumber][tBin].size() < 100) tMaxKStarCalls = fPairKStar4dVec[aAnalysisNumber][tBin].size();
//  else tMaxKStarCalls = 100;

  //definitely oversampling by commenting out the above
  //Currently, lowest bin only have 6 entries!
  tMaxKStarCalls = 16384;
//  if(fPairKStar4dVec[aAnalysisNumber][tBin].size() < tMaxKStarCalls) tMaxKStarCalls = fPairKStar4dVec[aAnalysisNumber][tBin].size();

  //Create the source Gaussians
  double tRoot2 = sqrt(2.);
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,tRoot2*par[1]);
  std::normal_distribution<double> tRSideSource(0.,tRoot2*par[1]);
  std::normal_distribution<double> tRLongSource(0.,tRoot2*par[1]);

  std::uniform_int_distribution<int> tRandomKStarElement;
  if(!fUseRandomKStarVectors) tRandomKStarElement = std::uniform_int_distribution<int>(0.0, fPairKStar4dVec[aAnalysisNumber][tBin].size()-1);

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);

  int tI;
  bool tCanInterp;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq;
  complex<double> tWaveFunctionSinglet, tWaveFunctionTriplet;

  vector<vector<double> > tMathematicaPairs;
  vector<double> tTempPair(3);

  int tNInterpolate = 0;
  int tNMathematica = 0;

//ChronoTimer tIntTimer;
//tIntTimer.Start();

  #pragma omp parallel for reduction(+: tCounter) reduction(+: tReturnCfContent) private(tI, tCanInterp, tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq) firstprivate(tKStar3Vec, tSource3Vec, tTempPair)
  for(int i=0; i<tMaxKStarCalls; i++)

//  for(int i=0; i<fPairKStar4dVec[aAnalysisNumber][tBin].size(); i++)
  {
//TODO Check randomization
//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
    if(!fUseRandomKStarVectors)
    {
      tI = tRandomKStarElement(generator);
      tKStar3Vec->SetXYZ(fPairKStar4dVec[aAnalysisNumber][tBin][tI][1],fPairKStar4dVec[aAnalysisNumber][tBin][tI][2],fPairKStar4dVec[aAnalysisNumber][tBin][tI][3]);
    }
    else SetRandomKStar3Vec(tKStar3Vec,aKStarMagMin,aKStarMagMax);

    tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric

    tTheta = tKStar3Vec->Angle(*tSource3Vec);
    tKStarMag = tKStar3Vec->Mag();
    tRStarMag = tSource3Vec->Mag();


//TODO make 1 and 2 histograms overlap by a bin or two!
//TODO
//TODO
//TODO!!!!!!!!!!!!! Must have separate InterpolateWfSquaredXiKchP and InterpolateWfSquaredXiKchM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //Note:  Running CanInterpAll is faster than checking CanInterpScatParams outside of for loop and comparing to CanInterpKRTh.....strange but true
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

    #pragma omp critical
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

  delete tKStar3Vec;
  delete tSource3Vec;

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
double GeneralFitter::GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
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
  int tBin = aKStarMagMin/tBinSize;

  //KStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0]
  //RStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1]
  //Theta    = fPairSample4dVec[aAnalysisNumber][tBin][i][2]


  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;
  tMaxKStarCalls = fPairSample4dVec[aAnalysisNumber][tBin].size();
//cout << "In GetFitCfContentCompletewStaticPairs: " << endl;
//cout << "\taKStarMagMin = " << aKStarMagMin << " and tBin = " << tBin << endl;
//cout << "\ttMaxKStarCalls = " << tMaxKStarCalls << endl;


  bool tCanInterp;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq;
  complex<double> tWaveFunctionSinglet, tWaveFunctionTriplet;

  vector<vector<double> > tMathematicaPairs;
  vector<double> tTempPair(3);

  int tNInterpolate = 0;
  int tNMathematica = 0;

//ChronoTimer tIntTimer;
//tIntTimer.Start();

  #pragma omp parallel for reduction(+: tCounter) reduction(+: tReturnCfContent) private(tCanInterp, tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq) firstprivate(tTempPair)
  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tKStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0];
    tRStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1];
    tTheta = fPairSample4dVec[aAnalysisNumber][tBin][i][2];

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

    #pragma omp critical
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
double GeneralFitter::GetFitCfContentSerialv2(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
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

  double tReturnCfContent = 0.;

  int tCounterGPU = 0;
  double tCfContentGPU = 0.;

  int tCounterCPU = 0;
  double tCfContentCPU = 0.;

//  int tMaxKStarCalls = 10000;
  int tMaxKStarCalls = 16384; //==2^14

  //Create the source Gaussians
  double tRoot2 = sqrt(2.);
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,tRoot2*par[1]);
  std::normal_distribution<double> tRSideSource(0.,tRoot2*par[1]);
  std::normal_distribution<double> tRLongSource(0.,tRoot2*par[1]);

  std::uniform_int_distribution<int> tRandomKStarElement;
  if(!fUseRandomKStarVectors) tRandomKStarElement = std::uniform_int_distribution<int>(0.0, fPairKStar4dVec[aAnalysisNumber][tBin].size()-1);

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
    if(!fUseRandomKStarVectors)
    {
      tI = tRandomKStarElement(generator);
      tKStar3Vec->SetXYZ(fPairKStar4dVec[aAnalysisNumber][tBin][tI][1],fPairKStar4dVec[aAnalysisNumber][tBin][tI][2],fPairKStar4dVec[aAnalysisNumber][tBin][tI][3]);
    }
    else SetRandomKStar3Vec(tKStar3Vec,aKStarMagMin,aKStarMagMax);
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
  vector<double> tGPUResults = InterpolateWfSquaredSerialv2(tPairsGPU,aKStarMagMin,aKStarMagMax,par[2],par[3],par[4]);
  for(int i=0; i<(int)tGPUResults.size(); i++)
  {
    tCfContentGPU += tGPUResults[i];
    tCounterGPU++;
  }

  tReturnCfContent = (tCfContentCPU + tCfContentGPU)/(tCounterCPU+tCounterGPU);
//  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));

  return tReturnCfContent;
}


//________________________________________________________________________________________________________________
td1dVec GeneralFitter::GetCoulombResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams, vector<double> &aKStarBinCenters, TH2* aTransformMatrix)
{
  TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/";
  TString tFileName, tFullFileName;
  TString tFileNameHFunction, tFullFileNameHFunction;

  switch(aResidualType) {
  case kResXiCKchP:
  case kResAXiCKchM:
    tFileName = TString("InterpHistsAttractive");
    tFileNameHFunction = TString("LednickyHFunction");
    fBohrRadius = -gBohrRadiusXiK;
    break;

  case kResXiCKchM:
  case kResAXiCKchP:
    tFileName = TString("InterpHistsRepulsive");
    tFileNameHFunction = TString("LednickyHFunction");
    fBohrRadius = gBohrRadiusXiK;
    break;


  case kResOmegaKchP:
  case kResAOmegaKchM:
    tFileName = TString("InterpHists_OmegaKchP");
    tFileNameHFunction = TString("LednickyHFunction_OmegaKchP");
    fBohrRadius = -gBohrRadiusOmegaK;
    break;


  case kResOmegaKchM:
  case kResAOmegaKchP:
    tFileName = TString("InterpHists_OmegaKchM");
    tFileNameHFunction = TString("LednickyHFunction_OmegaKchM");
    fBohrRadius = gBohrRadiusOmegaK;
    break;


  default:
    cout << "ERROR: GeneralFitter::GetCoulombResidualCorrelation: Invalid aResidualType = " << aResidualType << endl;
  }

  tFullFileName = tFileLocationBase + tFileName;
  tFullFileNameHFunction = tFileLocationBase + tFileNameHFunction;

  fWaveFunction->SetCurrentBohrRadius(aResidualType);
  LoadInterpHistFile(tFullFileName);
  LoadLednickyHFunctionFile(tFullFileNameHFunction);

  double tKStarBinWidth = aKStarBinCenters[1]-aKStarBinCenters[0];
  int tAnalysisNumber = 0;  //TODO

  vector<double> tParentCf(aKStarBinCenters.size(),0.);
  double tKStarMin, tKStarMax;
  for(unsigned int i=0; i<aKStarBinCenters.size(); i++)
  {
    tKStarMin = aKStarBinCenters[0]-tKStarBinWidth/2.;
    tKStarMax = aKStarBinCenters[0]+tKStarBinWidth/2.;

    tParentCf[i] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,aParentCfParams,tAnalysisNumber);
  }

  unsigned int tDaughterPairKStarBin, tParentPairKStarBin;
  double tDaughterPairKStar, tParentPairKStar;
  assert(tParentCf.size() == aKStarBinCenters.size());
  assert(tParentCf.size() == (unsigned int)aTransformMatrix->GetNbinsX());
  assert(tParentCf.size() == (unsigned int)aTransformMatrix->GetNbinsY());

  vector<double> tReturnResCf(tParentCf.size(),0.);
  vector<double> tNormVec(tParentCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

  for(unsigned int i=0; i<tParentCf.size(); i++)
  {
    tDaughterPairKStar = aKStarBinCenters[i];
    tDaughterPairKStarBin = aTransformMatrix->GetXaxis()->FindBin(tDaughterPairKStar);

    for(unsigned int j=0; j<tParentCf.size(); j++)
    {
      tParentPairKStar = aKStarBinCenters[j];
      tParentPairKStarBin = aTransformMatrix->GetYaxis()->FindBin(tParentPairKStar);

      tReturnResCf[i] += tParentCf[j]*aTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
      tNormVec[i] += aTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
    }
    tReturnResCf[i] /= tNormVec[i];
  }
  return tReturnResCf;
}



//________________________________________________________________________________________________________________
void GeneralFitter::CalculateFitFunction(int &npar, double &chi2, double *par)
{
  if(fVerbose) PrintCurrentParamValues(npar,par);
  //---------------------------------------------------------
  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  int tNFitParPerAnalysis;
  if(fIncludeSingletAndTriplet) tNFitParPerAnalysis = 8;
  else tNFitParPerAnalysis = 5;

  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  int tNbinsXToFitGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar);
  if(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetBinLowEdge(tNbinsXToFitGlobal) == fMaxFitKStar) tNbinsXToFitGlobal--;


  int tNbinsXToBuildMomResCrctn=0, tNbinsXToBuildResiduals=0;
  int tNbinsXToBuildGlobal;  // when applying momentum resolution corrections, many times you must go beyond fitting range to apply correction
  if(fApplyMomResCorrection) tNbinsXToBuildMomResCrctn = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetModelKStarTrueVsRecMixed()->GetNbinsX();
  if(fIncludeResidualCorrelations) tNbinsXToBuildResiduals = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetTransformMatrices()[0]->GetNbinsX();
  tNbinsXToBuildGlobal = std::max({tNbinsXToBuildMomResCrctn, tNbinsXToBuildResiduals, tNbinsXToFitGlobal});

//  vector<double> tPrimaryFitCfContentUnNorm(tNbinsXToBuildGlobal,0.);
  vector<double> tPrimaryFitCfContent(tNbinsXToBuildGlobal,0.);
  vector<double> tNumContent(tNbinsXToBuildGlobal,0.);
  vector<double> tDenContent(tNbinsXToBuildGlobal,0.);
  vector<double> tKStarBinCenters(tNbinsXToBuildGlobal,0.);

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

    if(!fAllOfSameCoulombType)
    {
      //assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
      if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
      else if(tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
      else fBohrRadius = 1000000000;
      fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
    }

    int tNbinsXToBuild;
    TH2* tMomResMatrix = NULL;
    if(fApplyMomResCorrection)
    {
      tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
      assert(tMomResMatrix);
      tNbinsXToBuild = tMomResMatrix->GetNbinsX();
      assert(tNbinsXToBuild == tNbinsXToBuildGlobal);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();
      TH1* tCf = tKStarCfLite->Cf();

      assert(tNum->GetXaxis()->GetBinWidth(1) == tDen->GetXaxis()->GetBinWidth(1));
      assert(tNum->GetXaxis()->GetBinWidth(1) == tCf->GetXaxis()->GetBinWidth(1));
      //make sure tNum and tDen and tCf have same bin size as tMomResMatrix
      if(fApplyMomResCorrection)
      {
        assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetXaxis()->GetBinWidth(1));
        assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetYaxis()->GetBinWidth(1));
      }
      //make sure tNum and tDen and tCf have same bin size as residuals
      if(fIncludeResidualCorrelations)
      {
        assert(tNum->GetXaxis()->GetBinWidth(1) == tFitPairAnalysis->GetTransformMatrices()[0]->GetXaxis()->GetBinWidth(1));
        assert(tNum->GetXaxis()->GetBinWidth(1) == tFitPairAnalysis->GetTransformMatrices()[0]->GetYaxis()->GetBinWidth(1));
      }

      //make sure tNum and tDen have same number of bins
      assert(tNum->GetNbinsX() == tDen->GetNbinsX());

      TAxis* tXaxisNum = tNum->GetXaxis();

      int tNbinsX = tNum->GetNbinsX();

      int tNbinsXToFit = tNum->FindBin(fMaxFitKStar);
      if(tNum->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;

      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tNbinsXToFit == tNbinsXToFitGlobal);
      int tNFitParams = tFitPartialAnalysis->GetNFitParams() +1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams = tNFitParPerAnalysis+1);

      int tLambdaMinuitParamNumber, tRadiusMinuitParamNumber, tRef0MinuitParamNumber, tImf0MinuitParamNumber, td0MinuitParamNumber, tNormMinuitParamNumber;
      int tRef02MinuitParamNumber, tImf02MinuitParamNumber, td02MinuitParamNumber;
      double *tPar;

      tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      if(!fIncludeSingletAndTriplet)
      {
        assert(tNFitParams == 6);
        tPar = new double[tNFitParams];

        tPar[0] = par[tLambdaMinuitParamNumber];
        tPar[1] = par[tRadiusMinuitParamNumber];
        tPar[2] = par[tRef0MinuitParamNumber];
        tPar[3] = par[tImf0MinuitParamNumber];
        tPar[4] = par[td0MinuitParamNumber];
        tPar[5] = par[tNormMinuitParamNumber];
      }

      else
      {
        assert(tNFitParams == 9);
        tPar = new double[tNFitParams];

        tRef02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef02)->GetMinuitParamNumber();
        tImf02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf02)->GetMinuitParamNumber();
        td02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd02)->GetMinuitParamNumber();

        tPar[0] = par[tLambdaMinuitParamNumber];
        tPar[1] = par[tRadiusMinuitParamNumber];
        tPar[2] = par[tRef0MinuitParamNumber];
        tPar[3] = par[tImf0MinuitParamNumber];
        tPar[4] = par[td0MinuitParamNumber];
        tPar[5] = par[tRef02MinuitParamNumber];
        tPar[6] = par[tImf02MinuitParamNumber];
        tPar[7] = par[td02MinuitParamNumber];
        tPar[8] = par[tNormMinuitParamNumber];
      }

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateFitFunction, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double x[1];
      bool tRejectOmega = tFitPartialAnalysis->RejectOmega();
      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);

      if(!fApplyMomResCorrection && !fIncludeResidualCorrelations) tNbinsXToBuild = tNbinsXToFit;
      double tKStarMin, tKStarMax;
      for(int ix=1; ix <= tNbinsXToBuild; ix++)
      {
        tKStarBinCenters[ix-1] = tXaxisNum->GetBinCenter(ix);
        x[0] = tKStarBinCenters[ix-1];

        tNumContent[ix-1] = tNum->GetBinContent(ix);
        tDenContent[ix-1] = tDen->GetBinContent(ix);

        tKStarMin = tXaxisNum->GetBinLowEdge(ix);
        tKStarMax = tXaxisNum->GetBinLowEdge(ix+1);

        if(fGeneralFitterType==kLedEq) tPrimaryFitCfContent[ix-1] = LednickyEq(x,tPar);
        else
        {
          if(!tAreParamsSame)
          {
            if(fUseStaticPairs) tPrimaryFitCfContent[ix-1] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
            else tPrimaryFitCfContent[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
          }
        }
      }

      vector<double> tFitCfContent;
      if(fIncludeResidualCorrelations) 
      {
        double tLambda_SigK = 0.78*tPar[0];  //for now, primary lambda scaled by some factor
        double *tPar_SigK = AdjustLambdaParam(tPar,tLambda_SigK,tNFitParams);
        td1dVec tResidual_SigK = GetNeutralResidualCorrelation(tPar_SigK,tKStarBinCenters,tFitPairAnalysis->GetTransformMatrices()[0]);

        double tLambda_Xi0K = 0.52*tPar[0];  //for now, primary lambda scaled by some factor
        double *tPar_Xi0K = AdjustLambdaParam(tPar,tLambda_Xi0K,tNFitParams);
        td1dVec tResidual_Xi0K = GetNeutralResidualCorrelation(tPar_Xi0K,tKStarBinCenters,tFitPairAnalysis->GetTransformMatrices()[2]);

        double tLambda_XiCK = 0.52*tPar[0];  //for now, primary lambda scaled by some factor
        double *tPar_XiCK = AdjustLambdaParam(tPar,tLambda_XiCK,tNFitParams);
        td1dVec tResidual_XiCK = GetCoulombResidualCorrelation(kResXiCKchP,tPar_XiCK,tKStarBinCenters,tFitPairAnalysis->GetTransformMatrices()[1]);  //TODO

        double tLambda_OmegaK = 0.02*tPar[0];  //for now, primary lambda scaled by some factor
        double *tPar_OmegaK = AdjustLambdaParam(tPar,tLambda_OmegaK,tNFitParams);
        td1dVec tResidual_OmegaK = GetCoulombResidualCorrelation(kResOmegaKchP,tPar_OmegaK,tKStarBinCenters,tFitPairAnalysis->GetTransformMatrices()[3]);  //TODO

        vector<double> tLambdas{tPar[0],tLambda_SigK,tLambda_Xi0K,tLambda_XiCK,tLambda_OmegaK};
        td2dVec tAllCfs{tPrimaryFitCfContent,tResidual_SigK,tResidual_Xi0K,tResidual_XiCK,tResidual_OmegaK};
        tFitCfContent = CombinePrimaryWithResiduals(tLambdas, tAllCfs);

        delete[] tPar_SigK;
        delete[] tPar_Xi0K;
      }
      else tFitCfContent = tPrimaryFitCfContent;

      vector<double> tCorrectedFitCfContent;
      if(fApplyMomResCorrection) tCorrectedFitCfContent = ApplyMomResCorrection(tFitCfContent, tKStarBinCenters, tMomResMatrix);
      else tCorrectedFitCfContent = tFitCfContent;

      if(fApplyNonFlatBackgroundCorrection)
      {
        TF1* tNonFlatBgd = tFitPartialAnalysis->GetNonFlatBackground(kLinear/*,0.40,0.90*/);
        ApplyNonFlatBackgroundCorrection(tCorrectedFitCfContent, tKStarBinCenters, tNonFlatBgd);
      }

      ApplyNormalization(tPar[5], tCorrectedFitCfContent);

      for(int ix=0; ix < tNbinsXToFit; ix++)
      {
        if(tRejectOmega && (tKStarBinCenters[ix] > tRejectOmegaLow) && (tKStarBinCenters[ix] < tRejectOmegaHigh)) {fChi2+=0;}
        else
        {
          if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tCorrectedFitCfContent[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
          {
            double tChi2 = 0.;
            if(fFitSharedAnalyses->GetFitType() == kChi2PML) tChi2 = GetPmlValue(tNumContent[ix],tDenContent[ix],tCorrectedFitCfContent[ix]);
            else if(fFitSharedAnalyses->GetFitType() == kChi2) tChi2 = GetChi2Value(ix+1,tCf,tPar);
            else tChi2 = 0.;

            fChi2Vec[iAnaly] += tChi2;
            fChi2 += tChi2;

            fNpFitsVec[iAnaly]++;
            fNpFits++;
          }
        }
      }

      delete[] tPar;
    }

  }

  delete[] tCurrentFitPar;

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
/*
  double *tParamsForHistograms = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tParamsForHistograms[i] = par[i];
  fFitSharedAnalyses->GetFitChi2Histograms()->FillHistograms(fChi2,tParamsForHistograms);
  delete[] tParamsForHistograms;
*/
}





//________________________________________________________________________________________________________________
void GeneralFitter::CalculateFakeChi2(int &npar, double &chi2, double *par)
{
  fNCalls++;
  cout << "\tfNCalls = " << fNCalls << endl;

  cout << "\t\tParameter update: " << endl;
  cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
  cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

  cout << "\t\t\tpar[2] = ReF0  = " << par[2] << endl;
  cout << "\t\t\tpar[3] = ImF0  = " << par[3] << endl;
  cout << "\t\t\tpar[4] = D0    = " << par[4] << endl;


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

    if(fUseStaticPairs) tmp = (fFakeCf->GetBinContent(ix) - par[8]*GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    else tmp = (fFakeCf->GetBinContent(ix) - par[8]*GetFitCfContentComplete(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    tChi2 += tmp*tmp;
    tNpfits++;
  }

  chi2 = tChi2;
cout << "tChi2 = " << tChi2 << endl << endl;
}




//________________________________________________________________________________________________________________
TH1* GeneralFitter::CreateFitHistogram(TString aName, int aAnalysisNumber)
{
  FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

  if(!fAllOfSameCoulombType)
  {
    //assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
    if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else if(tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
    else fBohrRadius = 1000000000;
    fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
  }

  int tNFitParams = tFitPairAnalysis->GetNFitParams(); //should be equal to 8

  TH1* tKStarCf = tFitPairAnalysis->GetKStarCf();
//  int tNbins = tKStarCf->GetNbinsX();
  int tNbins = tKStarCf->FindBin(fMaxFitKStar);
  if(tKStarCf->GetBinLowEdge(tNbins) == fMaxFitKStar) tNbins--;
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


  double tKStarMin, tKStarMax, tCfContentUnNorm, tCfContent;
  for(int ix=1; ix <= tNbins; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);

    if(fUseStaticPairs)
    {
      tCfContentUnNorm = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else
    {
      tCfContentUnNorm = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }

    tReturnHist->SetBinContent(ix,tCfContent);
  }

  delete[] tPar;
  delete[] tParError;

  return tReturnHist;
}


//________________________________________________________________________________________________________________
void GeneralFitter::DoFit()
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

//  arglist[0] = 0.0000000001;
//  fMinuit->mnexcm("SET EPS", arglist, 1, fErrFlg);

  // Set strategy to be used
  // 0 = economize (many params and/or fcn takes a long time to calculate and/or not interested in precise param erros)
  // 1 = default
  // 2 = Minuit allowed to waste calls in order to ensure precision (fcn evaluated in short time and/or param erros must be reliable)
  arglist[0] = 1;
  fMinuit->mnexcm("SET STR", arglist ,1,fErrFlg);

  // Now ready for minimization step
  arglist[0] = 50000;
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
/*
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    TH1* tHistTest = CreateFitHistogram("FitHis",iAnaly);
  }
*/
/*
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->SetFit(CreateFitFunction("fit",iAnaly));
  }
*/
}


//________________________________________________________________________________________________________________


