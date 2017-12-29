///////////////////////////////////////////////////////////////////////////
// CoulombFitter:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "CoulombFitter.h"

#ifdef __ROOT__
ClassImp(CoulombFitter)
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
CoulombFitter::CoulombFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar):
  fVerbose(true),
  fTurnOffCoulomb(false),
  fInterpHistsLoaded(false),
  fIncludeSingletAndTriplet(true),
  fUseRandomKStarVectors(false),
  fUseStaticPairs(false),

  fApplyNonFlatBackgroundCorrection(false), //TODO change deault to true here AND in CoulombFitter
  fApplyMomResCorrection(false), //TODO change default to true here AND LednickyFitter
  fIncludeResidualCorrelations(false),  //TODO change deault to true here AND in CoulombFitter
  fResidualsInitiated(false),
  fReturnPrimaryWithResidualsToAnalyses(false),
  fNonFlatBgdFitType(kLinear),

  fNCalls(0),
  fFakeCf(0),

  fFitSharedAnalyses(aFitSharedAnalyses),
  fMinuit(fFitSharedAnalyses->GetMinuitObject()),
  fNAnalyses(fFitSharedAnalyses->GetNFitPairAnalysis()),

  fSimCoulombCf(nullptr),
  fWaveFunction(0),
  fBohrRadius(gBohrRadiusXiK),

  fPairKStar4dVec(0),

  fBinSizeKStar(0.01),
  fNbinsKStar(30),
  fNPairsPerKStarBin(16384),
  fCurrentRadii(fNAnalyses, 1.),
  fPairSample4dVec(0),
  fInterpHistFile(0), fInterpHistFileLednickyHFunction(0),

  fLednickyHFunctionHist(0),

  fGTildeRealHist(0),
  fGTildeImagHist(0),

  fHyperGeo1F1RealHist(0),
  fHyperGeo1F1ImagHist(0),

  fMinInterpKStar(0), fMinInterpRStar(0), fMinInterpTheta(0),
  fMaxInterpKStar(0), fMaxInterpRStar(0), fMaxInterpTheta(0),

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
  int tNFitPartialAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetNFitPartialAnalysis();
  fCorrectedFitVecs.resize(fNAnalyses, td2dVec(tNFitPartialAnalysis));

  gRandom->SetSeed();

  fWaveFunction = new WaveFunction();
  SetCoulombAttributes(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetAnalysisType());

  for(int i=0; i<fNAnalyses; i++)
  {
    //fCfsToFit[i] = fFitSharedAnalyses->GetPairAnalysis(i)->GetCf();
  }

  omp_set_num_threads(3);

}



//________________________________________________________________________________________________________________
CoulombFitter::CoulombFitter(AnalysisType aAnalysisType, double aMaxFitKStar):
  fVerbose(true),
  fTurnOffCoulomb(false),
  fInterpHistsLoaded(false),
  fIncludeSingletAndTriplet(true),
  fUseRandomKStarVectors(false),
  fUseStaticPairs(false),

  fApplyNonFlatBackgroundCorrection(false), //TODO change deault to true here AND in CoulombFitter
  fApplyMomResCorrection(false), //TODO change default to true here AND LednickyFitter
  fIncludeResidualCorrelations(false),  //TODO change deault to true here AND in CoulombFitter
  fResidualsInitiated(false),
  fReturnPrimaryWithResidualsToAnalyses(false),
  fNonFlatBgdFitType(kLinear),

  fNCalls(0),
  fFakeCf(0),

  fFitSharedAnalyses(nullptr),
  fMinuit(nullptr),
  fNAnalyses(0),
  fCorrectedFitVecs(0),

  fSimCoulombCf(nullptr),
  fWaveFunction(0),
  fBohrRadius(-gBohrRadiusXiK),

  fPairKStar4dVec(0),

  fBinSizeKStar(0.01),
  fNbinsKStar(30),
  fNPairsPerKStarBin(16384),
  fCurrentRadii(0),
  fPairSample4dVec(0),
  fInterpHistFile(0), fInterpHistFileLednickyHFunction(0),

  fLednickyHFunctionHist(0),

  fGTildeRealHist(0),
  fGTildeImagHist(0),

  fHyperGeo1F1RealHist(0),
  fHyperGeo1F1ImagHist(0),

  fMinInterpKStar(0), fMinInterpRStar(0), fMinInterpTheta(0),
  fMaxInterpKStar(0), fMaxInterpRStar(0), fMaxInterpTheta(0),


  fCfsToFit(0),
  fFits(0),
  fMaxFitKStar(aMaxFitKStar),
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
//  int tNFitPartialAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetNFitPartialAnalysis();
//  fCorrectedFitVecs.resize(fNAnalyses, td2dVec(tNFitPartialAnalysis));

  gRandom->SetSeed();
  fWaveFunction = new WaveFunction();
  SetCoulombAttributes(aAnalysisType);
  omp_set_num_threads(3);

  fNAnalyses=1;
  fCurrentRadii = td1dVec(fNAnalyses, 1.);
  fUseRandomKStarVectors = true;
  SetUseStaticPairs(true,16384);

}


//________________________________________________________________________________________________________________
CoulombFitter::~CoulombFitter()
{
  cout << "CoulombFitter object is being deleted!!!" << endl;

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
CoulombType CoulombFitter::GetCoulombType(AnalysisType aAnalysisType)
{
//  assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
  if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) return kAttractive; //attractive
  else if(aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP) return kRepulsive; //repulsive
  else return kNeutral;
}


//________________________________________________________________________________________________________________
double CoulombFitter::GetBohrRadius(AnalysisType aAnalysisType)
{
  return fWaveFunction->GetBohrRadius(aAnalysisType);
}

//________________________________________________________________________________________________________________
void CoulombFitter::SetCoulombAttributes(AnalysisType aAnalysisType)
{
  fWaveFunction->SetCurrentAnalysisType(aAnalysisType);
  fBohrRadius = fWaveFunction->GetCurrentBohrRadius();
}


//________________________________________________________________________________________________________________
void CoulombFitter::LoadLednickyHFunctionFile(TString aFileBaseName)
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
void CoulombFitter::LoadInterpHistFile(TString aFileBaseName, TString aLednickyHFunctionFileBaseName)
{
ChronoTimer tTimer(kSec);
tTimer.Start();
cout << "Starting LoadInterpHistFile" << endl;

//--------------------------------------------------------------

  LoadLednickyHFunctionFile(aLednickyHFunctionFileBaseName);

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

  fInterpHistsLoaded = true;
}

//________________________________________________________________________________________________________________
void CoulombFitter::ExtractPairKStar3dVecFromSingleFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill)
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

    tKStarBinNumber = Interpolator::GetBinNumber(aBinSizeKStar, aNbinsKStar, tTupleKStarMag);
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
td3dVec CoulombFitter::BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
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

  fNbinsKStar = aNbinsKStar;
  fBinSizeKStar = (aKStarMax-aKStarMin)/fNbinsKStar;
cout << "fBinSizeKStar = " << fBinSizeKStar << endl;

  td3dVec tPairKStar3dVec;
    tPairKStar3dVec.clear();
  tPairKStar3dVec.resize(fNbinsKStar, td2dVec(0,td1dVec(0)));

cout << "Pre: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;

  //---------------------------------------------------
  TString tFileLocation;

  TString tArrayName = TString(cAnalysisBaseTags[aAnalysisType])+TString(cCentralityTags[aCentralityType]);
  TString tNtupleName = "PairKStarKStarCf_"+TString(cAnalysisBaseTags[aAnalysisType]);

  for(int iFile=0; iFile<17; iFile++) //Bm files
  {
    cout << "\t iFile = Bm" << iFile << endl;
    tFileLocation = aPairKStarNtupleDirName + TString("/") + aFileBaseName + TString("_Bm") + tRomanNumerals[iFile] + TString(".root");
    ExtractPairKStar3dVecFromSingleFile(tFileLocation,tArrayName,tNtupleName,fBinSizeKStar,fNbinsKStar,tPairKStar3dVec);
  }

  for(int iFile=0; iFile<10; iFile++) //Bp files
  {
    cout << "\t iFile = Bp" << iFile << endl;
    tFileLocation = aPairKStarNtupleDirName + TString("/") + aFileBaseName + TString("_Bp") + tRomanNumerals[iFile] + TString(".root");
    ExtractPairKStar3dVecFromSingleFile(tFileLocation,tArrayName,tNtupleName,fBinSizeKStar,fNbinsKStar,tPairKStar3dVec);
  }

  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Conversion complete in " << duration << " seconds" << endl;


  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<fNbinsKStar; i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

  return tPairKStar3dVec;
}

//________________________________________________________________________________________________________________
void CoulombFitter::WriteRow(ostream &aOutput, vector<double> &aRow)
{
  for(int i = 0; i < (int)aRow.size(); i++)
  {
    if( i < (int)aRow.size()-1) aOutput << aRow[i] << " ";
    else if(i == (int)aRow.size()-1) aOutput << aRow[i] << endl;
    else cout << "SOMETHING IS WRONG!!!!!\n";
  }
}



//________________________________________________________________________________________________________________
void CoulombFitter::WritePairKStar3dVecFile(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
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
void CoulombFitter::WriteAllPairKStar3dVecFiles(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
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
td3dVec CoulombFitter::BuildPairKStar3dVecFromTxt(TString aFileName)
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
void CoulombFitter::BuildPairKStar4dVecFromTxt(TString aFileBaseName)
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
void CoulombFitter::BuildPairKStar4dVecOnFly(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
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
void CoulombFitter::BuildPairSample4dVec(int aNPairsPerKStarBin, double aBinSize)
{
ChronoTimer tTimer(kSec);
tTimer.Start();

  fNPairsPerKStarBin = aNPairsPerKStarBin;
  fPairSample4dVec.resize(fNAnalyses, td3dVec(0, td2dVec(0, td1dVec(0))));

//  double tBinSize = 0.01;  //TODO make this automated
  fBinSizeKStar = aBinSize;
  fNbinsKStar = std::round(fMaxFitKStar/fBinSizeKStar);  //TODO make this general, ie subtract 1 if fMaxFitKStar is on bin edge (maybe, maybe not bx of iKStarBin<tNBinsKStar)

  //Create the source Gaussians
  double tRoot2 = sqrt(2.);  //need this scale to get 4 on denominator of exp in normal dist instead of 2
  double tRadius = 1.0;
  for(unsigned int i=0; i<fCurrentRadii.size(); i++) fCurrentRadii[i] = tRadius;
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
    for(int iKStarBin=0; iKStarBin<fNbinsKStar; iKStarBin++)
    {
      if(!fUseRandomKStarVectors) tRandomKStarElement = std::uniform_int_distribution<int>(0.0, fPairKStar4dVec[iAnaly][iKStarBin].size()-1);
      tKStarMagMin = iKStarBin*fBinSizeKStar;
      if(iKStarBin==0) tKStarMagMin=0.004;  //TODO here and in ChargedResidualCf
      tKStarMagMax = (iKStarBin+1)*fBinSizeKStar;
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
void CoulombFitter::UpdatePairRadiusParameter(double aNewRadius, int aAnalysisNumber)
{
  //TODO allow for change of MU also, not just SIGMA!
  double tScaleFactor = aNewRadius/fCurrentRadii[aAnalysisNumber];
  fCurrentRadii[aAnalysisNumber] = aNewRadius;

//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iKStarBin=0; iKStarBin<(int)fPairSample4dVec[aAnalysisNumber].size(); iKStarBin++)
  {
    for(int iPair=0; iPair<(int)fPairSample4dVec[aAnalysisNumber][iKStarBin].size(); iPair++)
    {
        fPairSample4dVec[aAnalysisNumber][iKStarBin][iPair][1] *= tScaleFactor;
    }
  }
}

//________________________________________________________________________________________________________________
void CoulombFitter::SetUseStaticPairs(bool aUseStaticPairs, int aNPairsPerKStarBin, double aBinSize)
{
  fUseStaticPairs = aUseStaticPairs;
  BuildPairSample4dVec(aNPairsPerKStarBin, aBinSize);
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetEta(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else return pow(((aKStar/hbarc)*fBohrRadius),-1);
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetGamowFactor(double aKStar)
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
complex<double> CoulombFitter::GetExpTerm(double aKStar, double aRStar, double aTheta)
{
  complex<double> tReturnValue = exp(-ImI*(aKStar/hbarc)*aRStar*cos(aTheta));
  return tReturnValue;
}

//________________________________________________________________________________________________________________
complex<double> CoulombFitter::BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0)
{
  complex<double> tF0 (aReF0,aImF0);
  double tKStar = aKStar/hbarc;
  complex<double> tScattAmp;

  if(aReF0==0. && aImF0==0. && aD0==0.) tScattAmp = complex<double>(0.,0.);
  else if(fTurnOffCoulomb)
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
double CoulombFitter::InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
{

  bool tDebug = true; //debug means use personal interpolation methods, instead of standard root ones

  //TODO put check to make sure file is open, not sure if assert(fInterpHistFile->IsOpen works);
  assert(fInterpHistsLoaded);
  //assert(fInterpHistFile->IsOpen());

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

  if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in CoulombFitter::InterpolateWfSquared !!!!!" << endl;
  assert(imag(tResultComplex) < std::numeric_limits< double >::min());

  return real(tResultComplex);

}

//________________________________________________________________________________________________________________
vector<double> CoulombFitter::InterpolateWfSquaredSerialv2(vector<vector<double> > &aPairs, double aKStarMagMin, double aKStarMagMax, double aReF0, double aImF0, double aD0)
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

      tHyperGeo1F1Real = Interpolator::TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = Interpolator::TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

      tGTildeReal = Interpolator::BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
      tGTildeImag = Interpolator::BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);

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

    if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in CoulombFitter::InterpolateWfSquaredSerialv2 !!!!!" << endl;
    assert(imag(tResultComplex) < std::numeric_limits< double >::min());

    tReturnVector.push_back(real(tResultComplex));
  }

  return tReturnVector;
}


//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpKStar(double aKStar)
{
  if(aKStar < fMinInterpKStar) return false;
  if(aKStar > fMaxInterpKStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpRStar(double aRStar)
{
  if(aRStar < fMinInterpRStar || aRStar > fMaxInterpRStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpTheta(double aTheta)
{
  if(aTheta < fMinInterpTheta || aTheta > fMaxInterpTheta) return false;
  return true;
}



//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0)
{
  if(CanInterpKStar(aKStar) && CanInterpRStar(aRStar) && CanInterpTheta(aTheta)) return true;
  return false;
}

//________________________________________________________________________________________________________________
void CoulombFitter::SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax)
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
double CoulombFitter::GetFitCfContent(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  omp_set_num_threads(6);

/*
cout << "In CoulombFitter::GetFitCfContent" << endl;
cout << "\t aKStarMagMin = " << aKStarMagMin << endl;
cout << "\t aKStarMagMax = " << aKStarMagMax << endl;
*/
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] != Norm !!!!!!!!!!!!!!!!!!!!!

  //should probably do x[0] /= hbarc, but let me test first


  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  //Probably fixed with use of std::round, but need to double check
//  double tBinSize = 0.01;
  int tBin = std::round(aKStarMagMin/fBinSizeKStar);
/*
cout << "In GetFitCfContent....." << endl;
cout << "\t tBin = " << tBin << endl;
cout << "\t aKStarMagMin = " << aKStarMagMin << "\t aKStarMagMax = " << aKStarMagMax << endl;
*/


  //KStarMag = fPairKStar4dVec[aAnalysisNumber][tBin][i][0]
  //KStarOut = fPairKStar4dVec[aAnalysisNumber][tBin][i][1]
  //KStarSide = fPairKStar4dVec[aAnalysisNumber][tBin][i][2]
  //KStarLong = fPairKStar4dVec[aAnalysisNumber][tBin][i][3]

  int tCounter = 0;
  double tReturnCfContent = 0.;
//cout << "fPairKStar4dVec[aAnalysisNumber][tBin].size() = " << fPairKStar4dVec[aAnalysisNumber][tBin].size() << endl << endl;

  int tMaxKStarCalls;
/*
  if(fPairKStar4dVec[aAnalysisNumber][tBin].size() < 100) tMaxKStarCalls = fPairKStar4dVec[aAnalysisNumber][tBin].size();
  else tMaxKStarCalls = 100;
*/
  //definitely oversampling by commenting out the above
  //Currently, lowest bin only have 6 entries!
  tMaxKStarCalls = 100000;

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
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSq;
  complex<double> tWaveFunction;

  td2dVec tMathematicaPairs;
  td1dVec tTempPair(3);

  int tNInterpolate = 0;
  int tNMathematica = 0;

  #pragma omp parallel for reduction(+: tCounter) reduction(+: tReturnCfContent) private(tI, tCanInterp, tTheta, tKStarMag, tRStarMag, tWaveFunctionSq) firstprivate(tKStar3Vec, tSource3Vec, tTempPair)
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

    //TODO: take out this restriction
//    while (tSource3Vec->Mag() > 20.) tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator));

    tRStarMag = tSource3Vec->Mag();


//TODO make 1 and 2 histograms overlap by a bin or two!
//TODO
//TODO
//TODO!!!!!!!!!!!!! Must have separate InterpolateWfSquaredXiKchP and InterpolateWfSquaredXiKchM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //Note:  Running CanInterpAll is faster than checking CanInterpScatParams outside of for loop and comparing to CanInterpKRTh.....strange but true
    tCanInterp = CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
    if(fTurnOffCoulomb || tCanInterp) 
    {
      tWaveFunctionSq = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tReturnCfContent += tWaveFunctionSq;
      tCounter++;
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

  tNInterpolate = tCounter;
  tNMathematica = tMathematicaPairs.size();

  for(int i=0; i<(int)tMathematicaPairs.size(); i++)
  {
    tWaveFunction = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0], tMathematicaPairs[i][1], tMathematicaPairs[i][2], par[2], par[3], par[4]);
    tWaveFunctionSq = norm(tWaveFunction);
    tReturnCfContent += tWaveFunctionSq;
    tCounter++;
  }

  delete tKStar3Vec;
  delete tSource3Vec;

  tReturnCfContent /= tCounter;

//  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));


  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\tIn GetFitCfContent" << endl;
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
    cout << "\t\t\t\taKStarMagMin = " << aKStarMagMin << endl;
    cout << "\t\t\t\taKStarMagMax = " << aKStarMagMax << endl;
    cout << "\t\t\t\tpar[0] = " << par[0] << endl;
    cout << "\t\t\t\tpar[1] = " << par[1] << endl;
    cout << "\t\t\t\tpar[2] = " << par[2] << endl;
    cout << "\t\t\t\tpar[3] = " << par[3] << endl;
    cout << "\t\t\t\tpar[4] = " << par[4] << endl << endl;
  }

  return tReturnCfContent;
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetFitCfContentwStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  omp_set_num_threads(6);

  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] != Norm !!!!!!!!!!!!!!!!

  if(abs(par[1]-fCurrentRadii[aAnalysisNumber]) > std::numeric_limits< double >::min()) UpdatePairRadiusParameter(par[1], aAnalysisNumber);
//  UpdatePairRadiusParametersGlobal(par[1]);  //TODO

  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  //Probably fixed with use of std::round, but need to double check
//  double tBinSize = aKStarMagMax-aKStarMagMin;
  int tBin = std::round(aKStarMagMin/fBinSizeKStar);

  //KStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0]
  //RStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1]
  //Theta    = fPairSample4dVec[aAnalysisNumber][tBin][i][2]

  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;
  tMaxKStarCalls = fPairSample4dVec[aAnalysisNumber][tBin].size();
//cout << "In GetFitCfContentwStaticPairs: " << endl;
//cout << "\taKStarMagMin = " << aKStarMagMin << " and tBin = " << tBin << endl;
//cout << "\ttMaxKStarCalls = " << tMaxKStarCalls << endl;

  bool tCanInterp;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSq;
  complex<double> tWaveFunction;

  td2dVec tMathematicaPairs;
  td1dVec tTempPair(3);

  int tNInterpolate = 0;
  int tNMathematica = 0;

  #pragma omp parallel for reduction(+: tCounter) reduction(+: tReturnCfContent) private(tCanInterp, tTheta, tKStarMag, tRStarMag, tWaveFunctionSq) firstprivate(tTempPair)
  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tKStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0];
    tRStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1];
    tTheta = fPairSample4dVec[aAnalysisNumber][tBin][i][2];

    tCanInterp = CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
    if(fTurnOffCoulomb || tCanInterp) 
    {
      tWaveFunctionSq = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tReturnCfContent += tWaveFunctionSq;
      tCounter++;
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

  tNInterpolate = tCounter;
  tNMathematica = tMathematicaPairs.size();

  for(int i=0; i<(int)tMathematicaPairs.size(); i++)
  {
    tWaveFunction = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0], tMathematicaPairs[i][1], tMathematicaPairs[i][2], par[2], par[3], par[4]);
    tWaveFunctionSq = norm(tWaveFunction);
    tReturnCfContent += tWaveFunctionSq;
    tCounter++;
  }

  tReturnCfContent /= tCounter;

//  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));


  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\tIn GetFitCfContentwStaticPairs" << endl;
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
    cout << "\t\t\t\taKStarMagMin = " << aKStarMagMin << endl;
    cout << "\t\t\t\taKStarMagMax = " << aKStarMagMax << endl;
    cout << "\t\t\t\tpar[0] = " << par[0] << endl;
    cout << "\t\t\t\tpar[1] = " << par[1] << endl;
    cout << "\t\t\t\tpar[2] = " << par[2] << endl;
    cout << "\t\t\t\tpar[3] = " << par[3] << endl;
    cout << "\t\t\t\tpar[4] = " << par[4] << endl << endl;
  }

  return tReturnCfContent;
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  assert(fIncludeSingletAndTriplet == true);

  omp_set_num_threads(6);
  //par[0] = kLambda
  //par[1] = kRadius
  //par[2] = kRef0
  //par[3] = kImf0
  //par[4] = kd0
  //par[5] = kRef02
  //par[6] = kImf02
  //par[7] = kd02
  //par[8] = kNorm


  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  //Probably fixed with use of std::round, but need to double check
//  double tBinSize = 0.01;
  int tBin = std::round(aKStarMagMin/fBinSizeKStar);

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
      tWaveFunctionSqSinglet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tWaveFunctionSqTriplet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[5],par[6],par[7]);

      tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
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
    tWaveFunctionSinglet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[2],par[3],par[4]);
    tWaveFunctionSqSinglet = norm(tWaveFunctionSinglet);

    tWaveFunctionTriplet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[5],par[6],par[7]);
    tWaveFunctionSqTriplet = norm(tWaveFunctionTriplet);

    tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
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
    cout << "\t\tIn GetFitCfContentComplete" << endl;
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
    cout << "\t\t\t\taKStarMagMin = " << aKStarMagMin << endl;
    cout << "\t\t\t\taKStarMagMax = " << aKStarMagMax << endl;
    cout << "\t\t\t\tpar[0] = " << par[0] << endl;
    cout << "\t\t\t\tpar[1] = " << par[1] << endl;
    cout << "\t\t\t\tpar[2] = " << par[2] << endl;
    cout << "\t\t\t\tpar[3] = " << par[3] << endl;
    cout << "\t\t\t\tpar[4] = " << par[4] << endl;
    cout << "\t\t\t\tpar[5] = " << par[5] << endl;
    cout << "\t\t\t\tpar[6] = " << par[6] << endl;
    cout << "\t\t\t\tpar[7] = " << par[7] << endl << endl;
  }

  return tReturnCfContent;
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  assert(fIncludeSingletAndTriplet == true);

  omp_set_num_threads(6);
  //par[0] = kLambda
  //par[1] = kRadius
  //par[2] = kRef0
  //par[3] = kImf0
  //par[4] = kd0
  //par[5] = kRef02
  //par[6] = kImf02
  //par[7] = kd02
  //par[8] = kNorm

  if(abs(par[1]-fCurrentRadii[aAnalysisNumber]) > std::numeric_limits< double >::min()) UpdatePairRadiusParameter(par[1], aAnalysisNumber);
//  UpdatePairRadiusParametersGlobal(par[1]);  //TODO

  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  //Probably fixed with use of std::round, but need to double check
//  double tBinSize = 0.01;
  int tBin = std::round(aKStarMagMin/fBinSizeKStar);

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
      tWaveFunctionSqSinglet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tWaveFunctionSqTriplet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[5],par[6],par[7]);

      tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
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
    tWaveFunctionSinglet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[2],par[3],par[4]);
    tWaveFunctionSqSinglet = norm(tWaveFunctionSinglet);

    tWaveFunctionTriplet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[5],par[6],par[7]);
    tWaveFunctionSqTriplet = norm(tWaveFunctionTriplet);

    tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
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
    cout << "\t\tIn GetFitCfContentCompletewStaticPairs" << endl;
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
    cout << "\t\t\t\taKStarMagMin = " << aKStarMagMin << endl;
    cout << "\t\t\t\taKStarMagMax = " << aKStarMagMax << endl;
    cout << "\t\t\t\tpar[0] = " << par[0] << endl;
    cout << "\t\t\t\tpar[1] = " << par[1] << endl;
    cout << "\t\t\t\tpar[2] = " << par[2] << endl;
    cout << "\t\t\t\tpar[3] = " << par[3] << endl;
    cout << "\t\t\t\tpar[4] = " << par[4] << endl;
    cout << "\t\t\t\tpar[5] = " << par[5] << endl;
    cout << "\t\t\t\tpar[6] = " << par[6] << endl;
    cout << "\t\t\t\tpar[7] = " << par[7] << endl << endl;
  }

  return tReturnCfContent;
}


//________________________________________________________________________________________________________________
double CoulombFitter::GetFitCfContentSerialv2(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
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
  //Probably fixed with use of std::round, but need to double check
//  double tBinSize = 0.01;
  int tBin = std::round(aKStarMagMin/fBinSizeKStar);

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
void CoulombFitter::PrintCurrentParamValues(int aNpar, double* aPar)
{
  for(int i=0; i<aNpar; i++) cout << "par[" << i << "] = " << aPar[i] << endl;
  cout << endl;
}


//________________________________________________________________________________________________________________
bool CoulombFitter::AreParamsSame(double *aCurrent, double *aNew, int aNEntries)
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
double CoulombFitter::GetChi2Value(int aKStarBin, TH1* aCfToFit, double aFitVal)
{
    double tChi = (aCfToFit->GetBinContent(aKStarBin) - aFitVal)/aCfToFit->GetBinError(aKStarBin);
    return tChi*tChi;
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetPmlValue(double aNumContent, double aDenContent, double aCfContent)
{
  double tTerm1 = aNumContent*log(  (aCfContent*(aNumContent+aDenContent)) / (aNumContent*(aCfContent+1))  );
  double tTerm2 = aDenContent*log(  (aNumContent+aDenContent) / (aDenContent*(aCfContent+1))  );
  double tChi2PML = -2.0*(tTerm1+tTerm2);
  return tChi2PML;
}


//________________________________________________________________________________________________________________
void CoulombFitter::CalculateChi2PML(int &npar, double &chi2, double *par)
{
ChronoTimer tTotalTimer;
tTotalTimer.Start();

  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;
  PrintCurrentParamValues(fFitSharedAnalyses->GetNMinuitParams(),par);
/*
  if(fIncludeSingletAndTriplet)
  {
    cout << "\t\tParameter update: " << endl;
    cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
    cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

    cout << "\t\t\tpar[2] = ReF0s  = " << par[2] << endl;
    cout << "\t\t\tpar[3] = ImF0s  = " << par[3] << endl;
    cout << "\t\t\tpar[4] = D0s    = " << par[4] << endl;

    cout << "\t\t\tpar[5] = ReF0t  = " << par[5] << endl;
    cout << "\t\t\tpar[6] = ImF0t  = " << par[6] << endl;
    cout << "\t\t\tpar[7] = D0t    = " << par[7] << endl;

    cout << "\t\t\tpar[8] = Norm1  = " << par[8] << endl;
    cout << "\t\t\tpar[9] = Norm2  = " << par[9] << endl;
    cout << "\t\t\tpar[10] = Norm3 = " << par[10] << endl;
    cout << "\t\t\tpar[11] = Norm4 = " << par[11] << endl;
    cout << "\t\t\tpar[12] = Norm5 = " << par[12] << endl;
    cout << "\t\t\tpar[13] = Norm6 = " << par[13] << endl;
    cout << "\t\t\tpar[14] = Norm7 = " << par[14] << endl;
    cout << "\t\t\tpar[15] = Norm8 = " << par[15] << endl;
    cout << "\t\t\tpar[16] = Norm9 = " << par[16] << endl;
    cout << "\t\t\tpar[17] = Norm10= " << par[17] << endl;
  }
  else
  {
    cout << "\t\tParameter update: " << endl;
    cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
    cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

    cout << "\t\t\tpar[2] = ReF0  = " << par[2] << endl;
    cout << "\t\t\tpar[3] = ImF0  = " << par[3] << endl;
    cout << "\t\t\tpar[4] = D0    = " << par[4] << endl;

    cout << "\t\t\tpar[8] = Norm1  = " << par[5] << endl;
    cout << "\t\t\tpar[9] = Norm2  = " << par[6] << endl;
    cout << "\t\t\tpar[10] = Norm3 = " << par[7] << endl;
    cout << "\t\t\tpar[11] = Norm4 = " << par[8] << endl;
    cout << "\t\t\tpar[12] = Norm5 = " << par[9] << endl;
    cout << "\t\t\tpar[13] = Norm6 = " << par[10] << endl;
    cout << "\t\t\tpar[14] = Norm7 = " << par[11] << endl;
    cout << "\t\t\tpar[15] = Norm8 = " << par[12] << endl;
    cout << "\t\t\tpar[16] = Norm9 = " << par[13] << endl;
    cout << "\t\t\tpar[17] = Norm10= " << par[14] << endl;
  }
*/
  //--------------------------------------------------------------

  int tNFitParPerAnalysis;
  if(fIncludeSingletAndTriplet) tNFitParPerAnalysis = 8;
  else tNFitParPerAnalysis = 5;

  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  int tNbinsXToFitGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar);
  if(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetBinLowEdge(tNbinsXToFitGlobal) == fMaxFitKStar) tNbinsXToFitGlobal--;

  vector<double> tCfContentUnNorm(tNbinsXToFitGlobal,0.);

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();
    SetCoulombAttributes(tAnalysisType);

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

      int tNbinsXToFit = tNum->FindBin(fMaxFitKStar);
      if(tNum->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;

      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tNbinsXToFit == tNbinsXToFitGlobal);
      int tNFitParams = tFitPartialAnalysis->GetNFitParams() + 1;  //the +1 accounts for the normalization parameter
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
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);
      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxisNum->GetBinLowEdge(ix);
        double tKStarMax = tXaxisNum->GetBinLowEdge(ix+1);

        double tNumContent = tNum->GetBinContent(ix);
        double tDenContent = tDen->GetBinContent(ix);

//        if(!tAreParamsSame) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
//        tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
        if(!tAreParamsSame)
        {
          if(fUseStaticPairs && fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fUseStaticPairs) tCfContentUnNorm[ix-1] = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else tCfContentUnNorm[ix-1] = GetFitCfContent(tKStarMin,tKStarMax,tPar,iAnaly);
        }

        double tCfContent;
        if(fIncludeSingletAndTriplet) tCfContent = tPar[8]*tCfContentUnNorm[ix-1];
        else tCfContent = tPar[5]*tCfContentUnNorm[ix-1];

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

  }

  delete[] tCurrentFitPar;

  if(std::isnan(fChi2) || std::isinf(fChi2))
  {
    cout << "WARNING: fChi2 = nan, setting it equal to 10^9-----------------------------------------" << endl << endl;
    fChi2 = pow(10,9);
  }

  chi2 = fChi2;
  if(fChi2 < fChi2GlobalMin) fChi2GlobalMin = fChi2;
//gObjectTable->Print();
cout << "fChi2 = " << fChi2 << endl;
cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;

tTotalTimer.Stop();
cout << "tTotalTimer: ";
tTotalTimer.PrintInterval();


  double *tParamsForHistograms = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tParamsForHistograms[i] = par[i];
  fFitSharedAnalyses->GetFitChi2Histograms()->FillHistograms(fChi2,tParamsForHistograms);
  delete[] tParamsForHistograms;
}


//________________________________________________________________________________________________________________
vector<double> CoulombFitter::ApplyMomResCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TH2* aMomResMatrix)
{
  //TODO probably rebin aMomResMatrix to match bin size of aCf
  //TODO do in both this AND LednickyFitter

  unsigned int tKStarRecBin, tKStarTrueBin;
  double tKStarRec, tKStarTrue;
  assert(aCf.size() == aKStarBinCenters.size());
  assert(aCf.size() == (unsigned int)aMomResMatrix->GetNbinsX());
  assert(aCf.size() == (unsigned int)aMomResMatrix->GetNbinsY());

  vector<double> tReturnCf(aCf.size(),0.);
  vector<double> tNormVec(aCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

  for(unsigned int i=0; i<aCf.size(); i++)
  {
    tKStarRec = aKStarBinCenters[i];
    tKStarRecBin = aMomResMatrix->GetYaxis()->FindBin(tKStarRec);

    for(unsigned int j=0; j<aCf.size(); j++)
    {
      tKStarTrue = aKStarBinCenters[j];
      tKStarTrueBin = aMomResMatrix->GetXaxis()->FindBin(tKStarTrue);

      tReturnCf[i] += aCf[j]*aMomResMatrix->GetBinContent(tKStarTrueBin,tKStarRecBin);
      tNormVec[i] += aMomResMatrix->GetBinContent(tKStarTrueBin,tKStarRecBin);
    }
    tReturnCf[i] /= tNormVec[i];
  }
  return tReturnCf;
}



//________________________________________________________________________________________________________________
void CoulombFitter::CalculateChi2PMLwMomResCorrection(int &npar, double &chi2, double *par)
{
ChronoTimer tTotalTimer;
tTotalTimer.Start();

  assert(fApplyMomResCorrection);

  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;

  if(fIncludeSingletAndTriplet)
  {
    cout << "\t\tParameter update: " << endl;
    cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
    cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

    cout << "\t\t\tpar[2] = ReF0s  = " << par[2] << endl;
    cout << "\t\t\tpar[3] = ImF0s  = " << par[3] << endl;
    cout << "\t\t\tpar[4] = D0s    = " << par[4] << endl;

    cout << "\t\t\tpar[5] = ReF0t  = " << par[5] << endl;
    cout << "\t\t\tpar[6] = ImF0t  = " << par[6] << endl;
    cout << "\t\t\tpar[7] = D0t    = " << par[7] << endl;

    cout << "\t\t\tpar[8] = Norm1  = " << par[8] << endl;
    cout << "\t\t\tpar[9] = Norm2  = " << par[9] << endl;
    cout << "\t\t\tpar[10] = Norm3 = " << par[10] << endl;
    cout << "\t\t\tpar[11] = Norm4 = " << par[11] << endl;
    cout << "\t\t\tpar[12] = Norm5 = " << par[12] << endl;
    cout << "\t\t\tpar[13] = Norm6 = " << par[13] << endl;
    cout << "\t\t\tpar[14] = Norm7 = " << par[14] << endl;
    cout << "\t\t\tpar[15] = Norm8 = " << par[15] << endl;
    cout << "\t\t\tpar[16] = Norm9 = " << par[16] << endl;
    cout << "\t\t\tpar[17] = Norm10= " << par[17] << endl;
  }
  else
  {
    cout << "\t\tParameter update: " << endl;
    cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
    cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

    cout << "\t\t\tpar[2] = ReF0  = " << par[2] << endl;
    cout << "\t\t\tpar[3] = ImF0  = " << par[3] << endl;
    cout << "\t\t\tpar[4] = D0    = " << par[4] << endl;

    cout << "\t\t\tpar[8] = Norm1  = " << par[5] << endl;
    cout << "\t\t\tpar[9] = Norm2  = " << par[6] << endl;
    cout << "\t\t\tpar[10] = Norm3 = " << par[7] << endl;
    cout << "\t\t\tpar[11] = Norm4 = " << par[8] << endl;
    cout << "\t\t\tpar[12] = Norm5 = " << par[9] << endl;
    cout << "\t\t\tpar[13] = Norm6 = " << par[10] << endl;
    cout << "\t\t\tpar[14] = Norm7 = " << par[11] << endl;
    cout << "\t\t\tpar[15] = Norm8 = " << par[12] << endl;
    cout << "\t\t\tpar[16] = Norm9 = " << par[13] << endl;
    cout << "\t\t\tpar[17] = Norm10= " << par[14] << endl;
  }

  //--------------------------------------------------------------

  int tNFitParPerAnalysis;
  if(fIncludeSingletAndTriplet) tNFitParPerAnalysis = 8;
  else tNFitParPerAnalysis = 5;

  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  int tNbinsXToFitGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar);
  if(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetBinLowEdge(tNbinsXToFitGlobal) == fMaxFitKStar) tNbinsXToFitGlobal--;

  int tNbinsXToBuildGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetModelKStarTrueVsRecMixed()->GetNbinsX();  // when applying momentum resolution corrections, many times
                                                                                                                     // you must go beyond fitting range to apply correction

  vector<double> tCfContentUnNorm(tNbinsXToBuildGlobal,0.);
  vector<double> tCfContent(tNbinsXToBuildGlobal,0.);
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

    TH2* tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
    assert(tMomResMatrix);
    int tNbinsXToBuild = tMomResMatrix->GetNbinsX();
    assert(tNbinsXToBuild == tNbinsXToBuildGlobal);

    SetCoulombAttributes(tAnalysisType);

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
//cout << "iPartAn = " << iPartAn << endl;
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();

      //make sure tNum and tDen have same bin size as tMomResMatrix
      assert(tNum->GetXaxis()->GetBinWidth(1) == tDen->GetXaxis()->GetBinWidth(1));
      assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetXaxis()->GetBinWidth(1));
      assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetYaxis()->GetBinWidth(1));

      //make sure tNum and tDen have same number of bins
      assert(tNum->GetNbinsX() == tDen->GetNbinsX());

      TAxis* tXaxisNum = tNum->GetXaxis();

      int tNbinsX = tNum->GetNbinsX();

      int tNbinsXToFit = tNum->FindBin(fMaxFitKStar);
      if(tNum->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;

      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tNbinsXToFit == tNbinsXToFitGlobal);
      int tNFitParams = tFitPartialAnalysis->GetNFitParams() + 1;  //the +1 accounts for the normalization parameter
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
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);
      for(int ix=1; ix <= tNbinsXToBuild; ix++)
      {
        double tKStarMin = tXaxisNum->GetBinLowEdge(ix);
        double tKStarMax = tXaxisNum->GetBinLowEdge(ix+1);

        tKStarBinCenters[ix-1] = tXaxisNum->GetBinCenter(ix);

        tNumContent[ix-1] = tNum->GetBinContent(ix);
        tDenContent[ix-1] = tDen->GetBinContent(ix);

//        if(!tAreParamsSame) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
//        tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
        if(!tAreParamsSame)
        {
          if(fUseStaticPairs && fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fUseStaticPairs) tCfContentUnNorm[ix-1] = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else tCfContentUnNorm[ix-1] = GetFitCfContent(tKStarMin,tKStarMax,tPar,iAnaly);
        }

        if(fIncludeSingletAndTriplet) tCfContent[ix-1] = tPar[8]*tCfContentUnNorm[ix-1];
        else tCfContent[ix-1] = tPar[5]*tCfContentUnNorm[ix-1];
      }

      vector<double> tCfContentwMomResCorrection = ApplyMomResCorrection(tCfContent, tKStarBinCenters, tMomResMatrix);
      for(int ix=0; ix < tNbinsXToFit; ix++)
      {
        if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tCfContentwMomResCorrection[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          double tTerm1 = tNumContent[ix]*log(  (tCfContentwMomResCorrection[ix]*(tNumContent[ix]+tDenContent[ix])) / (tNumContent[ix]*(tCfContentwMomResCorrection[ix]+1))  );
          double tTerm2 = tDenContent[ix]*log(  (tNumContent[ix]+tDenContent[ix]) / (tDenContent[ix]*(tCfContentwMomResCorrection[ix]+1))  );
          tmp = -2.0*(tTerm1+tTerm2);

          fChi2Vec[iAnaly] += tmp;
          fChi2 += tmp;

          fNpFitsVec[iAnaly]++;
          fNpFits++;
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
//gObjectTable->Print();
cout << "fChi2 = " << fChi2 << endl;
cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;

tTotalTimer.Stop();
cout << "tTotalTimer: ";
tTotalTimer.PrintInterval();


  double *tParamsForHistograms = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tParamsForHistograms[i] = par[i];
  fFitSharedAnalyses->GetFitChi2Histograms()->FillHistograms(fChi2,tParamsForHistograms);
  delete[] tParamsForHistograms;
}



//________________________________________________________________________________________________________________
void CoulombFitter::ApplyNonFlatBackgroundCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TF1* aNonFlatBgd)
{
  assert(aCf.size() == aKStarBinCenters.size());
  for(unsigned int i=0; i<aCf.size(); i++)
  {
    aCf[i] = aCf[i]*aNonFlatBgd->Eval(aKStarBinCenters[i]);
  }
}


//________________________________________________________________________________________________________________
void CoulombFitter::ApplyNormalization(double aNorm, td1dVec &aCf)
{
  for(unsigned int i=0; i<aCf.size(); i++) aCf[i] *= aNorm;
}



//________________________________________________________________________________________________________________
void CoulombFitter::CalculateFitFunction(int &npar, double &chi2, double *par)
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
    int tNbinsXToBuild=0;
    TH2* tMomResMatrix;
    if(fApplyMomResCorrection)
    {
      tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
      assert(tMomResMatrix);
      tNbinsXToBuild = tMomResMatrix->GetNbinsX();
      assert(tNbinsXToBuild == tNbinsXToBuildGlobal);
    }

    SetCoulombAttributes(tAnalysisType);

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

      //make sure tNum and tDen have same bin size as tMomResMatrix
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
      int tNFitParams = tFitPartialAnalysis->GetNFitParams() + 1;  //the +1 accounts for the normalization parameter
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
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      if(!fApplyMomResCorrection && !fIncludeResidualCorrelations) tNbinsXToBuild = tNbinsXToFit;

      vector<double> tFitCfContent;
      vector<double> tCorrectedFitCfContent;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);
      double tKStarMin, tKStarMax;
      for(int ix=1; ix <= tNbinsXToBuild; ix++)
      {
        tKStarMin = tXaxisNum->GetBinLowEdge(ix);
        tKStarMax = tXaxisNum->GetBinLowEdge(ix+1);

        tKStarBinCenters[ix-1] = tXaxisNum->GetBinCenter(ix);

        tNumContent[ix-1] = tNum->GetBinContent(ix);
        tDenContent[ix-1] = tDen->GetBinContent(ix);

        if(!tAreParamsSame)
        {
          if(fUseStaticPairs && fIncludeSingletAndTriplet) tPrimaryFitCfContent[ix-1] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fIncludeSingletAndTriplet) tPrimaryFitCfContent[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fUseStaticPairs) tPrimaryFitCfContent[ix-1] = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else tPrimaryFitCfContent[ix-1] = GetFitCfContent(tKStarMin,tKStarMax,tPar,iAnaly);
        }
      }

      if(fIncludeResidualCorrelations)
      {
        //TODO do stuff dealing with residual!  This is for later
      }
      else tFitCfContent = tPrimaryFitCfContent;

      if(fApplyMomResCorrection) tCorrectedFitCfContent = ApplyMomResCorrection(tFitCfContent, tKStarBinCenters, tMomResMatrix);
      else tCorrectedFitCfContent = tFitCfContent;

      bool tNormalizeBgdFitToCf=false;
      if(fApplyNonFlatBackgroundCorrection)
      {
        TF1* tNonFlatBgd = tFitPartialAnalysis->GetNonFlatBackground(fNonFlatBgdFitType, fFitSharedAnalyses->GetFitType(), tNormalizeBgdFitToCf);
        ApplyNonFlatBackgroundCorrection(tCorrectedFitCfContent, tKStarBinCenters, tNonFlatBgd);
      }

      fCorrectedFitVecs[iAnaly][iPartAn] = tCorrectedFitCfContent;
      if(fIncludeSingletAndTriplet) ApplyNormalization(tPar[8], tCorrectedFitCfContent);
      else ApplyNormalization(tPar[5], tCorrectedFitCfContent);


      for(int ix=0; ix < tNbinsXToFit; ix++)
      {
        if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tCorrectedFitCfContent[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          double tChi2 = 0.;
          if(fFitSharedAnalyses->GetFitType() == kChi2PML) tChi2 = GetPmlValue(tNumContent[ix],tDenContent[ix],tCorrectedFitCfContent[ix]);
          else if(fFitSharedAnalyses->GetFitType() == kChi2) tChi2 = GetChi2Value(ix+1,tCf,tCorrectedFitCfContent[ix]);
          else tChi2 = 0.;

          fChi2Vec[iAnaly] += tChi2;
          fChi2 += tChi2;

          fNpFitsVec[iAnaly]++;
          fNpFits++;
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
//gObjectTable->Print();

  tTotalTimer.Stop();
  if(fVerbose)
  {
    cout << "fChi2 = " << fChi2 << endl;
    cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl;

    cout << "tTotalTimer: ";
    tTotalTimer.PrintInterval();
    cout << "____________________________________________" << endl;
  }
/*
  double *tParamsForHistograms = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tParamsForHistograms[i] = par[i];
  fFitSharedAnalyses->GetFitChi2Histograms()->FillHistograms(fChi2,tParamsForHistograms);
  delete[] tParamsForHistograms;
*/
}


//________________________________________________________________________________________________________________
void CoulombFitter::CalculateFakeChi2(int &npar, double &chi2, double *par)
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

    if(fUseStaticPairs && fIncludeSingletAndTriplet) tmp = (fFakeCf->GetBinContent(ix) - par[8]*GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    else if(fIncludeSingletAndTriplet) tmp = (fFakeCf->GetBinContent(ix) - par[8]*GetFitCfContentComplete(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    else if(fUseStaticPairs) tmp = (fFakeCf->GetBinContent(ix) - par[5]*GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    else tmp = (fFakeCf->GetBinContent(ix) - par[5]*GetFitCfContent(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    tChi2 += tmp*tmp;
    tNpfits++;
  }

  chi2 = tChi2;
cout << "tChi2 = " << tChi2 << endl << endl;
}


//________________________________________________________________________________________________________________
double CoulombFitter::GetChi2(TH1* aFitHistogram)
{
  fChi2 = 0.;
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {

      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();
      TH1* tCfToFit = tKStarCfLite->Cf();

      TAxis* tXaxis = tCfToFit->GetXaxis();
      int tNbinsX = tCfToFit->GetNbinsX();
      int tNbinsXToFit = tCfToFit->FindBin(fMaxFitKStar);
      if(tCfToFit->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;
      int tNbinsXFitHist = aFitHistogram->FindBin(fMaxFitKStar);
      if(aFitHistogram->GetBinLowEdge(tNbinsXFitHist) == fMaxFitKStar) tNbinsXFitHist--;

      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tNbinsXToFit == tNbinsXFitHist);

      double tmp;
      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tCfContent = aFitHistogram->GetBinContent(ix);

        if(tCfToFit->GetBinContent(ix)!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          tmp = (tCfToFit->GetBinContent(ix) - tCfContent)/tCfToFit->GetBinError(ix);

          fChi2Vec[iAnaly] += tmp*tmp;
          fChi2 += tmp*tmp;

          fNpFitsVec[iAnaly]++;
          fNpFits++;
        }
      }
    }
  }
  return fChi2;
}


//________________________________________________________________________________________________________________
TH1* CoulombFitter::CreateFitHistogram(TString aName, int aAnalysisNumber)
{
  FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

  SetCoulombAttributes(tAnalysisType);

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

    if(fUseStaticPairs && fIncludeSingletAndTriplet)
    {
      tCfContentUnNorm = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else if(fIncludeSingletAndTriplet)
    {
      tCfContentUnNorm = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else if(fUseStaticPairs)
    {
      tCfContentUnNorm = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[5]*tCfContentUnNorm;
    }
    else
    {
      tCfContentUnNorm = GetFitCfContent(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[5]*tCfContentUnNorm;
    }
    tReturnHist->SetBinContent(ix,tCfContent);
  }

  delete[] tPar;
  delete[] tParError;

  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1* CoulombFitter::CreateFitHistogramSample(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm)
{
  cout << "Beginning CreateFitHistogramSample" << endl;
ChronoTimer tTimer;
tTimer.Start();

  SetCoulombAttributes(aAnalysisType);

  TH1D* tReturnHist = new TH1D(aName,aName,aNbinsK,aKMin,aKMax);

  double *tPar = new double[6];
  tPar[0] = aLambda;
  tPar[1] = aR;
  tPar[2] = aReF0;
  tPar[3] = aImF0;
  tPar[4] = aD0;
  tPar[5] = aNorm;

  int tAnalysisNumber = 0;  //TODO

  double tKStarMin, tKStarMax, tCfContentUnNorm, tCfContent, tCfContentv2;
  for(int ix=1; ix <= aNbinsK; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);


//---------------------------------------------
ChronoTimer tLoopTimer;
tLoopTimer.Start();

        tCfContentUnNorm = GetFitCfContent(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
        tCfContent = aNorm*tCfContentUnNorm;

tLoopTimer.Stop();
cout << "Finished GetFitCfContent ";
tLoopTimer.PrintInterval();
cout << endl;

/*
//---------------------------------------------
        std::clock_t start2 = std::clock();

        tCfContentv2 = GetFitCfContentSerialv2(tKStarMin,tKStarMax,tPar,tAnalysisNumber);

        double duration2 = (std::clock() - start2)/(double) CLOCKS_PER_SEC;
        cout << "Finished GetFitCfContentSerialv2 in " << duration2 << " seconds" << endl;


        cout << "tCfContent = " << tCfContent << endl;
        cout << "tCfContentv2 = " << tCfContentv2 << endl << endl;
//---------------------------------------------
*/

    tReturnHist->SetBinContent(ix,tCfContent);
    tReturnHist->SetBinError(ix,.1/ix);
  }

tTimer.Stop();
cout << "Finished CreateFitHistogramSample ";
tTimer.PrintInterval();

  delete[] tPar;

  fFakeCf = tReturnHist;  //TODO delete this

  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1* CoulombFitter::CreateFitHistogramSampleComplete(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double aNorm)
{
  cout << "Beginning CreateFitHistogramSample" << endl;
ChronoTimer tTimer;
tTimer.Start();

  SetCoulombAttributes(aAnalysisType);

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

  int tAnalysisNumber = 0;  //TODO

  double tKStarMin, tKStarMax, tCfContentUnNorm, tCfContent;
  for(int ix=1; ix <= aNbinsK; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);


//---------------------------------------------
//ChronoTimer tLoopTimer;
//tLoopTimer.Start();
    
    if(fUseStaticPairs && fIncludeSingletAndTriplet)
    {
      tCfContentUnNorm = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else if(fIncludeSingletAndTriplet)
    {
      tCfContentUnNorm = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else if(fUseStaticPairs)
    {
      tCfContentUnNorm = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else
    {
      tCfContentUnNorm = GetFitCfContent(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }

//tLoopTimer.Stop();
//cout << "Finished GetFitCfContentComplete ";
//tLoopTimer.PrintInterval();
//cout << endl;

    tReturnHist->SetBinContent(ix,tCfContent);
  }

tTimer.Stop();
cout << "Finished CreateFitHistogramSampleComplete ";
tTimer.PrintInterval();

  delete[] tPar;

  fFakeCf = tReturnHist;  //TODO delete this

  return tReturnHist;
}



//________________________________________________________________________________________________________________
td1dVec CoulombFitter::GetCoulombResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams, vector<double> &aKStarBinCenters, TH2* aTransformMatrix)
{
  TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/";
  TString tFileName, tFullFileName;
  TString tFileNameHFunction, tFullFileNameHFunction;

  switch(aResidualType) {
  case kResXiCKchP:
  case kResAXiCKchM:
    tFileName = TString("InterpHistsAttractive.root");
    tFileNameHFunction = TString("LednickyHFunction.root");
    fBohrRadius = -gBohrRadiusXiK;
    break;

  case kResXiCKchM:
  case kResAXiCKchP:
    tFileName = TString("InterpHistsRepulsive.root");
    tFileNameHFunction = TString("LednickyHFunction.root");
    fBohrRadius = gBohrRadiusXiK;
    break;


  case kResOmegaKchP:
  case kResAOmegaKchM:
    tFileName = TString("InterpHists_OmegaKchP.root");
    tFileNameHFunction = TString("LednickyHFunction_OmegaKchP.root");
    fBohrRadius = -gBohrRadiusOmegaK;
    break;


  case kResOmegaKchM:
  case kResAOmegaKchP:
    tFileName = TString("InterpHists_OmegaKchM.root");
    tFileNameHFunction = TString("LednickyHFunction_OmegaKchM.root");
    fBohrRadius = gBohrRadiusOmegaK;
    break;


  default:
    cout << "ERROR: CoulombFitter::GetCoulombResidualCorrelation: Invalid aResidualType = " << aResidualType << endl;
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

    tParentCf[i] = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,aParentCfParams,tAnalysisNumber);
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
void CoulombFitter::DoFit()
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


