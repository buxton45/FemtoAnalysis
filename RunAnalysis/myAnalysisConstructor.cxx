
#include "myAnalysisConstructor.h"
#include "TObjArray.h"
#include "AliESDtrack.h"
#ifdef __ROOT__
ClassImp(myAnalysisConstructor)
#endif

static const double PionMass = 0.13956995,
                    KchMass = 0.493677,
                    K0ShortMass = 0.497614,
                    ProtonMass = 0.938272013,
                    LambdaMass = 1.115683,
		    XiMass     = 1.32171;

//____________________________
const char* const myAnalysisConstructor::fAnalysisTags[] = {"LamK0", "ALamK0", "LamKchP", "ALamKchP", "LamKchM", "ALamKchM", "LamLam", "ALamALam", "LamALam", "LamPiP", "ALamPiP", "LamPiM", "ALamPiM", "XiKchP", "AXiKchP", "XiKchM", "AXiKchM"};

//____________________________
myAnalysisConstructor::myAnalysisConstructor() : 
  AliFemtoVertexMultAnalysis(),
  fAnalysisType(kLamK0),
  fCollectionOfCfs(0),
  fOutputName("Analysis"),
  fMultHist(0),
  fIsMCRun(kFALSE),
  fIsMBAnalysis(kFALSE),
  fImplementAvgSepCuts(kTRUE),
  fWritePairKinematics(kFALSE),
  fMinCent(-1000),
  fMaxCent(1000),
  BasicEvCut(0),
  EvCutEst(0),
  LamCut(0),
  ALamCut(0),
  KStarCf(0),
  AvgSepCf(0),
  SepCfs(0),
  AvgSepCfCowboysAndSailors(0),
  KStarCf2D(0),
  KStarCfMC(0),
  KStarModelCfs(0),
  K0Cut(0),
  V0PairCut(0),
  KchPCut(0),
  KchMCut(0),
  PiPCut(0),
  PiMCut(0),
  V0TrackPairCut(0),
  XiCut(0),
  AXiCut(0),
  XiTrackPairCut(0),

  LamCutNSigmaFilter(0),
  ALamCutNSigmaFilter(0),
  K0CutNSigmaFilter(0),
  fUseAliFemtoV0TrackCutNSigmaFilter(false),
  fUseCustomNSigmaFilters(false)
{
  SetVerboseMode(kFALSE);
  fMultHist = new TH1F("MultHist","MultHist",30,0,3000);
  fCollectionOfCfs = new AliFemtoCorrFctnCollection;
  SetEnablePairMonitors(fIsMCRun);
}

//____________________________
myAnalysisConstructor::myAnalysisConstructor(AnalysisType aAnalysisType, const char* name, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics) : 
  AliFemtoVertexMultAnalysis(),
  fAnalysisType(aAnalysisType),
  fCollectionOfCfs(0),
  fOutputName(name),
  fMultHist(0),
  fIsMCRun(aIsMCRun),
  fIsMBAnalysis(kFALSE),
  fImplementAvgSepCuts(aImplementAvgSepCuts),
  fWritePairKinematics(aWritePairKinematics),
  fMinCent(-1000),
  fMaxCent(1000),
  BasicEvCut(0),
  EvCutEst(0),
  LamCut(0),
  ALamCut(0),
  KStarCf(0),
  AvgSepCf(0),
  SepCfs(0),
  AvgSepCfCowboysAndSailors(0),
  KStarCf2D(0),
  KStarCfMC(0),
  KStarModelCfs(0),
  K0Cut(0),
  V0PairCut(0),
  KchPCut(0),
  KchMCut(0),
  PiPCut(0),
  PiMCut(0),
  V0TrackPairCut(0),
  XiCut(0),
  AXiCut(0),
  XiTrackPairCut(0),

  LamCutNSigmaFilter(0),
  ALamCutNSigmaFilter(0),
  K0CutNSigmaFilter(0),
  fUseAliFemtoV0TrackCutNSigmaFilter(false),
  fUseCustomNSigmaFilters(false)
{
  SetVerboseMode(kFALSE);
  SetNumEventsToMix(5);
  SetMinSizePartCollection(1);
  SetV0SharedDaughterCut(kTRUE);
  SetEnablePairMonitors(fIsMCRun);

  char tTitMultHist[101] = "MultHist_";
  strncat(tTitMultHist,fAnalysisTags[aAnalysisType],100);
  fMultHist = new TH1F(tTitMultHist,tTitMultHist,30,0,3000);

  fCollectionOfCfs = new AliFemtoCorrFctnCollection;
}

//____________________________
myAnalysisConstructor::myAnalysisConstructor(AnalysisType aAnalysisType, const char* name, unsigned int binsVertex, double minVertex, double maxVertex, unsigned int binsMult, double minMult, double maxMult, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics) : 
  AliFemtoVertexMultAnalysis(binsVertex,minVertex,maxVertex,binsMult,minMult,maxMult),
  fAnalysisType(aAnalysisType),
  fCollectionOfCfs(0),
  fOutputName(name),
  fIsMCRun(aIsMCRun),
  fIsMBAnalysis(kFALSE),
  fImplementAvgSepCuts(aImplementAvgSepCuts),
  fWritePairKinematics(aWritePairKinematics),
  fMinCent(-1000),
  fMaxCent(1000),
  fMultHist(0),
  BasicEvCut(0),
  EvCutEst(0),
  LamCut(0),
  ALamCut(0),
  KStarCf(0),
  AvgSepCf(0),
  SepCfs(0),
  AvgSepCfCowboysAndSailors(0),
  KStarCf2D(0),
  KStarCfMC(0),
  KStarModelCfs(0),
  K0Cut(0),
  V0PairCut(0),
  KchPCut(0),
  KchMCut(0),
  PiPCut(0),
  PiMCut(0),
  V0TrackPairCut(0),
  XiCut(0),
  AXiCut(0),
  XiTrackPairCut(0),

  LamCutNSigmaFilter(0),
  ALamCutNSigmaFilter(0),
  K0CutNSigmaFilter(0),
  fUseAliFemtoV0TrackCutNSigmaFilter(false),
  fUseCustomNSigmaFilters(false)
{
  SetVerboseMode(kFALSE);
  SetNumEventsToMix(5);
  SetMinSizePartCollection(1);
  SetV0SharedDaughterCut(kTRUE);
  SetEnablePairMonitors(fIsMCRun);

  fMinCent = minMult/10.;
  fMaxCent = maxMult/10.;

  fCollectionOfCfs = new AliFemtoCorrFctnCollection;

  char tTitMultHist[101] = "MultHist_";
  strncat(tTitMultHist,fAnalysisTags[aAnalysisType],100);
  fMultHist = new TH1F(tTitMultHist,tTitMultHist,30,0,3000);

  char tTitKStarCf[101] = "KStarCf_";
  strncat(tTitKStarCf,fAnalysisTags[aAnalysisType],100);
  if(fWritePairKinematics) KStarCf = CreateKStarCorrFctn(tTitKStarCf,62,0.,0.31); //TNtuple is huge, and I don't need data out to 1 GeV
  else KStarCf = CreateKStarCorrFctn(tTitKStarCf,200,0.,1.0);

  char tTitAvgSepCf[101] = "AvgSepCf_";
  strncat(tTitAvgSepCf,fAnalysisTags[aAnalysisType],100);
  AvgSepCf = CreateAvgSepCorrFctn(tTitAvgSepCf,200,0.,20.);

  char tTitSepCfs[101] = "SepCfs_";
  strncat(tTitSepCfs,fAnalysisTags[aAnalysisType],100);
  SepCfs = CreateSepCorrFctns(tTitSepCfs,10,0.,10.,200,0.,20.);

  char tTitAvgSepCfCowboysAndSailors[101] = "AvgSepCfCowboysAndSailors_";
  strncat(tTitAvgSepCfCowboysAndSailors,fAnalysisTags[aAnalysisType],100);
  AvgSepCfCowboysAndSailors = CreateAvgSepCorrFctnCowboysAndSailors(tTitAvgSepCfCowboysAndSailors,40,-5.,5.,200,0.,20.);

  char tTitKStarCf2D[101] = "KStarCf2D_";
  strncat(tTitKStarCf2D,fAnalysisTags[aAnalysisType],100);
  KStarCf2D = CreateKStarCorrFctn2D(tTitKStarCf2D,200,0.,1.0,2,-2.,2.);

  char tTitKStarCfMC[101] = "KStarCfTrue_";
  strncat(tTitKStarCfMC,fAnalysisTags[aAnalysisType],100);
  KStarCfMC = CreateKStarCorrFctnMC(tTitKStarCfMC,200,0.,1.0);

  //-----04/02/2016
  char tTitModelCorrFctnKStar[101] = "KStarModelCf_";
  strncat(tTitModelCorrFctnKStar,fAnalysisTags[aAnalysisType],100);
  KStarModelCfs = CreateModelCorrFctnKStar(tTitModelCorrFctnKStar,200,0.,1.0);

  if(fWritePairKinematics)
  {
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCf);
  }
  else
  {
    //cannot push_back fMultHist into fCollectionOfCfs because it is not of type AliFemtoCorrFctn
    //It cannot even be added via AliFemtoSimpleAnalysis::AddCorrFctn because it is not a cf
    //fMultHist is added in myAnalysisConstructor::ProcessEvent
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCf);
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)AvgSepCf);
    //fCollectionOfCfs->push_back((AliFemtoCorrFctn*)SepCfs);
    //fCollectionOfCfs->push_back((AliFemtoCorrFctn*)AvgSepCfCowboysAndSailors);
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCf2D);
  }

  if(fIsMCRun) 
  {
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCfMC);
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarModelCfs);
  }

  //Run the CreateXXXXXAnalysis, but do not set the analysis yet
  //This allows me to tweak the analysis in the Config file
  if(fAnalysisType == kLamK0) {CreateLamK0Analysis();}
  else if(fAnalysisType == kALamK0) {CreateALamK0Analysis();}

  else if(fAnalysisType == kLamKchP) {CreateLamKchPAnalysis();}
  else if(fAnalysisType == kALamKchP) {CreateALamKchPAnalysis();}
  else if(fAnalysisType == kLamKchM) {CreateLamKchMAnalysis();}
  else if(fAnalysisType == kALamKchM) {CreateALamKchMAnalysis();}

  else if(fAnalysisType == kLamLam) {CreateLamLamAnalysis();}
  else if(fAnalysisType == kALamALam) {CreateALamALamAnalysis();}
  else if(fAnalysisType == kLamALam) {CreateLamALamAnalysis();}

  else if(fAnalysisType == kLamPiP) {CreateLamPiPAnalysis();}
  else if(fAnalysisType == kALamPiP) {CreateALamPiPAnalysis();}
  else if(fAnalysisType == kLamPiM) {CreateLamPiMAnalysis();}
  else if(fAnalysisType == kALamPiM) {CreateALamPiMAnalysis();}

  else if(fAnalysisType == kXiKchP) {CreateXiKchPAnalysis();}
  else if(fAnalysisType == kAXiKchP) {CreateAXiKchPAnalysis();}
  else if(fAnalysisType == kXiKchM) {CreateXiKchMAnalysis();}
  else if(fAnalysisType == kAXiKchM) {CreateAXiKchMAnalysis();}

  else {cout << "ERROR in constructor:  No/Incorrect AnalysisType selected!!!!!!!!" << endl << endl << endl;}


}

//____________________________
//copy constructor - 30 June 2015
myAnalysisConstructor::myAnalysisConstructor(const myAnalysisConstructor& a) :
  AliFemtoVertexMultAnalysis(a),  //call the copy-constructor of the base
  fAnalysisType(kLamK0),
  fCollectionOfCfs(0),
  fOutputName(0),
  fMultHist(0),
  fIsMCRun(kFALSE),
  fIsMBAnalysis(kFALSE),
  fImplementAvgSepCuts(kFALSE),
  fWritePairKinematics(kFALSE),
  fMinCent(-1000),
  fMaxCent(1000),
  BasicEvCut(0),
  EvCutEst(0),
  LamCut(0),
  ALamCut(0),
  KStarCf(0),
  AvgSepCf(0),
  SepCfs(0),
  AvgSepCfCowboysAndSailors(0),
  KStarCf2D(0),
  KStarCfMC(0),
  KStarModelCfs(0),
  K0Cut(0),
  V0PairCut(0),
  KchPCut(0),
  KchMCut(0),
  PiPCut(0),
  PiMCut(0),
  V0TrackPairCut(0),
  XiCut(0),
  AXiCut(0),
  XiTrackPairCut(0),

  LamCutNSigmaFilter(0),
  ALamCutNSigmaFilter(0),
  K0CutNSigmaFilter(0),
  fUseAliFemtoV0TrackCutNSigmaFilter(false),
  fUseCustomNSigmaFilters(false)
{
  fCollectionOfCfs = new AliFemtoCorrFctnCollection;
  AliFemtoCorrFctnIterator iter;
  for(iter=a.fCollectionOfCfs->begin(); iter!=a.fCollectionOfCfs->end(); iter++)
  {
    AliFemtoCorrFctn* fctn = (*iter)->Clone();
    if(fctn) {AddCorrFctn(fctn);}
    else {cout << " myAnalysisConstructor::myAnalysisConstructor(const myAnalysisConstructor& a) - correlation function not found " << endl;}
  }
}

//____________________________
//assignment operator - 30 June 2015
myAnalysisConstructor& myAnalysisConstructor::operator=(const myAnalysisConstructor& TheOriginalAnalysis)
{
  if(this == &TheOriginalAnalysis) {return *this;}

  AliFemtoVertexMultAnalysis::operator=(TheOriginalAnalysis);  //call the assignment operator of the base

  fAnalysisType = kLamK0;
  fCollectionOfCfs = 0;
  fOutputName = "Analysis";
  fMultHist = 0;
  fIsMCRun = kFALSE;
  fIsMBAnalysis = kFALSE;
  fImplementAvgSepCuts = kFALSE;
  fWritePairKinematics = kFALSE;
  fMinCent = -1000;
  fMaxCent = 1000;
  BasicEvCut = 0;
  EvCutEst = 0;
  LamCut = 0;
  ALamCut = 0;
  KStarCf = 0;
  AvgSepCf = 0;
  SepCfs = 0;
  AvgSepCfCowboysAndSailors = 0;
  KStarCf2D = 0;
  KStarCfMC = 0;
  KStarModelCfs = 0;
  K0Cut = 0;
  V0PairCut = 0;
  KchPCut = 0;
  KchMCut = 0;
  PiPCut = 0;
  PiMCut = 0;
  V0TrackPairCut = 0;
  XiCut = 0;
  AXiCut = 0;
  XiTrackPairCut = 0;

  LamCutNSigmaFilter = 0;
  ALamCutNSigmaFilter = 0;
  K0CutNSigmaFilter = 0;
  fUseAliFemtoV0TrackCutNSigmaFilter = false;
  fUseCustomNSigmaFilters = false;

  if(fCollectionOfCfs) delete fCollectionOfCfs;
  fCollectionOfCfs = new AliFemtoCorrFctnCollection;
  AliFemtoCorrFctnIterator iter;
  for(iter=TheOriginalAnalysis.fCollectionOfCfs->begin(); iter!=TheOriginalAnalysis.fCollectionOfCfs->end(); iter++)
  {
    AliFemtoCorrFctn* fctn = (*iter)->Clone();
    if(fctn) AddCorrFctn(fctn);
  }

  return *this;
}

//____________________________
myAnalysisConstructor::~myAnalysisConstructor()
{

  AliFemtoCorrFctnIterator iter;
  for(iter=fCollectionOfCfs->begin(); iter!=fCollectionOfCfs->end(); iter++)
  {
    delete *iter;
  }
  delete fCollectionOfCfs;

}


//____________________________
void myAnalysisConstructor::CreateLamK0Analysis()
{ 
  //-----LamK0 analysis-----------------------------------------------------------------
  cout << "Setting up LamK0 analysis for " << fOutputName << endl;

  LamCut = CreateLambdaCut();
  K0Cut = CreateK0ShortCut();

  LamCutNSigmaFilter = CreateLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);
  K0CutNSigmaFilter = CreateK0ShortCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0PairCut = CreateV0PairCut(6.,0.,0.,6.);}
  else{V0PairCut = CreateV0PairCut(0.,0.,0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"LamK0"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetLamK0Analysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,LamCutNSigmaFilter,K0CutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else {SetAnalysis(BasicEvCut,LamCut,K0Cut,V0PairCut,fCollectionOfCfs);}
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,LamCutNSigmaFilter,K0CutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else {SetAnalysis(EvCutEst,LamCut,K0Cut,V0PairCut,fCollectionOfCfs);}
  }
}



//____________________________
void myAnalysisConstructor::CreateALamK0Analysis()
{ 
  //-----ALamK0 analysis-----------------------------------------------------------------
  cout << "Setting up ALamK0 analysis for " << fOutputName << endl;

  ALamCut = CreateAntiLambdaCut();
  K0Cut = CreateK0ShortCut();

  ALamCutNSigmaFilter = CreateAntiLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);
  K0CutNSigmaFilter = CreateK0ShortCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0PairCut = CreateV0PairCut(6.,0.,0.,6.);}
  else{V0PairCut = CreateV0PairCut(0.,0.,0.,0.);}


  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"ALamK0"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetALamK0Analysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,ALamCutNSigmaFilter,K0CutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else {SetAnalysis(BasicEvCut,ALamCut,K0Cut,V0PairCut,fCollectionOfCfs);}
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,ALamCutNSigmaFilter,K0CutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else {SetAnalysis(EvCutEst,ALamCut,K0Cut,V0PairCut,fCollectionOfCfs);}
  }
}



//____________________________
void myAnalysisConstructor::CreateLamKchPAnalysis()
{ 
  //-----LamKchP analysis-----------------------------------------------------------------
  cout << "Setting up LamKchP analysis for " << fOutputName << endl;

  LamCut = CreateLambdaCut();
  KchPCut = CreateKchCut(1);

  LamCutNSigmaFilter = CreateLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0TrackPairCut = CreateV0TrackPairCut(8.,0.);}
  else{V0TrackPairCut = CreateV0TrackPairCut(0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"LamKchP"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetLamKchPAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,LamCutNSigmaFilter,KchPCut,V0TrackPairCut,fCollectionOfCfs);}
    else {SetAnalysis(BasicEvCut,LamCut,KchPCut,V0TrackPairCut,fCollectionOfCfs);}
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,LamCutNSigmaFilter,KchPCut,V0TrackPairCut,fCollectionOfCfs);}
    else {SetAnalysis(EvCutEst,LamCut,KchPCut,V0TrackPairCut,fCollectionOfCfs);}
  }
}

//____________________________
void myAnalysisConstructor::CreateALamKchPAnalysis()
{ 
  //-----ALamKchP analysis-----------------------------------------------------------------
  cout << "Setting up ALamKchP analysis for " << fOutputName << endl;

  ALamCut = CreateAntiLambdaCut();
  KchPCut = CreateKchCut(1);

  ALamCutNSigmaFilter = CreateAntiLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0TrackPairCut = CreateV0TrackPairCut(8.,0.);}
  else{V0TrackPairCut = CreateV0TrackPairCut(0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"ALamKchP"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetALamKchPAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,ALamCutNSigmaFilter,KchPCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,ALamCut,KchPCut,V0TrackPairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,ALamCutNSigmaFilter,KchPCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,ALamCut,KchPCut,V0TrackPairCut,fCollectionOfCfs);
  }
}


//____________________________
void myAnalysisConstructor::CreateLamKchMAnalysis()
{ 
  //-----LamKchM analysis-----------------------------------------------------------------
  cout << "Setting up LamKchM analysis for " << fOutputName << endl;

  LamCut = CreateLambdaCut();
  KchMCut = CreateKchCut(-1);

  LamCutNSigmaFilter = CreateLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0TrackPairCut = CreateV0TrackPairCut(0.,8.);}
  else{V0TrackPairCut = CreateV0TrackPairCut(0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"LamKchM"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetLamKchMAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,LamCutNSigmaFilter,KchMCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,LamCut,KchMCut,V0TrackPairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,LamCutNSigmaFilter,KchMCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,LamCut,KchMCut,V0TrackPairCut,fCollectionOfCfs);
  }
}


//____________________________
void myAnalysisConstructor::CreateALamKchMAnalysis()
{ 
  //-----ALamKchM analysis-----------------------------------------------------------------
  cout << "Setting up ALamKchM analysis for " << fOutputName << endl;

  ALamCut = CreateAntiLambdaCut();
  KchMCut = CreateKchCut(-1);

  ALamCutNSigmaFilter = CreateAntiLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0TrackPairCut = CreateV0TrackPairCut(0.,8.);}
  else{V0TrackPairCut = CreateV0TrackPairCut(0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"ALamKchM"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetALamKchMAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,ALamCutNSigmaFilter,KchMCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,ALamCut,KchMCut,V0TrackPairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,ALamCutNSigmaFilter,KchMCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,ALamCut,KchMCut,V0TrackPairCut,fCollectionOfCfs);
  }
}


//____________________________
void myAnalysisConstructor::CreateLamLamAnalysis()
{ 
  //-----LamLam analysis-----------------------------------------------------------------
  cout << "Setting up LamLam analysis for " << fOutputName << endl;

  LamCut = CreateLambdaCut();
  LamCutNSigmaFilter = CreateLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0PairCut = CreateV0PairCut(3.,0.,0.,4.);}
  else{V0PairCut = CreateV0PairCut(0.,0.,0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"LamLam"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetLamLamAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,LamCutNSigmaFilter,LamCutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,LamCut,LamCut,V0PairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,LamCutNSigmaFilter,LamCutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,LamCut,LamCut,V0PairCut,fCollectionOfCfs);
  }
}

//____________________________
void myAnalysisConstructor::CreateALamALamAnalysis()
{ 
  //-----ALamALam analysis-----------------------------------------------------------------
  cout << "Setting up ALamALam analysis for " << fOutputName << endl;

  ALamCut = CreateAntiLambdaCut();
  ALamCutNSigmaFilter = CreateAntiLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0PairCut = CreateV0PairCut(4.,0.,0.,3.);}
  else{V0PairCut = CreateV0PairCut(0.,0.,0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"ALamALam"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetALamALamAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,ALamCutNSigmaFilter,ALamCutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,ALamCut,ALamCut,V0PairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,ALamCutNSigmaFilter,ALamCutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,ALamCut,ALamCut,V0PairCut,fCollectionOfCfs);
  }
}


//____________________________
void myAnalysisConstructor::CreateLamALamAnalysis()
{ 
  //-----LamALam analysis-----------------------------------------------------------------
  cout << "Setting up LamALam analysis for " << fOutputName << endl;

  LamCut = CreateLambdaCut();
  ALamCut = CreateAntiLambdaCut();

  LamCutNSigmaFilter = CreateLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);
  ALamCutNSigmaFilter = CreateAntiLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0PairCut = CreateV0PairCut(3.5,0.,0.,3.5);}
  else{V0PairCut = CreateV0PairCut(0.,0.,0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"LamALam"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetLamALamAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,LamCutNSigmaFilter,ALamCutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,LamCut,ALamCut,V0PairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,LamCutNSigmaFilter,ALamCutNSigmaFilter,V0PairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,LamCut,ALamCut,V0PairCut,fCollectionOfCfs);
  }
}

//____________________________
void myAnalysisConstructor::CreateLamPiPAnalysis()
{ 
  //-----LamPiP analysis-----------------------------------------------------------------
  cout << "Setting up LamPiP analysis for " << fOutputName << endl;

  LamCut = CreateLambdaCut();
  PiPCut = CreatePiCut(1);

  LamCutNSigmaFilter = CreateLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0TrackPairCut = CreateV0TrackPairCut(8.,0.);}
  else{V0TrackPairCut = CreateV0TrackPairCut(0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"LamPiP"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetLamPiPAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,LamCutNSigmaFilter,PiPCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,LamCut,PiPCut,V0TrackPairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,LamCutNSigmaFilter,PiPCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,LamCut,PiPCut,V0TrackPairCut,fCollectionOfCfs);
  }
}

//____________________________
void myAnalysisConstructor::CreateALamPiPAnalysis()
{ 
  //-----ALamPiP analysis-----------------------------------------------------------------
  cout << "Setting up ALamPiP analysis for " << fOutputName << endl;

  ALamCut = CreateAntiLambdaCut();
  PiPCut = CreatePiCut(1);

  ALamCutNSigmaFilter = CreateAntiLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0TrackPairCut = CreateV0TrackPairCut(8.,0.);}
  else{V0TrackPairCut = CreateV0TrackPairCut(0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"ALamPiP"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetALamPiPAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,ALamCutNSigmaFilter,PiPCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,ALamCut,PiPCut,V0TrackPairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,ALamCutNSigmaFilter,PiPCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,ALamCut,PiPCut,V0TrackPairCut,fCollectionOfCfs);
  }
}


//____________________________
void myAnalysisConstructor::CreateLamPiMAnalysis()
{ 
  //-----LamPiM analysis-----------------------------------------------------------------
  cout << "Setting up LamPiM analysis for " << fOutputName << endl;

  LamCut = CreateLambdaCut();
  PiMCut = CreatePiCut(-1);

  LamCutNSigmaFilter = CreateLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0TrackPairCut = CreateV0TrackPairCut(0.,8.);}
  else{V0TrackPairCut = CreateV0TrackPairCut(0.,0.);}

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"LamPiM"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetLamPiMAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,LamCutNSigmaFilter,PiMCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,LamCut,PiMCut,V0TrackPairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,LamCutNSigmaFilter,PiMCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(EvCutEst,LamCut,PiMCut,V0TrackPairCut,fCollectionOfCfs);
  }
}

//____________________________
void myAnalysisConstructor::CreateALamPiMAnalysis()
{ 
  //-----ALamPiM analysis-----------------------------------------------------------------
  cout << "Setting up ALamPiM analysis for " << fOutputName << endl;

  ALamCut = CreateAntiLambdaCut();
  PiMCut = CreatePiCut(-1);

  ALamCutNSigmaFilter = CreateAntiLambdaCutNSigmaFilter(fUseCustomNSigmaFilters);

  if(fImplementAvgSepCuts){V0TrackPairCut = CreateV0TrackPairCut(0.,8.);}
  else{V0TrackPairCut = CreateV0TrackPairCut(0.,0.);}


  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"ALamPiM"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetALamPiMAnalysis()
{
  if(fIsMBAnalysis)
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(BasicEvCut,ALamCutNSigmaFilter,PiMCut,V0TrackPairCut,fCollectionOfCfs);}
    else SetAnalysis(BasicEvCut,ALamCut,PiMCut,V0TrackPairCut,fCollectionOfCfs);
  }
  else
  {
    if(fUseAliFemtoV0TrackCutNSigmaFilter) {SetAnalysis(EvCutEst,ALamCutNSigmaFilter,PiMCut,V0TrackPairCut,fCollectionOfCfs);}
    SetAnalysis(EvCutEst,ALamCut,PiMCut,V0TrackPairCut,fCollectionOfCfs);
  }
}



//____________________________
void myAnalysisConstructor::CreateXiKchPAnalysis()
{ 
  //-----XiKchP analysis-----------------------------------------------------------------
  cout << "Setting up XiKchP analysis for " << fOutputName << endl;

  XiCut = CreateXiCut();
  KchPCut = CreateKchCut(1);

  XiTrackPairCut = CreateXiTrackPairCut();

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"XiKchP"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetXiKchPAnalysis()
{
  if(fIsMBAnalysis)
  {
    SetAnalysis(BasicEvCut,XiCut,KchPCut,XiTrackPairCut,fCollectionOfCfs);
  }
  else
  {
    SetAnalysis(EvCutEst,XiCut,KchPCut,XiTrackPairCut,fCollectionOfCfs);
  }
}

//____________________________
void myAnalysisConstructor::CreateAXiKchPAnalysis()
{ 
  //-----AXiKchP analysis-----------------------------------------------------------------
  cout << "Setting up AXiKchP analysis for " << fOutputName << endl;

  AXiCut = CreateAntiXiCut();
  KchPCut = CreateKchCut(1);

  XiTrackPairCut = CreateXiTrackPairCut();

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"AXiKchP"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetAXiKchPAnalysis()
{
  if(fIsMBAnalysis)
  {
    SetAnalysis(BasicEvCut,AXiCut,KchPCut,XiTrackPairCut,fCollectionOfCfs);
  }
  else
  {
    SetAnalysis(EvCutEst,AXiCut,KchPCut,XiTrackPairCut,fCollectionOfCfs);
  }
}


//____________________________
void myAnalysisConstructor::CreateXiKchMAnalysis()
{ 
  //-----XiKchM analysis-----------------------------------------------------------------
  cout << "Setting up XiKchM analysis for " << fOutputName << endl;

  XiCut = CreateXiCut();
  KchMCut = CreateKchCut(-1);

  XiTrackPairCut = CreateXiTrackPairCut();

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"XiKchM"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetXiKchMAnalysis()
{
  if(fIsMBAnalysis)
  {
    SetAnalysis(BasicEvCut,XiCut,KchMCut,XiTrackPairCut,fCollectionOfCfs);
  }
  else
  {
    SetAnalysis(EvCutEst,XiCut,KchMCut,XiTrackPairCut,fCollectionOfCfs);
  }
}

//____________________________
void myAnalysisConstructor::CreateAXiKchMAnalysis()
{ 
  //-----AXiKchM analysis-----------------------------------------------------------------
  cout << "Setting up AXiKchM analysis for " << fOutputName << endl;

  AXiCut = CreateAntiXiCut();
  KchMCut = CreateKchCut(-1);

  XiTrackPairCut = CreateXiTrackPairCut();

  //Check to see if min bias (-> CreateBasicEventCut) or centrality dependent (-> CreateEventCutEstimators)
  if(!strcmp(fOutputName,"AXiKchM"))  //strcmp returns 0 if the contents of both strings are equal
  {
    //Min-bias analysis, i.e. no centrality tag in name
    if( (fMinCent != 0.) && (fMaxCent != 100.)) {cout << "WARNING!!!!!!!!!!!" << endl << "Centrality limits imply this analysis (" << fOutputName << ") is NOT min bias!!!!!" << endl;}
    BasicEvCut = CreateBasicEventCut();
    fIsMBAnalysis = kTRUE;
  }
  else
  {
    //Centrality dependent analysis
    EvCutEst = CreateEventCutEstimators(fMinCent,fMaxCent);
    fIsMBAnalysis = kFALSE;
  }

}

//____________________________
void myAnalysisConstructor::SetAXiKchMAnalysis()
{
  if(fIsMBAnalysis)
  {
    SetAnalysis(BasicEvCut,AXiCut,KchMCut,XiTrackPairCut,fCollectionOfCfs);
  }
  else
  {
    SetAnalysis(EvCutEst,AXiCut,KchMCut,XiTrackPairCut,fCollectionOfCfs);
  }
}



//____________________________
void myAnalysisConstructor::SetCorrectAnalysis()
{
  if(fAnalysisType == kLamK0) {SetLamK0Analysis();}
  else if(fAnalysisType == kALamK0) {SetALamK0Analysis();}

  else if(fAnalysisType == kLamKchP) {SetLamKchPAnalysis();}
  else if(fAnalysisType == kALamKchP) {SetALamKchPAnalysis();}
  else if(fAnalysisType == kLamKchM) {SetLamKchMAnalysis();}
  else if(fAnalysisType == kALamKchM) {SetALamKchMAnalysis();}

  else if(fAnalysisType == kLamLam) {SetLamLamAnalysis();}
  else if(fAnalysisType == kALamALam) {SetALamALamAnalysis();}
  else if(fAnalysisType == kLamALam) {SetLamALamAnalysis();}

  else if(fAnalysisType == kLamPiP) {SetLamPiPAnalysis();}
  else if(fAnalysisType == kALamPiP) {SetALamPiPAnalysis();}
  else if(fAnalysisType == kLamPiM) {SetLamPiMAnalysis();}
  else if(fAnalysisType == kALamPiM) {SetALamPiMAnalysis();}

  else if(fAnalysisType == kXiKchP) {SetXiKchPAnalysis();}
  else if(fAnalysisType == kAXiKchP) {SetAXiKchPAnalysis();}
  else if(fAnalysisType == kXiKchM) {SetXiKchMAnalysis();}
  else if(fAnalysisType == kAXiKchM) {SetAXiKchMAnalysis();}

  else {cout << "ERROR in SetCorrectAnalysis:  No/Incorrect AnalysisType selected!!!!!!!!" << endl << endl << endl;}
}





//____________________________
void myAnalysisConstructor::ProcessEvent(const AliFemtoEvent* hbtEvent)
{
  double multiplicity = hbtEvent->UncorrectedNumberOfPrimaries();
  fMultHist->Fill(multiplicity);
  AliFemtoVertexMultAnalysis::ProcessEvent(hbtEvent);
}

//____________________________
TList* myAnalysisConstructor::GetOutputList()
{
  TList *olist = new TList();
  TObjArray *temp = new TObjArray();
  olist->SetName(fOutputName);
  temp->SetName(fOutputName);

  TList *tOutputList = AliFemtoSimpleAnalysis::GetOutputList(); 
  myAliFemtoV0TrackCut* p1cut = dynamic_cast <myAliFemtoV0TrackCut*> (fFirstParticleCut);
  if(p1cut)
    {
      tOutputList->Add(p1cut->GetPurityHisto());
      tOutputList->Add(p1cut->GetMisIDHisto());
    }
  myAliFemtoV0TrackCut* p2cut = dynamic_cast <myAliFemtoV0TrackCut*> (fSecondParticleCut);
  if(p2cut)
    {
      tOutputList->Add(p2cut->GetPurityHisto());
      tOutputList->Add(p2cut->GetMisIDHisto());
    }

  tOutputList->Add(GetMultHist());

  TListIter next(tOutputList);
  while (TObject *obj = next())
  {
    temp->Add(obj);
  }

  olist->Add(temp);    
  return olist;
}

//____________________________
AliFemtoBasicEventCut* myAnalysisConstructor::CreateBasicEventCut()
{
  AliFemtoBasicEventCut* mec = new AliFemtoBasicEventCut();
    //Accept events with the given multiplicity
    mec->SetEventMult(0,100000);
    //and z-vertex distance to the center of the TPC
    mec->SetVertZPos(-10.0,10.0);

    mec->AddCutMonitor(new AliFemtoCutMonitorEventMult("_EvPass"), new AliFemtoCutMonitorEventMult("_EvFail"));

  return mec;
}

//____________________________
TH1F *myAnalysisConstructor::GetMultHist()
{
  return fMultHist;
}

//____________________________
AliFemtoEventCutEstimators* myAnalysisConstructor::CreateEventCutEstimators(const float &aCentLow, const float &aCentHigh)
{
  AliFemtoEventCutEstimators* EvCutEst = new AliFemtoEventCutEstimators();
    EvCutEst->SetCentEst1Range(aCentLow,aCentHigh);
    EvCutEst->SetVertZPos(-8.0,8.0);

    EvCutEst->AddCutMonitor(new AliFemtoCutMonitorEventMult("_EvPass"), new AliFemtoCutMonitorEventMult("_EvFail"));

    EvCutEst->AddCutMonitor(new AliFemtoCutMonitorEventPartCollSize("_Part1",100,0,100,"_Part2",100,0,100));

  return EvCutEst;
}

//____________________________
myAliFemtoV0TrackCut* myAnalysisConstructor::CreateLambdaCut()
{
  // V0 Track Cut (1 = lambda)
  myAliFemtoV0TrackCut* v0cut1 = new myAliFemtoV0TrackCut();
    v0cut1->SetParticleType(0);  //  0=lambda -> daughters = proton(+) and pi-
    v0cut1->SetMass(LambdaMass);
    v0cut1->SetInvariantMassLambda(LambdaMass-0.0038,LambdaMass+0.0038);   //m_inv criteria for lambda's
    v0cut1->SetRemoveMisidentified(kTRUE);
    v0cut1->SetInvMassMisidentified(K0ShortMass-0.003677,K0ShortMass+0.003677);  //m_inv criteria to remove all lambda candidates fulfilling K0short hypothesis
    //v0cut1->SetInvMassMisidentified(K0ShortMass-0.013677,K0ShortMass+0.013677);  //m_inv criteria to remove all lambda candidates fulfilling K0short hypothesis
    v0cut1->SetMisIDHisto("MisIDLambdas",100,LambdaMass-0.035,LambdaMass+0.035);
    v0cut1->SetCalculatePurity(kTRUE);
    v0cut1->SetLooseInvMassCut(LambdaMass-0.035,LambdaMass+0.035);
    v0cut1->SetUseLooseInvMassCut(kTRUE);
    v0cut1->SetPurityHisto("LambdaPurity",100,LambdaMass-0.035,LambdaMass+0.035);
    v0cut1->SetEta(0.8); //|eta|<0.8 for V0s
    v0cut1->SetPt(0.4, 100);
    v0cut1->SetOnFlyStatus(kFALSE);
    v0cut1->SetMaxDcaV0(0.5); //  DCA of V0 to primary vertex must be less than 0.5 cm
    v0cut1->SetMaxCosPointingAngle(0.9993); //0.99 - Jai //0.998
    v0cut1->SetMaxV0DecayLength(60.0);
    //-----
    v0cut1->SetEtaDaughters(0.8); //|eta|<0.8 for daughters
    v0cut1->SetPtPosDaughter(0.5,99); //0.5 for protons
    v0cut1->SetPtNegDaughter(0.16,99); //0.16 for pions
    v0cut1->SetTPCnclsDaughters(80); //daughters required to have hits on at least 80 pad rows of TPC
    //v0cut1->SetNdofDaughters(4.0); //4.0
    v0cut1->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    v0cut1->SetMaxDcaV0Daughters(0.4); //DCA of v0 daughters at decay vertex
    v0cut1->SetMinDaughtersToPrimVertex(0.1,0.3);  //Note:  This (two arguments) only works with myAliFemtoV0TrackCut

    v0cut1->AddCutMonitor(new AliFemtoCutMonitorV0("_Lam_Pass"),new AliFemtoCutMonitorV0("_Lam_Fail"));

  return v0cut1;
}

//____________________________
AliFemtoV0TrackCutNSigmaFilter* myAnalysisConstructor::CreateLambdaCutNSigmaFilter(bool aUseCustom)
{
  // V0 Track Cut (1 = lambda)
  AliFemtoV0TrackCutNSigmaFilter* v0cut1 = new AliFemtoV0TrackCutNSigmaFilter();
    v0cut1->SetParticleType(0);  //  0=lambda -> daughters = proton(+) and pi-
    v0cut1->SetMass(LambdaMass);
    v0cut1->SetInvariantMassLambda(LambdaMass-0.0038,LambdaMass+0.0038);   //m_inv criteria for lambda's

    v0cut1->SetMinvPurityAidHistoV0("LambdaPurityAid","LambdaMinvBeforeFinalCut",100,LambdaMass-0.035,LambdaMass+0.035);

    v0cut1->SetEta(0.8); //|eta|<0.8 for V0s
    v0cut1->SetPt(0.4, 100);
    v0cut1->SetOnFlyStatus(kFALSE);
    v0cut1->SetMaxDcaV0(0.5); //  DCA of V0 to primary vertex must be less than 0.5 cm
    v0cut1->SetMaxCosPointingAngle(0.9993); //0.99 - Jai //0.998
    v0cut1->SetMaxV0DecayLength(60.0);
    //-----
    v0cut1->SetEtaDaughters(0.8); //|eta|<0.8 for daughters
    v0cut1->SetPtPosDaughter(0.5,99); //0.5 for protons
    v0cut1->SetPtNegDaughter(0.16,99); //0.16 for pions
    v0cut1->SetTPCnclsDaughters(80); //daughters required to have hits on at least 80 pad rows of TPC
    //v0cut1->SetNdofDaughters(4.0); //4.0
    v0cut1->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    v0cut1->SetMaxDcaV0Daughters(0.4); //DCA of v0 daughters at decay vertex
    v0cut1->SetMinDaughtersToPrimVertex(0.1,0.3);

    if(aUseCustom)
    {
      //for now, the custom filters will match the standard cuts in AliFemtoV0TrackCut
      //these also match my personal (proton) cuts in myAliFemtoV0TrackCut
      v0cut1->CreateCustomProtonNSigmaFilter();
      v0cut1->AddProtonTPCNSigmaCut(0.,0.8,3.);
      v0cut1->AddProtonTPCAndTOFNSigmaCut(0.8,1000.,3.,3.);
      v0cut1->AddProtonTPCNSigmaCut(0.8,1000.,3.);


      //the standard cuts in AliFemtoV0TrackCut
/*
      v0cut1->CreateCustomPionNSigmaFilter();
      v0cut1->AddPionTPCNSigmaCut(0.,1000.,3.);
*/

      //personal cuts in myAliFemtoV0TrackCut
      v0cut1->CreateCustomPionNSigmaFilter();
      v0cut1->AddPionTPCNSigmaCut(0.,0.8,3.);
      v0cut1->AddPionTPCAndTOFNSigmaCut(0.8,1000.,3.,3.);
      v0cut1->AddPionTPCNSigmaCut(0.8,1000.,3.);
    }

    v0cut1->AddCutMonitor(new AliFemtoCutMonitorV0("_Lam_Pass"),new AliFemtoCutMonitorV0("_Lam_Fail"));

  return v0cut1;
}

//____________________________
myAliFemtoV0TrackCut* myAnalysisConstructor::CreateAntiLambdaCut()
{
  // V0 Track Cut (2 = anti-lambda)
  myAliFemtoV0TrackCut* v0cut2 = new myAliFemtoV0TrackCut();
    v0cut2->SetParticleType(1);  //1=anti-lambda -> daughters = anti-proton(-) and pi+
    v0cut2->SetMass(LambdaMass);
    v0cut2->SetInvariantMassLambda(LambdaMass-0.0038,LambdaMass+0.0038);   //m_inv criteria for anti-lambda's
    v0cut2->SetRemoveMisidentified(kTRUE);
    v0cut2->SetInvMassMisidentified(K0ShortMass-0.003677,K0ShortMass+0.003677);  //m_inv criteria to remove all anti-lambda candidates fulfilling K0short hypothesis
    //v0cut2->SetInvMassMisidentified(K0ShortMass-0.013677,K0ShortMass+0.013677);  //m_inv criteria to remove all anti-lambda candidates fulfilling K0short hypothesis
    v0cut2->SetMisIDHisto("MisIDAntiLambdas",100,LambdaMass-0.035,LambdaMass+0.035);
    v0cut2->SetCalculatePurity(kTRUE);
    v0cut2->SetLooseInvMassCut(LambdaMass-0.035,LambdaMass+0.035);
    v0cut2->SetUseLooseInvMassCut(kTRUE);
    v0cut2->SetPurityHisto("AntiLambdaPurity",100,LambdaMass-0.035,LambdaMass+0.035);
    v0cut2->SetEta(0.8);
    v0cut2->SetPt(0.4,100);
    v0cut2->SetOnFlyStatus(kFALSE); //kTRUE
    v0cut2->SetMaxDcaV0(0.5);
    v0cut2->SetMaxCosPointingAngle(0.9993); //0.99 - Jai
    v0cut2->SetMaxV0DecayLength(60.0);
    //-----
    v0cut2->SetEtaDaughters(0.8);
    v0cut2->SetPtPosDaughter(0.16,99); //0.16 for pions
    v0cut2->SetPtNegDaughter(0.3,99);  //0.3 for anti-protons
    v0cut2->SetTPCnclsDaughters(80);
    //v0cut2->SetNdofDaughters(4.0); //4.0
    v0cut2->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    v0cut2->SetMaxDcaV0Daughters(0.4); //1.5 Jai, 0.6
    v0cut2->SetMinDaughtersToPrimVertex(0.3,0.1);  //Note:  This (two arguments) only works with myAliFemtoV0TrackCut

    v0cut2->AddCutMonitor(new AliFemtoCutMonitorV0("_ALam_Pass"),new AliFemtoCutMonitorV0("_ALam_Fail"));

  return v0cut2;
}


//____________________________
AliFemtoV0TrackCutNSigmaFilter* myAnalysisConstructor::CreateAntiLambdaCutNSigmaFilter(bool aUseCustom)
{
  // V0 Track Cut (2 = anti-lambda)
  AliFemtoV0TrackCutNSigmaFilter* v0cut2 = new AliFemtoV0TrackCutNSigmaFilter();
    v0cut2->SetParticleType(1);  //1=anti-lambda -> daughters = anti-proton(-) and pi+
    v0cut2->SetMass(LambdaMass);
    v0cut2->SetInvariantMassLambda(LambdaMass-0.0038,LambdaMass+0.0038);   //m_inv criteria for anti-lambda's

    v0cut2->SetMinvPurityAidHistoV0("AntiLambdaPurityAid","AntiLambdaMinvBeforeFinalCut",100,LambdaMass-0.035,LambdaMass+0.035);

    v0cut2->SetEta(0.8);
    v0cut2->SetPt(0.4,100);
    v0cut2->SetOnFlyStatus(kFALSE); //kTRUE
    v0cut2->SetMaxDcaV0(0.5);
    v0cut2->SetMaxCosPointingAngle(0.9993); //0.99 - Jai
    v0cut2->SetMaxV0DecayLength(60.0);
    //-----
    v0cut2->SetEtaDaughters(0.8);
    v0cut2->SetPtPosDaughter(0.16,99); //0.16 for pions
    v0cut2->SetPtNegDaughter(0.3,99);  //0.3 for anti-protons
    v0cut2->SetTPCnclsDaughters(80);
    //v0cut2->SetNdofDaughters(4.0); //4.0
    v0cut2->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    v0cut2->SetMaxDcaV0Daughters(0.4); //1.5 Jai, 0.6
    v0cut2->SetMinDaughtersToPrimVertex(0.3,0.1); 

    if(aUseCustom)
    {
      //for now, the custom filters will match the standard cuts in AliFemtoV0TrackCut
      //these also match my personal (proton) cuts in myAliFemtoV0TrackCut
      v0cut2->CreateCustomProtonNSigmaFilter();
      v0cut2->AddProtonTPCNSigmaCut(0.,0.8,3.);
      v0cut2->AddProtonTPCAndTOFNSigmaCut(0.8,1000.,3.,3.);
      v0cut2->AddProtonTPCNSigmaCut(0.8,1000.,3.);


      //the standard cuts in AliFemtoV0TrackCut
/*
      v0cut2->CreateCustomPionNSigmaFilter();
      v0cut2->AddPionTPCNSigmaCut(0.,1000.,3.);
*/

      //personal cuts in myAliFemtoV0TrackCut
      v0cut2->CreateCustomPionNSigmaFilter();
      v0cut2->AddPionTPCNSigmaCut(0.,0.8,3.);
      v0cut2->AddPionTPCAndTOFNSigmaCut(0.8,1000.,3.,3.);
      v0cut2->AddPionTPCNSigmaCut(0.8,1000.,3.);
    }

    v0cut2->AddCutMonitor(new AliFemtoCutMonitorV0("_ALam_Pass"),new AliFemtoCutMonitorV0("_ALam_Fail"));

  return v0cut2;
}


//____________________________
myAliFemtoV0TrackCut* myAnalysisConstructor::CreateK0ShortCut()
{
  myAliFemtoV0TrackCut* k0cut1 = new myAliFemtoV0TrackCut();
    k0cut1->SetParticleType(2);  //  2=K0Short -> daughters = pi+ and pi-
    k0cut1->SetMass(K0ShortMass);
    k0cut1->SetInvariantMassK0Short(K0ShortMass-0.013677,K0ShortMass+0.020323);  //m_inv criteria for K0shorts
    k0cut1->SetRemoveMisidentified(kTRUE);
    k0cut1->SetInvMassMisidentified(LambdaMass-0.005683,LambdaMass+0.005683);  //m_inv criteria to remove all K0short candidates fulfilling (anti-)lambda hypothesis
    k0cut1->SetMisIDHisto("MisIDK0Short1",100,K0ShortMass-0.070,K0ShortMass+0.070);
    k0cut1->SetCalculatePurity(kTRUE);
    k0cut1->SetLooseInvMassCut(K0ShortMass-0.070,K0ShortMass+0.070);
    k0cut1->SetUseLooseInvMassCut(kTRUE);
    k0cut1->SetPurityHisto("K0ShortPurity1",100,K0ShortMass-0.070,K0ShortMass+0.070);
    k0cut1->SetEta(0.8); //|eta|<0.8 for V0s
    k0cut1->SetPt(0.2, 100);
    k0cut1->SetOnFlyStatus(kFALSE);
    k0cut1->SetMaxDcaV0(0.3); //  DCA of V0 to primary vertex
    k0cut1->SetMaxCosPointingAngle(0.9993); //0.99 - Jai //0.998
    k0cut1->SetMaxV0DecayLength(30.0);
    //-----
    k0cut1->SetEtaDaughters(0.8); //|eta|<0.8 for daughters
    k0cut1->SetPtPosDaughter(0.15,99); //
    k0cut1->SetPtNegDaughter(0.15,99); //
    k0cut1->SetTPCnclsDaughters(80); //daughters required to have hits on at least 80 pad rows of TPC
    //k0cut1->SetNdofDaughters(4.0); //4.0
    k0cut1->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    k0cut1->SetMaxDcaV0Daughters(0.3); //DCA of v0 daughters at decay vertex
    k0cut1->SetMinDaughtersToPrimVertex(0.3,0.3);  //Note:  This (two arguments) only works with myAliFemtoV0TrackCut

    k0cut1->AddCutMonitor(new AliFemtoCutMonitorV0("_K0_Pass"),new AliFemtoCutMonitorV0("_K0_Fail"));

  return k0cut1;
}

//____________________________
AliFemtoV0TrackCutNSigmaFilter* myAnalysisConstructor::CreateK0ShortCutNSigmaFilter(bool aUseCustom)
{
  AliFemtoV0TrackCutNSigmaFilter* k0cut1 = new AliFemtoV0TrackCutNSigmaFilter();
    k0cut1->SetParticleType(2);  //  2=K0Short -> daughters = pi+ and pi-
    k0cut1->SetMass(K0ShortMass);
    k0cut1->SetInvariantMassK0s(K0ShortMass-0.013677,K0ShortMass+0.020323);  //m_inv criteria for K0shorts

    k0cut1->SetMinvPurityAidHistoV0("K0ShortPurityAid","K0ShortMinvBeforeFinalCut",100,K0ShortMass-0.070,K0ShortMass+0.070);

    k0cut1->SetEta(0.8); //|eta|<0.8 for V0s
    k0cut1->SetPt(0.2, 100);
    k0cut1->SetOnFlyStatus(kFALSE);
    k0cut1->SetMaxDcaV0(0.3); //  DCA of V0 to primary vertex
    k0cut1->SetMaxCosPointingAngle(0.9993); //0.99 - Jai //0.998
    k0cut1->SetMaxV0DecayLength(30.0);
    //-----
    k0cut1->SetEtaDaughters(0.8); //|eta|<0.8 for daughters
    k0cut1->SetPtPosDaughter(0.15,99); //
    k0cut1->SetPtNegDaughter(0.15,99); //
    k0cut1->SetTPCnclsDaughters(80); //daughters required to have hits on at least 80 pad rows of TPC
    //k0cut1->SetNdofDaughters(4.0); //4.0
    k0cut1->SetStatusDaughters(AliESDtrack::kTPCrefit/* | AliESDtrack::kITSrefit*/);
    k0cut1->SetMaxDcaV0Daughters(0.3); //DCA of v0 daughters at decay vertex
    k0cut1->SetMinDaughtersToPrimVertex(0.3,0.3); 

    if(aUseCustom)
    {
      //the standard cuts in AliFemtoV0TrackCut
/*
      k0cut1->CreateCustomPionNSigmaFilter();
      k0cut1->AddPionTPCNSigmaCut(0.,1000.,3.);
*/

      //personal cuts used in myAliFemtoV0TrackCut
      k0cut1->CreateCustomPionNSigmaFilter();
      k0cut1->AddPionTPCNSigmaCut(0.,0.8,3.);
      k0cut1->AddPionTPCAndTOFNSigmaCut(0.8,1000.,3.,3.);
      k0cut1->AddPionTPCNSigmaCut(0.8,1000.,3.);
    }

    k0cut1->AddCutMonitor(new AliFemtoCutMonitorV0("_K0_Pass"),new AliFemtoCutMonitorV0("_K0_Fail"));

  return k0cut1;
}



//____________________________
myAliFemtoESDTrackCut* myAnalysisConstructor::CreateKchCut(const int aCharge)
{
  myAliFemtoESDTrackCut* kaontc1 = new myAliFemtoESDTrackCut();
    kaontc1->SetPidProbPion(0.0,0.1);
    kaontc1->SetPidProbMuon(0.0,0.8);
    kaontc1->SetPidProbKaon(0.2,1.001);
    kaontc1->SetPidProbProton(0.0,0.1);
    kaontc1->SetMostProbableKaon();  //this uses P().Mag() as first argument to IsKaonNSigma()
    //kaontc1->SetMostProbable(11);  //this looks for Kaons, and uses Pt() as first argument to IsKaonNSigma
    kaontc1->SetCharge(aCharge);
  // so we set the correct mass
    kaontc1->SetMass(KchMass);
  // we select low pt
    kaontc1->SetPt(0.14,1.5);
    kaontc1->SetEta(-0.8,0.8);
//    kaontc1->SetStatus(AliESDtrack::kTPCrefit|AliESDtrack::kITSrefit);  //This cuts out all particles when used in conjunction with SetFilterBit(7)
    kaontc1->SetminTPCncls(80);
    kaontc1->SetRemoveKinks(kTRUE);
    kaontc1->SetLabel(kFALSE);
    kaontc1->SetMaxITSChiNdof(3.0);
    kaontc1->SetMaxTPCChiNdof(4.0);
    kaontc1->SetMaxSigmaToVertex(3.0);
    kaontc1->SetMaxImpactXY(2.4);
    kaontc1->SetMaxImpactZ(3.0);

    kaontc1->SetElectronRejection(true);  // 25/02/2016
    kaontc1->SetPionRejection(true);  // 25/02/2016
  //Cut monitor
  char pass[20];
  char fail[20];
  if(aCharge == 1)
  {
    sprintf(pass, "_KchP_Pass");
    sprintf(fail, "_KchP_Fail");
  }
  else
  {
    sprintf(pass, "_KchM_Pass");
    sprintf(fail, "_KchM_Fail");
  }
  AliFemtoCutMonitorParticleYPt *cutPass = new AliFemtoCutMonitorParticleYPt(pass, 0.13957);
  AliFemtoCutMonitorParticleYPt *cutFail = new AliFemtoCutMonitorParticleYPt(fail, 0.13957);
  kaontc1->AddCutMonitor(cutPass, cutFail);

  AliFemtoCutMonitorParticlePID *cutPIDPass = new AliFemtoCutMonitorParticlePID(pass, 1);
  AliFemtoCutMonitorParticlePID *cutPIDFail = new AliFemtoCutMonitorParticlePID(fail, 1);
  kaontc1->AddCutMonitor(cutPIDPass, cutPIDFail);

  return kaontc1;
}

//____________________________
myAliFemtoESDTrackCut* myAnalysisConstructor::CreatePiCut(const int aCharge)
{
  myAliFemtoESDTrackCut* piontc1 = new myAliFemtoESDTrackCut();
    piontc1->SetPidProbPion(0.2,1.001);
    piontc1->SetPidProbMuon(0.0,0.8);
    piontc1->SetPidProbKaon(0.0,0.1);
    piontc1->SetPidProbProton(0.0,0.1);
    piontc1->SetMostProbablePion();
    piontc1->SetCharge(aCharge);
  // so we set the correct mass
    piontc1->SetMass(PionMass);
  // we select low pt
    piontc1->SetPt(0.1,2.0);
    piontc1->SetEta(-0.8,0.8);
//    piontc1->SetStatus(AliESDtrack::kTPCrefit|AliESDtrack::kITSrefit);    //This cuts out all pions when used in conjunction with SetFilterBit(7)
    piontc1->SetminTPCncls(80);
    piontc1->SetRemoveKinks(kTRUE);
    piontc1->SetLabel(kFALSE);
    piontc1->SetMaxITSChiNdof(3.0);
    piontc1->SetMaxTPCChiNdof(2.0);
    piontc1->SetMaxSigmaToVertex(3.0);
    piontc1->SetMaxImpactXY(2.4);
    piontc1->SetMaxImpactZ(3.2);
  //Cut monitor
  char pass[20];
  char fail[20];
  if(aCharge == 1)
  {
    sprintf(pass, "_PiP_Pass");
    sprintf(fail, "_PiP_Fail");
  }
  else
  {
    sprintf(pass, "_PiM_Pass");
    sprintf(fail, "_PiM_Fail");
  }
  AliFemtoCutMonitorParticleYPt *cutPass = new AliFemtoCutMonitorParticleYPt(pass, 0.13957);
  AliFemtoCutMonitorParticleYPt *cutFail = new AliFemtoCutMonitorParticleYPt(fail, 0.13957);
  piontc1->AddCutMonitor(cutPass, cutFail);

  AliFemtoCutMonitorParticlePID *cutPIDPass = new AliFemtoCutMonitorParticlePID(pass, 0);
  AliFemtoCutMonitorParticlePID *cutPIDFail = new AliFemtoCutMonitorParticlePID(fail, 0);
  piontc1->AddCutMonitor(cutPIDPass, cutPIDFail);

  return piontc1;
}


//____________________________
AliFemtoXiTrackCut* myAnalysisConstructor::CreateXiCut()
{
  //NOTE: the SetMass call actually is important
  //      This should be set to the mass of the particle of interest, here the Xi
  //      Be sure to not accidentally set it again in the Lambda cuts (for instance, when copy/pasting the lambda cuts from above!)

  //Xi -> Lam Pi-

  AliFemtoXiTrackCut* tXiCut = new AliFemtoXiTrackCut();

  //Xi Cuts
/*
  tXiCut->SetChargeXi(-1);
  tXiCut->SetParticleTypeXi(0);  //kXiMinus = 0
  tXiCut->SetPtXi(0.8,100);
//  tXiCut->SetEtaXi(0.8);
  tXiCut->SetMass(XiMass);
  tXiCut->SetInvariantMassXi(XiMass-0.003,XiMass+0.003);
//  tXiCut->SetMaxDecayLengthXi(15.0);
  tXiCut->SetMinCosPointingAngleXi(0.9992);
//  tXiCut->SetMaxDcaXi(0.5);  //not sure what this should be.  Isn't anywhere in analysis notes?
  tXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tXiCut->SetMaxDcaXiDaughters(0.3);


  //Bachelor cuts (here = PiM)
  tXiCut->SetMinDcaXiBac(0.03);
//  tXiCut->SetEtaBac(0.8);
  tXiCut->SetTPCnclsBac(70);
//  tXiCut->SetPtBac(0.16,99);
  tXiCut->SetPtBac(0.,99);
  tXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?



  //Lambda cuts (regular V0)
  tXiCut->SetParticleType(0); //0=lambda
  tXiCut->SetMinDcaV0(0.1);
  tXiCut->SetInvariantMassLambda(LambdaMass-0.005,LambdaMass+0.005);
  tXiCut->SetMinCosPointingAngle(0.998);
//  tXiCut->SetEta(0.8);
  tXiCut->SetEta(10.);
//  tXiCut->SetPt(0.4,100);
  tXiCut->SetPt(0.0,100);
  tXiCut->SetOnFlyStatus(kFALSE);
  tXiCut->SetMaxV0DecayLength(60.0);
    //Lambda daughter cuts
    tXiCut->SetMinDaughtersToPrimVertex(0.1,0.1);
    tXiCut->SetMaxDcaV0Daughters(0.8);
//    tXiCut->SetEtaDaughters(0.8);
//    tXiCut->SetPtPosDaughter(0.5,99); //0.5 for protons
//    tXiCut->SetPtNegDaughter(0.16,99); //0.16 for pions
    tXiCut->SetPtPosDaughter(0.,99); //0.5 for protons
    tXiCut->SetPtNegDaughter(0.,99); //0.16 for pions
    tXiCut->SetTPCnclsDaughters(70);
    tXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
*/


//------------------------ Bare bones Xi finder ----------------------------------------
/*
  tXiCut->SetChargeXi(-1);
  tXiCut->SetParticleTypeXi(0);  //kXiMinus = 0
  tXiCut->SetPtXi(0.0,100);
  tXiCut->SetEtaXi(100.);
  tXiCut->SetMass(XiMass);
  tXiCut->SetInvariantMassXi(XiMass-0.01,XiMass+0.01);
  tXiCut->SetMaxDecayLengthXi(100.);
  tXiCut->SetMinCosPointingAngleXi(0.);
  tXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tXiCut->SetMaxDcaXiDaughters(100.);


  //Bachelor cuts (here = PiM)
  tXiCut->SetMinDcaXiBac(0.);
  tXiCut->SetEtaBac(100.);
  tXiCut->SetTPCnclsBac(0);
  tXiCut->SetPtBac(0.,100.);
  tXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //Lambda cuts (regular V0)
  tXiCut->SetParticleType(0); //0=lambda
  tXiCut->SetMinDcaV0(0.);
  tXiCut->SetInvariantMassLambda(LambdaMass-0.01,LambdaMass+0.01);
  tXiCut->SetMinCosPointingAngle(0.);
  tXiCut->SetEta(100.);
  tXiCut->SetPt(0.0,100);
  tXiCut->SetOnFlyStatus(kFALSE);
  tXiCut->SetMaxV0DecayLength(100.);
    //Lambda daughter cuts
    tXiCut->SetMinDaughtersToPrimVertex(0.,0.);
    tXiCut->SetMaxDcaV0Daughters(100.);
    tXiCut->SetEtaDaughters(100.);
    tXiCut->SetPtPosDaughter(0.,99); //0.5 for protons
    tXiCut->SetPtNegDaughter(0.,99); //0.16 for pions
    tXiCut->SetTPCnclsDaughters(0);
    tXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
*/
//--------------------------------------------------------------------------------------


// ######################### Tightening cuts #############################################

/*
  tXiCut->SetChargeXi(-1);
  tXiCut->SetParticleTypeXi(0);  //kXiMinus = 0
  tXiCut->SetPtXi(0.8,100);
  tXiCut->SetEtaXi(0.8);
  tXiCut->SetMass(XiMass);
  tXiCut->SetInvariantMassXi(XiMass-0.008,XiMass+0.008);
  tXiCut->SetMaxDecayLengthXi(100.);
  tXiCut->SetMinCosPointingAngleXi(0.9992);
  tXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tXiCut->SetMaxDcaXiDaughters(0.3);


  //Bachelor cuts (here = PiM)
  tXiCut->SetMinDcaXiBac(0.03);
  tXiCut->SetEtaBac(0.8);
  tXiCut->SetTPCnclsBac(70);
  tXiCut->SetPtBac(0.,100.);
  tXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //Lambda cuts (regular V0)
  tXiCut->SetParticleType(0); //0=lambda
  tXiCut->SetMinDcaV0(0.1);
  tXiCut->SetInvariantMassLambda(LambdaMass-0.005,LambdaMass+0.005);
  tXiCut->SetMinCosPointingAngle(0.998);
  tXiCut->SetEta(0.8);
  tXiCut->SetPt(0.0,100);
  tXiCut->SetOnFlyStatus(kFALSE);
  tXiCut->SetMaxV0DecayLength(100.);
    //Lambda daughter cuts
    tXiCut->SetMinDaughtersToPrimVertex(0.1,0.1);
    tXiCut->SetMaxDcaV0Daughters(0.8);
    tXiCut->SetEtaDaughters(0.8);
    tXiCut->SetPtPosDaughter(0.,99); //0.5 for protons
    tXiCut->SetPtNegDaughter(0.,99); //0.16 for pions
    tXiCut->SetTPCnclsDaughters(70);
    tXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
*/
// #######################################################################################


// %%%%%%%%%%%%%%%%%%%%%%%% Version 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*

  tXiCut->SetChargeXi(-1);
  tXiCut->SetParticleTypeXi(0);  //kXiMinus = 0
  tXiCut->SetPtXi(0.8,100);
  tXiCut->SetEtaXi(0.8);
  tXiCut->SetMass(XiMass);
  tXiCut->SetInvariantMassXi(XiMass-0.008,XiMass+0.008);
  tXiCut->SetMaxDecayLengthXi(100.);
  tXiCut->SetMinCosPointingAngleXi(0.9992);
  tXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tXiCut->SetMaxDcaXiDaughters(0.3);


  //Bachelor cuts (here = PiM)
  tXiCut->SetMinDcaXiBac(0.03);
  tXiCut->SetEtaBac(0.8);
  tXiCut->SetTPCnclsBac(70);
  tXiCut->SetPtBac(0.,100.);
  tXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //Lambda cuts (regular V0)
  tXiCut->SetParticleType(0); //0=lambda
  tXiCut->SetMinDcaV0(0.1);
  tXiCut->SetInvariantMassLambda(LambdaMass-0.005,LambdaMass+0.005);
  tXiCut->SetMinCosPointingAngle(0.998);
  tXiCut->SetEta(0.8);
  tXiCut->SetPt(0.0,100);
  tXiCut->SetOnFlyStatus(kFALSE);
  tXiCut->SetMaxV0DecayLength(100.);
    //Lambda daughter cuts
    tXiCut->SetMinDaughtersToPrimVertex(0.1,0.1);
    tXiCut->SetMaxDcaV0Daughters(0.8);
    tXiCut->SetEtaDaughters(0.8);
    tXiCut->SetPtPosDaughter(0.,99); //0.5 for protons
    tXiCut->SetPtNegDaughter(0.,99); //0.16 for pions
    tXiCut->SetTPCnclsDaughters(70);
    tXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
*/
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%%%%% Version 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  tXiCut->SetChargeXi(-1);
  tXiCut->SetParticleTypeXi(0);  //kXiMinus = 0
  tXiCut->SetPtXi(0.8,100);
  tXiCut->SetEtaXi(0.8);
  tXiCut->SetMass(XiMass);
  tXiCut->SetInvariantMassXi(XiMass-0.003,XiMass+0.003);
  tXiCut->SetMaxDecayLengthXi(100.);
  tXiCut->SetMinCosPointingAngleXi(0.9992);
  tXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tXiCut->SetMaxDcaXiDaughters(0.3);


  //Bachelor cuts (here = PiM)
  tXiCut->SetMinDcaXiBac(0.03);
  tXiCut->SetEtaBac(0.8);
  tXiCut->SetTPCnclsBac(70);
  tXiCut->SetPtBac(0.,100.);
  tXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //Lambda cuts (regular V0)
  tXiCut->SetParticleType(0); //0=lambda
  tXiCut->SetMinDcaV0(0.1);
  tXiCut->SetInvariantMassLambda(LambdaMass-0.005,LambdaMass+0.005);
  tXiCut->SetMinCosPointingAngle(0.998);
  tXiCut->SetEta(0.8);
  tXiCut->SetPt(0.0,100);
  tXiCut->SetOnFlyStatus(kFALSE);
  tXiCut->SetMaxV0DecayLength(100.);
    //Lambda daughter cuts
    tXiCut->SetMinDaughtersToPrimVertex(0.1,0.1);
    tXiCut->SetMaxDcaV0Daughters(0.8);
    tXiCut->SetEtaDaughters(0.8);
    tXiCut->SetPtPosDaughter(0.,99); //0.5 for protons
    tXiCut->SetPtNegDaughter(0.,99); //0.16 for pions
    tXiCut->SetTPCnclsDaughters(70);
    tXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  tXiCut->SetMinvPurityAidHistoXi("XiPurityAid","XiMinvBeforeFinalCut",100,XiMass-0.035,XiMass+0.035);
  tXiCut->SetMinvPurityAidHistoV0("LambdaPurityAid","LambdaMinvBeforeFinalCut",100,LambdaMass-0.035,LambdaMass+0.035);

  tXiCut->AddCutMonitor(new AliFemtoCutMonitorXi("_Xi_Pass"),new AliFemtoCutMonitorXi("_Xi_Fail"));

  return tXiCut;
}

//____________________________
AliFemtoXiTrackCut* myAnalysisConstructor::CreateAntiXiCut()
{
  //NOTE: the SetMass call actually is important
  //      This should be set to the mass of the particle of interest, here the Xi
  //      Be sure to not accidentally set it again in the Lambda cuts (for instance, when copy/pasting the lambda cuts from above!)

  //AXi -> ALam Pi+

  AliFemtoXiTrackCut* tAXiCut = new AliFemtoXiTrackCut();

  //Xi Cuts
/*
  tAXiCut->SetChargeXi(1);
  tAXiCut->SetParticleTypeXi(1); //kXiPlus = 1
  tAXiCut->SetPtXi(0.8,100);
//  tAXiCut->SetEtaXi(0.8);
  tAXiCut->SetMass(XiMass);
  tAXiCut->SetInvariantMassXi(XiMass-0.003,XiMass+0.003);
//  tAXiCut->SetMaxDecayLengthXi(15.0);
  tAXiCut->SetMinCosPointingAngleXi(0.9992);
//  tAXiCut->SetMaxDcaXi(0.5);  //not sure what this should be.  Isn't anywhere in analysis notes?
  tAXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tAXiCut->SetMaxDcaXiDaughters(0.3);


  //Bachelor cuts (here = PiP)
  tAXiCut->SetMinDcaXiBac(0.03);
//  tAXiCut->SetEtaBac(0.8);
  tAXiCut->SetTPCnclsBac(70);
//  tAXiCut->SetPtBac(0.16,99);
  tAXiCut->SetPtBac(0.,99);
  tAXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //AntiLambda cuts (regular V0)
  tAXiCut->SetParticleType(1); //1=anti-lambda
  tAXiCut->SetMinDcaV0(0.1);
  tAXiCut->SetInvariantMassLambda(LambdaMass-0.005,LambdaMass+0.005);
  tAXiCut->SetMinCosPointingAngle(0.998);
//  tAXiCut->SetEta(0.8);
  tAXiCut->SetEta(10);
//  tAXiCut->SetPt(0.4,100);
  tAXiCut->SetPt(0.,100);
  tAXiCut->SetOnFlyStatus(kFALSE);
  tAXiCut->SetMaxV0DecayLength(60.0);
    //Lambda daughter cuts
    tAXiCut->SetMinDaughtersToPrimVertex(0.1,0.1);
    tAXiCut->SetMaxDcaV0Daughters(0.8);
//    tAXiCut->SetEtaDaughters(0.8);
//    tAXiCut->SetPtPosDaughter(0.16,99); //0.16 for pions
//    tAXiCut->SetPtNegDaughter(0.3,99); //0.5 for anti-protons
    tAXiCut->SetPtPosDaughter(0.,99); //0.16 for pions
    tAXiCut->SetPtNegDaughter(0.,99); //0.5 for anti-protons
    tAXiCut->SetTPCnclsDaughters(70);
    tAXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
*/

// ------------------------------ Bare bones AXi finder ---------------------------------------------
/*
  tAXiCut->SetChargeXi(1);
  tAXiCut->SetParticleTypeXi(1); //kXiPlus = 1
  tAXiCut->SetPtXi(0.0,100);
  tAXiCut->SetEtaXi(100.);
  tAXiCut->SetMass(XiMass);
  tAXiCut->SetInvariantMassXi(XiMass-0.01,XiMass+0.01);
  tAXiCut->SetMaxDecayLengthXi(100.0);
  tAXiCut->SetMinCosPointingAngleXi(0.);
  tAXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tAXiCut->SetMaxDcaXiDaughters(100.);


  //Bachelor cuts (here = PiP)
  tAXiCut->SetMinDcaXiBac(0.);
  tAXiCut->SetEtaBac(100.);
  tAXiCut->SetTPCnclsBac(0);
  tAXiCut->SetPtBac(0.,100);
  tAXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //AntiLambda cuts (regular V0)
  tAXiCut->SetParticleType(1); //1=anti-lambda
  tAXiCut->SetMinDcaV0(0.);
  tAXiCut->SetInvariantMassLambda(LambdaMass-0.01,LambdaMass+0.01);
  tAXiCut->SetMinCosPointingAngle(0.);
  tAXiCut->SetEta(100.);
  tAXiCut->SetPt(0.,100);
  tAXiCut->SetOnFlyStatus(kFALSE);
  tAXiCut->SetMaxV0DecayLength(100.);
    //Lambda daughter cuts
    tAXiCut->SetMinDaughtersToPrimVertex(0.,0.);
    tAXiCut->SetMaxDcaV0Daughters(100.);
    tAXiCut->SetEtaDaughters(100.);
    tAXiCut->SetPtPosDaughter(0.,99); //0.16 for pions
    tAXiCut->SetPtNegDaughter(0.,99); //0.5 for anti-protons
    tAXiCut->SetTPCnclsDaughters(0);
    tAXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
*/
// --------------------------------------------------------------------------------------------------

// ######################### Tightening cuts #############################################
/*
  tAXiCut->SetChargeXi(1);
  tAXiCut->SetParticleTypeXi(1); //kXiPlus = 1
  tAXiCut->SetPtXi(0.0,100);
  tAXiCut->SetEtaXi(0.8);
  tAXiCut->SetMass(XiMass);
  tAXiCut->SetInvariantMassXi(XiMass-0.008,XiMass+0.008);
  tAXiCut->SetMaxDecayLengthXi(100.0);
  tAXiCut->SetMinCosPointingAngleXi(0.9992);
  tAXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tAXiCut->SetMaxDcaXiDaughters(100.);


  //Bachelor cuts (here = PiP)
  tAXiCut->SetMinDcaXiBac(0.);
  tAXiCut->SetEtaBac(0.8);
  tAXiCut->SetTPCnclsBac(70);
  tAXiCut->SetPtBac(0.,100);
  tAXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //AntiLambda cuts (regular V0)
  tAXiCut->SetParticleType(1); //1=anti-lambda
  tAXiCut->SetMinDcaV0(0.);
  tAXiCut->SetInvariantMassLambda(LambdaMass-0.005,LambdaMass+0.005);
  tAXiCut->SetMinCosPointingAngle(0.998);
  tAXiCut->SetEta(0.8);
  tAXiCut->SetPt(0.,100);
  tAXiCut->SetOnFlyStatus(kFALSE);
  tAXiCut->SetMaxV0DecayLength(100.);
    //Lambda daughter cuts
    tAXiCut->SetMinDaughtersToPrimVertex(0.,0.);
    tAXiCut->SetMaxDcaV0Daughters(100.);
    tAXiCut->SetEtaDaughters(0.8);
    tAXiCut->SetPtPosDaughter(0.,99); //0.16 for pions
    tAXiCut->SetPtNegDaughter(0.,99); //0.5 for anti-protons
    tAXiCut->SetTPCnclsDaughters(70);
    tAXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
*/
// #######################################################################################

// %%%%%%%%%%%%%%%%%%%%%%%% Version 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*
  tAXiCut->SetChargeXi(1);
  tAXiCut->SetParticleTypeXi(1); //kXiPlus = 1
  tAXiCut->SetPtXi(0.8,100);
  tAXiCut->SetEtaXi(0.8);
  tAXiCut->SetMass(XiMass);
  tAXiCut->SetInvariantMassXi(XiMass-0.008,XiMass+0.008);
  tAXiCut->SetMaxDecayLengthXi(100.0);
  tAXiCut->SetMinCosPointingAngleXi(0.9992);
  tAXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tAXiCut->SetMaxDcaXiDaughters(0.3);


  //Bachelor cuts (here = PiP)
  tAXiCut->SetMinDcaXiBac(0.03);
  tAXiCut->SetEtaBac(0.8);
  tAXiCut->SetTPCnclsBac(70);
  tAXiCut->SetPtBac(0.,100);
  tAXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //AntiLambda cuts (regular V0)
  tAXiCut->SetParticleType(1); //1=anti-lambda
  tAXiCut->SetMinDcaV0(0.1);
  tAXiCut->SetInvariantMassLambda(LambdaMass-0.005,LambdaMass+0.005);
  tAXiCut->SetMinCosPointingAngle(0.998);
  tAXiCut->SetEta(0.8);
  tAXiCut->SetPt(0.,100);
  tAXiCut->SetOnFlyStatus(kTRUE);  //!!!!!!!!!!!!!!!!!!!!!!! Set to kTRUE!
  tAXiCut->SetMaxV0DecayLength(100.);
    //Lambda daughter cuts
    tAXiCut->SetMinDaughtersToPrimVertex(0.1,0.1);
    tAXiCut->SetMaxDcaV0Daughters(0.8);
    tAXiCut->SetEtaDaughters(0.8);
    tAXiCut->SetPtPosDaughter(0.,99); //0.16 for pions
    tAXiCut->SetPtNegDaughter(0.,99); //0.5 for anti-protons
    tAXiCut->SetTPCnclsDaughters(70);
    tAXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?
*/
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// %%%%%%%%%%%%%%%%%%%%%%%% Version 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  tAXiCut->SetChargeXi(1);
  tAXiCut->SetParticleTypeXi(1); //kXiPlus = 1
  tAXiCut->SetPtXi(0.8,100);
  tAXiCut->SetEtaXi(0.8);
  tAXiCut->SetMass(XiMass);
  tAXiCut->SetInvariantMassXi(XiMass-0.003,XiMass+0.003);
  tAXiCut->SetMaxDecayLengthXi(100.0);
  tAXiCut->SetMinCosPointingAngleXi(0.9992);
  tAXiCut->SetMaxDcaXi(100);
    //XiDaughters
    tAXiCut->SetMaxDcaXiDaughters(0.3);


  //Bachelor cuts (here = PiP)
  tAXiCut->SetMinDcaXiBac(0.03);
  tAXiCut->SetEtaBac(0.8);
  tAXiCut->SetTPCnclsBac(70);
  tAXiCut->SetPtBac(0.,100);
  tAXiCut->SetStatusBac(AliESDtrack::kTPCrefit);  //yes or no?


  //AntiLambda cuts (regular V0)
  tAXiCut->SetParticleType(1); //1=anti-lambda
  tAXiCut->SetMinDcaV0(0.1);
  tAXiCut->SetInvariantMassLambda(LambdaMass-0.005,LambdaMass+0.005);
  tAXiCut->SetMinCosPointingAngle(0.998);
  tAXiCut->SetEta(0.8);
  tAXiCut->SetPt(0.,100);
  tAXiCut->SetOnFlyStatus(kTRUE);  //!!!!!!!!!!!!!!!!!!!!!!! Set to kTRUE!
  tAXiCut->SetMaxV0DecayLength(100.);
    //Lambda daughter cuts
    tAXiCut->SetMinDaughtersToPrimVertex(0.1,0.1);
    tAXiCut->SetMaxDcaV0Daughters(0.8);
    tAXiCut->SetEtaDaughters(0.8);
    tAXiCut->SetPtPosDaughter(0.,99); //0.16 for pions
    tAXiCut->SetPtNegDaughter(0.,99); //0.5 for anti-protons
    tAXiCut->SetTPCnclsDaughters(70);
    tAXiCut->SetStatusDaughters(AliESDtrack::kTPCrefit);  //yes or no?

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  tAXiCut->SetMinvPurityAidHistoXi("AXiPurityAid","AXiMinvBeforeFinalCut",100,XiMass-0.035,XiMass+0.035);
  tAXiCut->SetMinvPurityAidHistoV0("AntiLambdaPurityAid","AntiLambdaMinvBeforeFinalCut",100,LambdaMass-0.035,LambdaMass+0.035);

  tAXiCut->AddCutMonitor(new AliFemtoCutMonitorXi("_AXi_Pass"),new AliFemtoCutMonitorXi("_AXi_Fail"));

  return tAXiCut;
}



//____________________________
AliFemtoV0PairCut* myAnalysisConstructor::CreateV0PairCut(double aMinAvgSepPosPos, double aMinAvgSepPosNeg, double aMinAvgSepNegPos, double aMinAvgSepNegNeg)
{
  AliFemtoV0PairCut *v0pc1 = new AliFemtoV0PairCut();  //K0Short-lambda
//  v0pc1->SetV0Max(0.25);
//  v0pc1->SetShareFractionMax(0.05)  //how do I implement this in AliFemtoV0PairCut?
  v0pc1->SetRemoveSameLabel(kTRUE);
  v0pc1->SetTPCExitSepMinimum(-1.0);  //Default is 0, but for some reason distExitPos(Neg) always end up as 0?

  v0pc1->SetMinAvgSeparation(0,aMinAvgSepPosPos);
  v0pc1->SetMinAvgSeparation(1,aMinAvgSepPosNeg);
  v0pc1->SetMinAvgSeparation(2,aMinAvgSepNegPos);
  v0pc1->SetMinAvgSeparation(3,aMinAvgSepNegNeg);

  if(fIsMCRun)
  {
    AliFemtoPairOriginMonitor *cutPass = new AliFemtoPairOriginMonitor("Pass");
    v0pc1->AddCutMonitorPass(cutPass);
  }

  return v0pc1;
}

//____________________________
AliFemtoV0TrackPairCut* myAnalysisConstructor::CreateV0TrackPairCut(double aMinAvgSepTrackPos, double aMinAvgSepTrackNeg)
{
  AliFemtoV0TrackPairCut *v0TrackPairCut1 = new AliFemtoV0TrackPairCut();
    v0TrackPairCut1->SetShareQualityMax(1.0);
    v0TrackPairCut1->SetShareFractionMax(1.0);
    v0TrackPairCut1->SetTPCOnly(kTRUE);
    v0TrackPairCut1->SetDataType(AliFemtoPairCut::kAOD);
    v0TrackPairCut1->SetTPCEntranceSepMinimum(0.00001);
    v0TrackPairCut1->SetTPCExitSepMinimum(-1.);
//    v0TrackPairCut1->SetKstarCut(0.04,AliFemtoV0TrackPairCut::kAntiLambda,AliFemtoV0TrackPairCut::kAntiProton); //1 - antilambda, 3 - antiproton
//    v0TrackPairCut1->SetMinAvgSeparation(0,0); //0 - track-pos, 1 - track-neg
//    v0TrackPairCut1->SetMinAvgSeparation(1,11);
    v0TrackPairCut1->SetRemoveSameLabel(kTRUE);

  v0TrackPairCut1->SetMinAvgSeparation(0,aMinAvgSepTrackPos);
  v0TrackPairCut1->SetMinAvgSeparation(1,aMinAvgSepTrackNeg);

  if(fIsMCRun)
  {
    AliFemtoPairOriginMonitor *cutPass = new AliFemtoPairOriginMonitor("Pass");
    v0TrackPairCut1->AddCutMonitorPass(cutPass);
  }

  return v0TrackPairCut1;
}


//____________________________
AliFemtoXiTrackPairCut* myAnalysisConstructor::CreateXiTrackPairCut()
{
  AliFemtoXiTrackPairCut *tXiTrackPairCut = new AliFemtoXiTrackPairCut();

  return tXiTrackPairCut;
}




//____________________________
myAliFemtoKStarCorrFctn* myAnalysisConstructor::CreateKStarCorrFctn(const char* name, unsigned int bins, double min, double max)
{
  myAliFemtoKStarCorrFctn *cf = new myAliFemtoKStarCorrFctn(name,bins,min,max);
  cf->CalculatePairKinematics(fWritePairKinematics);
  return cf;
}

//____________________________
myAliFemtoAvgSepCorrFctn* myAnalysisConstructor::CreateAvgSepCorrFctn(const char* name, unsigned int bins, double min, double max)
{
  myAliFemtoAvgSepCorrFctn *cf = new myAliFemtoAvgSepCorrFctn(name,bins,min,max);
  return cf;
}

//____________________________
myAliFemtoSepCorrFctns* myAnalysisConstructor::CreateSepCorrFctns(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY)
{
  myAliFemtoSepCorrFctns *cf = new myAliFemtoSepCorrFctns(name,binsX,minX,maxX,binsY,minY,maxY);
  return cf;
}

//____________________________
myAliFemtoAvgSepCorrFctnCowboysAndSailors* myAnalysisConstructor::CreateAvgSepCorrFctnCowboysAndSailors(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY)
{
  myAliFemtoAvgSepCorrFctnCowboysAndSailors *cf = new myAliFemtoAvgSepCorrFctnCowboysAndSailors(name,binsX,minX,maxX,binsY,minY,maxY);
  return cf;
}

//____________________________
myAliFemtoKStarCorrFctn2D* myAnalysisConstructor::CreateKStarCorrFctn2D(const char* name, unsigned int nbinsKStar, double KStarLo, double KStarHi, unsigned int nbinsY, double YLo, double YHi)
{
  myAliFemtoKStarCorrFctn2D *cf = new myAliFemtoKStarCorrFctn2D(name,nbinsKStar,KStarLo,KStarHi,nbinsY,YLo,YHi);
  return cf;
}

//____________________________
myAliFemtoKStarCorrFctnMC* myAnalysisConstructor::CreateKStarCorrFctnMC(const char* name, unsigned int bins, double min, double max)
{
  myAliFemtoKStarCorrFctnMC *cf = new myAliFemtoKStarCorrFctnMC(name,bins,min,max);
  return cf;
}

//____________________________
myAliFemtoModelCorrFctnKStar* myAnalysisConstructor::CreateModelCorrFctnKStar(const char* name, unsigned int bins, double min, double max)
{
  AliFemtoModelWeightGeneratorBasicLednicky *tGenerator = new AliFemtoModelWeightGeneratorBasicLednicky();
  tGenerator->SetIdenticalParticles(false);
  tGenerator->SetParamAlpha(0.);
  if(fAnalysisType == kLamKchP || fAnalysisType == kALamKchM)
  {
    tGenerator->SetParamLambda(0.1403);
    tGenerator->SetParamRadius(4.241);
    tGenerator->SetParamRef0(-1.981);
    tGenerator->SetParamImf0(0.8138);
    tGenerator->SetParamd0(2.621);
    tGenerator->SetParamNorm(1.);    
  }
  else if(fAnalysisType == kLamKchM || fAnalysisType == kALamKchP)
  {
    tGenerator->SetParamLambda(0.331);
    tGenerator->SetParamRadius(4.107);
    tGenerator->SetParamRef0(0.1362);
    tGenerator->SetParamImf0(0.4482);
    tGenerator->SetParamd0(6.666);
    tGenerator->SetParamNorm(1.);    
  }

  AliFemtoModelManager *tManager = new AliFemtoModelManager();
  tManager->AcceptWeightGenerator((AliFemtoModelWeightGenerator*)tGenerator);

  myAliFemtoModelCorrFctnKStar *cf = new myAliFemtoModelCorrFctnKStar(name,bins,min,max);
    cf->SetAnalysisType(fAnalysisType);
  cf->ConnectToManager(tManager);

  return cf;
}






//____________________________
void myAnalysisConstructor::SetAnalysis(AliFemtoEventCut* aEventCut, AliFemtoParticleCut* aPartCut1, AliFemtoParticleCut* aPartCut2, AliFemtoPairCut* aPairCut, AliFemtoCorrFctnCollection* aCollectionOfCfs)
{
  SetEventCut(aEventCut);
  SetFirstParticleCut(aPartCut1);
  SetSecondParticleCut(aPartCut2);
  SetPairCut(aPairCut);

  AliFemtoCorrFctnIterator iter;
  for(iter=aCollectionOfCfs->begin(); iter!=aCollectionOfCfs->end(); iter++)
  {
    AddCorrFctn(*iter);
  }
}


//____________________________
void myAnalysisConstructor::SetImplementAvgSepCuts(bool aImplementAvgSepCuts)
{
  fImplementAvgSepCuts = aImplementAvgSepCuts;
}

