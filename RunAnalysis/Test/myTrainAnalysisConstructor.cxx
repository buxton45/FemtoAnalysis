
#include "myTrainAnalysisConstructor.h"
#include "TObjArray.h"
#include "AliESDtrack.h"
#ifdef __ROOT__
ClassImp(myTrainAnalysisConstructor)
#endif

static const double PionMass = 0.13956995,
                    KchMass = 0.493677,
                    K0ShortMass = 0.497614,
                    ProtonMass = 0.938272013,
                    LambdaMass = 1.115683,
		    XiMass     = 1.32171;

//____________________________
const char* const myTrainAnalysisConstructor::fAnalysisTags[] = {"LamK0", "ALamK0", "LamKchP", "ALamKchP", "LamKchM", "ALamKchM", "LamLam", "ALamALam", "LamALam", "LamPiP", "ALamPiP", "LamPiM", "ALamPiM", "XiKchP", "AXiKchP", "XiKchM", "AXiKchM"};

//____________________________
myTrainAnalysisConstructor::myTrainAnalysisConstructor() : 
  AliFemtoVertexMultAnalysis(),

  fAnalysisType(kLamK0),
  fGeneralAnalysisType(kV0V0),
  fParticlePDGType1(kPDGLam),
  fParticlePDGType2(kPDGK0),
  fGeneralParticleType1(kV0),
  fGeneralParticleType2(kV0),

  fOutputName("Analysis"),
  fMultHist(NULL),
  fImplementAvgSepCuts(kTRUE),
  fWritePairKinematics(kFALSE),
  fIsMCRun(kFALSE),
  fIsMBAnalysis(kFALSE),
  fBuildMultHist(kFALSE),
  fMinCent(-1000),
  fMaxCent(1000),

  fCollectionOfCfs(NULL),

  BasicEvCut(NULL),
  EvCutEst(NULL),

  XiCut1(NULL),
  XiCut2(NULL),

  V0Cut1(NULL),
  V0Cut2(NULL),
  
  TrackCut1(NULL),
  TrackCut2(NULL),
  
  V0PairCut(NULL),
  V0TrackPairCut(NULL),
  XiTrackPairCut(NULL),

  KStarCf(NULL),
  AvgSepCf(NULL),
/*
  SepCfs(0),
  AvgSepCfCowboysAndSailors(0),
  KStarCf2D(0),
*/
  KStarModelCfs(NULL),

  LamCutNSigmaFilter(NULL),
  ALamCutNSigmaFilter(NULL),
  K0CutNSigmaFilter(NULL)

{
  SetParticleTypes(fAnalysisType);
  SetVerboseMode(kFALSE);
  SetMultHist("");
  fCollectionOfCfs = new AliFemtoCorrFctnCollection;
  SetEnablePairMonitors(fIsMCRun);
}

//____________________________
myTrainAnalysisConstructor::myTrainAnalysisConstructor(AnalysisType aAnalysisType, const char* name, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics) : 
  AliFemtoVertexMultAnalysis(),

  fAnalysisType(aAnalysisType),
  fGeneralAnalysisType(kV0V0),
  fParticlePDGType1(kPDGLam),
  fParticlePDGType2(kPDGK0),
  fGeneralParticleType1(kV0),
  fGeneralParticleType2(kV0),

  fOutputName(name),
  fMultHist(NULL),
  fImplementAvgSepCuts(aImplementAvgSepCuts),
  fWritePairKinematics(aWritePairKinematics),
  fIsMCRun(aIsMCRun),
  fIsMBAnalysis(kFALSE),
  fBuildMultHist(kFALSE),
  fMinCent(-1000),
  fMaxCent(1000),

  fCollectionOfCfs(NULL),

  BasicEvCut(NULL),
  EvCutEst(NULL),

  XiCut1(NULL),
  XiCut2(NULL),

  V0Cut1(NULL),
  V0Cut2(NULL),
  
  TrackCut1(NULL),
  TrackCut2(NULL),
  
  V0PairCut(NULL),
  V0TrackPairCut(NULL),
  XiTrackPairCut(NULL),

  KStarCf(NULL),
  AvgSepCf(NULL),
/*
  SepCfs(0),
  AvgSepCfCowboysAndSailors(0),
  KStarCf2D(0),
*/
  KStarModelCfs(NULL),

  LamCutNSigmaFilter(NULL),
  ALamCutNSigmaFilter(NULL),
  K0CutNSigmaFilter(NULL)

{
  SetParticleTypes(fAnalysisType);
  SetVerboseMode(kFALSE);
  SetNumEventsToMix(5);
  SetMinSizePartCollection(1);
  SetV0SharedDaughterCut(kTRUE);
  SetEnablePairMonitors(fIsMCRun);

  SetMultHist(fAnalysisTags[aAnalysisType]);

  fCollectionOfCfs = new AliFemtoCorrFctnCollection;
}

//____________________________
myTrainAnalysisConstructor::myTrainAnalysisConstructor(AnalysisType aAnalysisType, const char* name, unsigned int binsVertex, double minVertex, double maxVertex, unsigned int binsMult, double minMult, double maxMult, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics) : 
  AliFemtoVertexMultAnalysis(binsVertex,minVertex,maxVertex,binsMult,minMult,maxMult),

  fAnalysisType(aAnalysisType),
  fGeneralAnalysisType(kV0V0),
  fParticlePDGType1(kPDGLam),
  fParticlePDGType2(kPDGK0),
  fGeneralParticleType1(kV0),
  fGeneralParticleType2(kV0),

  fOutputName(name),
  fMultHist(NULL),
  fImplementAvgSepCuts(aImplementAvgSepCuts),
  fWritePairKinematics(aWritePairKinematics),
  fIsMCRun(aIsMCRun),
  fIsMBAnalysis(kFALSE),
  fBuildMultHist(kFALSE),
  fMinCent(-1000),
  fMaxCent(1000),

  fCollectionOfCfs(NULL),

  BasicEvCut(NULL),
  EvCutEst(NULL),

  XiCut1(NULL),
  XiCut2(NULL),

  V0Cut1(NULL),
  V0Cut2(NULL),
  
  TrackCut1(NULL),
  TrackCut2(NULL),
  
  V0PairCut(NULL),
  V0TrackPairCut(NULL),
  XiTrackPairCut(NULL),

  KStarCf(NULL),
  AvgSepCf(NULL),
/*
  SepCfs(0),
  AvgSepCfCowboysAndSailors(0),
  KStarCf2D(0),
*/
  KStarModelCfs(NULL),

  LamCutNSigmaFilter(NULL),
  ALamCutNSigmaFilter(NULL),
  K0CutNSigmaFilter(NULL)

{
  SetParticleTypes(fAnalysisType);
  SetVerboseMode(kFALSE);
  SetNumEventsToMix(5);
  SetMinSizePartCollection(1);
  SetV0SharedDaughterCut(kTRUE);
  SetEnablePairMonitors(fIsMCRun);
  SetMultHist(fAnalysisTags[aAnalysisType]);

  fMinCent = minMult/10.;
  fMaxCent = maxMult/10.;

  fCollectionOfCfs = new AliFemtoCorrFctnCollection;

  if(fWritePairKinematics) KStarCf = CreateCorrFctnKStar(fAnalysisTags[aAnalysisType],62,0.,0.31); //TNtuple is huge, and I don't need data out to 1 GeV
  else KStarCf = CreateCorrFctnKStar(fAnalysisTags[aAnalysisType],200,0.,1.0);

  AvgSepCf = CreateAvgSepCorrFctn(fAnalysisTags[aAnalysisType],200,0.,20.);

/*
  char tTitSepCfs[101] = "SepCfs_";
  strncat(tTitSepCfs,fAnalysisTags[aAnalysisType],100);
  SepCfs = CreateSepCorrFctns(tTitSepCfs,10,0.,10.,200,0.,20.);

  char tTitAvgSepCfCowboysAndSailors[101] = "AvgSepCfCowboysAndSailors_";
  strncat(tTitAvgSepCfCowboysAndSailors,fAnalysisTags[aAnalysisType],100);
  AvgSepCfCowboysAndSailors = CreateAvgSepCorrFctnCowboysAndSailors(tTitAvgSepCfCowboysAndSailors,40,-5.,5.,200,0.,20.);

  char tTitKStarCf2D[101] = "KStarCf2D_";
  strncat(tTitKStarCf2D,fAnalysisTags[aAnalysisType],100);
  KStarCf2D = CreateCorrFctnKStar2D(tTitKStarCf2D,200,0.,1.0,2,-2.,2.);
*/

  KStarModelCfs = CreateModelCorrFctnKStarFull(fAnalysisTags[aAnalysisType],200,0.,1.0);

  if(fWritePairKinematics)
  {
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCf);
  }
  else
  {
    //cannot push_back fMultHist into fCollectionOfCfs because it is not of type AliFemtoCorrFctn
    //It cannot even be added via AliFemtoSimpleAnalysis::AddCorrFctn because it is not a cf
    //fMultHist is added in myTrainAnalysisConstructor::ProcessEvent
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCf);
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)AvgSepCf);
/*
    //fCollectionOfCfs->push_back((AliFemtoCorrFctn*)SepCfs);
    //fCollectionOfCfs->push_back((AliFemtoCorrFctn*)AvgSepCfCowboysAndSailors);
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarCf2D);
*/
  }

  if(fIsMCRun) 
  {
    fCollectionOfCfs->push_back((AliFemtoCorrFctn*)KStarModelCfs);
  }

}

/*
//____________________________
//copy constructor - 30 June 2015
myTrainAnalysisConstructor::myTrainAnalysisConstructor(const myTrainAnalysisConstructor& a) :
  AliFemtoVertexMultAnalysis(a),  //call the copy-constructor of the base
  fAnalysisType(kLamK0),
  fCollectionOfCfs(0),
  fOutputName(0),
  fMultHist(0),
  fIsMCRun(kFALSE),
  fIsMBAnalysis(kFALSE),
  fBuildMultHist(kFALSE),
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
  K0CutNSigmaFilter(0)

{
  fCollectionOfCfs = new AliFemtoCorrFctnCollection;
  AliFemtoCorrFctnIterator iter;
  for(iter=a.fCollectionOfCfs->begin(); iter!=a.fCollectionOfCfs->end(); iter++)
  {
    AliFemtoCorrFctn* fctn = (*iter)->Clone();
    if(fctn) {AddCorrFctn(fctn);}
    else {cout << " myTrainAnalysisConstructor::myTrainAnalysisConstructor(const myTrainAnalysisConstructor& a) - correlation function not found " << endl;}
  }
}
*/

/*
//____________________________
//assignment operator - 30 June 2015
myTrainAnalysisConstructor& myTrainAnalysisConstructor::operator=(const myTrainAnalysisConstructor& TheOriginalAnalysis)
{
  if(this == &TheOriginalAnalysis) {return *this;}

  AliFemtoVertexMultAnalysis::operator=(TheOriginalAnalysis);  //call the assignment operator of the base

  fAnalysisType = kLamK0;
  fCollectionOfCfs = 0;
  fOutputName = "Analysis";
  fMultHist = 0;
  fIsMCRun = kFALSE;
  fIsMBAnalysis = kFALSE;
  fBuildMultHist = kFALSE;
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
*/
//____________________________
myTrainAnalysisConstructor::~myTrainAnalysisConstructor()
{

  AliFemtoCorrFctnIterator iter;
  for(iter=fCollectionOfCfs->begin(); iter!=fCollectionOfCfs->end(); iter++)
  {
    delete *iter;
  }
  delete fCollectionOfCfs;

}

//____________________________
void myTrainAnalysisConstructor::SetParticleTypes(AnalysisType aAnType)
{
  switch(aAnType) {
  case kLamK0:
    fGeneralAnalysisType = kV0V0;
    fParticlePDGType1 = kPDGLam;
    fParticlePDGType2 = kPDGK0;
    break;

  case kALamK0:
    fGeneralAnalysisType = kV0V0;
    fParticlePDGType1 = kPDGALam;
    fParticlePDGType2 = kPDGK0;
    break;

  case kLamKchP:
    fGeneralAnalysisType = kV0Track;
    fParticlePDGType1 = kPDGLam;
    fParticlePDGType2 = kPDGKchP;
    break;

  case kALamKchP:
    fGeneralAnalysisType = kV0Track;
    fParticlePDGType1 = kPDGALam;
    fParticlePDGType2 = kPDGKchP;
    break;

  case kLamKchM:
    fGeneralAnalysisType = kV0Track;
    fParticlePDGType1 = kPDGLam;
    fParticlePDGType2 = kPDGKchM;
    break;

  case kALamKchM:
    fGeneralAnalysisType = kV0Track;
    fParticlePDGType1 = kPDGALam;
    fParticlePDGType2 = kPDGKchM;
    break;

  case kLamLam:
    fGeneralAnalysisType = kV0V0;
    fParticlePDGType1 = kPDGLam;
    fParticlePDGType2 = kPDGLam;
    break;

  case kALamALam:
    fGeneralAnalysisType = kV0V0;
    fParticlePDGType1 = kPDGALam;
    fParticlePDGType2 = kPDGALam;
    break;

  case kLamALam:
    fGeneralAnalysisType = kV0V0;
    fParticlePDGType1 = kPDGLam;
    fParticlePDGType2 = kPDGALam;
    break;

  case kLamPiP:
    fGeneralAnalysisType = kV0Track;
    fParticlePDGType1 = kPDGLam;
    fParticlePDGType2 = kPDGPiP;
    break;

  case kALamPiP:
    fGeneralAnalysisType = kV0Track;
    fParticlePDGType1 = kPDGALam;
    fParticlePDGType2 = kPDGPiP;
    break;

  case kLamPiM:
    fGeneralAnalysisType = kV0Track;
    fParticlePDGType1 = kPDGLam;
    fParticlePDGType2 = kPDGPiM;
    break;

  case kALamPiM:
    fGeneralAnalysisType = kV0Track;
    fParticlePDGType1 = kPDGALam;
    fParticlePDGType2 = kPDGPiM;
    break;

  case kXiKchP:
    fGeneralAnalysisType = kXiTrack;
    fParticlePDGType1 = kPDGXiC;
    fParticlePDGType2 = kPDGKchP;
    break;

  case kAXiKchP:
    fGeneralAnalysisType = kXiTrack;
    fParticlePDGType1 = kPDGAXiC;
    fParticlePDGType2 = kPDGKchP;
    break;

  case kXiKchM:
    fGeneralAnalysisType = kXiTrack;
    fParticlePDGType1 = kPDGXiC;
    fParticlePDGType2 = kPDGKchM;
    break;

  case kAXiKchM:
    fGeneralAnalysisType = kXiTrack;
    fParticlePDGType1 = kPDGAXiC;
    fParticlePDGType2 = kPDGKchM;
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::SetParticleTypes: Invalid AnalysisType"
            "selection '" << aAnType << endl;
  }

  switch(fGeneralAnalysisType) {
  case kV0V0:
    fGeneralParticleType1 = kV0;
    fGeneralParticleType2 = kV0;
    break;

  case kV0Track:
    fGeneralParticleType1 = kV0;
    fGeneralParticleType2 = kTrack;
    break;

  case kXiTrack:
    fGeneralParticleType1 = kCascade;
    fGeneralParticleType2 = kTrack;
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::SetParticleTypes" << endl;
  }



}


//____________________________
void myTrainAnalysisConstructor::SetParticleCut1(ParticlePDGType aParticleType, bool aUseCustom)
{
  switch(aParticleType) {
  case kPDGLam:
  case kPDGALam:
  case kPDGK0:
    V0Cut1 = CreateV0Cut(aParticleType, aUseCustom);
    break;

  case kPDGXiC:
    XiCut1 = CreateXiCut();
    break;

  case kPDGAXiC:
    XiCut1 = CreateAntiXiCut();
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::SetParticleCut1"
            "selection '" << aParticleType << endl;
  }
}

//____________________________
void myTrainAnalysisConstructor::SetParticleCut2(ParticlePDGType aParticleType, bool aUseCustom)
{
  switch(aParticleType) {
  case kPDGLam:
  case kPDGALam:
  case kPDGK0:
    V0Cut2 = CreateV0Cut(aParticleType, aUseCustom);
    break;

  case kPDGKchP:
  case kPDGKchM:
  case kPDGPiP:
  case kPDGPiM:
    TrackCut2 = CreateTrackCut(aParticleType, aUseCustom);
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::SetParticleCut2"
            "selection '" << aParticleType << endl;
  }
}

//____________________________
void myTrainAnalysisConstructor::SetParticleCuts(bool aUseCustom1, bool aUseCustom2)
{
  SetParticleCut1(fParticlePDGType1, aUseCustom1);
  SetParticleCut2(fParticlePDGType2, aUseCustom2);
}


//____________________________
void myTrainAnalysisConstructor::ProcessEvent(const AliFemtoEvent* hbtEvent)
{
  double multiplicity = hbtEvent->UncorrectedNumberOfPrimaries();
  if(fBuildMultHist) fMultHist->Fill(multiplicity);
  AliFemtoVertexMultAnalysis::ProcessEvent(hbtEvent);
}

//____________________________
TList* myTrainAnalysisConstructor::GetOutputList()
{
  TList *olist = new TList();
  TObjArray *temp = new TObjArray();
  olist->SetName(fOutputName);
  temp->SetName(fOutputName);

  TList *tOutputList = AliFemtoSimpleAnalysis::GetOutputList(); 

  if(fBuildMultHist) tOutputList->Add(GetMultHist());

  TListIter next(tOutputList);
  while (TObject *obj = next())
  {
    temp->Add(obj);
  }

  olist->Add(temp);    
  return olist;
}

//____________________________
AliFemtoBasicEventCut* myTrainAnalysisConstructor::CreateBasicEventCut()
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
void myTrainAnalysisConstructor::SetMultHist(const char* name, int aNbins, double aMin, double aMax)
{
  fMultHist = new TH1F(
                       TString::Format("MultHist_%s", name), "Multiplicity",
                       aNbins, aMin, aMax);
  fBuildMultHist = true;
}

//____________________________
TH1F *myTrainAnalysisConstructor::GetMultHist()
{
  return fMultHist;
}

//____________________________
AliFemtoEventCutEstimators* myTrainAnalysisConstructor::CreateEventCutEstimators(const float &aCentLow, const float &aCentHigh)
{
  AliFemtoEventCutEstimators* EvCutEst = new AliFemtoEventCutEstimators();
    EvCutEst->SetCentEst1Range(aCentLow,aCentHigh);
    EvCutEst->SetVertZPos(-8.0,8.0);

    EvCutEst->AddCutMonitor(new AliFemtoCutMonitorEventMult("_EvPass"), new AliFemtoCutMonitorEventMult("_EvFail"));

    EvCutEst->AddCutMonitor(new AliFemtoCutMonitorEventPartCollSize("_Part1",100,0,100,"_Part2",100,0,100));

  return EvCutEst;
}

//____________________________
AliFemtoV0TrackCutNSigmaFilter* myTrainAnalysisConstructor::CreateLambdaCut(bool aUseCustom)
{
  bool tRemoveMisID = true;
  bool tUseSimpleMisID = true;
  bool tUseCustomMisID = false;

  AliFemtoV0TrackCutNSigmaFilter* v0cut1 = new AliFemtoV0TrackCutNSigmaFilter();
    v0cut1->SetParticleType(0);  //  0=lambda -> daughters = proton(+) and pi-
    v0cut1->SetMass(LambdaMass);
    v0cut1->SetInvariantMassLambda(LambdaMass-0.0038,LambdaMass+0.0038);   //m_inv criteria for lambda's

    v0cut1->SetLooseInvMassCut(true, LambdaMass-0.035,LambdaMass+0.035);
    v0cut1->SetMinvPurityAidHistoV0("LambdaPurityAid","LambdaMinvBeforeFinalCut",100,LambdaMass-0.035,LambdaMass+0.035);

    //Misidentification cuts -----*****-----*****-----*****-----*****-----*****-----*****
    v0cut1->SetRemoveMisidentified(tRemoveMisID);
    v0cut1->SetInvMassReject(AliFemtoV0TrackCut::kK0s, K0ShortMass-0.003677,K0ShortMass+0.003677, tRemoveMisID);  //m_inv criteria to remove all lambda candidates fulfilling K0short hypothesis
    v0cut1->SetUseSimpleMisIDCut(tUseSimpleMisID);
    if(!tUseSimpleMisID && tUseCustomMisID)
    {
      v0cut1->CreateCustomV0Rejection(AliFemtoV0TrackCut::kK0s);
      v0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                           0.,0.8,3.,  //positive daughter
                                           0.,0.8,3.); //negative daughter
      v0cut1->AddTPCAndTOFNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                                 0.8,1000.,3.,3.,  //positive daughter
                                                 0.8,1000.,3.,3.); //negative daughter
      v0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                           0.8,1000.,3.,  //positive daughter
                                           0.8,1000.,3.); //negative daughter
    }
    v0cut1->SetBuildMisIDHistograms(true);
      v0cut1->SetMisIDHisto(AliFemtoV0TrackCut::kLambda,100,LambdaMass-0.035,LambdaMass+0.035);
      v0cut1->SetMisIDHisto(AliFemtoV0TrackCut::kK0s,100,K0ShortMass-0.070,K0ShortMass+0.070);

    //-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****

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
      //--Proton(+) daughter selection filter
      //for now, the custom filters will match the standard cuts in AliFemtoV0TrackCut
      //these also match my personal (proton) cuts in myAliFemtoV0TrackCut
      v0cut1->CreateCustomProtonNSigmaFilter();
      v0cut1->AddProtonTPCNSigmaCut(0.,0.8,3.);
      v0cut1->AddProtonTPCAndTOFNSigmaCut(0.8,1000.,3.,3.);
      v0cut1->AddProtonTPCNSigmaCut(0.8,1000.,3.);


      //--Pion(-) daughter selection filter
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
AliFemtoV0TrackCutNSigmaFilter* myTrainAnalysisConstructor::CreateAntiLambdaCut(bool aUseCustom)
{
  bool tRemoveMisID = true;
  bool tUseSimpleMisID = true;
  bool tUseCustomMisID = false;

  AliFemtoV0TrackCutNSigmaFilter* v0cut2 = new AliFemtoV0TrackCutNSigmaFilter();
    v0cut2->SetParticleType(1);  //1=anti-lambda -> daughters = anti-proton(-) and pi+
    v0cut2->SetMass(LambdaMass);
    v0cut2->SetInvariantMassLambda(LambdaMass-0.0038,LambdaMass+0.0038);   //m_inv criteria for anti-lambda's

    v0cut2->SetLooseInvMassCut(true, LambdaMass-0.035,LambdaMass+0.035);
    v0cut2->SetMinvPurityAidHistoV0("AntiLambdaPurityAid","AntiLambdaMinvBeforeFinalCut",100,LambdaMass-0.035,LambdaMass+0.035);

    //Misidentification cuts -----*****-----*****-----*****-----*****-----*****-----*****
    v0cut2->SetRemoveMisidentified(tRemoveMisID);
    v0cut2->SetInvMassReject(AliFemtoV0TrackCut::kK0s, K0ShortMass-0.003677,K0ShortMass+0.003677, tRemoveMisID);  //m_inv criteria to remove all anti-lambda candidates fulfilling K0short hypothesis
    v0cut2->SetUseSimpleMisIDCut(tUseSimpleMisID);
    if(!tUseSimpleMisID && tUseCustomMisID)
    {
      v0cut2->CreateCustomV0Rejection(AliFemtoV0TrackCut::kK0s);
      v0cut2->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                           0.,0.8,3.,  //positive daughter
                                           0.,0.8,3.); //negative daughter
      v0cut2->AddTPCAndTOFNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                                 0.8,1000.,3.,3.,  //positive daughter
                                                 0.8,1000.,3.,3.); //negative daughter
      v0cut2->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kK0s,
                                           0.8,1000.,3.,  //positive daughter
                                           0.8,1000.,3.); //negative daughter
    }
    v0cut2->SetBuildMisIDHistograms(true);
      v0cut2->SetMisIDHisto(AliFemtoV0TrackCut::kAntiLambda,100,LambdaMass-0.035,LambdaMass+0.035);
      v0cut2->SetMisIDHisto(AliFemtoV0TrackCut::kK0s,100,K0ShortMass-0.070,K0ShortMass+0.070);
    //-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****

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
      //--(Anti)Proton(-) daughter selection filter
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
      //--Pion(+) daughter selection filter
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
AliFemtoV0TrackCutNSigmaFilter* myTrainAnalysisConstructor::CreateK0ShortCut(bool aUseCustom)
{
  bool tRemoveMisID = true;
  bool tUseSimpleMisID = true;
  bool tUseCustomMisID = false;

  AliFemtoV0TrackCutNSigmaFilter* k0cut1 = new AliFemtoV0TrackCutNSigmaFilter();
    k0cut1->SetParticleType(2);  //  2=K0Short -> daughters = pi+ and pi-
    k0cut1->SetMass(K0ShortMass);
    k0cut1->SetInvariantMassK0s(K0ShortMass-0.013677,K0ShortMass+0.020323);  //m_inv criteria for K0shorts

    k0cut1->SetLooseInvMassCut(true, K0ShortMass-0.070,K0ShortMass+0.070);
    k0cut1->SetMinvPurityAidHistoV0("K0ShortPurityAid","K0ShortMinvBeforeFinalCut",100,K0ShortMass-0.070,K0ShortMass+0.070);

    //Misidentification cuts -----*****-----*****-----*****-----*****-----*****-----*****
    k0cut1->SetRemoveMisidentified(tRemoveMisID);
    k0cut1->SetInvMassReject(AliFemtoV0TrackCut::kLambda, LambdaMass-0.005683,LambdaMass+0.005683, tRemoveMisID);  //m_inv criteria to remove all K0short candidates fulfilling (anti-)lambda hypothesis
    k0cut1->SetInvMassReject(AliFemtoV0TrackCut::kAntiLambda, LambdaMass-0.005683,LambdaMass+0.005683, tRemoveMisID);  //m_inv criteria to remove all K0short candidates fulfilling (anti-)lambda hypothesis
    k0cut1->SetUseSimpleMisIDCut(tUseSimpleMisID);
    if(!tUseSimpleMisID && tUseCustomMisID)
    {
      //Lambda rejection
      k0cut1->CreateCustomV0Rejection(AliFemtoV0TrackCut::kLambda);
        //Positive daughter (Proton)
        k0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kLambda,1,0.,0.8,3.);
        k0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kLambda,1,0.8,1000.,3.);
        //Negative daughter (Pion)
        k0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kLambda,-1,0.,0.5,3.);
        k0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kLambda,-1,0.5,1000.,3.);

      //AntiLambda rejection
      k0cut1->CreateCustomV0Rejection(AliFemtoV0TrackCut::kAntiLambda);
        //Positive daughter (Pion)
        k0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kAntiLambda,1,0.,0.5,3.);
        k0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kAntiLambda,1,0.5,1000.,3.);
        //Negative daughter (AntiProton)
        k0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kAntiLambda,-1,0.,0.8,3.);
        k0cut1->AddTPCNSigmaCutToV0Rejection(AliFemtoV0TrackCut::kAntiLambda,-1,0.8,1000.,3.);
    }
    k0cut1->SetBuildMisIDHistograms(true);
      k0cut1->SetMisIDHisto(AliFemtoV0TrackCut::kLambda,100,LambdaMass-0.035,LambdaMass+0.035);
      k0cut1->SetMisIDHisto(AliFemtoV0TrackCut::kAntiLambda,100,LambdaMass-0.035,LambdaMass+0.035);
      k0cut1->SetMisIDHisto(AliFemtoV0TrackCut::kK0s,100,K0ShortMass-0.070,K0ShortMass+0.070);
    //-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****

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
      //--Pion(+) daughter selection filter
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
AliFemtoV0TrackCutNSigmaFilter* myTrainAnalysisConstructor::CreateV0Cut(ParticlePDGType aType, bool aUseCustom)
{
  AliFemtoV0TrackCutNSigmaFilter* tReturnCut = new AliFemtoV0TrackCutNSigmaFilter();
  switch(aType) {
  case kPDGLam:
    tReturnCut = CreateLambdaCut(aUseCustom);
    break;

  case kPDGALam:
    tReturnCut = CreateAntiLambdaCut(aUseCustom);
    break;

  case kPDGK0:
    tReturnCut = CreateK0ShortCut(aUseCustom);
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::CreateV0Cut: Invalid ParticlePDGType"
            "selection '" << aType << endl;
  }

  return tReturnCut;
}

//____________________________
AliFemtoESDTrackCutNSigmaFilter* myTrainAnalysisConstructor::CreateKchCut(const int aCharge, bool aUseCustom)
{
  AliFemtoESDTrackCutNSigmaFilter* kaontc1 = new AliFemtoESDTrackCutNSigmaFilter();
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


  if(aUseCustom)
  {
    //Kaon filter
    kaontc1->CreateCustomNSigmaFilter(AliFemtoESDTrackCutNSigmaFilter::kKaon);
      kaontc1->AddTPCNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kKaon,0.0,0.5,2.0);
      kaontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kKaon,0.5,0.8,3.0,2.0);
      kaontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kKaon,0.8,1.0,3.0,1.5);
      kaontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kKaon,1.0,1.5,3.0,1.0);

    //Electron filter for removing misidentified
    kaontc1->CreateCustomNSigmaFilter(AliFemtoESDTrackCutNSigmaFilter::kElectron);
      kaontc1->AddTPCNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kElectron,0.,99.,3.0);

    //Pion filter for removing misidentified
    kaontc1->CreateCustomNSigmaFilter(AliFemtoESDTrackCutNSigmaFilter::kPion);
      kaontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.0,0.65,3.0,3.0);
      kaontc1->AddTPCNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.0,0.35,3.0);
      kaontc1->AddTPCNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.35,0.5,3.0);
      kaontc1->AddTPCNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.5,0.65,2.0);

      kaontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.65,1.5,5.0,3.0);
      kaontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,1.5,99.,5.0,2.0);
  }

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
AliFemtoESDTrackCutNSigmaFilter* myTrainAnalysisConstructor::CreatePiCut(const int aCharge, bool aUseCustom)
{
  AliFemtoESDTrackCutNSigmaFilter* piontc1 = new AliFemtoESDTrackCutNSigmaFilter();
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

  if(aUseCustom)
  {
    //Pion filter
    piontc1->CreateCustomNSigmaFilter(AliFemtoESDTrackCutNSigmaFilter::kPion);
      piontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.0,0.65,3.0,3.0);
      piontc1->AddTPCNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.0,0.35,3.0);
      piontc1->AddTPCNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.35,0.5,3.0);
      piontc1->AddTPCNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.5,0.65,2.0);

      piontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,0.65,1.5,5.0,3.0);
      piontc1->AddTPCAndTOFNSigmaCut(AliFemtoESDTrackCutNSigmaFilter::kPion,1.5,99.,5.0,2.0);
  }

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


AliFemtoESDTrackCutNSigmaFilter* myTrainAnalysisConstructor::CreateTrackCut(ParticlePDGType aType, bool aUseCustom)
{
  AliFemtoESDTrackCutNSigmaFilter* tReturnCut = new AliFemtoESDTrackCutNSigmaFilter();
  switch(aType) {
  case kPDGKchP:
    tReturnCut = CreateKchCut(1, aUseCustom);
    break;

  case kPDGKchM:
    tReturnCut = CreateKchCut(-1, aUseCustom);
    break;

  case kPDGPiP:
    tReturnCut = CreatePiCut(1, aUseCustom);
    break;

  case kPDGPiM:
    tReturnCut = CreatePiCut(-1, aUseCustom);
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::CreateTrackCut: Invalid ParticleType"
            "selection '" << aType << endl;
  }

  return tReturnCut;
}

//____________________________
AliFemtoXiTrackCut* myTrainAnalysisConstructor::CreateXiCut()
{
  //NOTE: the SetMass call actually is important
  //      This should be set to the mass of the particle of interest, here the Xi
  //      Be sure to not accidentally set it again in the Lambda cuts (for instance, when copy/pasting the lambda cuts from above!)

  //Xi -> Lam Pi-

  AliFemtoXiTrackCut* tXiCut = new AliFemtoXiTrackCut();

  //Xi Cuts
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

  tXiCut->SetMinvPurityAidHistoXi("XiPurityAid","XiMinvBeforeFinalCut",100,XiMass-0.035,XiMass+0.035);
  tXiCut->SetMinvPurityAidHistoV0("LambdaPurityAid","LambdaMinvBeforeFinalCut",100,LambdaMass-0.035,LambdaMass+0.035);

  tXiCut->AddCutMonitor(new AliFemtoCutMonitorXi("_Xi_Pass"),new AliFemtoCutMonitorXi("_Xi_Fail"));

  return tXiCut;
}

//____________________________
AliFemtoXiTrackCut* myTrainAnalysisConstructor::CreateAntiXiCut()
{
  //NOTE: the SetMass call actually is important
  //      This should be set to the mass of the particle of interest, here the Xi
  //      Be sure to not accidentally set it again in the Lambda cuts (for instance, when copy/pasting the lambda cuts from above!)

  //AXi -> ALam Pi+

  AliFemtoXiTrackCut* tAXiCut = new AliFemtoXiTrackCut();

  //Xi Cuts
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

  tAXiCut->SetMinvPurityAidHistoXi("AXiPurityAid","AXiMinvBeforeFinalCut",100,XiMass-0.035,XiMass+0.035);
  tAXiCut->SetMinvPurityAidHistoV0("AntiLambdaPurityAid","AntiLambdaMinvBeforeFinalCut",100,LambdaMass-0.035,LambdaMass+0.035);

  tAXiCut->AddCutMonitor(new AliFemtoCutMonitorXi("_AXi_Pass"),new AliFemtoCutMonitorXi("_AXi_Fail"));

  return tAXiCut;
}


//____________________________
AliFemtoXiTrackCut* myTrainAnalysisConstructor::CreateCascadeCut(ParticlePDGType aType)
{
  AliFemtoXiTrackCut *tReturnCut = new AliFemtoXiTrackCut();
  switch(aType) {
  case kPDGXiC:
    tReturnCut = CreateXiCut();
    break;

  case kPDGAXiC:
    tReturnCut = CreateAntiXiCut();
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::CreateCascadeCut: Invalid ParticleType"
            "selection '" << aType << endl;

  }

  return tReturnCut;
}


//____________________________
AliFemtoV0PairCut* myTrainAnalysisConstructor::CreateV0PairCut(double aMinAvgSepPosPos, double aMinAvgSepPosNeg, double aMinAvgSepNegPos, double aMinAvgSepNegNeg)
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
AliFemtoV0TrackPairCut* myTrainAnalysisConstructor::CreateV0TrackPairCut(double aMinAvgSepTrackPos, double aMinAvgSepTrackNeg)
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
AliFemtoXiTrackPairCut* myTrainAnalysisConstructor::CreateXiTrackPairCut()
{
  AliFemtoXiTrackPairCut *tXiTrackPairCut = new AliFemtoXiTrackPairCut();

  return tXiTrackPairCut;
}


//____________________________
void myTrainAnalysisConstructor::CreatePairCut(double aArg1, double aArg2, double aArg3, double aArg4)
{
  switch(fGeneralAnalysisType) {
  case kV0V0:
    V0PairCut = CreateV0PairCut(aArg1,aArg2,aArg3,aArg4);
    break;

  case kV0Track:
    V0TrackPairCut = CreateV0TrackPairCut(aArg1,aArg2);
    break;

  case kXiTrack:
    XiTrackPairCut = CreateXiTrackPairCut();
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::CreatePairCut" << endl;
  }
}


//____________________________
AliFemtoCorrFctnKStar* myTrainAnalysisConstructor::CreateCorrFctnKStar(const char* name, unsigned int bins, double min, double max)
{
  AliFemtoCorrFctnKStar *cf = new AliFemtoCorrFctnKStar(TString::Format("KStarCf_%s",name),bins,min,max);
    cf->SetCalculatePairKinematics(fWritePairKinematics);
    cf->SetBuildkTBinned(false);
    cf->SetBuildmTBinned(false);
    cf->SetBuild3d(false);
  return cf;
}

//____________________________
AliFemtoAvgSepCorrFctn* myTrainAnalysisConstructor::CreateAvgSepCorrFctn(const char* name, unsigned int bins, double min, double max)
{
  AliFemtoAvgSepCorrFctn *cf = new AliFemtoAvgSepCorrFctn(TString::Format("AvgSepCf_%s", name),bins,min,max);

  switch(fGeneralAnalysisType) {
  case kV0V0:
    cf->SetPairType(AliFemtoAvgSepCorrFctn::kV0s);
    break;

  case kV0Track:
    cf->SetPairType(AliFemtoAvgSepCorrFctn::kTrackV0);
    break;

  case kXiTrack:
    cf->SetPairType(AliFemtoAvgSepCorrFctn::kTrackV0); //TODO
    break;

  default:
    cerr << "E-myTrainAnalysisConstructor::CreateAvgSepCorrFctn" << endl;
  }
  return cf;
}

/*
//____________________________
myAliFemtoSepCorrFctns* myTrainAnalysisConstructor::CreateSepCorrFctns(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY)
{
  myAliFemtoSepCorrFctns *cf = new myAliFemtoSepCorrFctns(name,binsX,minX,maxX,binsY,minY,maxY);
  return cf;
}

//____________________________
myAliFemtoAvgSepCorrFctnCowboysAndSailors* myTrainAnalysisConstructor::CreateAvgSepCorrFctnCowboysAndSailors(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY)
{
  myAliFemtoAvgSepCorrFctnCowboysAndSailors *cf = new myAliFemtoAvgSepCorrFctnCowboysAndSailors(name,binsX,minX,maxX,binsY,minY,maxY);
  return cf;
}

//____________________________
myAliFemtoKStarCorrFctn2D* myTrainAnalysisConstructor::CreateCorrFctnKStar2D(const char* name, unsigned int nbinsKStar, double KStarLo, double KStarHi, unsigned int nbinsY, double YLo, double YHi)
{
  myAliFemtoKStarCorrFctn2D *cf = new myAliFemtoKStarCorrFctn2D(name,nbinsKStar,KStarLo,KStarHi,nbinsY,YLo,YHi);
  return cf;
}
*/

//____________________________
AliFemtoModelCorrFctnKStarFull* myTrainAnalysisConstructor::CreateModelCorrFctnKStarFull(const char* name, unsigned int bins, double min, double max)
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

  AliFemtoModelCorrFctnKStarFull *cf = new AliFemtoModelCorrFctnKStarFull(TString::Format("KStarModelCf_%s",name),bins,min,max);
    cf->SetRemoveMisidentified(false);
    cf->SetExpectedPDGCodes((int)fParticlePDGType1,(int)fParticlePDGType2);
  cf->ConnectToManager(tManager);
  return cf;
}






//____________________________
void myTrainAnalysisConstructor::SetAnalysis(AliFemtoEventCut* aEventCut, AliFemtoParticleCut* aPartCut1, AliFemtoParticleCut* aPartCut2, AliFemtoPairCut* aPairCut, AliFemtoCorrFctnCollection* aCollectionOfCfs)
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
void myTrainAnalysisConstructor::SetAnalysis(AliFemtoEventCut* aEventCut, AliFemtoParticleCut* aPartCut1, AliFemtoParticleCut* aPartCut2, AliFemtoPairCut* aPairCut)
{
  SetEventCut(aEventCut);
  SetFirstParticleCut(aPartCut1);
  SetSecondParticleCut(aPartCut2);
  SetPairCut(aPairCut);

  AliFemtoCorrFctnIterator iter;
  for(iter=fCollectionOfCfs->begin(); iter!=fCollectionOfCfs->end(); iter++)
  {
    AddCorrFctn(*iter);
  }
}

//____________________________
void myTrainAnalysisConstructor::SetImplementAvgSepCuts(bool aImplementAvgSepCuts)
{
  fImplementAvgSepCuts = aImplementAvgSepCuts;
}

