/////////////////////////////////////////////////////////////////////////////////
//									       //
//  myTrainAnalysisConstructor - initiates an AliFemtoVertexMultAnalysis and   //
//  the particle cuts to make running multiple analyses in a single taks       //
//  easier.  It will also add additional objects to the tOutputList (just      //
//  as myAliFemtoVertexMultAnalysis does).  Using this method, I derive        //
//  from the AliFemtoVertexMultAnalysis class, instead of re-writting it       //
//  to include my additional functionality (like in myAliFemto...              //
/////////////////////////////////////////////////////////////////////////////////

#ifndef MYTRAINANALYSISCONSTRUCTOR_H
#define MYTRAINANALYSISCONSTRUCTOR_H

#include "AliFemtoSimpleAnalysis.h"
#include "AliFemtoVertexMultAnalysis.h"

#include "AliFemtoEventCut.h"
#include "AliFemtoBasicEventCut.h"
#include "AliFemtoEventCutEstimators.h"
#include "AliFemtoCutMonitorEventMult.h"
#include "AliFemtoCutMonitorV0.h"
#include "AliFemtoCutMonitorXi.h"

#include "AliFemtoParticleCut.h"
#include "AliFemtoV0TrackCut.h"
#include "AliFemtoBasicTrackCut.h"
#include "AliFemtoESDTrackCut.h"
#include "AliFemtoAODTrackCut.h"
#include "AliFemtoCutMonitorParticleYPt.h"
#include "AliFemtoCutMonitorParticlePID.h"

#include "AliFemtoXiTrackCut.h"
#include "AliFemtoXiTrackPairCut.h"

#include "AliFemtoPairCut.h"
#include "AliFemtoV0PairCut.h"
#include "AliFemtoV0TrackPairCut.h"
#include "AliFemtoPairOriginMonitor.h"

#include "AliFemtoCorrFctn.h"
#include "AliFemtoCorrFctnKStar.h"
#include "AliFemtoCorrFctnCollection.h"
#include "AliFemtoAvgSepCorrFctn.h"

/*
#include "myAliFemtoSepCorrFctns.h"
#include "myAliFemtoAvgSepCorrFctnCowboysAndSailors.h"
#include "myAliFemtoKStarCorrFctn2D.h"
*/

#include "AliFemtoNSigmaFilter.h"
#include "AliFemtoV0TrackCutNSigmaFilter.h"
#include "AliFemtoESDTrackCutNSigmaFilter.h"

#include "AliFemtoCutMonitorEventPartCollSize.h"

#include "AliFemtoModelWeightGeneratorBasicLednicky.h"
#include "AliFemtoModelCorrFctnKStarFull.h"

#include <string>
#include <iostream>
#include <stdio.h>
#include <typeinfo>


class myTrainAnalysisConstructor : public AliFemtoVertexMultAnalysis {

public:
  enum AnalysisType {kLamK0=0, kALamK0=1, 
                     kLamKchP=2, kALamKchP=3, kLamKchM=4, kALamKchM=5, 
                     kLamLam=6, kALamALam=7, kLamALam=8, 
                     kLamPiP=9, kALamPiP=10, kLamPiM=11, kALamPiM=12, 
                     kXiKchP=13, kAXiKchP=14, kXiKchM=15, kAXiKchM=16};

  enum GeneralAnalysisType {kV0V0=0, kV0Track=1, kXiTrack=2};

  enum ParticlePDGType {kPDGProt   = 2212,  kPDGAntiProt = -2212, 
		        kPDGPiP    = 211,   kPDGPiM      = -211, 
                        kPDGK0     = 310,
                        kPDGKchP   = 321,   kPDGKchM     = -321,
		        kPDGLam    = 3122,  kPDGALam     = -3122,
		        kPDGSigma  = 3212,  kPDGASigma   = -3212,
		        kPDGXiC    = 3312,  kPDGAXiC     = -3312,
		        kPDGXi0    = 3322,  kPDGAXi0     = -3322,
		        kPDGOmega  = 3334,  kPDGAOmega   = -3334,
                        kPDGNull      = 0                        };

  enum GeneralParticleType {kV0=0, kTrack=1, kCascade=2};

/*
  struct AnalysisParams;
  struct EventCutParams;
  struct V0CutParams;
  struct ESDCutParams;
  struct XiCutParams;
  struct PairCutParams;
*/

  myTrainAnalysisConstructor();
  myTrainAnalysisConstructor(AnalysisType aAnalysisType, const char* name, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics=false);
  myTrainAnalysisConstructor(AnalysisType aAnalysisType, const char* name, unsigned int binsVertex, double minVertex, double maxVertex, unsigned int binsMult, double minMult, double maxMult, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics=false);
    //Since I am using rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality), 
      // in AliFemtoEventReaderAOD.cxx this causes tEvent->SetNormalizedMult(lrint(10*cent->GetCentralityPercentile("V0A"))), i.e. fNormalizedMult in [0,1000]
      // Therefore, since fNormalizedMult is presumably >= -1, in AliFemtoEvent.cxx the call UncorrectedNumberOfPrimaries returns fNormalizedMult
    //LONG STORY SHORT:  the inputs for multiplicity in the above are actually for 10*centrality (i.e. 0-100 for 0-10% centrality)
    //Note:  fNormalizedMult is typically in range [0,1000] (as can be seen in AliFemtoEventReaderAOD.cxx).  This appears true when SetUseMultiplicity is set to kCentrality, kCentralityV0A, kCentralityV0C, kCentralityZNA, kCentralityZNC, kCentralityCL1, kCentralityCL0, kCentralityTRK, kCentralityTKL, kCentralityCND, kCentralityNPA, kCentralityFMD.
      // fNormalizedMult WILL NOT be [0,1000] when SetUseMultiplicity is set to kGlobalCount, kReference, kTPCOnlyRef, and kVZERO 



  myTrainAnalysisConstructor(const myTrainAnalysisConstructor& TheOriginalAnalysis);  //copy constructor - 30 June 2015 //TODO
  myTrainAnalysisConstructor& operator=(const myTrainAnalysisConstructor& TheOriginalAnalysis);  //assignment operator - 30 June 2015 //TODO
  virtual ~myTrainAnalysisConstructor();

  virtual void ProcessEvent(const AliFemtoEvent* ProcessThisEvent);  //will add fMultHist to the process event of AliFemtoVertexMultAnalysis
  virtual TList* GetOutputList();

  void SetParticleTypes(AnalysisType aAnType);
  void SetParticleCut1(ParticlePDGType aParticlePDGType, bool aUseCustom);
  void SetParticleCut2(ParticlePDGType aParticlePDGType, bool aUseCustom);
  void SetParticleCuts(bool aUseCustom1, bool aUseCustom2);

  AliFemtoBasicEventCut* CreateBasicEventCut();
  AliFemtoEventCutEstimators* CreateEventCutEstimators(const float &aCentLow, const float &aCentHigh);

  AliFemtoV0TrackCut* CreateLambdaCut();
  AliFemtoV0TrackCut* CreateAntiLambdaCut();
  AliFemtoV0TrackCut* CreateK0ShortCut();

  AliFemtoV0TrackCutNSigmaFilter* CreateLambdaCut(bool aUseCustom);
  AliFemtoV0TrackCutNSigmaFilter* CreateAntiLambdaCut(bool aUseCustom);
  AliFemtoV0TrackCutNSigmaFilter* CreateK0ShortCut(bool aUseCustom);
  AliFemtoV0TrackCutNSigmaFilter* CreateV0Cut(ParticlePDGType aType, bool aUseCustom);
  template<typename T>
  T* CloneV0Cut(T* aCut);

  AliFemtoESDTrackCutNSigmaFilter* CreateKchCut(const int aCharge, bool aUseCustom);
  AliFemtoESDTrackCutNSigmaFilter* CreatePiCut(const int aCharge, bool aUseCustom);
  AliFemtoESDTrackCutNSigmaFilter* CreateTrackCut(ParticlePDGType aType, bool aUseCustom);

  AliFemtoXiTrackCut* CreateXiCut();
  AliFemtoXiTrackCut* CreateAntiXiCut();
  AliFemtoXiTrackCut* CreateCascadeCut(ParticlePDGType aType);

  AliFemtoV0PairCut* CreateV0PairCut(double aMinAvgSepPosPos, double aMinAvgSepPosNeg, double aMinAvgSepNegPos, double aMinAvgSepNegNeg);
  AliFemtoV0TrackPairCut* CreateV0TrackPairCut(double aMinAvgSepTrackPos, double aMinAvgSepTrackNeg);
  AliFemtoXiTrackPairCut* CreateXiTrackPairCut();
  void CreatePairCut(double aArg1=0.0, double aArg2=0.0, double aArg3=0.0, double aArg4=0.0); 

  AliFemtoCorrFctnKStar* CreateCorrFctnKStar(const char* name, unsigned int bins, double min, double max);
  AliFemtoAvgSepCorrFctn* CreateAvgSepCorrFctn(const char* name, unsigned int bins, double min, double max);
/*
  myAliFemtoSepCorrFctns* CreateSepCorrFctns(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY);
  myAliFemtoAvgSepCorrFctnCowboysAndSailors *CreateAvgSepCorrFctnCowboysAndSailors(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY);
  myAliFemtoKStarCorrFctn2D* CreateCorrFctnKStar2D(const char* name, unsigned int nbinsKStar, double KStarLo, double KStarHi, unsigned int nbinsY, double YLo, double YHi);
*/

  AliFemtoModelCorrFctnKStarFull* CreateModelCorrFctnKStarFull(const char* name, unsigned int bins, double min, double max);    //TODO check that enum to int is working

  void AddCutMonitors(AliFemtoEventCut* aEventCut, AliFemtoParticleCut* aPartCut1, AliFemtoParticleCut* aPartCut2, AliFemtoPairCut* aPairCut);
  void SetAnalysis(AliFemtoEventCut* aEventCut, AliFemtoParticleCut* aPartCut1, AliFemtoParticleCut* aPartCut2, AliFemtoPairCut* aPairCut, AliFemtoCorrFctnCollection* aCollectionOfCfs);
  void SetAnalysis(AliFemtoEventCut* aEventCut, AliFemtoParticleCut* aPartCut1, AliFemtoParticleCut* aPartCut2, AliFemtoPairCut* aPairCut);

  void SetMultHist(const char* name, int aNbins=30, double aMin=0., double aMax=3000);
  TH1F *GetMultHist();

  void SetImplementAvgSepCuts(bool aImplementAvgSepCuts);

  //-----25/02/2016
  void SetRemoveMisidentifiedMCParticles(bool aRemove);
  AliFemtoCorrFctnCollection* GetCollectionOfCfs();

protected:
  static const char* const fAnalysisTags[];

  AnalysisType fAnalysisType;
  GeneralAnalysisType fGeneralAnalysisType;
  ParticlePDGType fParticlePDGType1, fParticlePDGType2;
  GeneralParticleType fGeneralParticleType1, fGeneralParticleType2;
  const char* fOutputName;		      /* name given to output directory for specific analysis*/
  TH1F* fMultHist;			      //histogram of event multiplicities to ensure event cuts are properly implemented
  bool fImplementAvgSepCuts;		      //Self-explanatory, set to kTRUE when I want Avg Sep cuts implemented
  bool fWritePairKinematics;
  bool fIsMCRun;
  bool fIsMBAnalysis;
  bool fBuildMultHist;

  double fMinCent, fMaxCent;

  AliFemtoCorrFctnCollection* fCollectionOfCfs;

  //----------------------------------------
  //-Event cuts
  AliFemtoBasicEventCut *BasicEvCut;
  AliFemtoEventCutEstimators *EvCutEst;

  AliFemtoXiTrackCut *XiCut1, *XiCut2;
  AliFemtoV0TrackCutNSigmaFilter *V0Cut1, *V0Cut2;
  AliFemtoESDTrackCutNSigmaFilter *TrackCut1, *TrackCut2;

  AliFemtoV0PairCut *V0PairCut;
  AliFemtoV0TrackPairCut *V0TrackPairCut;
  AliFemtoXiTrackPairCut *XiTrackPairCut;

  AliFemtoCorrFctnKStar *KStarCf;
  AliFemtoAvgSepCorrFctn *AvgSepCf;

  /* Not yet built
  AliFemtoSepCorrFctns *SepCfs;
  myAliFemtoAvgSepCorrFctnCowboysAndSailors *AvgSepCfCowboysAndSailors;
  myAliFemtoKStarCorrFctn2D *KStarCf2D;
*/
  AliFemtoModelCorrFctnKStarFull *KStarModelCfs;



#ifdef __ROOT__
  ClassDef(myTrainAnalysisConstructor, 0)
#endif

};

inline void myTrainAnalysisConstructor::SetRemoveMisidentifiedMCParticles(bool aRemove) {KStarModelCfs->SetRemoveMisidentified(aRemove);}
inline AliFemtoCorrFctnCollection* myTrainAnalysisConstructor::GetCollectionOfCfs() {return fCollectionOfCfs;}

/*
struct myTrainAnalysisConstructor::AnalysisParams
{
  unsigned int nBinsVertex;
  double minVertex,
         maxVertex;

  unsigned int nBinsMult;
  double minMult, 
         maxMult;

  AnalysisType analysisType;
  GeneralAnalysisType generalAnalysisType;

  unsigned int nEventsToMix;
  unsigned int minCollectionSize;

  bool verbose;
  bool enablePairMonitors;
  bool isMCRun;


  Bool_t group_output_objects;
  Bool_t is_mc_analysis;
};

struct myTrainAnalysisConstructor::EventCutParams
{
  double minCentrality,
         maxCentrality;

  double minMult,
         maxMult;

  double minVertexZ,
         maxVertexZ;
};

struct myTrainAnalysisConstructor::V0CutParams
{
  ParticlePDGType particlePDGType;
  GeneralParticleType generalParticleType;

  int v0Type;  //0=kLambda, 1=kAntiLambda, 2=kK0s

  double mass;
  double minInvariantMass,
         maxInvariantMass;

  bool useLooseInvMassCut;
  double minLooseInvMass,
         maxLooseInvMass;

  int nBinsPurity;
  double minPurityMass,
         maxPurityMass;

  bool useCustomFilter;

  bool removeMisID;
  double minInvMassReject,
         maxInvMassReject;

  bool useSimpleMisID;
  bool buildMisIDHistograms;

  double eta;
  double minPt,
         maxPt;
  bool onFlyStatus;
  double maxDcaV0;
  double minCosPointingAngle;
  double maxV0DecayLength;

  double etaDaughters;
  double minPtPosDaughter,
         maxPtPosDaughter;
  double minPtNegDaughter,
         maxPtNegDaughter;
  int minTPCnclsDaughters;
  double maxDcaV0Daughters;
  double minPosDaughterToPrimVertex,
         minNegDaughterToPrimVertex;

};

struct myTrainAnalysisConstructor::ESDCutParams
{
  double pidProbPion;
  double pidProbMuon;
  double pidProbKaon;
  double pidProbProton;
  int mostProbable;
  int charge;
  double mass;

  double minPt,
         maxPt;
  double eta;
  int minTPCncls;

  bool removeKinks;
  bool setLabel;
  double maxITSChiNdof;
  double maxTPCChiNdof;
  double maxSigmaToVertex;
  double maxImpactXY;
  double maxImpactZ;

  bool useCustomFilter;
  bool useElectronRejection;
  bool usePionRejection;
};

struct myTrainAnalysisConstructor::XiCutParams
{
  int charge;
  int xiType;
  double minPt,
         maxPt;
  double eta;
  double mass;
  double minInvariantMass,
         maxInvariantMass;

  double maxDecayLengthXi;
  double minCosPointingAngleXi;
  double maxDcaXi;
  double maxDcaXiDaughters;

  double minDcaXiBac;
  double etaBac;
  double minPtBac,
         maxPtBac;

  int v0Type;
  double minDcaV0;
  double minInvMassV0,
         maxInvMassV0;
  double minCosPointingAngleV0;
  double etaV0;
  double minPtV0,
         maxPtV0;
  bool onFlyStatusV0;
  double maxV0DecayLength;
  double minV0DaughtersToPrimVertex,
         maxV0DaughtersToPrimVertex;
  double maxDcaV0Daughters;
  double etaV0Daughters;
  double minPtPosV0Daughter,
         maxPtPosV0Daughter;
  double minPtNegV0Daughter,
         maxPtNegV0Daughter;

  int minTPCnclsV0Daughters;

  bool setPurityAidXi;
  bool setPurityAidV0;
};

struct myTrainAnalysisConstructor::PairCutParams
{
  bool removeSameLabel;
  double shareQualityMax;
  double shareFractionMax;
  bool tpcOnly;

  double tpcExitSepMinimum;
  double tpcEntranceSepMinimum;

  double minAvgSepPosPos,
         minAvgSepPosNeg,
         minAvgSepNegPos,
         minAvgSepNegNeg;

  double minAvgSepTrackPos,
         minAvgSepTrackNeg;
};
*/

#endif
