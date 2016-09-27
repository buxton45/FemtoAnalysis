// AliFemtoAnalysisLambdaKaon.h

#ifndef ALIFEMTOANALYSISLAMBDAKAON_H
#define ALIFEMTOANALYSISLAMBDAKAON_H

#include "AliFemtoVertexMultAnalysis.h"

#include "AliFemtoBasicEventCut.h"
#include "AliFemtoEventCutEstimators.h"
#include "AliFemtoCutMonitorEventMult.h"
#include "AliFemtoCutMonitorV0.h"
#include "AliFemtoCutMonitorXi.h"

#include "AliFemtoBasicTrackCut.h"
#include "AliFemtoESDTrackCut.h"
#include "AliFemtoAODTrackCut.h"
#include "AliFemtoCutMonitorParticleYPt.h"
#include "AliFemtoCutMonitorParticlePID.h"

#include "AliFemtoXiTrackCut.h"
#include "AliFemtoXiTrackPairCut.h"

#include "AliFemtoV0PairCut.h"
#include "AliFemtoV0TrackPairCut.h"
#include "AliFemtoPairOriginMonitor.h"

#include "AliFemtoCorrFctnKStar.h"
#include "AliFemtoAvgSepCorrFctn.h"


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


class AliFemtoAnalysisLambdaKaon : public AliFemtoVertexMultAnalysis {

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

  struct AnalysisParams;
  struct EventCutParams;
  struct V0CutParams;
  struct ESDCutParams;
  struct XiCutParams;
  struct PairCutParams;

  AliFemtoAnalysisLambdaKaon(AnalysisType aAnalysisType, const char* name, unsigned int binsVertex, double minVertex, double maxVertex, unsigned int binsMult, double minMult, double maxMult, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics=false);
    //Since I am using rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality), 
      // in AliFemtoEventReaderAOD.cxx this causes tEvent->SetNormalizedMult(lrint(10*cent->GetCentralityPercentile("V0A"))), i.e. fNormalizedMult in [0,1000]
      // Therefore, since fNormalizedMult is presumably >= -1, in AliFemtoEvent.cxx the call UncorrectedNumberOfPrimaries returns fNormalizedMult
    //LONG STORY SHORT:  the inputs for multiplicity in the above are actually for 10*centrality (i.e. 0-100 for 0-10% centrality)
    //Note:  fNormalizedMult is typically in range [0,1000] (as can be seen in AliFemtoEventReaderAOD.cxx).  This appears true when SetUseMultiplicity is set to kCentrality, kCentralityV0A, kCentralityV0C, kCentralityZNA, kCentralityZNC, kCentralityCL1, kCentralityCL0, kCentralityTRK, kCentralityTKL, kCentralityCND, kCentralityNPA, kCentralityFMD.
      // fNormalizedMult WILL NOT be [0,1000] when SetUseMultiplicity is set to kGlobalCount, kReference, kTPCOnlyRef, and kVZERO 

  virtual ~AliFemtoAnalysisLambdaKaon();

  void SetParticleTypes(AnalysisType aAnType);

  void AddCustomV0SelectionFilters(ParticlePDGType aV0Type, AliFemtoV0TrackCutNSigmaFilter* aCut);
  void AddCustomV0RejectionFilters(ParticlePDGType aV0Type, AliFemtoV0TrackCutNSigmaFilter* aCut);

  AliFemtoV0TrackCutNSigmaFilter* CreateV0Cut(V0CutParams &aCutParams);

  //------Builders for default cut objects
  AnalysisParams DefaultAnalysisParams();
  EventCutParams DefaultEventCutParams();

  V0CutParams DefaultLambdaCutParams();
  V0CutParams DefaultAntiLambdaCutParams();
  V0CutParams DefaultK0ShortCutParams();

  ESDCutParams DefaultKchCutParams(int aCharge);

  XiCutParams DefaultXiCutParams();
  XiCutParams DefaultAXiCutParams();

  PairCutParams DefaultPairParams();
  //----------------------------------------






protected:
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

  AliFemtoModelCorrFctnKStarFull *KStarModelCfs;





#ifdef __ROOT__
  ClassDef(AliFemtoAnalysisLambdaKaon, 0)
#endif

};

struct AliFemtoAnalysisLambdaKaon::AnalysisParams
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
};

struct AliFemtoAnalysisLambdaKaon::EventCutParams
{
  double minCentrality,
         maxCentrality;

  double minMult,
         maxMult;

  double minVertexZ,
         maxVertexZ;
};

struct AliFemtoAnalysisLambdaKaon::V0CutParams
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
  bool useCustomMisID;

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

struct AliFemtoAnalysisLambdaKaon::ESDCutParams
{
  ParticlePDGType particlePDGType;
  GeneralParticleType generalParticleType;

  double minPidProbPion,
         maxPidProbPion;
  double minPidProbMuon,
         maxPidPronMuon;
  double minPidProbKaon,
         maxPidProbKaon;
  double minPidProbProton,
         maxPidProbProton;
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

struct AliFemtoAnalysisLambdaKaon::XiCutParams
{
  ParticlePDGType particlePDGType;
  GeneralParticleType generalParticleType;

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
  int minTPCnclsBac;
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

struct AliFemtoAnalysisLambdaKaon::PairCutParams
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

#endif
