////////////////////////////////////////////////////////////////////////////
//									  //
//  myAnalysisConstructor - initiates an AliFemtoVertexMultAnalysis and   //
//  the particle cuts to make running multiple analyses in a single taks  //
//  easier.  It will also add additional objects to the tOutputList (just //
//  as myAliFemtoVertexMultAnalysis does).  Using this method, I derive   //
//  from the AliFemtoVertexMultAnalysis class, instead of re-writting it  //
//  to include my additional functionality (like in myAliFemto...         //
////////////////////////////////////////////////////////////////////////////

#ifndef MYANALYSISCONSTRUCTOR_H
#define MYANALYSISCONSTRUCTOR_H

#include "AliFemtoSimpleAnalysis.h"
#include "AliFemtoVertexMultAnalysis.h"

#include "AliFemtoEventCut.h"
#include "AliFemtoBasicEventCut.h"
#include "AliFemtoEventCutEstimators.h"
#include "AliFemtoCutMonitorEventMult.h"
#include "AliFemtoCutMonitorV0.h"
#include "AliFemtoCutMonitorXi.h"

#include "AliFemtoParticleCut.h"
#include "myAliFemtoV0TrackCut.h"
#include "AliFemtoBasicTrackCut.h"
#include "AliFemtoESDTrackCut.h"
#include "myAliFemtoESDTrackCut.h"
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
#include "AliFemtoCorrFctnCollection.h"
#include "myAliFemtoKStarCorrFctn.h"
#include "myAliFemtoAvgSepCorrFctn.h"
#include "myAliFemtoSepCorrFctns.h"
#include "myAliFemtoAvgSepCorrFctnCowboysAndSailors.h"
#include "myAliFemtoKStarCorrFctn2D.h"
#include "myAliFemtoKStarCorrFctnMC.h"

//-----17/12/2015----------------------
#include "AliFemtoV0TrackCutNSigmaFilter.h"
#include "AliFemtoNSigmaFilter.h"

//-----01/02/2016
#include "AliFemtoCutMonitorEventPartCollSize.h"

//-----04/02/2016
#include "AliFemtoModelWeightGeneratorBasicLednicky.h"
#include "myAliFemtoModelCorrFctnKStar.h"

#include <string>
#include <iostream>
#include <stdio.h>

class myAnalysisConstructor : public AliFemtoVertexMultAnalysis {

public:
  enum AnalysisType {kLamK0=0, kALamK0=1, kLamKchP=2, kALamKchP=3, kLamKchM=4, kALamKchM=5, kLamLam=6, kALamALam=7, kLamALam=8, kLamPiP=9, kALamPiP=10, kLamPiM=11, kALamPiM=12, kXiKchP=13, kAXiKchP=14, kXiKchM=15, kAXiKchM=16};

  myAnalysisConstructor();
  myAnalysisConstructor(AnalysisType aAnalysisType, const char* name, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics=false);
  myAnalysisConstructor(AnalysisType aAnalysisType, const char* name, unsigned int binsVertex, double minVertex, double maxVertex, unsigned int binsMult, double minMult, double maxMult, bool aIsMCRun, bool aImplementAvgSepCuts, bool aWritePairKinematics=false);
    //Since I am using rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality), 
      // in AliFemtoEventReaderAOD.cxx this causes tEvent->SetNormalizedMult(lrint(10*cent->GetCentralityPercentile("V0A"))), i.e. fNormalizedMult in [0,1000]
      // Therefore, since fNormalizedMult is presumably >= -1, in AliFemtoEvent.cxx the call UncorrectedNumberOfPrimaries returns fNormalizedMult
    //LONG STORY SHORT:  the inputs for multiplicity in the above are actually for 10*centrality (i.e. 0-100 for 0-10% centrality)
    //Note:  fNormalizedMult is typically in range [0,1000] (as can be seen in AliFemtoEventReaderAOD.cxx).  This appears true when SetUseMultiplicity is set to kCentrality, kCentralityV0A, kCentralityV0C, kCentralityZNA, kCentralityZNC, kCentralityCL1, kCentralityCL0, kCentralityTRK, kCentralityTKL, kCentralityCND, kCentralityNPA, kCentralityFMD.
      // fNormalizedMult WILL NOT be [0,1000] when SetUseMultiplicity is set to kGlobalCount, kReference, kTPCOnlyRef, and kVZERO 



  myAnalysisConstructor(const myAnalysisConstructor& TheOriginalAnalysis);  //copy constructor - 30 June 2015
  myAnalysisConstructor& operator=(const myAnalysisConstructor& TheOriginalAnalysis);  //assignment operator - 30 June 2015
  virtual ~myAnalysisConstructor();

  virtual void ProcessEvent(const AliFemtoEvent* ProcessThisEvent);  //will add fMultHist to the process event of AliFemtoVertexMultAnalysis
  virtual TList* GetOutputList();


  //I want to create the analysis BEFORE I set the analysis.  Once I set the analysis, things cannot be changed.
  //This allows me to tweak the cuts before finalizing everything (the lesson I learned from SetImplementAvgSepCuts)
  void CreateLamK0Analysis();
    void SetLamK0Analysis();
  void CreateALamK0Analysis();
    void SetALamK0Analysis();

  void CreateLamKchPAnalysis();
    void SetLamKchPAnalysis();
  void CreateALamKchPAnalysis();
    void SetALamKchPAnalysis();
  void CreateLamKchMAnalysis();
    void SetLamKchMAnalysis();
  void CreateALamKchMAnalysis();
    void SetALamKchMAnalysis();

  void CreateLamLamAnalysis();
    void SetLamLamAnalysis();
  void CreateALamALamAnalysis();
    void SetALamALamAnalysis();
  void CreateLamALamAnalysis();
    void SetLamALamAnalysis();

  void CreateLamPiPAnalysis();
    void SetLamPiPAnalysis();
  void CreateALamPiPAnalysis();
    void SetALamPiPAnalysis();
  void CreateLamPiMAnalysis();
    void SetLamPiMAnalysis();
  void CreateALamPiMAnalysis();
    void SetALamPiMAnalysis();

  void CreateXiKchPAnalysis();
    void SetXiKchPAnalysis();
  void CreateAXiKchPAnalysis();
    void SetAXiKchPAnalysis();
  void CreateXiKchMAnalysis();
    void SetXiKchMAnalysis();
  void CreateAXiKchMAnalysis();
    void SetAXiKchMAnalysis();

  void SetCorrectAnalysis();

  AliFemtoBasicEventCut* CreateBasicEventCut();
  AliFemtoEventCutEstimators* CreateEventCutEstimators(const float &aCentLow, const float &aCentHigh);

  myAliFemtoV0TrackCut* CreateLambdaCut();
  AliFemtoV0TrackCutNSigmaFilter* CreateLambdaCutNSigmaFilter(bool aUseCustom);  //-----17/12/2015
  myAliFemtoV0TrackCut* CreateAntiLambdaCut();
  AliFemtoV0TrackCutNSigmaFilter* CreateAntiLambdaCutNSigmaFilter(bool aUseCustom); //-----17/12/2015
  myAliFemtoV0TrackCut* CreateK0ShortCut();
  AliFemtoV0TrackCutNSigmaFilter* CreateK0ShortCutNSigmaFilter(bool aUseCustom); //-----17/12/2015
  myAliFemtoESDTrackCut* CreateKchCut(const int aCharge);
  myAliFemtoESDTrackCut* CreatePiCut(const int aCharge);



  AliFemtoXiTrackCut* CreateXiCut();
  AliFemtoXiTrackCut* CreateAntiXiCut();


  AliFemtoV0PairCut* CreateV0PairCut(double aMinAvgSepPosPos, double aMinAvgSepPosNeg, double aMinAvgSepNegPos, double aMinAvgSepNegNeg);
  AliFemtoV0TrackPairCut* CreateV0TrackPairCut(double aMinAvgSepTrackPos, double aMinAvgSepTrackNeg);
  AliFemtoXiTrackPairCut* CreateXiTrackPairCut();

  myAliFemtoKStarCorrFctn* CreateKStarCorrFctn(const char* name, unsigned int bins, double min, double max);
  myAliFemtoAvgSepCorrFctn* CreateAvgSepCorrFctn(const char* name, unsigned int bins, double min, double max);
  myAliFemtoSepCorrFctns* CreateSepCorrFctns(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY);
  myAliFemtoAvgSepCorrFctnCowboysAndSailors *CreateAvgSepCorrFctnCowboysAndSailors(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY);
  myAliFemtoKStarCorrFctn2D* CreateKStarCorrFctn2D(const char* name, unsigned int nbinsKStar, double KStarLo, double KStarHi, unsigned int nbinsY, double YLo, double YHi);
  myAliFemtoKStarCorrFctnMC* CreateKStarCorrFctnMC(const char* name, unsigned int bins, double min, double max);

  myAliFemtoModelCorrFctnKStar* CreateModelCorrFctnKStar(const char* name, unsigned int bins, double min, double max);  //-----04/02/2016


  void SetAnalysis(AliFemtoEventCut* aEventCut, AliFemtoParticleCut* aPartCut1, AliFemtoParticleCut* aPartCut2, AliFemtoPairCut* aPairCut, AliFemtoCorrFctnCollection* aCollectionOfCfs);

  TH1F *GetMultHist();

  void SetImplementAvgSepCuts(bool aImplementAvgSepCuts);

  //-----17/12/2015
  void SetUseAliFemtoV0TrackCutNSigmaFilter(bool aUse);
  void SetUseCustomNSigmaFilters(bool aUseCustom);

  //-----25/02/2016
  void SetRemoveMisidentifiedMCParticles(bool aRemove);


protected:
  static const char* const fAnalysisTags[];

  AnalysisType fAnalysisType;
  const char* fOutputName;		      /* name given to output directory for specific analysis*/
  TH1F* fMultHist;			      //histogram of event multiplicities to ensure event cuts are properly implemented
  bool fImplementAvgSepCuts;		      //Self-explanatory, set to kTRUE when I want Avg Sep cuts implemented
  bool fWritePairKinematics;
  bool fIsMCRun;
  bool fIsMBAnalysis;

  double fMinCent, fMaxCent;

  AliFemtoCorrFctnCollection* fCollectionOfCfs;

  //----------------------------------------
  //-----Variables common to all analyses
  AliFemtoBasicEventCut *BasicEvCut;
  AliFemtoEventCutEstimators *EvCutEst;

  myAliFemtoV0TrackCut *LamCut;
  myAliFemtoV0TrackCut *ALamCut;

  myAliFemtoKStarCorrFctn *KStarCf;
  myAliFemtoAvgSepCorrFctn *AvgSepCf;
  myAliFemtoSepCorrFctns *SepCfs;
  myAliFemtoAvgSepCorrFctnCowboysAndSailors *AvgSepCfCowboysAndSailors;
  myAliFemtoKStarCorrFctn2D *KStarCf2D;
  myAliFemtoKStarCorrFctnMC *KStarCfMC;
  myAliFemtoModelCorrFctnKStar *KStarModelCfs;  //-----04/02/2016
  //-----Specific for (A)LamK0 analyses
  myAliFemtoV0TrackCut *K0Cut;
  AliFemtoV0PairCut *V0PairCut;

  //-----Specific for (A)LamKchPM analyses
  myAliFemtoESDTrackCut *KchPCut;
  myAliFemtoESDTrackCut *KchMCut;

  //-----Specific for (A)LamPiPM analyses
  myAliFemtoESDTrackCut *PiPCut;
  myAliFemtoESDTrackCut *PiMCut;

  AliFemtoV0TrackPairCut *V0TrackPairCut;

  //-----Specific for (A)XiKchPM analyses
  AliFemtoXiTrackCut *XiCut;
  AliFemtoXiTrackCut *AXiCut;
  AliFemtoXiTrackPairCut *XiTrackPairCut;

  //-----17/12/2015
  AliFemtoV0TrackCutNSigmaFilter *LamCutNSigmaFilter;
  AliFemtoV0TrackCutNSigmaFilter *ALamCutNSigmaFilter;
  AliFemtoV0TrackCutNSigmaFilter *K0CutNSigmaFilter;

  bool fUseAliFemtoV0TrackCutNSigmaFilter;
  bool fUseCustomNSigmaFilters;



#ifdef __ROOT__
  ClassDef(myAnalysisConstructor, 0)
#endif

};

inline void myAnalysisConstructor::SetUseAliFemtoV0TrackCutNSigmaFilter(bool aUse) {fUseAliFemtoV0TrackCutNSigmaFilter = aUse;}
inline void myAnalysisConstructor::SetUseCustomNSigmaFilters(bool aUseCustom) {fUseCustomNSigmaFilters = aUseCustom;}

inline void myAnalysisConstructor::SetRemoveMisidentifiedMCParticles(bool aRemove) {KStarModelCfs->SetRemoveMisidentified(aRemove);}

#endif
