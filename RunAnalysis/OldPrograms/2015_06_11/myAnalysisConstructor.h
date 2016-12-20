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

#include "AliFemtoBasicEventCut.h"
#include "AliFemtoEventCutEstimators.h"
#include "AliFemtoCutMonitorEventMult.h"
#include "AliFemtoCutMonitorV0.h"

#include "myAliFemtoV0TrackCut.h"
#include "AliFemtoBasicTrackCut.h"
#include "AliFemtoESDTrackCut.h"
#include "AliFemtoAODTrackCut.h"
#include "AliFemtoCutMonitorParticleYPt.h"

#include "AliFemtoV0PairCut.h"
#include "AliFemtoV0TrackPairCut.h"

#include "myAliFemtoKStarCorrFctn.h"
#include "myAliFemtoAvgSepCorrFctn.h"
#include "myAliFemtoSepCorrFctns.h"

class myAnalysisConstructor : public AliFemtoVertexMultAnalysis {

public:

  myAnalysisConstructor();
  myAnalysisConstructor(const char* name);
  myAnalysisConstructor(const char* name, unsigned int binsVertex, double minVertex, double maxVertex, unsigned int binsMult, double minMult, double maxMult);
//  myAnalysisConstructor(const myAnalysisConstructor& TheOriginalAnalysis);
//  myAnalysisConstructor& operator=(const myAnalysisConstructor& TheOriginalAnalysis);
  virtual ~myAnalysisConstructor();

  virtual void ProcessEvent(const AliFemtoEvent* ProcessThisEvent);  //will add fMultHist to the process event of AliFemtoVertexMultAnalysis
  virtual TList* GetOutputList();

  AliFemtoBasicEventCut* CreateBasicEventCut();
  AliFemtoEventCutEstimators* CreateEventCutEstimators(const float &aCentLow, const float &aCentHigh);

  myAliFemtoV0TrackCut* CreateLambdaCut();
  myAliFemtoV0TrackCut* CreateAntiLambdaCut();
  myAliFemtoV0TrackCut* CreateK0ShortCut();
  AliFemtoESDTrackCut* CreateKchCut(const int aCharge);
  AliFemtoESDTrackCut* CreatePiCut(const int aCharge);

  AliFemtoV0PairCut* CreateV0PairCut(double aMinAvgSepPosPos, double aMinAvgSepPosNeg, double aMinAvgSepNegPos, double aMinAvgSepNegNeg);
  AliFemtoV0TrackPairCut* CreateV0TrackPairCut(double aMinAvgSepTrackPos, double aMinAvgSepTrackNeg);

  myAliFemtoKStarCorrFctn* CreateKStarCorrFctn(const char* name, unsigned int bins, double min, double max);
  myAliFemtoAvgSepCorrFctn* CreateAvgSepCorrFctn(const char* name, unsigned int bins, double min, double max);
  myAliFemtoSepCorrFctns* CreateSepCorrFctns(const char* name, unsigned int binsX, double minX, double maxX, unsigned int binsY, double minY, double maxY);


  void SetAnalysis(AliFemtoBasicEventCut* aEventCut, myAliFemtoV0TrackCut* aPartCut1, myAliFemtoV0TrackCut* aPartCut2, AliFemtoV0PairCut* aPairCut, myAliFemtoKStarCorrFctn* aCF);
  void SetAnalysis(AliFemtoEventCutEstimators* aEventCut, myAliFemtoV0TrackCut* aPartCut1, myAliFemtoV0TrackCut* aPartCut2, AliFemtoV0PairCut* aPairCut, myAliFemtoKStarCorrFctn* aCF);
  void SetAnalysis(AliFemtoBasicEventCut* aEventCut, myAliFemtoV0TrackCut* aPartCut1, AliFemtoESDTrackCut* aPartCut2, AliFemtoV0TrackPairCut* aPairCut, myAliFemtoKStarCorrFctn* aCF);
  void SetAnalysis(AliFemtoEventCutEstimators* aEventCut, myAliFemtoV0TrackCut* aPartCut1, AliFemtoESDTrackCut* aPartCut2, AliFemtoV0TrackPairCut* aPairCut, myAliFemtoKStarCorrFctn* aCF);
  //---should expand this to include a list of correlation functions

  TH1F *GetMultHist();


protected:
  const char* fOutputName;		      /* name given to output directory for specific analysis*/
  TH1F* fMultHist;			      //histogram of event multiplicities to ensure event cuts are properly implemented


#ifdef __ROOT__
  ClassDef(myAnalysisConstructor, 0)
#endif

};

#endif
