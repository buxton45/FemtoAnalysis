//
//  ConfigFemtoAnalysis.cxx
//
//
//  Returns a pointer to the created AliFemtoManager


#if !defined(__CINT__) || defined(__MAKECINT__)
#include "AliFemtoManager.h"
#include "AliFemtoEventReaderESDChain.h"
#include "AliFemtoEventReaderAODChain.h"
#include "AliFemtoShareQualityTPCEntranceSepPairCut.h"
#include "AliFemtoQinvCorrFctn.h"
#include "AliFemtoShareQualityCorrFctn.h"
#include "AliFemtoTPCInnerCorrFctn.h"

#include <AliFemtoTrack.h>
#include "AliFemtoV0TrackCut.h"
#include <AliFemtoAvgSepCorrFctn.h>

#include "KStarCF.h"


#include "AliFemtoBasicTrackCut.h"
#include "AliFemtoV0PairCut.h"
#include "myAliFemtoKStarCorrFctn.h"
#include "myAliFemtoAvgSepCorrFctn.h"
#include "myAliFemtoV0TrackCut.h"
#include "myAliFemtoSepCorrFctns.h"

#include "myAnalysisConstructor.h"
#endif

AliFemtoManager* ConfigFemtoAnalysis() 
{
  bool ImplementAvgSepCuts = false;

  const double PionMass = 0.13956995,
               KaonMass = 0.493677,
               ProtonMass = 0.938272013,
               LambdaMass = 1.115683;

  gROOT->LoadMacro("myAliFemtoV0TrackCut.cxx+g");
  gROOT->LoadMacro("myAliFemtoKStarCorrFctn.cxx+g");
  gROOT->LoadMacro("myAliFemtoAvgSepCorrFctn.cxx+g");
  gROOT->LoadMacro("/home/Analysis/K0Lam/KStarCF.cxx+g");
  gROOT->LoadMacro("myAnalysisConstructor.cxx+g");
  gROOT->LoadMacro("myAliFemtoSepCorrFctns.cxx+g");

  //Setup the event reader for ALICE AOD
  AliFemtoEventReaderAODChain *rdr = new AliFemtoEventReaderAODChain();
    rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality);  //Sets the type of the event multiplicity estimator
    rdr->SetFilterBit(7);
    //rdr->SetCentralityPreSelection(0, 900);
    rdr->SetReadV0(1);  //Read V0 information from the AOD and put it into V0Collection
    rdr->SetEPVZERO(kTRUE);  //to get event plane angle from VZERO
    rdr->SetCentralityFlattening(kTRUE);

  //Setup the manager
  AliFemtoManager *mgr = new AliFemtoManager();
    //Point to the data source - the reader
    mgr->SetEventReader(rdr);


  //Setup the analyses
  //------------------------------- Lambda-K0Short -----------------------------------------------
  myAnalysisConstructor *analy_LamK0 = new myAnalysisConstructor("LamK0",8, -8.0, 8.0, 20, 0, 1000);
    AliFemtoBasicEventCut *EvCut1 = analy_LamK0->CreateBasicEventCut();
    myAliFemtoV0TrackCut *LamCut1 = analy_LamK0->CreateLambdaCut();
    myAliFemtoV0TrackCut *K0Cut1 = analy_LamK0->CreateK0ShortCut();
    if(ImplementAvgSepCuts){AliFemtoV0PairCut *PairCut1 = analy_LamK0->CreateV0PairCut(6.,0.,0.,6.);}
    else{AliFemtoV0PairCut *PairCut1 = analy_LamK0->CreateV0PairCut(0.,0.,0.,0.);}
    myAliFemtoKStarCorrFctn *KStarCF_LamK0 = analy_LamK0->CreateKStarCorrFctn("KStarCF_LamK0",75,0.,0.4);
    analy_LamK0->SetAnalysis(EvCut1,LamCut1,K0Cut1,PairCut1,KStarCF_LamK0);
    myAliFemtoAvgSepCorrFctn *AvgSepCF_LamK0 = analy_LamK0->CreateAvgSepCorrFctn("AvgSepCF_LamK0",200,0.,20.);
      analy_LamK0->AddCorrFctn(AvgSepCF_LamK0);
    myAliFemtoSepCorrFctns *SepCFs_LamK0 = analy_LamK0->CreateSepCorrFctns("SepCFs_LamK0",10,0.,10.,200,0.,20.);
      analy_LamK0->AddCorrFctn(SepCFs_LamK0);


	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_LamK0_0010 = new myAnalysisConstructor("LamK0_0010",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamK0_0010 = analy_LamK0_0010->CreateEventCutEstimators(0.,10.);
	  myAliFemtoV0TrackCut *LamCut_LamK0_0010 = analy_LamK0_0010->CreateLambdaCut();
	  myAliFemtoV0TrackCut *K0Cut_LamK0_0010 = analy_LamK0_0010->CreateK0ShortCut();
          if(ImplementAvgSepCuts){AliFemtoV0PairCut *PairCut_LamK0_0010 = analy_LamK0_0010->CreateV0PairCut(6.,0.,0.,6.);}
          else{AliFemtoV0PairCut *PairCut_LamK0_0010 = analy_LamK0_0010->CreateV0PairCut(0.,0.,0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamK0_0010 = analy_LamK0_0010->CreateKStarCorrFctn("KStarCF_LamK0_0010",75,0.,0.4);
	  analy_LamK0_0010->SetAnalysis(EvCutEst_LamK0_0010,LamCut_LamK0_0010,K0Cut_LamK0_0010,PairCut_LamK0_0010,KStarCF_LamK0_0010);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamK0_0010 = analy_LamK0_0010->CreateAvgSepCorrFctn("AvgSepCF_LamK0_0010",200,0.,20.);
            analy_LamK0_0010->AddCorrFctn(AvgSepCF_LamK0_0010);
          myAliFemtoSepCorrFctns *SepCFs_LamK0_0010 = analy_LamK0_0010->CreateSepCorrFctns("SepCFs_LamK0_0010",10,0.,10.,200,0.,20.);
            analy_LamK0_0010->AddCorrFctn(SepCFs_LamK0_0010);


	myAnalysisConstructor *analy_LamK0_1030 = new myAnalysisConstructor("LamK0_1030",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamK0_1030 = analy_LamK0_1030->CreateEventCutEstimators(10.,30.);
	  myAliFemtoV0TrackCut *LamCut_LamK0_1030 = analy_LamK0_1030->CreateLambdaCut();
	  myAliFemtoV0TrackCut *K0Cut_LamK0_1030 = analy_LamK0_1030->CreateK0ShortCut();
          if(ImplementAvgSepCuts){AliFemtoV0PairCut *PairCut_LamK0_1030 = analy_LamK0_1030->CreateV0PairCut(6.,0.,0.,6.);}
          else{AliFemtoV0PairCut *PairCut_LamK0_1030 = analy_LamK0_1030->CreateV0PairCut(0.,0.,0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamK0_1030 = analy_LamK0_1030->CreateKStarCorrFctn("KStarCF_LamK0_1030",75,0.,0.4);
	  analy_LamK0_1030->SetAnalysis(EvCutEst_LamK0_1030,LamCut_LamK0_1030,K0Cut_LamK0_1030,PairCut_LamK0_1030,KStarCF_LamK0_1030);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamK0_1030 = analy_LamK0_1030->CreateAvgSepCorrFctn("AvgSepCF_LamK0_1030",200,0.,20.);
            analy_LamK0_1030->AddCorrFctn(AvgSepCF_LamK0_1030);
          myAliFemtoSepCorrFctns *SepCFs_LamK0_1030 = analy_LamK0_1030->CreateSepCorrFctns("SepCFs_LamK0_1030",10,0.,10.,200,0.,20.);
            analy_LamK0_1030->AddCorrFctn(SepCFs_LamK0_1030);


	myAnalysisConstructor *analy_LamK0_3050 = new myAnalysisConstructor("LamK0_3050",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamK0_3050 = analy_LamK0_3050->CreateEventCutEstimators(30.,50.);
	  myAliFemtoV0TrackCut *LamCut_LamK0_3050 = analy_LamK0_3050->CreateLambdaCut();
	  myAliFemtoV0TrackCut *K0Cut_LamK0_3050 = analy_LamK0_3050->CreateK0ShortCut();
          if(ImplementAvgSepCuts){AliFemtoV0PairCut *PairCut_LamK0_3050 = analy_LamK0_3050->CreateV0PairCut(6.,0.,0.,6.);}
          else{AliFemtoV0PairCut *PairCut_LamK0_3050 = analy_LamK0_3050->CreateV0PairCut(0.,0.,0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamK0_3050 = analy_LamK0_3050->CreateKStarCorrFctn("KStarCF_LamK0_3050",75,0.,0.4);
	  analy_LamK0_3050->SetAnalysis(EvCutEst_LamK0_3050,LamCut_LamK0_3050,K0Cut_LamK0_3050,PairCut_LamK0_3050,KStarCF_LamK0_3050);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamK0_3050 = analy_LamK0_3050->CreateAvgSepCorrFctn("AvgSepCF_LamK0_3050",200,0.,20.);
            analy_LamK0_3050->AddCorrFctn(AvgSepCF_LamK0_3050);
          myAliFemtoSepCorrFctns *SepCFs_LamK0_3050 = analy_LamK0_3050->CreateSepCorrFctns("SepCFs_LamK0_3050",10,0.,10.,200,0.,20.);
            analy_LamK0_3050->AddCorrFctn(SepCFs_LamK0_3050);



  //------------------------------- AntiLambda-K0Short -----------------------------------------------
  myAnalysisConstructor *analy_ALamK0 = new myAnalysisConstructor("ALamK0",8, -8.0, 8.0, 20, 0, 1000);
    AliFemtoBasicEventCut *EvCut2 = analy_ALamK0->CreateBasicEventCut();
    myAliFemtoV0TrackCut *ALamCut2 = analy_ALamK0->CreateAntiLambdaCut();
    myAliFemtoV0TrackCut *K0Cut2 = analy_ALamK0->CreateK0ShortCut();
    if(ImplementAvgSepCuts){AliFemtoV0PairCut *PairCut2 = analy_ALamK0->CreateV0PairCut(6.,0.,0.,6.);}
    else{AliFemtoV0PairCut *PairCut2 = analy_ALamK0->CreateV0PairCut(0.,0.,0.,0.);}
    myAliFemtoKStarCorrFctn *KStarCF2 = analy_ALamK0->CreateKStarCorrFctn("ALamK0KStarCF2",75,0.,0.4);
    analy_ALamK0->SetAnalysis(EvCut2,ALamCut2,K0Cut2,PairCut2,KStarCF2);
    myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamK0 = analy_ALamK0->CreateAvgSepCorrFctn("AvgSepCF_ALamK0",200,0.,20.);
      analy_ALamK0->AddCorrFctn(AvgSepCF_ALamK0);
    myAliFemtoSepCorrFctns *SepCFs_ALamK0 = analy_ALamK0->CreateSepCorrFctns("SepCFs_ALamK0",10,0.,10.,200,0.,20.);
      analy_ALamK0->AddCorrFctn(SepCFs_ALamK0);

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_ALamK0_0010 = new myAnalysisConstructor("ALamK0_0010",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamK0_0010 = analy_ALamK0_0010->CreateEventCutEstimators(0.,10.);
	  myAliFemtoV0TrackCut *ALamCut_ALamK0_0010 = analy_ALamK0_0010->CreateAntiLambdaCut();
	  myAliFemtoV0TrackCut *K0Cut_ALamK0_0010 = analy_ALamK0_0010->CreateK0ShortCut();
          if(ImplementAvgSepCuts){AliFemtoV0PairCut *PairCut_ALamK0_0010 = analy_ALamK0_0010->CreateV0PairCut(6.,0.,0.,6.);}
          else{AliFemtoV0PairCut *PairCut_ALamK0_0010 = analy_ALamK0_0010->CreateV0PairCut(0.,0.,0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamK0_0010 = analy_ALamK0_0010->CreateKStarCorrFctn("KStarCF_ALamK0_0010",75,0.,0.4);
	  analy_ALamK0_0010->SetAnalysis(EvCutEst_ALamK0_0010,ALamCut_ALamK0_0010,K0Cut_ALamK0_0010,PairCut_ALamK0_0010,KStarCF_ALamK0_0010);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamK0_0010 = analy_ALamK0_0010->CreateAvgSepCorrFctn("AvgSepCF_ALamK0_0010",200,0.,20.);
            analy_ALamK0_0010->AddCorrFctn(AvgSepCF_ALamK0_0010);
          myAliFemtoSepCorrFctns *SepCFs_ALamK0_0010 = analy_ALamK0_0010->CreateSepCorrFctns("SepCFs_ALamK0_0010",10,0.,10.,200,0.,20.);
            analy_ALamK0_0010->AddCorrFctn(SepCFs_ALamK0_0010);

	myAnalysisConstructor *analy_ALamK0_1030 = new myAnalysisConstructor("ALamK0_1030",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamK0_1030 = analy_ALamK0_1030->CreateEventCutEstimators(10.,30.);
	  myAliFemtoV0TrackCut *ALamCut_ALamK0_1030 = analy_ALamK0_1030->CreateAntiLambdaCut();
	  myAliFemtoV0TrackCut *K0Cut_ALamK0_1030 = analy_ALamK0_1030->CreateK0ShortCut();
          if(ImplementAvgSepCuts){AliFemtoV0PairCut *PairCut_ALamK0_1030 = analy_ALamK0_1030->CreateV0PairCut(6.,0.,0.,6.);}
          else{AliFemtoV0PairCut *PairCut_ALamK0_1030 = analy_ALamK0_1030->CreateV0PairCut(0.,0.,0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamK0_1030 = analy_ALamK0_1030->CreateKStarCorrFctn("KStarCF_ALamK0_1030",75,0.,0.4);
	  analy_ALamK0_1030->SetAnalysis(EvCutEst_ALamK0_1030,ALamCut_ALamK0_1030,K0Cut_ALamK0_1030,PairCut_ALamK0_1030,KStarCF_ALamK0_1030);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamK0_1030 = analy_ALamK0_1030->CreateAvgSepCorrFctn("AvgSepCF_ALamK0_1030",200,0.,20.);
            analy_ALamK0_1030->AddCorrFctn(AvgSepCF_ALamK0_1030);
          myAliFemtoSepCorrFctns *SepCFs_ALamK0_1030 = analy_ALamK0_1030->CreateSepCorrFctns("SepCFs_ALamK0_1030",10,0.,10.,200,0.,20.);
            analy_ALamK0_1030->AddCorrFctn(SepCFs_ALamK0_1030);

	myAnalysisConstructor *analy_ALamK0_3050 = new myAnalysisConstructor("ALamK0_3050",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamK0_3050 = analy_ALamK0_3050->CreateEventCutEstimators(30.,50.);
	  myAliFemtoV0TrackCut *ALamCut_ALamK0_3050 = analy_ALamK0_3050->CreateAntiLambdaCut();
	  myAliFemtoV0TrackCut *K0Cut_ALamK0_3050 = analy_ALamK0_3050->CreateK0ShortCut();
          if(ImplementAvgSepCuts){AliFemtoV0PairCut *PairCut_ALamK0_3050 = analy_ALamK0_3050->CreateV0PairCut(6.,0.,0.,6.);}
          else{AliFemtoV0PairCut *PairCut_ALamK0_3050 = analy_ALamK0_3050->CreateV0PairCut(0.,0.,0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamK0_3050 = analy_ALamK0_3050->CreateKStarCorrFctn("KStarCF_ALamK0_3050",75,0.,0.4);
	  analy_ALamK0_3050->SetAnalysis(EvCutEst_ALamK0_3050,ALamCut_ALamK0_3050,K0Cut_ALamK0_3050,PairCut_ALamK0_3050,KStarCF_ALamK0_3050);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamK0_3050 = analy_ALamK0_3050->CreateAvgSepCorrFctn("AvgSepCF_ALamK0_3050",200,0.,20.);
            analy_ALamK0_3050->AddCorrFctn(AvgSepCF_ALamK0_3050);
          myAliFemtoSepCorrFctns *SepCFs_ALamK0_3050 = analy_ALamK0_3050->CreateSepCorrFctns("SepCFs_ALamK0_3050",10,0.,10.,200,0.,20.);
            analy_ALamK0_3050->AddCorrFctn(SepCFs_ALamK0_3050);


  //------------------------------- Lambda-K+ -----------------------------------------------
  myAnalysisConstructor *analy_LamKchP = new myAnalysisConstructor("LamKchP",8, -8.0, 8.0, 20, 0, 1000);
    AliFemtoBasicEventCut *EvCut_LamKchP = analy_LamKchP->CreateBasicEventCut();
    myAliFemtoV0TrackCut *LamCut_LamKchP = analy_LamKchP->CreateLambdaCut();
    AliFemtoESDTrackCut *KchPCut_LamKchP = analy_LamKchP->CreateKchCut(1);
    if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_LamKchP = analy_LamKchP->CreateV0TrackPairCut(8.,0.);}
    else{AliFemtoV0TrackPairCut *PairCut_LamKchP = analy_LamKchP->CreateV0TrackPairCut(0.,0.);}
    myAliFemtoKStarCorrFctn *KStarCF_LamKchP = analy_LamKchP->CreateKStarCorrFctn("LamKchPKStarCF",75,0.,0.4);
    analy_LamKchP->SetAnalysis(EvCut_LamKchP,LamCut_LamKchP,KchPCut_LamKchP,PairCut_LamKchP,KStarCF_LamKchP);
    myAliFemtoAvgSepCorrFctn *AvgSepCF_LamKchP = analy_LamKchP->CreateAvgSepCorrFctn("AvgSepCF_LamKchP",200,0.,20.);
      analy_LamKchP->AddCorrFctn(AvgSepCF_LamKchP);
    myAliFemtoSepCorrFctns *SepCFs_LamKchP = analy_LamKchP->CreateSepCorrFctns("SepCFs_LamKchP",10,0.,10.,200,0.,20.);
      analy_LamKchP->AddCorrFctn(SepCFs_LamKchP);

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_LamKchP_0010 = new myAnalysisConstructor("LamKchP_0010",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamKchP_0010 = analy_LamKchP_0010->CreateEventCutEstimators(0.,10.);
	  myAliFemtoV0TrackCut *LamCut_LamKchP_0010 = analy_LamKchP_0010->CreateLambdaCut();
	  AliFemtoESDTrackCut *KchPCut_LamKchP_0010 = analy_LamKchP_0010->CreateKchCut(1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_LamKchP_0010 = analy_LamKchP_0010->CreateV0TrackPairCut(8.,0.);}
          else{AliFemtoV0TrackPairCut *PairCut_LamKchP_0010 = analy_LamKchP_0010->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamKchP_0010 = analy_LamKchP_0010->CreateKStarCorrFctn("KStarCF_LamKchP_0010",75,0.,0.4);
	  analy_LamKchP_0010->SetAnalysis(EvCutEst_LamKchP_0010,LamCut_LamKchP_0010,KchPCut_LamKchP_0010,PairCut_LamKchP_0010,KStarCF_LamKchP_0010);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamKchP_0010 = analy_LamKchP_0010->CreateAvgSepCorrFctn("AvgSepCF_LamKchP_0010",200,0.,20.);
            analy_LamKchP_0010->AddCorrFctn(AvgSepCF_LamKchP_0010);
          myAliFemtoSepCorrFctns *SepCFs_LamKchP_0010 = analy_LamKchP_0010->CreateSepCorrFctns("SepCFs_LamKchP_0010",10,0.,10.,200,0.,20.);
            analy_LamKchP_0010->AddCorrFctn(SepCFs_LamKchP_0010);

	myAnalysisConstructor *analy_LamKchP_1030 = new myAnalysisConstructor("LamKchP_1030",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamKchP_1030 = analy_LamKchP_1030->CreateEventCutEstimators(10.,30.);
	  myAliFemtoV0TrackCut *LamCut_LamKchP_1030 = analy_LamKchP_1030->CreateLambdaCut();
	  AliFemtoESDTrackCut *KchPCut_LamKchP_1030 = analy_LamKchP_1030->CreateKchCut(1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_LamKchP_1030 = analy_LamKchP_1030->CreateV0TrackPairCut(8.,0.);}
          else{AliFemtoV0TrackPairCut *PairCut_LamKchP_1030 = analy_LamKchP_1030->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamKchP_1030 = analy_LamKchP_1030->CreateKStarCorrFctn("KStarCF_LamKchP_1030",75,0.,0.4);
	  analy_LamKchP_1030->SetAnalysis(EvCutEst_LamKchP_1030,LamCut_LamKchP_1030,KchPCut_LamKchP_1030,PairCut_LamKchP_1030,KStarCF_LamKchP_1030);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamKchP_1030 = analy_LamKchP_1030->CreateAvgSepCorrFctn("AvgSepCF_LamKchP_1030",200,0.,20.);
            analy_LamKchP_1030->AddCorrFctn(AvgSepCF_LamKchP_1030);
          myAliFemtoSepCorrFctns *SepCFs_LamKchP_1030 = analy_LamKchP_1030->CreateSepCorrFctns("SepCFs_LamKchP_1030",10,0.,10.,200,0.,20.);
            analy_LamKchP_1030->AddCorrFctn(SepCFs_LamKchP_1030);

	myAnalysisConstructor *analy_LamKchP_3050 = new myAnalysisConstructor("LamKchP_3050",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamKchP_3050 = analy_LamKchP_3050->CreateEventCutEstimators(30.,50.);
	  myAliFemtoV0TrackCut *LamCut_LamKchP_3050 = analy_LamKchP_3050->CreateLambdaCut();
	  AliFemtoESDTrackCut *KchPCut_LamKchP_3050 = analy_LamKchP_3050->CreateKchCut(1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_LamKchP_3050 = analy_LamKchP_3050->CreateV0TrackPairCut(8.,0.);}
          else{AliFemtoV0TrackPairCut *PairCut_LamKchP_3050 = analy_LamKchP_3050->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamKchP_3050 = analy_LamKchP_3050->CreateKStarCorrFctn("KStarCF_LamKchP_3050",75,0.,0.4);
	  analy_LamKchP_3050->SetAnalysis(EvCutEst_LamKchP_3050,LamCut_LamKchP_3050,KchPCut_LamKchP_3050,PairCut_LamKchP_3050,KStarCF_LamKchP_3050);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamKchP_3050 = analy_LamKchP_3050->CreateAvgSepCorrFctn("AvgSepCF_LamKchP_3050",200,0.,20.);
            analy_LamKchP_3050->AddCorrFctn(AvgSepCF_LamKchP_3050);
          myAliFemtoSepCorrFctns *SepCFs_LamKchP_3050 = analy_LamKchP_3050->CreateSepCorrFctns("SepCFs_LamKchP_3050",10,0.,10.,200,0.,20.);
            analy_LamKchP_3050->AddCorrFctn(SepCFs_LamKchP_3050);

  //------------------------------- Lambda-K- -----------------------------------------------
  myAnalysisConstructor *analy_LamKchM = new myAnalysisConstructor("LamKchM",8, -8.0, 8.0, 20, 0, 1000);
    AliFemtoBasicEventCut *EvCut_LamKchM = analy_LamKchM->CreateBasicEventCut();
    myAliFemtoV0TrackCut *LamCut_LamKchM = analy_LamKchM->CreateLambdaCut();
    AliFemtoESDTrackCut *KchMCut_LamKchM = analy_LamKchM->CreateKchCut(-1);
    if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_LamKchM = analy_LamKchM->CreateV0TrackPairCut(0.,8.);}
    else{AliFemtoV0TrackPairCut *PairCut_LamKchM = analy_LamKchM->CreateV0TrackPairCut(0.,0.);}
    myAliFemtoKStarCorrFctn *KStarCF_LamKchM = analy_LamKchM->CreateKStarCorrFctn("LamKchMKStarCF",75,0.,0.4);
    analy_LamKchM->SetAnalysis(EvCut_LamKchM,LamCut_LamKchM,KchMCut_LamKchM,PairCut_LamKchM,KStarCF_LamKchM);
    myAliFemtoAvgSepCorrFctn *AvgSepCF_LamKchM = analy_LamKchM->CreateAvgSepCorrFctn("AvgSepCF_LamKchM",200,0.,20.);
      analy_LamKchM->AddCorrFctn(AvgSepCF_LamKchM);
    myAliFemtoSepCorrFctns *SepCFs_LamKchM = analy_LamKchM->CreateSepCorrFctns("SepCFs_LamKchM",10,0.,10.,200,0.,20.);
      analy_LamKchM->AddCorrFctn(SepCFs_LamKchM);

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_LamKchM_0010 = new myAnalysisConstructor("LamKchM_0010",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamKchM_0010 = analy_LamKchM_0010->CreateEventCutEstimators(0.,10.);
	  myAliFemtoV0TrackCut *LamCut_LamKchM_0010 = analy_LamKchM_0010->CreateLambdaCut();
	  AliFemtoESDTrackCut *KchMCut_LamKchM_0010 = analy_LamKchM_0010->CreateKchCut(-1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_LamKchM_0010 = analy_LamKchM_0010->CreateV0TrackPairCut(0.,8.);}
          else{AliFemtoV0TrackPairCut *PairCut_LamKchM_0010 = analy_LamKchM_0010->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamKchM_0010 = analy_LamKchM_0010->CreateKStarCorrFctn("KStarCF_LamKchM_0010",75,0.,0.4);
	  analy_LamKchM_0010->SetAnalysis(EvCutEst_LamKchM_0010,LamCut_LamKchM_0010,KchMCut_LamKchM_0010,PairCut_LamKchM_0010,KStarCF_LamKchM_0010);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamKchM_0010 = analy_LamKchM_0010->CreateAvgSepCorrFctn("AvgSepCF_LamKchM_0010",200,0.,20.);
            analy_LamKchM_0010->AddCorrFctn(AvgSepCF_LamKchM_0010);
          myAliFemtoSepCorrFctns *SepCFs_LamKchM_0010 = analy_LamKchM_0010->CreateSepCorrFctns("SepCFs_LamKchM_0010",10,0.,10.,200,0.,20.);
            analy_LamKchM_0010->AddCorrFctn(SepCFs_LamKchM_0010);

	myAnalysisConstructor *analy_LamKchM_1030 = new myAnalysisConstructor("LamKchM_1030",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamKchM_1030 = analy_LamKchM_1030->CreateEventCutEstimators(10.,30.);
	  myAliFemtoV0TrackCut *LamCut_LamKchM_1030 = analy_LamKchM_1030->CreateLambdaCut();
	  AliFemtoESDTrackCut *KchMCut_LamKchM_1030 = analy_LamKchM_1030->CreateKchCut(-1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_LamKchM_1030 = analy_LamKchM_1030->CreateV0TrackPairCut(0.,8.);}
          else{AliFemtoV0TrackPairCut *PairCut_LamKchM_1030 = analy_LamKchM_1030->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamKchM_1030 = analy_LamKchM_1030->CreateKStarCorrFctn("KStarCF_LamKchM_1030",75,0.,0.4);
	  analy_LamKchM_1030->SetAnalysis(EvCutEst_LamKchM_1030,LamCut_LamKchM_1030,KchMCut_LamKchM_1030,PairCut_LamKchM_1030,KStarCF_LamKchM_1030);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamKchM_1030 = analy_LamKchM_1030->CreateAvgSepCorrFctn("AvgSepCF_LamKchM_1030",200,0.,20.);
            analy_LamKchM_1030->AddCorrFctn(AvgSepCF_LamKchM_1030);
          myAliFemtoSepCorrFctns *SepCFs_LamKchM_1030 = analy_LamKchM_1030->CreateSepCorrFctns("SepCFs_LamKchM_1030",10,0.,10.,200,0.,20.);
            analy_LamKchM_1030->AddCorrFctn(SepCFs_LamKchM_1030);

	myAnalysisConstructor *analy_LamKchM_3050 = new myAnalysisConstructor("LamKchM_3050",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_LamKchM_3050 = analy_LamKchM_3050->CreateEventCutEstimators(30.,50.);
	  myAliFemtoV0TrackCut *LamCut_LamKchM_3050 = analy_LamKchM_3050->CreateLambdaCut();
	  AliFemtoESDTrackCut *KchMCut_LamKchM_3050 = analy_LamKchM_3050->CreateKchCut(-1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_LamKchM_3050 = analy_LamKchM_3050->CreateV0TrackPairCut(0.,8.);}
          else{AliFemtoV0TrackPairCut *PairCut_LamKchM_3050 = analy_LamKchM_3050->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_LamKchM_3050 = analy_LamKchM_3050->CreateKStarCorrFctn("KStarCF_LamKchM_3050",75,0.,0.4);
	  analy_LamKchM_3050->SetAnalysis(EvCutEst_LamKchM_3050,LamCut_LamKchM_3050,KchMCut_LamKchM_3050,PairCut_LamKchM_3050,KStarCF_LamKchM_3050);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_LamKchM_3050 = analy_LamKchM_3050->CreateAvgSepCorrFctn("AvgSepCF_LamKchM_3050",200,0.,20.);
            analy_LamKchM_3050->AddCorrFctn(AvgSepCF_LamKchM_3050);
          myAliFemtoSepCorrFctns *SepCFs_LamKchM_3050 = analy_LamKchM_3050->CreateSepCorrFctns("SepCFs_LamKchM_3050",10,0.,10.,200,0.,20.);
            analy_LamKchM_3050->AddCorrFctn(SepCFs_LamKchM_3050);

  //------------------------------- AntiLambda-K+ -----------------------------------------------
  myAnalysisConstructor *analy_ALamKchP = new myAnalysisConstructor("ALamKchP",8, -8.0, 8.0, 20, 0, 1000);
    AliFemtoBasicEventCut *EvCut_ALamKchP = analy_ALamKchP->CreateBasicEventCut();
    myAliFemtoV0TrackCut *ALamCut_ALamKchP = analy_ALamKchP->CreateAntiLambdaCut();
    AliFemtoESDTrackCut *KchPCut_ALamKchP = analy_ALamKchP->CreateKchCut(1);
    if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_ALamKchP = analy_ALamKchP->CreateV0TrackPairCut(8.,0.);}
    else{AliFemtoV0TrackPairCut *PairCut_ALamKchP = analy_ALamKchP->CreateV0TrackPairCut(0.,0.);}
    myAliFemtoKStarCorrFctn *KStarCF_ALamKchP = analy_ALamKchP->CreateKStarCorrFctn("ALamKchPKStarCF",75,0.,0.4);
    analy_ALamKchP->SetAnalysis(EvCut_ALamKchP,ALamCut_ALamKchP,KchPCut_ALamKchP,PairCut_ALamKchP,KStarCF_ALamKchP);
    myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamKchP = analy_ALamKchP->CreateAvgSepCorrFctn("AvgSepCF_ALamKchP",200,0.,20.);
      analy_ALamKchP->AddCorrFctn(AvgSepCF_ALamKchP);
    myAliFemtoSepCorrFctns *SepCFs_ALamKchP = analy_ALamKchP->CreateSepCorrFctns("SepCFs_ALamKchP",10,0.,10.,200,0.,20.);
      analy_ALamKchP->AddCorrFctn(SepCFs_ALamKchP);

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_ALamKchP_0010 = new myAnalysisConstructor("ALamKchP_0010",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamKchP_0010 = analy_ALamKchP_0010->CreateEventCutEstimators(0.,10.);
	  myAliFemtoV0TrackCut *ALamCut_ALamKchP_0010 = analy_ALamKchP_0010->CreateAntiLambdaCut();
	  AliFemtoESDTrackCut *KchPCut_ALamKchP_0010 = analy_ALamKchP_0010->CreateKchCut(1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_ALamKchP_0010 = analy_ALamKchP_0010->CreateV0TrackPairCut(8.,0.);}
          else{AliFemtoV0TrackPairCut *PairCut_ALamKchP_0010 = analy_ALamKchP_0010->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamKchP_0010 = analy_ALamKchP_0010->CreateKStarCorrFctn("KStarCF_ALamKchP_0010",75,0.,0.4);
	  analy_ALamKchP_0010->SetAnalysis(EvCutEst_ALamKchP_0010,ALamCut_ALamKchP_0010,KchPCut_ALamKchP_0010,PairCut_ALamKchP_0010,KStarCF_ALamKchP_0010);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamKchP_0010 = analy_ALamKchP_0010->CreateAvgSepCorrFctn("AvgSepCF_ALamKchP_0010",200,0.,20.);
            analy_ALamKchP_0010->AddCorrFctn(AvgSepCF_ALamKchP_0010);
          myAliFemtoSepCorrFctns *SepCFs_ALamKchP_0010 = analy_ALamKchP_0010->CreateSepCorrFctns("SepCFs_ALamKchP_0010",10,0.,10.,200,0.,20.);
            analy_ALamKchP_0010->AddCorrFctn(SepCFs_ALamKchP_0010);

	myAnalysisConstructor *analy_ALamKchP_1030 = new myAnalysisConstructor("ALamKchP_1030",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamKchP_1030 = analy_ALamKchP_1030->CreateEventCutEstimators(10.,30.);
	  myAliFemtoV0TrackCut *ALamCut_ALamKchP_1030 = analy_ALamKchP_1030->CreateAntiLambdaCut();
	  AliFemtoESDTrackCut *KchPCut_ALamKchP_1030 = analy_ALamKchP_1030->CreateKchCut(1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_ALamKchP_1030 = analy_ALamKchP_1030->CreateV0TrackPairCut(8.,0.);}
          else{AliFemtoV0TrackPairCut *PairCut_ALamKchP_1030 = analy_ALamKchP_1030->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamKchP_1030 = analy_ALamKchP_1030->CreateKStarCorrFctn("KStarCF_ALamKchP_1030",75,0.,0.4);
	  analy_ALamKchP_1030->SetAnalysis(EvCutEst_ALamKchP_1030,ALamCut_ALamKchP_1030,KchPCut_ALamKchP_1030,PairCut_ALamKchP_1030,KStarCF_ALamKchP_1030);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamKchP_1030 = analy_ALamKchP_1030->CreateAvgSepCorrFctn("AvgSepCF_ALamKchP_1030",200,0.,20.);
            analy_ALamKchP_1030->AddCorrFctn(AvgSepCF_ALamKchP_1030);
          myAliFemtoSepCorrFctns *SepCFs_ALamKchP_1030 = analy_ALamKchP_1030->CreateSepCorrFctns("SepCFs_ALamKchP_1030",10,0.,10.,200,0.,20.);
            analy_ALamKchP_1030->AddCorrFctn(SepCFs_ALamKchP_1030);

	myAnalysisConstructor *analy_ALamKchP_3050 = new myAnalysisConstructor("ALamKchP_3050",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamKchP_3050 = analy_ALamKchP_3050->CreateEventCutEstimators(30.,50.);
	  myAliFemtoV0TrackCut *ALamCut_ALamKchP_3050 = analy_ALamKchP_3050->CreateAntiLambdaCut();
	  AliFemtoESDTrackCut *KchPCut_ALamKchP_3050 = analy_ALamKchP_3050->CreateKchCut(1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_ALamKchP_3050 = analy_ALamKchP_3050->CreateV0TrackPairCut(8.,0.);}
          else{AliFemtoV0TrackPairCut *PairCut_ALamKchP_3050 = analy_ALamKchP_3050->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamKchP_3050 = analy_ALamKchP_3050->CreateKStarCorrFctn("KStarCF_ALamKchP_3050",75,0.,0.4);
	  analy_ALamKchP_3050->SetAnalysis(EvCutEst_ALamKchP_3050,ALamCut_ALamKchP_3050,KchPCut_ALamKchP_3050,PairCut_ALamKchP_3050,KStarCF_ALamKchP_3050);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamKchP_3050 = analy_ALamKchP_3050->CreateAvgSepCorrFctn("AvgSepCF_ALamKchP_3050",200,0.,20.);
            analy_ALamKchP_3050->AddCorrFctn(AvgSepCF_ALamKchP_3050);
          myAliFemtoSepCorrFctns *SepCFs_ALamKchP_3050 = analy_ALamKchP_3050->CreateSepCorrFctns("SepCFs_ALamKchP_3050",10,0.,10.,200,0.,20.);
            analy_ALamKchP_3050->AddCorrFctn(SepCFs_ALamKchP_3050);

  //------------------------------- AntiLambda-K- -----------------------------------------------
  myAnalysisConstructor *analy_ALamKchM = new myAnalysisConstructor("ALamKchM",8, -8.0, 8.0, 20, 0, 1000);
    AliFemtoBasicEventCut *EvCut_ALamKchM = analy_ALamKchM->CreateBasicEventCut();
    myAliFemtoV0TrackCut *ALamCut_ALamKchM = analy_ALamKchM->CreateAntiLambdaCut();
    AliFemtoESDTrackCut *KchMCut_ALamKchM = analy_ALamKchM->CreateKchCut(-1);
    if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_ALamKchM = analy_ALamKchM->CreateV0TrackPairCut(0.,8.);}
    else{AliFemtoV0TrackPairCut *PairCut_ALamKchM = analy_ALamKchM->CreateV0TrackPairCut(0.,0.);}
    myAliFemtoKStarCorrFctn *KStarCF_ALamKchM = analy_ALamKchM->CreateKStarCorrFctn("ALamKchMKStarCF",75,0.,0.4);
    analy_ALamKchM->SetAnalysis(EvCut_ALamKchM,ALamCut_ALamKchM,KchMCut_ALamKchM,PairCut_ALamKchM,KStarCF_ALamKchM);
    myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamKchM = analy_ALamKchM->CreateAvgSepCorrFctn("AvgSepCF_ALamKchM",200,0.,20.);
      analy_ALamKchM->AddCorrFctn(AvgSepCF_ALamKchM);
    myAliFemtoSepCorrFctns *SepCFs_ALamKchM = analy_ALamKchM->CreateSepCorrFctns("SepCFs_ALamKchM",10,0.,10.,200,0.,20.);
      analy_ALamKchM->AddCorrFctn(SepCFs_ALamKchM);

	//_______Centrality dependent______________________________________________________
	myAnalysisConstructor *analy_ALamKchM_0010 = new myAnalysisConstructor("ALamKchM_0010",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamKchM_0010 = analy_ALamKchM_0010->CreateEventCutEstimators(0.,10.);
	  myAliFemtoV0TrackCut *ALamCut_ALamKchM_0010 = analy_ALamKchM_0010->CreateAntiLambdaCut();
	  AliFemtoESDTrackCut *KchMCut_ALamKchM_0010 = analy_ALamKchM_0010->CreateKchCut(-1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_ALamKchM_0010 = analy_ALamKchM_0010->CreateV0TrackPairCut(0.,8.);}
          else{AliFemtoV0TrackPairCut *PairCut_ALamKchM_0010 = analy_ALamKchM_0010->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamKchM_0010 = analy_ALamKchM_0010->CreateKStarCorrFctn("KStarCF_ALamKchM_0010",75,0.,0.4);
	  analy_ALamKchM_0010->SetAnalysis(EvCutEst_ALamKchM_0010,ALamCut_ALamKchM_0010,KchMCut_ALamKchM_0010,PairCut_ALamKchM_0010,KStarCF_ALamKchM_0010);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamKchM_0010 = analy_ALamKchM_0010->CreateAvgSepCorrFctn("AvgSepCF_ALamKchM_0010",200,0.,20.);
            analy_ALamKchM_0010->AddCorrFctn(AvgSepCF_ALamKchM_0010);
          myAliFemtoSepCorrFctns *SepCFs_ALamKchM_0010 = analy_ALamKchM_0010->CreateSepCorrFctns("SepCFs_ALamKchM_0010",10,0.,10.,200,0.,20.);
            analy_ALamKchM_0010->AddCorrFctn(SepCFs_ALamKchM_0010);

	myAnalysisConstructor *analy_ALamKchM_1030 = new myAnalysisConstructor("ALamKchM_1030",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamKchM_1030 = analy_ALamKchM_1030->CreateEventCutEstimators(10.,30.);
	  myAliFemtoV0TrackCut *ALamCut_ALamKchM_1030 = analy_ALamKchM_1030->CreateAntiLambdaCut();
	  AliFemtoESDTrackCut *KchMCut_ALamKchM_1030 = analy_ALamKchM_1030->CreateKchCut(-1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_ALamKchM_1030 = analy_ALamKchM_1030->CreateV0TrackPairCut(0.,8.);}
          else{AliFemtoV0TrackPairCut *PairCut_ALamKchM_1030 = analy_ALamKchM_1030->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamKchM_1030 = analy_ALamKchM_1030->CreateKStarCorrFctn("KStarCF_ALamKchM_1030",75,0.,0.4);
	  analy_ALamKchM_1030->SetAnalysis(EvCutEst_ALamKchM_1030,ALamCut_ALamKchM_1030,KchMCut_ALamKchM_1030,PairCut_ALamKchM_1030,KStarCF_ALamKchM_1030);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamKchM_1030 = analy_ALamKchM_1030->CreateAvgSepCorrFctn("AvgSepCF_ALamKchM_1030",200,0.,20.);
            analy_ALamKchM_1030->AddCorrFctn(AvgSepCF_ALamKchM_1030);
          myAliFemtoSepCorrFctns *SepCFs_ALamKchM_1030 = analy_ALamKchM_1030->CreateSepCorrFctns("SepCFs_ALamKchM_1030",10,0.,10.,200,0.,20.);
            analy_ALamKchM_1030->AddCorrFctn(SepCFs_ALamKchM_1030);

	myAnalysisConstructor *analy_ALamKchM_3050 = new myAnalysisConstructor("ALamKchM_3050",8, -8.0, 8.0, 20, 0, 1000);
	  AliFemtoEventCutEstimators *EvCutEst_ALamKchM_3050 = analy_ALamKchM_3050->CreateEventCutEstimators(30.,50.);
	  myAliFemtoV0TrackCut *ALamCut_ALamKchM_3050 = analy_ALamKchM_3050->CreateAntiLambdaCut();
	  AliFemtoESDTrackCut *KchMCut_ALamKchM_3050 = analy_ALamKchM_3050->CreateKchCut(-1);
          if(ImplementAvgSepCuts){AliFemtoV0TrackPairCut *PairCut_ALamKchM_3050 = analy_ALamKchM_3050->CreateV0TrackPairCut(0.,8.);}
          else{AliFemtoV0TrackPairCut *PairCut_ALamKchM_3050 = analy_ALamKchM_3050->CreateV0TrackPairCut(0.,0.);}
	  myAliFemtoKStarCorrFctn *KStarCF_ALamKchM_3050 = analy_ALamKchM_3050->CreateKStarCorrFctn("KStarCF_ALamKchM_3050",75,0.,0.4);
	  analy_ALamKchM_3050->SetAnalysis(EvCutEst_ALamKchM_3050,ALamCut_ALamKchM_3050,KchMCut_ALamKchM_3050,PairCut_ALamKchM_3050,KStarCF_ALamKchM_3050);
          myAliFemtoAvgSepCorrFctn *AvgSepCF_ALamKchM_3050 = analy_ALamKchM_3050->CreateAvgSepCorrFctn("AvgSepCF_ALamKchM_3050",200,0.,20.);
            analy_ALamKchM_3050->AddCorrFctn(AvgSepCF_ALamKchM_3050);
          myAliFemtoSepCorrFctns *SepCFs_ALamKchM_1030 = analy_ALamKchM_1030->CreateSepCorrFctns("SepCFs_ALamKchM_1030",10,0.,10.,200,0.,20.);
            analy_ALamKchM_1030->AddCorrFctn(SepCFs_ALamKchM_1030);

  // Add the analyses to the manager
  mgr->AddAnalysis(analy_LamK0);
    mgr->AddAnalysis(analy_LamK0_0010);
    mgr->AddAnalysis(analy_LamK0_1030);
    mgr->AddAnalysis(analy_LamK0_3050);
  mgr->AddAnalysis(analy_ALamK0);
    mgr->AddAnalysis(analy_ALamK0_0010);
    mgr->AddAnalysis(analy_ALamK0_1030);
    mgr->AddAnalysis(analy_ALamK0_3050);


  return mgr;
}


