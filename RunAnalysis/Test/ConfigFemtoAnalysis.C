///
/// \file PWGCF/FEMTOSCOPY/macros/Train/LambdaKaonFemto/ConfigFemtoAnalysis.C
/// \brief The configuration macro which sets up Lambda-Kaon analyses
/// \ author Jesse Buxton, Ohio State University,
///
/// \ based largely off of work done by Andrew Kubera in
/// \ file PWGCF/FEMTOSCOPY/macros/Train/PionPionFemto/ConfigFemtoAnalysis.C
/// \   Andrew Kubera, Ohio State University, andrew.kubera@cern.ch
///

#if !defined(__CINT__) || defined(__MAKECINT__)

#include "AliFemtoAnalysisLambdaKaon.h"

#include "AliFemtoManager.h"
#include "AliFemtoEventReaderESDChain.h"
#include "AliFemtoEventReaderAODChain.h"

#include <TROOT.h>
#endif

typedef AliFemtoAnalysisLambdaKaon AFALK;


AliFemtoManager* ConfigFemtoAnalysis() 
{
  const double PionMass = 0.13956995,
               KaonMass = 0.493677,
               ProtonMass = 0.938272013,
               LambdaMass = 1.115683;


  //Setup the manager
  AliFemtoManager *mgr = new AliFemtoManager();
  //Setup the event reader for ALICE AOD
  AliFemtoEventReaderAODChain *rdr = new AliFemtoEventReaderAODChain();
    rdr->SetUseMultiplicity(AliFemtoEventReaderAOD::kCentrality);  //Sets the type of the event multiplicity estimator
    rdr->SetFilterBit(7);
    //rdr->SetCentralityPreSelection(0, 900);
    rdr->SetReadV0(1);  //Read V0 information from the AOD and put it into V0Collection
    rdr->SetEPVZERO(kTRUE);  //to get event plane angle from VZERO
    rdr->SetCentralityFlattening(kFALSE);
    rdr->SetPrimaryVertexCorrectionTPCPoints(ImplementVertexCorrection);//TODO
    rdr->SetReadMC(RunMC);//TODO
  mgr->SetEventReader(rdr);

}
