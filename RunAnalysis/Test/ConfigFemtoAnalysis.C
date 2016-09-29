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

bool DEFAULT_DO_KT = kFALSE;

struct MacroParams {
  std::vector<int> centrality_ranges;
  std::vector<AliFemtoAnalysisLambdaKaon::AnalysisType> pair_codes;
  float qinv_bin_size_MeV;
  float qinv_max_GeV;
  bool do_qinv_cf;
  bool do_q3d_cf;
  bool do_deltaeta_deltaphi_cf;
  bool do_avg_sep_cf;
  bool do_kt_q3d;
  bool do_kt_qinv;
  bool do_ylm_cf; // not implemented yet
  int filter_bit;
  AliFemtoEventReaderAOD::EventMult multiplicity;
  bool dca_global_track;
};

void SetPairCodes(AliFemtoAnalysisLambdaKaon::AnalysisType aAnType, MacroParams &aMacroConfig);

AliFemtoAnalysisLambdaKaon* 
CreateCorrectAnalysis(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::AnalysisType aAnType,
  AliFemtoAnalysisLambdaKaon::AnalysisParams &aAnParams,
  AliFemtoAnalysisLambdaKaon::EventCutParams &aEvCutParams,
  AliFemtoAnalysisLambdaKaon::PairCutParams &aPairCutParams
);

void
BuildConfiguration(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::AnalysisParams &aAnParams,
  AliFemtoAnalysisLambdaKaon::EventCutParams &aEvCutParams,
  AliFemtoAnalysisLambdaKaon::PairCutParams &aPairCutParams,
  MacroParams &aMac
);

void
BuildParticleConfigurations(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::V0CutParams &aV0CutParams1,
  AliFemtoAnalysisLambdaKaon::V0CutParams &aV0CutParams2
);

void
BuildParticleConfigurations(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::V0CutParams &aV0CutParams1,
  AliFemtoAnalysisLambdaKaon::ESDCutParams &aESDCutParams2
);

void
BuildParticleConfigurations(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::XiCutParams &aXiCutParams1,
  AliFemtoAnalysisLambdaKaon::ESDCutParams &aESDCutParams2
);

void
BuildParticleConfiguration(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::V0CutParams &aV0CutParams
);

void
BuildParticleConfiguration(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::ESDCutParams &aESDCutParams
);


void
BuildParticleConfiguration(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::XiCutParams &aXiCutParams
);



AliFemtoManager* ConfigFemtoAnalysis(const TString& aParamString="") 
{
  std::cout << "[ConfigFemtoAnalysis (LambdaKaon)]\n";

  // Get the default configurations
  AFALK::AnalysisParams tAnalysisConfig = AFALK::DefaultAnalysisParams();
  AFALK::EventCutParams tEventCutConfig = AFALK::DefaultEventCutParams();
  AFALK::PairCutParams tPairCutConfig = AFALK::DefaultPairParams();

  // default Macro config
  MacroParams tMacroConfig;
  tMacroConfig.do_qinv_cf = true;
  tMacroConfig.do_q3d_cf = true;
  tMacroConfig.do_deltaeta_deltaphi_cf = false;
  tMacroConfig.do_avg_sep_cf = false;
  tMacroConfig.do_kt_q3d = tMacroConfig.do_kt_qinv = DEFAULT_DO_KT;
  tMacroConfig.do_ylm_cf = false;
  tMacroConfig.qinv_bin_size_MeV = 5.0f;
  tMacroConfig.qinv_max_GeV = 1.0f;
  tMacroConfig.filter_bit = 7;
  tMacroConfig.multiplicity = AliFemtoEventReaderAOD::kCentrality;
  tMacroConfig.dca_global_track = true;

  // Read parameter string and update configurations
  BuildConfiguration(aParamString,tAnalysisConfig,tEventCutConfig,tPairCutConfig,tMacroConfig);

  // Begin to build the manager and analyses
  AliFemtoManager *tManager = new AliFemtoManager();

  //Setup the event reader for ALICE AOD
  AliFemtoEventReaderAODChain *rdr = new AliFemtoEventReaderAODChain();
    rdr->SetUseMultiplicity(tMacroConfig.multiplicity);  //Sets the type of the event multiplicity estimator
    rdr->SetFilterBit(tMacroConfig.filter_bit);
    //rdr->SetCentralityPreSelection(0, 900);
    rdr->SetReadV0(1);  //Read V0 information from the AOD and put it into V0Collection
    rdr->SetEPVZERO(kTRUE);  //to get event plane angle from VZERO
    rdr->SetCentralityFlattening(kFALSE);
    rdr->SetPrimaryVertexCorrectionTPCPoints(tAnalysisConfig.implementVertexCorrection);
    rdr->SetReadMC(tAnalysisConfig.isMCRun);
  tManager->SetEventReader(rdr);


  if(tMacroConfig.centrality_ranges.empty())
  {
    tMacroConfig.centrality_ranges.push_back(0.);
    tMacroConfig.centrality_ranges.push_back(10.);
  }

  // Identify all sister analyses that go along with tAnalysisConfig.analysisType
  SetPairCodes(tAnalysisConfig.analysisType, tMacroConfig);


  // loop over centrality ranges
  for(unsigned int iCent = 0; iCent+1 < tMacroConfig.centrality_ranges.size(); iCent += 2)
  {
    const int tMultLow  = tMacroConfig.centrality_ranges[iCent],
              tMultHigh = tMacroConfig.centrality_ranges[iCent+1];

    //loop over pair types
    for(unsigned int iPair = 0; iPair < tMacroConfig.pair_codes.size(); iPair++)
    {
      
      //Build unique analysis for each pair type in each centrality bin
      AliFemtoAnalysisLambdaKaon *tAnalysis = CreateCorrectAnalysis(aParamString,tMacroConfig.pair_codes[iPair],tAnalysisConfig,tEventCutConfig,tPairCutConfig);
      //TODO get pair cut to change for LamKchP and LamKchM
    }

  }




  return tManager;
}



void SetPairCodes(AliFemtoAnalysisLambdaKaon::AnalysisType aAnType, MacroParams &aMacroConfig)
{
  aMacroConfig.pair_codes.clear();

  switch(aAnType) {
  case AliFemtoAnalysisLambdaKaon::kLamK0:
  case AliFemtoAnalysisLambdaKaon::kALamK0:
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kLamK0);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kALamK0);
    break;

  case AliFemtoAnalysisLambdaKaon::kLamKchP:
  case AliFemtoAnalysisLambdaKaon::kALamKchP:
  case AliFemtoAnalysisLambdaKaon::kLamKchM:
  case AliFemtoAnalysisLambdaKaon::kALamKchM:
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kLamKchP);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kALamKchP);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kLamKchM);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kALamKchM);
    break;

  case AliFemtoAnalysisLambdaKaon::kLamLam:
  case AliFemtoAnalysisLambdaKaon::kALamALam:
  case AliFemtoAnalysisLambdaKaon::kLamALam:
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kLamLam);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kALamALam);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kLamALam);
    break;

  case AliFemtoAnalysisLambdaKaon::kLamPiP:
  case AliFemtoAnalysisLambdaKaon::kALamPiP:
  case AliFemtoAnalysisLambdaKaon::kLamPiM:
  case AliFemtoAnalysisLambdaKaon::kALamPiM:
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kLamPiP);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kALamPiP);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kLamPiM);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kALamPiM);
    break;

  case AliFemtoAnalysisLambdaKaon::kXiKchP:
  case AliFemtoAnalysisLambdaKaon::kAXiKchP:
  case AliFemtoAnalysisLambdaKaon::kXiKchM:
  case AliFemtoAnalysisLambdaKaon::kAXiKchM:
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kXiKchP);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kAXiKchP);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kXiKchM);
    aMacroConfig.pair_codes.push_back(AliFemtoAnalysisLambdaKaon::kAXiKchM);
    break;


  default:
    continue;
  }
}

AliFemtoAnalysisLambdaKaon* 
CreateCorrectAnalysis(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::AnalysisType aAnType,
  AliFemtoAnalysisLambdaKaon::AnalysisParams &aAnParams,
  AliFemtoAnalysisLambdaKaon::EventCutParams &aEvCutParams,
  AliFemtoAnalysisLambdaKaon::PairCutParams &aPairCutParams
)
{
  AliFemtoAnalysisLambdaKaon *tAnalysis;

  aAnParams.analysisType = aAnType;

  AFALK::V0CutParams tV0CutConfig1,
                     tV0CutConfig2;
  AFALK::ESDCutParams tESDCutConfig;
  AFALK::XiCutParams tXiCutConfig;

  switch(aAnParams.generalAnalysisType) {

  case AliFemtoAnalysisLambdaKaon::kV0V0:
    switch(aAnType) {
    case AliFemtoAnalysisLambdaKaon::kLamK0:
      tV0CutConfig1 = AFALK::DefaultLambdaCutParams();
      tV0CutConfig2 = AFALK::DefaultK0ShortCutParams();
      break;

    case AliFemtoAnalysisLambdaKaon::kALamK0:
      tV0CutConfig1 = AFALK::DefaultAntiLambdaCutParams();
      tV0CutConfig2 = AFALK::DefaultK0ShortCutParams();
      break;

    case AliFemtoAnalysisLambdaKaon::kLamLam:
      tV0CutConfig1 = AFALK::DefaultLambdaCutParams();
      tV0CutConfig2 = AFALK::DefaultLambdaCutParams();
      break;

    case AliFemtoAnalysisLambdaKaon::kALamALam:
      tV0CutConfig1 = AFALK::DefaultAntiLambdaCutParams();
      tV0CutConfig2 = AFALK::DefaultAntiLambdaCutParams();
      break;

    case AliFemtoAnalysisLambdaKaon::kLamALam:
      tV0CutConfig1 = AFALK::DefaultLambdaCutParams();
      tV0CutConfig2 = AFALK::DefaultAntiLambdaCutParams();
      break;
    }
    BuildParticleConfiguration(aText,tV0CutConfig1);
    BuildParticleConfiguration(aText,tV0CutConfig2);
    tAnalysis = new AliFemtoAnalysisLambdaKaon(aAnParams,aEvCutParams,aPairCutParams,tV0CutConfig1,tV0CutConfig2);
    break;


  case AliFemtoAnalysisLambdaKaon::kV0Track:
    switch(aAnType) {
    case AliFemtoAnalysisLambdaKaon::kLamKchP:
      tV0CutConfig1 = AFALK::DefaultLambdaCutParams();
      tESDCutConfig = AFALK::DefaultKchCutParams(1);
      break;

    case AliFemtoAnalysisLambdaKaon::kALamKchP:
      tV0CutConfig1 = AFALK::DefaultAntiLambdaCutParams();
      tESDCutConfig = AFALK::DefaultKchCutParams(1);
      break;

    case AliFemtoAnalysisLambdaKaon::kLamKchM:
      tV0CutConfig1 = AFALK::DefaultLambdaCutParams();
      tESDCutConfig = AFALK::DefaultKchCutParams(-1);
      break;

    case AliFemtoAnalysisLambdaKaon::kALamKchM:
      tV0CutConfig1 = AFALK::DefaultAntiLambdaCutParams();
      tESDCutConfig = AFALK::DefaultKchCutParams(-1);
      break;

    case AliFemtoAnalysisLambdaKaon::kLamPiP:
      tV0CutConfig1 = AFALK::DefaultLambdaCutParams();
      tESDCutConfig = AFALK::DefaultPiCutParams(1);
      break;

    case AliFemtoAnalysisLambdaKaon::kALamPiP:
      tV0CutConfig1 = AFALK::DefaultAntiLambdaCutParams();
      tESDCutConfig = AFALK::DefaultPiCutParams(1);
      break;

    case AliFemtoAnalysisLambdaKaon::kLamPiM:
      tV0CutConfig1 = AFALK::DefaultLambdaCutParams();
      tESDCutConfig = AFALK::DefaultPiCutParams(-1);
      break;

    case AliFemtoAnalysisLambdaKaon::kALamPiM:
      tV0CutConfig1 = AFALK::DefaultAntiLambdaCutParams();
      tESDCutConfig = AFALK::DefaultPiCutParams(-1);
      break;
    }
    BuildParticleConfiguration(aText,tV0CutConfig1);
    BuildParticleConfiguration(aText,tESDCutConfig);
    tAnalysis = new AliFemtoAnalysisLambdaKaon(aAnParams,aEvCutParams,aPairCutParams,tV0CutConfig1,tESDCutConfig);
    break;


  case AliFemtoAnalysisLambdaKaon::kXiTrack:
    switch(aAnType) {
    case AliFemtoAnalysisLambdaKaon::kXiKchP:
      tXiCutConfig = AFALK::DefaultXiCutParams();
      tESDCutConfig = AFALK::DefaultKchCutParams(1);
      break;

    case AliFemtoAnalysisLambdaKaon::kAXiKchP:
      tXiCutConfig = AFALK::DefaultAXiCutParams();
      tESDCutConfig = AFALK::DefaultKchCutParams(1);
      break;

    case AliFemtoAnalysisLambdaKaon::kXiKchM:
      tXiCutConfig = AFALK::DefaultXiCutParams();
      tESDCutConfig = AFALK::DefaultKchCutParams(-1);
      break;

    case AliFemtoAnalysisLambdaKaon::kAXiKchM:
      tXiCutConfig = AFALK::DefaultAXiCutParams();
      tESDCutConfig = AFALK::DefaultKchCutParams(-1);
      break;
    }
    BuildParticleConfiguration(aText,tXiCutConfig);
    BuildParticleConfiguration(aText,tESDCutConfig);
    tAnalysis = new AliFemtoAnalysisLambdaKaon(aAnParams,aEvCutParams,aPairCutParams,tXiCutConfig,tESDCutConfig);
    break;
  }

  return tAnalysis;
}


void
BuildConfiguration(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::AnalysisParams &aAnParams,
  AliFemtoAnalysisLambdaKaon::EventCutParams &aEvCutParams,
  AliFemtoAnalysisLambdaKaon::PairCutParams &aPairCutParams,
  MacroParams &aMac)
{
  std::cout << "I-BuildConfiguration:" << TBase64::Encode(text) << " \n";

  const TString tAnalysisVarName = "aAnParams",//
                tEvCutVarName    = "aEvCutParams",
                tPairCutVarName  = "aPairCutParams",
                tMacroVarName    = "aMac";//

  TObjArray* tLines = aText.Tokenize("\n;");
  TIter tNextLine(tLines);
  TObject *tLineObj = NULL;

  while(tLineObj = tNextLine())
  {
    const TString tLine = ((TObjString*)tLineObj)->String().Strip(TString::kBoth, ' ');
    TString tCmd("");

    switch(tLine[0]) {
    case '@':  //Analysis Params
      tCmd = tAnalysisVarName + "." + tLine(1, tLine.Length() - 1);
      break;

    case '&':  //Event Cut Params
      tCmd = tEvCutVarName + "." + tLine(1, tLine.Length() - 1);
      break;

    case '%':  //Pair Cut Params
      tCmd = tPairCutVarName + "." + tLine(1, tLine.Length() - 1);
      break;

    case '~':  //Macro Params
      tCmd = tMacroVarName + "." + tLine(1, tLine.Length() - 1);
      break;

    case '[':  //Centrality in Macro Params
    {
      unsigned int tRangeEnd = tLine.Index("]");
      if(tRangeEnd == -1) tRangeEnd = tLine.Length();

      TString tCentralityRanges = tLine(1, tRangeEnd - 1);
      TObjArray *tRangeGroups = tCentralityRanges.Tokenize(",");
      TIter tNextRangeGroup(tRangeGroups);
      TObjString *tRangeGroup = NULL;

      while(tRangeGroup = (TObjString*)tNextRangeGroup())
      {
        TObjArray *tSubRange = tRangeGroup->String().Tokenize(":");
        TIter tNextSubRange(tSubRange);
        TObjString *tSubRange_it = (TObjString*)tNextSubRange();
        TString tPrev = TString::Format("%0.2d", tSubRange_it->String().Atoi());
        while(tSubRange_it = (TObjString*)tNextSubRange())
        {
          TString tNext = TString::Format("%0.2d", tSubRange_it->String().Atoi());

          tCmd = tMacroVarName + ".centrality_ranges.push_back(" + tPrev + ");";
          gROOT->ProcessLineFast(tCmd);

          tCmd = tMacroVarName + ".centrality_ranges.push_back(" + tNext + ");";
          gROOT->ProcessLineFast(tCmd);
          tPrev = tNext;
        }
      }
    }
      continue;

    default:
      continue;
    }

    tCmd += ";";
    cout << "I-BuildConfiguration: `" << tCmd << "`\n";
    gROOT->ProcessLineFast(tCmd);
  }

}




void
BuildParticleConfigurations(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::V0CutParams &aV0CutParams1,
  AliFemtoAnalysisLambdaKaon::V0CutParams &aV0CutParams2
)
{
  std::cout << "I-BuildParticleConfigurations:" << TBase64::Encode(text) << " \n";

  const TString tV0CutVarName1   = "aV0CutParams1",
                tV0CutVarName2   = "aV0CutParams2";

  TObjArray* tLines = aText.Tokenize("\n;");
  TIter tNextLine(tLines);
  TObject *tLineObj = NULL;

  while(tLineObj = tNextLine())
  {
    const TString tLine = ((TObjString*)tLineObj)->String().Strip(TString::kBoth, ' ');
    TString tCmd("");

    switch(tLine[0]) {
    case '$':  //Particle Cut Params
      switch(tLine[1]) {
      case '1':  //Particle 1
        tCmd = tV0CutVarName1 + "." + tLine(2, tLine.Length() - 1);
        break;

      case '2':  //Particle 2
        tCmd = tV0CutVarName2 + "." + tLine(2, tLine.Length() - 1);
        break;
      }
      break;

    default:
      continue;
    }
    tCmd += ";";
    cout << "I-BuildConfiguration: `" << tCmd << "`\n";
    gROOT->ProcessLineFast(tCmd);
  }
}

void
BuildParticleConfigurations(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::V0CutParams &aV0CutParams1,
  AliFemtoAnalysisLambdaKaon::ESDCutParams &aESDCutParams2
)
{
  std::cout << "I-BuildParticleConfigurations:" << TBase64::Encode(text) << " \n";

  const TString tV0CutVarName1    = "aV0CutParams1",
                tESDCutVarName2   = "aESDCutParams2";

  TObjArray* tLines = aText.Tokenize("\n;");
  TIter tNextLine(tLines);
  TObject *tLineObj = NULL;

  while(tLineObj = tNextLine())
  {
    const TString tLine = ((TObjString*)tLineObj)->String().Strip(TString::kBoth, ' ');
    TString tCmd("");

    switch(tLine[0]) {
    case '$':  //Particle Cut Params
      switch(tLine[1]) {
      case '1':  //Particle 1
        tCmd = tV0CutVarName1 + "." + tLine(2, tLine.Length() - 1);
        break;

      case '2':  //Particle 2
        tCmd = tESDCutVarName2 + "." + tLine(2, tLine.Length() - 1);
        break;
      }
      break;

    default:
      continue;
    }
    tCmd += ";";
    cout << "I-BuildConfiguration: `" << tCmd << "`\n";
    gROOT->ProcessLineFast(tCmd);
  }
}

void
BuildParticleConfigurations(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::XiCutParams &aXiCutParams1,
  AliFemtoAnalysisLambdaKaon::ESDCutParams &aESDCutParams2
)
{
  std::cout << "I-BuildParticleConfigurations:" << TBase64::Encode(text) << " \n";

  const TString tXiCutVarName1    = "aXiCutParams1",
                tESDCutVarName2   = "aESDCutParams2";

  TObjArray* tLines = aText.Tokenize("\n;");
  TIter tNextLine(tLines);
  TObject *tLineObj = NULL;

  while(tLineObj = tNextLine())
  {
    const TString tLine = ((TObjString*)tLineObj)->String().Strip(TString::kBoth, ' ');
    TString tCmd("");

    switch(tLine[0]) {
    case '$':  //Particle Cut Params
      switch(tLine[1]) {
      case '1':  //Particle 1
        tCmd = tXiCutVarName1 + "." + tLine(2, tLine.Length() - 1);
        break;

      case '2':  //Particle 2
        tCmd = tESDCutVarName2 + "." + tLine(2, tLine.Length() - 1);
        break;
      }
      break;

    default:
      continue;
    }
    tCmd += ";";
    cout << "I-BuildConfiguration: `" << tCmd << "`\n";
    gROOT->ProcessLineFast(tCmd);
  }
}




void
BuildParticleConfiguration(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::V0CutParams &aV0CutParams
)
{
  std::cout << "I-BuildParticleConfiguration:" << TBase64::Encode(text) << " \n";

  const TString tV0CutVarName = "aV0CutParams";

  TString tDesiredName;

  switch(aV0CutParams.particlePDGType) {
  case AliFemtoAnalysisLambdaKaon::kPDGLam:
    tDesiredName = TString("Lam");
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGALam:
    tDesiredName = TString("ALam");
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGK0:
    tDesiredName = TString("K0s");
    break;

  default:
    tDesiredName = TString("");
  }

  TObjArray* tLines = aText.Tokenize("\n;");
  TIter tNextLine(tLines);
  TObject *tLineObj = NULL;

  while(tLineObj = tNextLine())
  {
    const TString tLine = ((TObjString*)tLineObj)->String().Strip(TString::kBoth, ' ');
    TString tCmd("");

    if(tLine[0] == '$')
    {
      TObjArray* tCutFullLine = tLine.Tokenize("|");
      const TString tParticleType = ((TObjString*)tCutFullLine->At(1))->String().Strip(TString::kBoth, ' ');
      const TString tParticleCut = ((TObjString*)tCutFullLine->At(2))->String().Strip(TString::kBoth, ' ');

      if(tParticleType.EqualTo(tDesiredName)) tCmd = tV0CutVarName + "." + tParticleCut(0, tParticleCut.Length() - 1);
    }
    tCmd += ";";
    cout << "I-BuildConfiguration: `" << tCmd << "`\n";
    gROOT->ProcessLineFast(tCmd);
  }
}


void
BuildParticleConfiguration(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::ESDCutParams &aESDCutParams
)
{
  std::cout << "I-BuildParticleConfiguration:" << TBase64::Encode(text) << " \n";

  const TString tESDCutVarName = "aESDCutParams";

  TString tDesiredName;

  switch(aESDCutParams.particlePDGType) {
  case AliFemtoAnalysisLambdaKaon::kPDGKchP:
    tDesiredName = TString("KchP");
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGKchM:
    tDesiredName = TString("KchP");
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGPiP:
    tDesiredName = TString("PiP");
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGPiM:
    tDesiredName = TString("PiM");
    break;

  default:
    tDesiredName = TString("");
  }

  TObjArray* tLines = aText.Tokenize("\n;");
  TIter tNextLine(tLines);
  TObject *tLineObj = NULL;

  while(tLineObj = tNextLine())
  {
    const TString tLine = ((TObjString*)tLineObj)->String().Strip(TString::kBoth, ' ');
    TString tCmd("");

    if(tLine[0] == '$')
    {
      TObjArray* tCutFullLine = tLine.Tokenize("|");
      const TString tParticleType = ((TObjString*)tCutFullLine->At(1))->String().Strip(TString::kBoth, ' ');
      const TString tParticleCut = ((TObjString*)tCutFullLine->At(2))->String().Strip(TString::kBoth, ' ');

      if(tParticleType.EqualTo(tDesiredName)) tCmd = tESDCutVarName + "." + tParticleCut(0, tParticleCut.Length() - 1);
    }
    tCmd += ";";
    cout << "I-BuildConfiguration: `" << tCmd << "`\n";
    gROOT->ProcessLineFast(tCmd);
  }
}


void
BuildParticleConfiguration(
  const TString &aText,
  AliFemtoAnalysisLambdaKaon::XiCutParams &aXiCutParams
)
{
  std::cout << "I-BuildParticleConfiguration:" << TBase64::Encode(text) << " \n";

  const TString tXiCutVarName = "aXiCutParams";

  TString tDesiredName;

  switch(aXiCutParams.particlePDGType) {
  case AliFemtoAnalysisLambdaKaon::kPDGXiC:
    tDesiredName = TString("Xi");
    break;

  case AliFemtoAnalysisLambdaKaon::kPDGAXiC:
    tDesiredName = TString("AXi");
    break;

  default:
    tDesiredName = TString("");
  }

  TObjArray* tLines = aText.Tokenize("\n;");
  TIter tNextLine(tLines);
  TObject *tLineObj = NULL;

  while(tLineObj = tNextLine())
  {
    const TString tLine = ((TObjString*)tLineObj)->String().Strip(TString::kBoth, ' ');
    TString tCmd("");

    if(tLine[0] == '$')
    {
      TObjArray* tCutFullLine = tLine.Tokenize("|");
      const TString tParticleType = ((TObjString*)tCutFullLine->At(1))->String().Strip(TString::kBoth, ' ');
      const TString tParticleCut = ((TObjString*)tCutFullLine->At(2))->String().Strip(TString::kBoth, ' ');

      if(tParticleType.EqualTo(tDesiredName)) tCmd = tXiCutVarName + "." + tParticleCut(0, tParticleCut.Length() - 1);
    }
    tCmd += ";";
    cout << "I-BuildConfiguration: `" << tCmd << "`\n";
    gROOT->ProcessLineFast(tCmd);
  }
}

