///////////////////////////////////////////////////////////////////////////
// Types:                                                                //
///////////////////////////////////////////////////////////////////////////


#include "Types.h"




//________________________________________________________________________________________________________________
//enum ParticlePDGType
const int cPDGValues[18] = {2212,-2212,211,-211,310,321,-321,3122,-3122,3212,-3212,3312,-3312,3322,-3322,3334,-3334,0};


//enum AnalysisType
const char* const cAnalysisBaseTags[17] = {"LamK0", "ALamK0", "LamKchP", "ALamKchP", "LamKchM", "ALamKchM", "XiKchP", "AXiKchP", "XiKchM", "AXiKchM", "LamLam", "ALamALam", "LamALam", "LamPiP", "ALamPiP", "LamPiM", "ALamPiM"};

const char* const cAnalysisRootTags[17] = {"#LambdaK^{0}", "#bar{#Lambda}K^{0}", "#LambdaK+", "#bar{#Lambda}K+", "#LambdaK-", "#bar{#Lambda}K-", "#XiK+", "#bar{#Xi}K+", "#XiK-", "#bar{#Xi}K-", "#Lambda#Lambda", "#bar{#Lambda}#bar{#Lambda}", "#Lambda#bar{#Lambda}", "#Lambda#pi+", "#bar{#Lambda}#pi+", "#Lambda#pi-", "#bar{#Lambda}#pi-"};

//enum BFieldType
const char* const cBFieldTags[5] = {"_Bp1", "_Bp2", "_Bm1", "_Bm2", "_Bm3"};

//enum CentralityType
const char* const cCentralityTags[4] = {"_0010", "_1030", "_3050", ""};

//enum ParticleType
const char* const cParticleTags[11] = {"Lam", "ALam", "K0", "KchP", "KchM", "Xi", "AXi", "PiP", "PiM", "Proton", "AntiProton"};

const char* const cRootParticleTags[11] = {"#Lambda", "#bar{#Lambda}", "K^{0}", "K+", "K-", "#Xi-", "#bar{#Xi}+", "#pi+", "#pi-", "p", "#bar{p}"};

//enum KStarTrueVsRecType
const char* const cKStarTrueVsRecTypeTags[4] = {"Same","RotSame","Mixed","RotMixed"};

const char* cKStarCfBaseTagNum = "NumKStarCf_";
const char* cKStarCfBaseTagDen = "DenKStarCf_";

const char* cKStarCfMCTrueBaseTagNum = "NumKStarCfTrue_";
const char* cKStarCfMCTrueBaseTagDen = "DenKStarCfTrue_";

const char* cKStar2dCfBaseTagNum = "NumKStarCf2D_";
const char* cKStar2dCfBaseTagDen = "DenKStarCf2D_";

  //
const char* cModelKStarCfNumTrueBaseTag = "NumTrueKStarModelCf_";
const char* cModelKStarCfDenBaseTag = "DenKStarModelCf_";

const char* cModelKStarCfNumTrueIdealBaseTag = "NumTrueIdealKStarModelCf_";
const char* cModelKStarCfDenIdealBaseTag = "DenIdealKStarModelCf_";

const char* cModelKStarCfNumTrueUnitWeightsBaseTag = "NumTrueKStarModelCfUnitWeightsKStarModelCf_";
const char* cModelKStarCfNumTrueIdealUnitWeightsBaseTag = "NumTrueIdealKStarModelCfUnitWeightsKStarModelCf_";

const char* cModelKStarCfNumFakeBaseTag = "NumFakeKStarModelCf_";
const char* cModelKStarCfNumFakeIdealBaseTag = "NumFakeIdealKStarModelCf_";

//const char* cModelKStarTrueVsRecBaseTag = "QgenQrecKStarModelCf_";
const char* cModelKStarTrueVsRecSameBaseTag = "fKTrueKRecSameKStarModelCf_";
const char* cModelKStarTrueVsRecRotSameBaseTag = "fKTrueKRecRotSameKStarModelCf_";
const char* cModelKStarTrueVsRecMixedBaseTag = "fKTrueKRecMixedKStarModelCf_";
const char* cModelKStarTrueVsRecRotMixedBaseTag = "fKTrueKRecRotMixedKStarModelCf_";
  //

const char* cLambdaPurityTag = "LambdaPurity";
const char* cAntiLambdaPurityTag = "AntiLambdaPurity";
const char* cK0ShortPurityTag = "K0ShortPurity1";
const char* cXiPurityTag = "XiPurityAid";
const char* cAXiPurityTag = "AXiPurityAid";

//enum DaughterPairType----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*
const char* const cDaughterPairTags[7] = {"PosPos", "PosNeg", "NegPos", "NegNeg", "TrackPos", "TrackNeg", "TrackTrack"};


const char* const cAvgSepCfBaseTagsNum[7] = {"NumPosPosAvgSepCf_", "NumPosNegAvgSepCf_", "NumNegPosAvgSepCf_", "NumNegNegAvgSepCf_", "NumTrackPosAvgSepCf_", "NumTrackNegAvgSepCf_", "NumTrackTrackAvgSepCf_"};
const char* const cAvgSepCfBaseTagsDen[7] = {"DenPosPosAvgSepCf_", "DenPosNegAvgSepCf_", "DenNegPosAvgSepCf_", "DenNegNegAvgSepCf_", "DenTrackPosAvgSepCf_", "DenTrackNegAvgSepCf_", "DenTrackTrackAvgSepCf_"};



const char* const cSepCfsBaseTagsNum[7] = {"NumPosPosSepCfs_", "NumPosNegSepCfs_", "NumNegPosSepCfs_", "NumNegNegSepCfs_", "NumTrackPosSepCfs_", "NumTrackNegSepCfs_", "NumTrackTrackSepCfs_"};
const char* const cSepCfsBaseTagsDen[7] = {"DenPosPosSepCfs_", "DenPosNegSepCfs_", "DenNegPosSepCfs_", "DenNegNegSepCfs_", "DenTrackPosSepCfs_", "DenTrackNegSepCfs_", "DenTrackTrackSepCfs_"};

const char* const cAvgSepCfCowboysAndSailorsBaseTagsNum[7] = {"NumPosPosAvgSepCfCowboysAndSailors_", "NumPosNegAvgSepCfCowboysAndSailors_", "NumNegPosAvgSepCfCowboysAndSailors_", "NumNegNegAvgSepCfCowboysAndSailors_", "NumTrackPosAvgSepCfCowboysAndSailors_", "NumTrackNegAvgSepCfCowboysAndSailors_", "NumTrackTrackAvgSepCfCowboysAndSailors_"};
const char* const cAvgSepCfCowboysAndSailorsBaseTagsDen[7] = {"DenPosPosAvgSepCfCowboysAndSailors_", "DenPosNegAvgSepCfCowboysAndSailors_", "DenNegPosAvgSepCfCowboysAndSailors_", "DenNegNegAvgSepCfCowboysAndSailors_", "DenTrackPosAvgSepCfCowboysAndSailors_", "DenTrackNegAvgSepCfCowboysAndSailors_", "DenTrackTrackAvgSepCfCowboysAndSailors_"};

//----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*
const char* const cParameterNames[9] = {"Lambda", "Radius", "Ref0", "Imf0", "d0", "Ref02", "Imf02", "d02", "Norm"};

//________________________________________________________________________________________________________________
//const double cLamK0_0010StartValues[6] = {0.15,1.72,-0.14,0.17,8.9,1.};
//const double cLamK0_0010StartValues[6] = {0.42,2.07,-0.16,-0.03,7.4,1.};
const double cLamK0_0010StartValues[6] = {0.15,5.00,-0.14,0.17,8.9,0.2};
const double cLamK0_1030StartValues[6] = {0.15,4.00,-0.14,0.17,8.9,0.2};
const double cLamK0_3050StartValues[6] = {0.15,3.00,-0.14,0.17,8.9,0.2};
const double* cLamK0StartValues[3] = {cLamK0_0010StartValues,cLamK0_1030StartValues,cLamK0_3050StartValues};

//const double cALamK0_0010StartValues[6] = {0.15,1.72,-0.14,0.17,8.9,1.};
//const double cALamK0_0010StartValues[6] = {0.42,2.07,-0.16,-0.03,7.4,1.};
const double cALamK0_0010StartValues[6] = {0.15,5.00,-0.14,0.17,8.9,0.2};
const double cALamK0_1030StartValues[6] = {0.15,4.00,-0.14,0.17,8.9,0.2};
const double cALamK0_3050StartValues[6] = {0.15,3.00,-0.14,0.17,8.9,0.2};
const double* cALamK0StartValues[3] = {cALamK0_0010StartValues,cALamK0_1030StartValues,cALamK0_3050StartValues};

//-----

const double cLamKchP_0010StartValues[6] = {0.19,4.36,-1.28,0.69,1.79,0.2};
const double cLamKchP_1030StartValues[6] = {0.19,4.00,-1.28,0.69,1.79,0.2};
const double cLamKchP_3050StartValues[6] = {0.19,3.00,-1.28,0.69,1.79,0.2};
const double* cLamKchPStartValues[3] = {cLamKchP_0010StartValues,cLamKchP_1030StartValues,cLamKchP_3050StartValues};

const double cALamKchP_0010StartValues[6] = {0.49,3.94,0.09,0.27,7.79,0.2};
const double cALamKchP_1030StartValues[6] = {0.49,4.00,0.09,0.27,7.79,0.2};
const double cALamKchP_3050StartValues[6] = {0.49,3.00,0.09,0.27,7.79,0.2};
const double* cALamKchPStartValues[3] = {cALamKchP_0010StartValues,cALamKchP_1030StartValues,cALamKchP_3050StartValues};

const double cLamKchM_0010StartValues[6] = {0.49,3.94,0.09,0.27,7.79,0.2};
const double cLamKchM_1030StartValues[6] = {0.49,4.00,0.09,0.27,7.79,0.2};
const double cLamKchM_3050StartValues[6] = {0.49,3.00,0.09,0.27,7.79,0.2};
const double* cLamKchMStartValues[3] = {cLamKchM_0010StartValues,cLamKchM_1030StartValues,cLamKchM_3050StartValues};

const double cALamKchM_0010StartValues[6] = {0.19,4.36,-1.28,0.69,1.79,0.2};
const double cALamKchM_1030StartValues[6] = {0.19,4.00,-1.28,0.69,1.79,0.2};
const double cALamKchM_3050StartValues[6] = {0.19,3.00,-1.28,0.69,1.79,0.2};
const double* cALamKchMStartValues[3] = {cALamKchM_0010StartValues,cALamKchM_1030StartValues,cALamKchM_3050StartValues};

//-----

const double cXiKchP_0010StartValues[6] = {0.19,4.36,-1.28,0.69,1.79,0.2};
const double cXiKchP_1030StartValues[6] = {0.19,4.00,-1.28,0.69,1.79,0.2};
const double cXiKchP_3050StartValues[6] = {0.19,3.00,-1.28,0.69,1.79,0.2};
const double* cXiKchPStartValues[3] = {cXiKchP_0010StartValues,cXiKchP_1030StartValues,cXiKchP_3050StartValues};

const double cAXiKchP_0010StartValues[6] = {0.49,3.94,0.09,0.27,7.79,0.2};
const double cAXiKchP_1030StartValues[6] = {0.49,4.00,0.09,0.27,7.79,0.2};
const double cAXiKchP_3050StartValues[6] = {0.49,3.00,0.09,0.27,7.79,0.2};
const double* cAXiKchPStartValues[3] = {cAXiKchP_0010StartValues,cAXiKchP_1030StartValues,cAXiKchP_3050StartValues};

const double cXiKchM_0010StartValues[6] = {0.49,3.94,0.09,0.27,7.79,0.2};
const double cXiKchM_1030StartValues[6] = {0.49,4.00,0.09,0.27,7.79,0.2};
const double cXiKchM_3050StartValues[6] = {0.49,3.00,0.09,0.27,7.79,0.2};
const double* cXiKchMStartValues[3] = {cXiKchM_0010StartValues,cXiKchM_1030StartValues,cXiKchM_3050StartValues};

const double cAXiKchM_0010StartValues[6] = {0.19,4.36,-1.28,0.69,1.79,0.2};
const double cAXiKchM_1030StartValues[6] = {0.19,4.00,-1.28,0.69,1.79,0.2};
const double cAXiKchM_3050StartValues[6] = {0.19,3.00,-1.28,0.69,1.79,0.2};
const double* cAXiKchMStartValues[3] = {cAXiKchM_0010StartValues,cAXiKchM_1030StartValues,cAXiKchM_3050StartValues};

//-----

const double** cStartValues[10] = {cLamK0StartValues, cALamK0StartValues, cLamKchPStartValues, cALamKchPStartValues, cLamKchMStartValues, cALamKchMStartValues, cXiKchPStartValues, cAXiKchPStartValues, cXiKchMStartValues, cAXiKchMStartValues};


//-----
const double hbarc = 0.197327;
const double gBohrRadiusXiK = 75.23349845;

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


