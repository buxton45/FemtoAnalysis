///////////////////////////////////////////////////////////////////////////
// Types:                                                                //
///////////////////////////////////////////////////////////////////////////


#include "Types.h"
#include <cassert>



//________________________________________________________________________________________________________________
//enum ParticlePDGType
const int cPDGValues[26] = {2212,-2212,211,-211,311,321,-321,3122,-3122,3212,-3212,3312,-3312,3322,-3322,3334,-3334, 3224,-3224,3114,-3114,3214,-3214,313,-313,0};
const char* const cPDGRootNames[26] {"p", "#bar{p}", "#pi+", "#pi-", "K^{0}_{S}", "K+", "K-", "#Lambda", "#bar{#Lambda}", "#Sigma^{0}", "#bar{#Sigma}^{0}", "#Xi-", "#bar{#Xi}+", "#Xi^{0}", "#bar{#Xi}^{0}", "#Omega-", "#bar{#Omega}+", "#Sigma*+", "#bar{#Sigma*}-", "#Sigma*-", "#bar{#Sigma*}+", "#Sigma*^{0}", "#bar{#Sigma*}^{0}", "K*^{0}", "#bar{K*}^{0}", "", };

const char* const GetPDGRootName(ParticlePDGType aType)
{
  int tPosition = -1;
  for(int i=0; i<26; i++)
  {
    if(cPDGValues[i] == aType) tPosition = i;
  }
  assert(tPosition > -1);
  return cPDGRootNames[tPosition];
}


//enum AnalysisType
const char* const cAnalysisBaseTags[63] = {
"LamK0", "ALamK0", 
"LamKchP", "ALamKchM", "LamKchM", "ALamKchP", 
"XiKchP", "AXiKchM", "XiKchM", "AXiKchP", 
"XiK0", "AXiK0", 
"LamLam", "ALamALam", "LamALam", 
"LamPiP", "ALamPiM", "LamPiM", "ALamPiP", 
//----- Residual Types -----
"Sig0KchP", "ASig0KchM", "Sig0KchM", "ASig0KchP", 
"Xi0KchP", "AXi0KchM", "Xi0KchM", "AXi0KchP", 
"XiKchP", "AXiKchM", "XiKchM", "AXiKchP", 
"OmegaKchP", "AOmegaKchM", "OmegaKchM", "AOmegaKchP",
"SigStPKchP", "ASigStMKchM", "SigStPKchM", "ASigStMKchP", 
"SigStMKchP", "ASigStPKchM", "SigStMKchM", "ASigStPKchP", 
"SigSt0KchP", "ASigSt0KchM", "SigSt0KchM", "ASigSt0KchP", 
"LamKSt0", "ALamAKSt0", "LamAKSt0", "ALamKSt0", 
"Sig0KSt0", "ASigma0AKSt0", "Sig0AKSt0", "ASig0KSt0", 
"Xi0KSt0", "AXi0AKSt0", "Xi0AKSt0", "AXi0KSt0", 
"XiKSt0", "AXiAKSt0", "XiAKSt0", "AXiKSt0"

};

const char* const cAnalysisRootTags[63] = {
"#LambdaK^{0}_{S}", "#bar{#Lambda}K^{0}_{S}", 
"#LambdaK+", "#bar{#Lambda}K-", "#LambdaK-", "#bar{#Lambda}K+", 
"#Xi-K+", "#bar{#Xi}+K-", "#Xi-K-", "#bar{#Xi}+K+",
"#XiK^{0}_{S}", "#bar{#Xi}K^{0}_{S}", 
"#Lambda#Lambda", "#bar{#Lambda}#bar{#Lambda}", "#Lambda#bar{#Lambda}", 
"#Lambda#pi+", "#bar{#Lambda}#pi-", "#Lambda#pi-", "#bar{#Lambda}#pi+", 
//----- Residual Types -----
"#Sigma^{0}K+", "#bar{#Sigma}^{0}K-", "#Sigma^{0}K-", "#bar{#Sigma}^{0}K+", 
"#Xi^{0}K+", "#bar{#Xi}^{0}K-", "#Xi^{0}K-", "#bar{#Xi}^{0}K+", 
"#Xi-K+", "#bar{#Xi}+K-", "#Xi-K-", "#bar{#Xi}+K+", 
"#Omega-K+", "#bar{#Omega}+K-", "#Omega-K-", "#bar{#Omega}+K+",
"#Sigma*^{+}K+", "#bar{#Sigma*}^{-}K-", "#Sigma*^{+}K-", "#bar{#Sigma*}^{-}K+", 
"#Sigma*^{-}K+", "#bar{#Sigma*}^{+}K-", "#Sigma*^{-}K-", "#bar{#Sigma*}^{+}K+", 
"#Sigma*^{0}K+", "#bar{#Sigma*}^{0}K-", "#Sigma*^{0}K-", "#bar{#Sigma*}^{0}K+", 
"#LambdaK*^{0}", "#bar{#Lambda}#bar{K*}^{0}", "#Lambda#bar{K*}^{0}", "#bar{#Lambda}K*^{0}", 
"#Sigma^{0}K*^{0}", "#bar{#Sigma}^{0}#bar{K*}^{0}", "#Sigma^{0}#bar{K*}^{0}", "#bar{#Sigma}^{0}K*^{0}", 
"#Xi^{0}K*^{0}", "#bar{#Xi}^{0}#bar{K*}^{0}", "#Xi^{0}#bar{K*}^{0}", "#bar{#Xi}^{0}K*^{0}", 
"#Xi-K*^{0}", "#bar{#Xi}+#bar{K*}^{0}", "#Xi-#bar{K*}^{0}", "#bar{#Xi}+K*^{0}"

};

/*
//enum ResidualType
const char* const cResidualBaseTags[44] = {
"Sig0KchP", "ASig0KchM", "Sig0KchM", "ASig0KchP", 
"Xi0KchP", "AXi0KchM", "Xi0KchM", "AXi0KchP", 
"XiKchP", "AXiKchM", "XiKchM", "AXiKchP", 
"OmegaKchP", "AOmegaKchM", "OmegaKchM", "AOmegaKchP",
"SigStPKchP", "ASigStMKchM", "SigStPKchM", "ASigStMKchP", 
"SigStMKchP", "ASigStPKchM", "SigStMKchM", "ASigStPKchP", 
"SigSt0KchP", "ASigSt0KchM", "SigSt0KchM", "ASigSt0KchP", 
"LamKSt0", "ALamAKSt0", "LamAKSt0", "ALamKSt0", 
"Sig0KSt0", "ASigma0AKSt0", "Sig0AKSt0", "ASig0KSt0", 
"Xi0KSt0", "AXi0AKSt0", "Xi0AKSt0", "AXi0KSt0", 
"XiKSt0", "AXiAKSt0", "XiAKSt0", "AXiKSt0"
};


const char* const cResidualRootTags[44] = {
"#Sigma^{0}K+", "#bar{#Sigma}^{0}K-", "#Sigma^{0}K-", "#bar{#Sigma}^{0}K+", 
"#Xi^{0}K+", "#bar{#Xi}^{0}K-", "#Xi^{0}K-", "#bar{#Xi}^{0}K+", 
"#Xi-K+", "#bar{#Xi}+K-", "#Xi-K-", "#bar{#Xi}+K+", 
"#Omega-K+", "#bar{#Omega}+K-", "#Omega-K-", "#bar{#Omega}+K+",
"#Sigma*^{+}K+", "#bar{#Sigma*}^{-}K-", "#Sigma*^{+}K-", "#bar{#Sigma*}^{-}K+", 
"#Sigma*^{-}K+", "#bar{#Sigma*}^{+}K-", "#Sigma*^{-}K-", "#bar{#Sigma*}^{+}K+", 
"#Sigma*^{0}K+", "#bar{#Sigma*}^{0}K-", "#Sigma*^{0}K-", "#bar{#Sigma*}^{0}K+", 
"#LambdaK*^{0}", "#bar{#Lambda}#bar{K*}^{0}", "#Lambda#bar{K*}^{0}", "#bar{#Lambda}K*^{0}", 
"#Sigma^{0}K*^{0}", "#bar{#Sigma}^{0}#bar{K*}^{0}", "#Sigma^{0}#bar{K*}^{0}", "#bar{#Sigma}^{0}K*^{0}", 
"#Xi^{0}K*^{0}", "#bar{#Xi}^{0}#bar{K*}^{0}", "#Xi^{0}#bar{K*}^{0}", "#bar{#Xi}^{0}K*^{0}", 
"#Xi-K*^{0}", "#bar{#Xi}+#bar{K*}^{0}", "#Xi-#bar{K*}^{0}", "#bar{#Xi}+K*^{0}"
};
*/

//enum BFieldType
const char* const cBFieldTags[7] = {"_FemtoPlus", "_FemtoMinus", "_Bp1", "_Bp2", "_Bm1", "_Bm2", "_Bm3"};

//enum CentralityType
const char* const cCentralityTags[4] = {"_0010", "_1030", "_3050", ""};
const char* const cPrettyCentralityTags[4] = {"0-10%", "10-30%", "30-50%", ""};

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
/*
const char* cModelKStarTrueVsRecSameBaseTag = "fKTrueKRecSameKStarModelCf_";
const char* cModelKStarTrueVsRecRotSameBaseTag = "fKTrueKRecRotSameKStarModelCf_";
const char* cModelKStarTrueVsRecMixedBaseTag = "fKTrueKRecMixedKStarModelCf_";
const char* cModelKStarTrueVsRecRotMixedBaseTag = "fKTrueKRecRotMixedKStarModelCf_";
*/
const char* cModelKStarTrueVsRecSameBaseTag = "KTrueVsKRecSameKStarModelCf_";
const char* cModelKStarTrueVsRecRotSameBaseTag = "KTrueVsKRecRotSameKStarModelCf_";
const char* cModelKStarTrueVsRecMixedBaseTag = "KTrueVsKRecMixedKStarModelCf_";
const char* cModelKStarTrueVsRecRotMixedBaseTag = "KTrueVsKRecRotMixedKStarModelCf_";

  //

/*
const char* cLambdaPurityTag = "LambdaPurity";
const char* cAntiLambdaPurityTag = "AntiLambdaPurity";
const char* cK0ShortPurityTag = "K0ShortPurity";
*/
const char* cLambdaPurityTag = "LambdaPurityAid";
const char* cAntiLambdaPurityTag = "AntiLambdaPurityAid";
const char* cK0ShortPurityTag = "K0ShortPurityAid";

const char* cXiPurityTag = "XiPurityAid";
const char* cAXiPurityTag = "AXiPurityAid";

//enum DaughterPairType----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*
const char* const cDaughterPairTags[10] = {"PosPos", "PosNeg", "NegPos", "NegNeg", "TrackPos", "TrackNeg", "TrackTrack", "TrackBac", "BacPos", "BacNeg"};

/*
const char* const cAvgSepCfBaseTagsNum[7] = {"NumPosPosAvgSepCf_", "NumPosNegAvgSepCf_", "NumNegPosAvgSepCf_", "NumNegNegAvgSepCf_", "NumTrackPosAvgSepCf_", "NumTrackNegAvgSepCf_", "NumTrackTrackAvgSepCf_"};
const char* const cAvgSepCfBaseTagsDen[7] = {"DenPosPosAvgSepCf_", "DenPosNegAvgSepCf_", "DenNegPosAvgSepCf_", "DenNegNegAvgSepCf_", "DenTrackPosAvgSepCf_", "DenTrackNegAvgSepCf_", "DenTrackTrackAvgSepCf_"};
*/
const char* const cAvgSepCfBaseTagsNum[8] = {"NumV0sPosPosAvgSepCf_", "NumV0sPosNegAvgSepCf_", "NumV0sNegPosAvgSepCf_", "NumV0sNegNegAvgSepCf_", "NumV0TrackPosAvgSepCf_", "NumV0TrackNegAvgSepCf_", "NumTrackTrackAvgSepCf_", "NumXiTrackBacAvgSepCf_"};
const char* const cAvgSepCfBaseTagsDen[8] = {"DenV0sPosPosAvgSepCf_", "DenV0sPosNegAvgSepCf_", "DenV0sNegPosAvgSepCf_", "DenV0sNegNegAvgSepCf_", "DenV0TrackPosAvgSepCf_", "DenV0TrackNegAvgSepCf_", "DenTrackTrackAvgSepCf_", "DenXiTrackBacAvgSepCf_"};


const char* const cSepCfsBaseTagsNum[8] = {"NumPosPosSepCfs_", "NumPosNegSepCfs_", "NumNegPosSepCfs_", "NumNegNegSepCfs_", "NumTrackPosSepCfs_", "NumTrackNegSepCfs_", "NumTrackTrackSepCfs_", "NumTrackBacSepCfs_"};
const char* const cSepCfsBaseTagsDen[8] = {"DenPosPosSepCfs_", "DenPosNegSepCfs_", "DenNegPosSepCfs_", "DenNegNegSepCfs_", "DenTrackPosSepCfs_", "DenTrackNegSepCfs_", "DenTrackTrackSepCfs_", "DenTrackBacSepCfs_"};

const char* const cAvgSepCfCowboysAndSailorsBaseTagsNum[8] = {"NumPosPosAvgSepCfCowboysAndSailors_", "NumPosNegAvgSepCfCowboysAndSailors_", "NumNegPosAvgSepCfCowboysAndSailors_", "NumNegNegAvgSepCfCowboysAndSailors_", "NumTrackPosAvgSepCfCowboysAndSailors_", "NumTrackNegAvgSepCfCowboysAndSailors_", "NumTrackTrackAvgSepCfCowboysAndSailors_", "NumTrackBacAvgSepCfCowboysAndSailors_"};
const char* const cAvgSepCfCowboysAndSailorsBaseTagsDen[8] = {"DenPosPosAvgSepCfCowboysAndSailors_", "DenPosNegAvgSepCfCowboysAndSailors_", "DenNegPosAvgSepCfCowboysAndSailors_", "DenNegNegAvgSepCfCowboysAndSailors_", "DenTrackPosAvgSepCfCowboysAndSailors_", "DenTrackNegAvgSepCfCowboysAndSailors_", "DenTrackTrackAvgSepCfCowboysAndSailors_", "DenTrackBacAvgSepCfCowboysAndSailors_"};

//----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*
const char* const cParameterNames[9] = {"Lambda", "Radius", "Ref0", "Imf0", "d0", "Ref02", "Imf02", "d02", "Norm"};

//________________________________________________________________________________________________________________

const double cLamK0_0010StartValues[6] = {0.4,3.0,-0.15,0.18,0.0,0.2};
const double cLamK0_1030StartValues[6] = {0.4,2.3,-0.15,0.18,0.0,0.2};
const double cLamK0_3050StartValues[6] = {0.4,1.7,-0.15,0.18,0.0,0.2};
const double* cLamK0StartValues[3] = {cLamK0_0010StartValues,cLamK0_1030StartValues,cLamK0_3050StartValues};

const double cALamK0_0010StartValues[6] = {0.4,3.0,-0.15,0.18,0.0,0.2};
const double cALamK0_1030StartValues[6] = {0.4,2.3,-0.15,0.18,0.0,0.2};
const double cALamK0_3050StartValues[6] = {0.4,1.7,-0.15,0.18,0.0,0.2};
const double* cALamK0StartValues[3] = {cALamK0_0010StartValues,cALamK0_1030StartValues,cALamK0_3050StartValues};
/*
const double cLamK0_0010StartValues[6] = {0.4,4.5,-0.25,0.25,0.0,0.2};
const double cLamK0_1030StartValues[6] = {0.4,4.0,-0.25,0.25,0.0,0.2};
const double cLamK0_3050StartValues[6] = {0.4,3.5,-0.25,0.25,0.0,0.2};
const double* cLamK0StartValues[3] = {cLamK0_0010StartValues,cLamK0_1030StartValues,cLamK0_3050StartValues};

const double cALamK0_0010StartValues[6] = {0.4,4.5,-0.25,0.25,0.0,0.2};
const double cALamK0_1030StartValues[6] = {0.4,4.0,-0.25,0.25,0.0,0.2};
const double cALamK0_3050StartValues[6] = {0.4,3.5,-0.25,0.25,0.0,0.2};
const double* cALamK0StartValues[3] = {cALamK0_0010StartValues,cALamK0_1030StartValues,cALamK0_3050StartValues};
*/
/*
const double cLamK0_0010StartValues[6] = {0.35,3.81,-0.22,0.43,3.16,0.1};
const double cLamK0_1030StartValues[6] = {0.15,2.00,-0.22,0.43,3.16,0.1};
const double cLamK0_3050StartValues[6] = {0.67,3.52,-0.22,0.43,3.16,0.1};
const double* cLamK0StartValues[3] = {cLamK0_0010StartValues,cLamK0_1030StartValues,cLamK0_3050StartValues};

const double cALamK0_0010StartValues[6] = {0.25,3.81,-0.22,0.43,3.16,0.1};
const double cALamK0_1030StartValues[6] = {0.16,2.00,-0.22,0.43,3.16,0.1};
const double cALamK0_3050StartValues[6] = {0.23,3.52,-0.22,0.43,3.16,0.1};
const double* cALamK0StartValues[3] = {cALamK0_0010StartValues,cALamK0_1030StartValues,cALamK0_3050StartValues};
*/
//________________________________________________________________________________________________________________
const double cLamKchP_0010StartValues[6] = {0.4,4.5,-0.5,0.5,0.0,0.2};
const double cLamKchP_1030StartValues[6] = {0.4,4.0,-0.5,0.5,0.0,0.2};
const double cLamKchP_3050StartValues[6] = {0.4,3.5,-0.5,0.5,0.0,0.2};
const double* cLamKchPStartValues[3] = {cLamKchP_0010StartValues,cLamKchP_1030StartValues,cLamKchP_3050StartValues};

const double cALamKchM_0010StartValues[6] = {0.4,4.5,-0.5,0.5,0.0,0.2};
const double cALamKchM_1030StartValues[6] = {0.4,4.0,-0.5,0.5,0.0,0.2};
const double cALamKchM_3050StartValues[6] = {0.4,3.5,-0.5,0.5,0.0,0.2};
/*
const double cLamKchP_0010StartValues[6] = {0.2,4.00,-1.315,0.5159,0.0,0.2};
const double cLamKchP_1030StartValues[6] = {0.2,3.75,-1.315,0.5159,0.0,0.2};
const double cLamKchP_3050StartValues[6] = {0.2,3.50,-1.315,0.5159,0.0,0.2};
const double* cLamKchPStartValues[3] = {cLamKchP_0010StartValues,cLamKchP_1030StartValues,cLamKchP_3050StartValues};

const double cALamKchM_0010StartValues[6] = {0.2,4.00,-1.315,0.5159,0.0,0.2};
const double cALamKchM_1030StartValues[6] = {0.2,3.75,-1.315,0.5159,0.0,0.2};
const double cALamKchM_3050StartValues[6] = {0.2,3.50,-1.315,0.5159,0.0,0.2};
*/
/*
const double cLamKchP_0010StartValues[6] = {0.38,4.05,-0.69,0.39,0.64,0.1};
const double cLamKchP_1030StartValues[6] = {0.48,3.92,-0.69,0.39,0.64,0.1};
const double cLamKchP_3050StartValues[6] = {0.64,3.72,-0.69,0.39,0.64,0.1};
const double* cLamKchPStartValues[3] = {cLamKchP_0010StartValues,cLamKchP_1030StartValues,cLamKchP_3050StartValues};

const double cALamKchM_0010StartValues[6] = {0.37,4.05,-0.69,0.39,0.64,0.1};
const double cALamKchM_1030StartValues[6] = {0.41,3.92,-0.69,0.39,0.64,0.1};
const double cALamKchM_3050StartValues[6] = {0.62,3.72,-0.69,0.39,0.64,0.1};

//const double cALamKchM_0010StartValues[6] = {0.37,cLamKchP_0010StartValues[1],cLamKchP_0010StartValues[2],cLamKchP_0010StartValues[3],cLamKchP_0010StartValues[4],cLamKchP_0010StartValues[5]};
//const double cALamKchM_1030StartValues[6] = {0.41,cLamKchP_1030StartValues[1],cLamKchP_1030StartValues[2],cLamKchP_1030StartValues[3],cLamKchP_1030StartValues[4],cLamKchP_1030StartValues[5]};
//const double cALamKchM_3050StartValues[6] = {0.62,cLamKchP_3050StartValues[1],cLamKchP_3050StartValues[2],cLamKchP_3050StartValues[3],cLamKchP_3050StartValues[4],cLamKchP_3050StartValues[5]};
*/
const double* cALamKchMStartValues[3] = {cALamKchM_0010StartValues,cALamKchM_1030StartValues,cALamKchM_3050StartValues};

//--------------------------------------

const double cLamKchM_0010StartValues[6] = {0.4,4.5,0.5,0.5,0.0,0.2};
const double cLamKchM_1030StartValues[6] = {0.4,4.0,0.5,0.5,0.0,0.2};
const double cLamKchM_3050StartValues[6] = {0.4,3.5,0.5,0.5,0.0,0.2};
const double* cLamKchMStartValues[3] = {cLamKchM_0010StartValues,cLamKchM_1030StartValues,cLamKchM_3050StartValues};

const double cALamKchP_0010StartValues[6] = {0.4,4.5,0.5,0.5,0.0,0.2};
const double cALamKchP_1030StartValues[6] = {0.4,4.0,0.5,0.5,0.0,0.2};
const double cALamKchP_3050StartValues[6] = {0.4,3.5,0.5,0.5,0.0,0.2};
const double* cALamKchPStartValues[3] = {cALamKchP_0010StartValues,cALamKchP_1030StartValues,cALamKchP_3050StartValues};
/*
const double cLamKchM_0010StartValues[6] = {0.45,4.79,0.18,0.45,-5.0,0.1};
const double cLamKchM_1030StartValues[6] = {0.40,4.00,0.18,0.45,-5.0,0.1};
const double cLamKchM_3050StartValues[6] = {0.20,2.11,0.18,0.45,-5.0,0.1};
const double* cLamKchMStartValues[3] = {cLamKchM_0010StartValues,cLamKchM_1030StartValues,cLamKchM_3050StartValues};

const double cALamKchP_0010StartValues[6] = {0.48,4.79,0.18,0.45,-5.0,0.1};
const double cALamKchP_1030StartValues[6] = {0.49,4.00,0.18,0.45,-5.0,0.1};
const double cALamKchP_3050StartValues[6] = {0.22,2.11,0.18,0.45,-5.0,0.1};
const double* cALamKchPStartValues[3] = {cALamKchP_0010StartValues,cALamKchP_1030StartValues,cALamKchP_3050StartValues};
*/
//________________________________________________________________________________________________________________

const double cXiKchP_0010StartValues[6] = {0.5,3.0,-0.5,0.5,0.,0.2};
const double cXiKchP_1030StartValues[6] = {0.19,4.00,-1.28,0.69,1.79,0.2};
const double cXiKchP_3050StartValues[6] = {0.19,3.00,-1.28,0.69,1.79,0.2};
const double* cXiKchPStartValues[3] = {cXiKchP_0010StartValues,cXiKchP_1030StartValues,cXiKchP_3050StartValues};

const double cAXiKchP_0010StartValues[6] = {0.5,3.0,0.5,0.5,0.,0.2};
const double cAXiKchP_1030StartValues[6] = {0.49,4.00,0.09,0.27,7.79,0.2};
const double cAXiKchP_3050StartValues[6] = {0.49,3.00,0.09,0.27,7.79,0.2};
const double* cAXiKchPStartValues[3] = {cAXiKchP_0010StartValues,cAXiKchP_1030StartValues,cAXiKchP_3050StartValues};

const double cXiKchM_0010StartValues[6] = {0.5,3.0,0.5,0.5,0.,0.2};
const double cXiKchM_1030StartValues[6] = {0.49,4.00,0.09,0.27,7.79,0.2};
const double cXiKchM_3050StartValues[6] = {0.49,3.00,0.09,0.27,7.79,0.2};
const double* cXiKchMStartValues[3] = {cXiKchM_0010StartValues,cXiKchM_1030StartValues,cXiKchM_3050StartValues};

const double cAXiKchM_0010StartValues[6] = {0.5,3.0,-0.5,0.5,0.,0.2};
const double cAXiKchM_1030StartValues[6] = {0.19,4.00,-1.28,0.69,1.79,0.2};
const double cAXiKchM_3050StartValues[6] = {0.19,3.00,-1.28,0.69,1.79,0.2};
const double* cAXiKchMStartValues[3] = {cAXiKchM_0010StartValues,cAXiKchM_1030StartValues,cAXiKchM_3050StartValues};

//-----

const double** cStartValues[10] = {cLamK0StartValues, cALamK0StartValues, cLamKchPStartValues, cALamKchMStartValues, cLamKchMStartValues, cALamKchPStartValues, cXiKchPStartValues, cAXiKchMStartValues, cXiKchMStartValues, cAXiKchPStartValues};


//________________________________________________________________________________________________________________

const double cLamK0_0010SysErrors[5] = {0.116, 0.329, 0.043, 0.059, 2.836};
const double cLamK0_1030SysErrors[5] = {0.116, 0.324, 0.043, 0.059, 2.836};
const double cLamK0_3050SysErrors[5] = {0.116, 0.280, 0.043, 0.059, 2.836};
const double* cLamK0SysErrors[3] = {cLamK0_0010SysErrors,cLamK0_1030SysErrors,cLamK0_3050SysErrors};

const double cALamK0_0010SysErrors[5] = {0.116, 0.329, 0.043, 0.059, 2.836};
const double cALamK0_1030SysErrors[5] = {0.116, 0.324, 0.043, 0.059, 2.836};
const double cALamK0_3050SysErrors[5] = {0.116, 0.280, 0.043, 0.059, 2.836};
const double* cALamK0SysErrors[3] = {cALamK0_0010SysErrors,cALamK0_1030SysErrors,cALamK0_3050SysErrors};

//________________________________________________________________________________________________________________
const double cLamKchP_0010SysErrors[5] = {0.220, 0.830, 0.223 ,0.111, 1.621};
const double cLamKchP_1030SysErrors[5] = {0.241, 0.663, 0.223, 0.111, 1.621};
const double cLamKchP_3050SysErrors[5] = {0.204, 0.420, 0.223, 0.111, 1.621};
const double* cLamKchPSysErrors[3] = {cLamKchP_0010SysErrors,cLamKchP_1030SysErrors,cLamKchP_3050SysErrors};

const double cALamKchM_0010SysErrors[5] = {0.217, 0.830, 0.223, 0.111, 1.621};
const double cALamKchM_1030SysErrors[5] = {0.201, 0.663, 0.223, 0.111, 1.621};
const double cALamKchM_3050SysErrors[5] = {0.203, 0.420, 0.223, 0.111, 1.621};
const double* cALamKchMSysErrors[3] = {cALamKchM_0010SysErrors,cALamKchM_1030SysErrors,cALamKchM_3050SysErrors};

//--------------------------------------

const double cLamKchM_0010SysErrors[5] = {0.186, 1.375, 0.095, 0.184, 7.658};
const double cLamKchM_1030SysErrors[5] = {0.198, 0.978, 0.095, 0.184, 7.658};
const double cLamKchM_3050SysErrors[5] = {0.132, 0.457, 0.095, 0.184, 7.658};
const double* cLamKchMSysErrors[3] = {cLamKchM_0010SysErrors,cLamKchM_1030SysErrors,cLamKchM_3050SysErrors};

const double cALamKchP_0010SysErrors[5] = {0.152, 1.375, 0.095, 0.184, 7.658};
const double cALamKchP_1030SysErrors[5] = {0.148, 0.978, 0.095 ,0.184, 7.658};
const double cALamKchP_3050SysErrors[5] = {0.106, 0.457, 0.095, 0.184, 7.658};
const double* cALamKchPSysErrors[3] = {cALamKchP_0010SysErrors,cALamKchP_1030SysErrors,cALamKchP_3050SysErrors};
//________________________________________________________________________________________________________________

const double cXiKchP_0010SysErrors[5] = {0.,0.,0.,0.,0.};
const double cXiKchP_1030SysErrors[5] = {0.,0.,0.,0.,0.};
const double cXiKchP_3050SysErrors[5] = {0.,0.,0.,0.,0.};
const double* cXiKchPSysErrors[3] = {cXiKchP_0010SysErrors,cXiKchP_1030SysErrors,cXiKchP_3050SysErrors};

const double cAXiKchP_0010SysErrors[5] = {0.,0.,0.,0.,0.};
const double cAXiKchP_1030SysErrors[5] = {0.,0.,0.,0.,0.};
const double cAXiKchP_3050SysErrors[5] = {0.,0.,0.,0.,0.};
const double* cAXiKchPSysErrors[3] = {cAXiKchP_0010SysErrors,cAXiKchP_1030SysErrors,cAXiKchP_3050SysErrors};

const double cXiKchM_0010SysErrors[5] = {0.,0.,0.,0.,0.};
const double cXiKchM_1030SysErrors[5] = {0.,0.,0.,0.,0.};
const double cXiKchM_3050SysErrors[5] = {0.,0.,0.,0.,0.};
const double* cXiKchMSysErrors[3] = {cXiKchM_0010SysErrors,cXiKchM_1030SysErrors,cXiKchM_3050SysErrors};

const double cAXiKchM_0010SysErrors[5] = {0.,0.,0.,0.,0.};
const double cAXiKchM_1030SysErrors[5] = {0.,0.,0.,0.,0.};
const double cAXiKchM_3050SysErrors[5] = {0.,0.,0.,0.,0.};
const double* cAXiKchMSysErrors[3] = {cAXiKchM_0010SysErrors,cAXiKchM_1030SysErrors,cAXiKchM_3050SysErrors};

//-----

const double** cSysErrors[10] = {cLamK0SysErrors, cALamK0SysErrors, cLamKchPSysErrors, cALamKchMSysErrors, cLamKchMSysErrors, cALamKchPSysErrors, cXiKchPSysErrors, cAXiKchMSysErrors, cXiKchMSysErrors, cAXiKchPSysErrors};



//-----
const double hbarc = 0.197327;
const double gBohrRadiusXiK = 75.233503766;
const double gBohrRadiusOmegaK = 70.942912779;
const double gBohrRadiusSigStPK = 74.328580067;
const double gBohrRadiusSigStMK = 74.267628885;

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


