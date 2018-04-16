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

//enum ParticleType
const char* const cParticleTags[25] = {
"Proton", "AntiProton", 
"PiP", "PiM", 
"K0", 
"KchP", "KchM", 
"Lam", "ALam", 
"Sig0", "ASig0",
"Xi", "AXi",
"Xi0", "AXi0",
"Omega", "AOmega", 

"kSigStP", "kASigStM", 
"kSigStM", "kASigStP", 
"kSigSt0", "kASigSt0", 
"kKSt0", "kAKSt0"
};


const char* const cRootParticleTags[25] = {
"p", "#bar{p}", 
"#pi+", "#pi-", 
"K^{0}_{S}", 
"K+", "K-", 
"#Lambda", "#bar{#Lambda}", 
"#Sigma^{0}", "#bar{#Sigma}^{0}", 
"#Xi-", "#bar{#Xi}+", 
"#Xi^{0}", "#bar{#Xi}^{0}", 
"#Omega-", "#bar{#Omega}+", 

"#Sigma*+", "#bar{#Sigma*}-", 
"#Sigma*-", "#bar{#Sigma*}+", 
"#Sigma*^{0}", "#bar{#Sigma*}^{0}", 
"K*^{0}", "#bar{K*}^{0}"
};



//enum AnalysisType
const char* const cAnalysisBaseTags[85] = {
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
"XiKSt0", "AXiAKSt0", "XiAKSt0", "AXiKSt0",

"Sig0K0", "ASig0K0", 
"Xi0K0", "AXi0K0",
"XiK0", "AXiK0",
"OmegaK0", "AOmegaK0",
"SigStPK0", "ASigStMK0",
"SigStMK0", "ASigStPK0", 
"SigSt0K0", "ASigSt0K0",

"LamKSt0ToLamK0", "ALamKSt0ToALamK0", 
"Sig0KSt0ToLamK0", "ASigma0KSt0ToALamK0", 
"Xi0KSt0ToLamK0", "AXi0KSt0ToALamK0",
"XiKSt0ToLamK0", "AXiKSt0ToALamK0"

};

const char* const cAnalysisRootTags[85] = {
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
"#Xi-K*^{0}", "#bar{#Xi}+#bar{K*}^{0}", "#Xi-#bar{K*}^{0}", "#bar{#Xi}+K*^{0}", 

"#Sigma^{0}K^{0}_{S}", "#bar{#Sigma}^{0}K^{0}_{S}", 
"#Xi^{0}K^{0}_{S}", "#bar{#Xi}^{0}K^{0}_{S}",
"#Xi-K^{0}_{S}", "#bar{#Xi}+K^{0}_{S}",
"#Omega-K^{0}_{S}", "#bar{#Omega}+K^{0}_{S}",
"#Sigma*^{+}K^{0}_{S}", "#bar{#Sigma*}^{-}K^{0}_{S}",
"#Sigma*^{-}K^{0}_{S}", "#bar{#Sigma*}^{+}K^{0}_{S}",
"#Sigma*^{0}K^{0}_{S}", "#bar{#Sigma*}^{0}K^{0}_{S}0",

"#LambdaK*^{0}", "#bar{#Lambda}#bar{K*}^{0}",
"#Sigma^{0}K*^{0}", "#bar{#Sigma}^{0}#bar{K*}^{0}",
"#Xi^{0}K*^{0}", "#bar{#Xi}^{0}#bar{K*}^{0}",
"#Xi-K*^{0}", "#bar{#Xi}+#bar{K*}^{0}"

};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and no MaxDecayLengthPrimary! ------------------------------------//
const double cAnalysisLambdaFactors2[85] = {
/*kLamK0=*/0.162, /*kALamK0=*/0.167,
/*kLamKchP=*/0.152, /*kALamKchM=*/0.156, /*kLamKchM=*/0.151, /*kALamKchP=*/0.156,
/*kXiKchP=*/1.000, /*kAXiKchM=*/1.000, /*kXiKchM=*/1.000, /*kAXiKchP=*/1.000,
/*kXiK0=*/1.000, /*kAXiK0=*/1.000,
/*kLamLam=*/1.000, /*kALamALam=*/1.000, /*kLamALam=*/1.000,
/*kLamPiP=*/1.000, /*kALamPiM=*/1.000, /*kLamPiM=*/1.000, /*kALamPiP=*/1.000,

//----- Residual Types -----
/*kResSig0KchP=*/0.104, /*kResASig0KchM=*/0.107, /*kResSig0KchM=*/0.103, /*kResASig0KchP=*/0.108,
/*kResXi0KchP=*/0.075, /*kResAXi0KchM=*/0.071, /*kResXi0KchM=*/0.074, /*kResAXi0KchP=*/0.071,
/*kResXiCKchP=*/0.073, /*kResAXiCKchM=*/0.069, /*kResXiCKchM=*/0.072, /*kResAXiCKchP=*/0.069,
/*kResOmegaKchP=*/0.000, /*kResAOmegaKchM=*/0.000, /*kResOmegaKchM=*/0.000, /*kResAOmegaKchP=*/0.000,
/*kResSigStPKchP=*/0.048, /*kResASigStMKchM=*/0.048, /*kResSigStPKchM=*/0.048, /*kResASigStMKchP=*/0.049,
/*kResSigStMKchP=*/0.044, /*kResASigStPKchM=*/0.047, /*kResSigStMKchM=*/0.043, /*kResASigStPKchP=*/0.047,
/*kResSigSt0KchP=*/0.044, /*kResASigSt0KchM=*/0.042, /*kResSigSt0KchM=*/0.043, /*kResASigSt0KchP=*/0.043,

/*kResLamKSt0=*/0.041, /*kResALamAKSt0=*/0.043, /*kResLamAKSt0=*/0.041, /*kResALamKSt0=*/0.043,
/*kResSig0KSt0=*/0.037, /*kResASig0AKSt0=*/0.038, /*kResSig0AKSt0=*/0.037, /*kResASig0KSt0=*/0.038,
/*kResXi0KSt0=*/0.026, /*kResAXi0AKSt0=*/0.025, /*kResXi0AKSt0=*/0.026, /*kResAXi0KSt0=*/0.025,
/*kResXiCKSt0=*/0.026, /*kResAXiCAKSt0=*/0.024, /*kResXiCAKSt0=*/0.026, /*kResAXiCKSt0=*/0.024, 

/*kResSig0K0=*/0.113, /*kResASig0K0=*/0.117,
/*kResXi0K0=*/0.081, /*kResAXi0K0=*/0.077,
/*kResXiCK0=*/0.078, /*kResAXiCK0=*/0.075,
/*kResOmegaK0=*/0.000, /*kResAOmegaK0=*/0.000,
/*kResSigStPK0=*/0.052, /*kResASigStMK0=*/0.053,
/*kResSigStMK0=*/0.047, /*kResASigStPK0=*/0.051,
/*kResSigSt0K0=*/0.047, /*kResASigSt0K0=*/0.046,

/*kResLamKSt0ToLamK0=*/0.020, /*kResALamKSt0ToALamK0=*/0.021,
/*kResSig0KSt0ToLamK0=*/0.018, /*kResASig0KSt0ToALamK0=*/0.018,
/*kResXi0KSt0ToLamK0=*/0.013, /*kResAXi0KSt0ToALamK0=*/0.012,
/*kResXiCKSt0ToLamK0=*/0.012, /*kResAXiCKSt0ToALamK0=*/0.012
};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=5! ------------------------------------//
//---------NEW:ASSUMING SOME OF "OTHER" FEED-DOWN BIN ARE PRIMARY------------------------
const double cAnalysisLambdaFactors[85] = {
/*kLamK0=*/0.217, /*kALamK0=*/0.222,
/*kLamKchP=*/0.188, /*kALamKchM=*/0.192, /*kLamKchM=*/0.188, /*kALamKchP=*/0.193,
/*kXiKchP=*/1.000, /*kAXiKchM=*/1.000, /*kXiKchM=*/1.000, /*kAXiKchP=*/1.000,
/*kXiK0=*/1.000, /*kAXiK0=*/1.000,
/*kLamLam=*/1.000, /*kALamALam=*/1.000, /*kLamALam=*/1.000,
/*kLamPiP=*/1.000, /*kALamPiM=*/1.000, /*kLamPiM=*/1.000, /*kALamPiP=*/1.000,

//----- Residual Types -----
/*kResSig0KchP=*/0.099, /*kResASig0KchM=*/0.102, /*kResSig0KchM=*/0.099, /*kResASig0KchP=*/0.103,
/*kResXi0KchP=*/0.072, /*kResAXi0KchM=*/0.067, /*kResXi0KchM=*/0.071, /*kResAXi0KchP=*/0.068,
/*kResXiCKchP=*/0.069, /*kResAXiCKchM=*/0.065, /*kResXiCKchM=*/0.068, /*kResAXiCKchP=*/0.066,
/*kResOmegaKchP=*/0.000, /*kResAOmegaKchM=*/0.000, /*kResOmegaKchM=*/0.000, /*kResAOmegaKchP=*/0.000,
/*kResSigStPKchP=*/0.046, /*kResASigStMKchM=*/0.046, /*kResSigStPKchM=*/0.046, /*kResASigStMKchP=*/0.046,
/*kResSigStMKchP=*/0.042, /*kResASigStPKchM=*/0.045, /*kResSigStMKchM=*/0.041, /*kResASigStPKchP=*/0.045,
/*kResSigSt0KchP=*/0.042, /*kResASigSt0KchM=*/0.040, /*kResSigSt0KchM=*/0.041, /*kResASigSt0KchP=*/0.041,

/*kResLamKSt0=*/0.039, /*kResALamAKSt0=*/0.041, /*kResLamAKSt0=*/0.039, /*kResALamKSt0=*/0.041,
/*kResSig0KSt0=*/0.035, /*kResASig0AKSt0=*/0.036, /*kResSig0AKSt0=*/0.035, /*kResASig0KSt0=*/0.036,
/*kResXi0KSt0=*/0.025, /*kResAXi0AKSt0=*/0.024, /*kResXi0AKSt0=*/0.025, /*kResAXi0KSt0=*/0.024,
/*kResXiCKSt0=*/0.024, /*kResAXiCAKSt0=*/0.023, /*kResXiCAKSt0=*/0.024, /*kResAXiCKSt0=*/0.023, 

/*kResSig0K0=*/0.107, /*kResASig0K0=*/0.111,
/*kResXi0K0=*/0.077, /*kResAXi0K0=*/0.073,
/*kResXiCK0=*/0.075, /*kResAXiCK0=*/0.071,
/*kResOmegaK0=*/0.000, /*kResAOmegaK0=*/0.000,
/*kResSigStPK0=*/0.050, /*kResASigStMK0=*/0.050,
/*kResSigStMK0=*/0.045, /*kResASigStPK0=*/0.049,
/*kResSigSt0K0=*/0.045, /*kResASigSt0K0=*/0.044,

/*kResLamKSt0ToLamK0=*/0.019, /*kResALamKSt0ToALamK0=*/0.020,
/*kResSig0KSt0ToLamK0=*/0.017, /*kResASig0KSt0ToALamK0=*/0.017,
/*kResXi0KSt0ToLamK0=*/0.012, /*kResAXi0KSt0ToALamK0=*/0.011,
/*kResXiCKSt0ToLamK0=*/0.012, /*kResAXiCKSt0ToALamK0=*/0.011
};


//------------ WHEN ONLY INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=5! ------------------------------------//
const double cAnalysisLambdaFactorsOLD[85] = {
/*kLamK0=*/0.235, /*kALamK0=*/0.242,
/*kLamKchP=*/0.228, /*kALamKchM=*/0.233, /*kLamKchM=*/0.227, /*kALamKchP=*/0.234,
/*kXiKchP=*/1.000, /*kAXiKchM=*/1.000, /*kXiKchM=*/1.000, /*kAXiKchP=*/1.000,
/*kXiK0=*/1.000, /*kAXiK0=*/1.000,
/*kLamLam=*/1.000, /*kALamALam=*/1.000, /*kLamALam=*/1.000,
/*kLamPiP=*/1.000, /*kALamPiM=*/1.000, /*kLamPiM=*/1.000, /*kALamPiP=*/1.000,

//----- Residual Types -----
/*kResSig0KchP=*/0.099, /*kResASig0KchM=*/0.102, /*kResSig0KchM=*/0.099, /*kResASig0KchP=*/0.103,
/*kResXi0KchP=*/0.072, /*kResAXi0KchM=*/0.067, /*kResXi0KchM=*/0.071, /*kResAXi0KchP=*/0.068,
/*kResXiCKchP=*/0.069, /*kResAXiCKchM=*/0.065, /*kResXiCKchM=*/0.068, /*kResAXiCKchP=*/0.066,
/*kResOmegaKchP=*/0.000, /*kResAOmegaKchM=*/0.000, /*kResOmegaKchM=*/0.000, /*kResAOmegaKchP=*/0.000,
/*kResSigStPKchP=*/0.000, /*kResASigStMKchM=*/0.000, /*kResSigStPKchM=*/0.000, /*kResASigStMKchP=*/0.000,
/*kResSigStMKchP=*/0.000, /*kResASigStPKchM=*/0.000, /*kResSigStMKchM=*/0.000, /*kResASigStPKchP=*/0.000,
/*kResSigSt0KchP=*/0.000, /*kResASigSt0KchM=*/0.000, /*kResSigSt0KchM=*/0.000, /*kResASigSt0KchP=*/0.000,

/*kResLamKSt0=*/0.000, /*kResALamAKSt0=*/0.000, /*kResLamAKSt0=*/0.000, /*kResALamKSt0=*/0.000,
/*kResSig0KSt0=*/0.000, /*kResASig0AKSt0=*/0.000, /*kResSig0AKSt0=*/0.000, /*kResASig0KSt0=*/0.000,
/*kResXi0KSt0=*/0.000, /*kResAXi0AKSt0=*/0.000, /*kResXi0AKSt0=*/0.000, /*kResAXi0KSt0=*/0.000,
/*kResXiCKSt0=*/0.000, /*kResAXiCAKSt0=*/0.000, /*kResXiCAKSt0=*/0.000, /*kResAXiCKSt0=*/0.000,

/*kResSig0K0=*/0.107, /*kResASig0K0=*/0.111,
/*kResXi0K0=*/0.077, /*kResAXi0K0=*/0.073,
/*kResXiCK0=*/0.075, /*kResAXiCK0=*/0.071,
/*kResOmegaK0=*/0.000, /*kResAOmegaK0=*/0.000,
/*kResSigStPK0=*/0.000, /*kResASigStMK0=*/0.000,
/*kResSigStMK0=*/0.000, /*kResASigStPK0=*/0.000,
/*kResSigSt0K0=*/0.000, /*kResASigSt0K0=*/0.000,

/*kResLamKSt0ToLamK0=*/0.000, /*kResALamAKSt0ToALamK0=*/0.000,
/*kResSig0KSt0ToLamK0=*/0.000, /*kResASig0AKSt0ToALamK0=*/0.000,
/*kResXi0KSt0ToLamK0=*/0.000, /*kResAXi0AKSt0ToALamK0=*/0.000,
/*kResXiCKSt0ToLamK0=*/0.000, /*kResAXiCAKSt0ToALamK0=*/0.000
};

const double cmTFactorsFromTherminator[85] = {
/*kLamK0=*/1.512, /*kALamK0=*/1.502,
/*kLamKchP=*/1.404, /*kALamKchM=*/1.388, /*kLamKchM=*/1.404, /*kALamKchP=*/1.388,
/*kXiKchP=*/1.000, /*kAXiKchM=*/1.000, /*kXiKchM=*/1.000, /*kAXiKchP=*/1.000,
/*kXiK0=*/1.000, /*kAXiK0=*/1.000,
/*kLamLam=*/1.000, /*kALamALam=*/1.000, /*kLamALam=*/1.000,
/*kLamPiP=*/1.000, /*kALamPiM=*/1.000, /*kLamPiM=*/1.000, /*kALamPiP=*/1.000,

//----- Residual Types -----
/*kResSig0KchP=*/1.453, /*kResASig0KchM=*/1.438, /*kResSig0KchM=*/1.453, /*kResASig0KchP=*/1.438,
/*kResXi0KchP=*/1.536, /*kResAXi0KchM=*/1.520, /*kResXi0KchM=*/1.536, /*kResAXi0KchP=*/1.521,
/*kResXiCKchP=*/1.540, /*kResAXiCKchM=*/1.525, /*kResXiCKchM=*/1.540, /*kResAXiCKchP=*/1.525,
/*kResOmegaKchP=*/1.768, /*kResAOmegaKchM=*/1.755, /*kResOmegaKchM=*/1.768, /*kResAOmegaKchP=*/1.759,
/*kResSigStPKchP=*/1.585, /*kResASigStMKchM=*/1.570, /*kResSigStPKchM=*/1.585, /*kResASigStMKchP=*/1.570,
/*kResSigStMKchP=*/1.589, /*kResASigStPKchM=*/1.573, /*kResSigStMKchM=*/1.589, /*kResASigStPKchP=*/1.575,
/*kResSigSt0KchP=*/1.586, /*kResASigSt0KchM=*/1.570, /*kResSigSt0KchM=*/1.585, /*kResASigSt0KchP=*/1.571,

/*kResLamKSt0=*/1.668, /*kResALamAKSt0=*/1.653, /*kResLamAKSt0=*/1.669, /*kResALamKSt0=*/1.653,
/*kResSig0KSt0=*/1.714, /*kResASig0AKSt0=*/1.698, /*kResSig0AKSt0=*/1.715, /*kResASig0KSt0=*/1.698,
/*kResXi0KSt0=*/1.788, /*kResAXi0AKSt0=*/1.773, /*kResXi0AKSt0=*/1.788, /*kResAXi0KSt0=*/1.772,
/*kResXiCKSt0=*/1.792, /*kResAXiCAKSt0=*/1.776, /*kResXiCAKSt0=*/1.792, /*kResAXiCKSt0=*/1.777,

/*kResSig0K0=*/1.560, /*kResASig0K0=*/1.551,
/*kResXi0K0=*/1.641, /*kResAXi0K0=*/1.629,
/*kResXiCK0=*/1.643, /*kResAXiCK0=*/1.636,
/*kResOmegaK0=*/1.856, /*kResAOmegaK0=*/1.852,
/*kResSigStPK0=*/1.689, /*kResASigStMK0=*/1.678,
/*kResSigStMK0=*/1.691, /*kResASigStPK0=*/1.683,
/*kResSigSt0K0=*/1.688, /*kResASigSt0K0=*/1.678,

/*kResLamKSt0ToLamK0=*/1.765, /*kResALamAKSt0ToALamK0=*/1.755,
/*kResSig0KSt0ToLamK0=*/1.805, /*kResASig0AKSt0ToALamK0=*/1.798,
/*kResXi0KSt0ToLamK0=*/1.873, /*kResAXi0AKSt0ToALamK0=*/1.867,
/*kResXiCKSt0ToLamK0=*/1.882, /*kResAXiCAKSt0ToALamK0=*/1.869
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


//enum KStarTrueVsRecType
const char* const cKStarTrueVsRecTypeTags[4] = {"Same","RotSame","Mixed","RotMixed"};

//enum NonFlatBgdFitType
const char* const cNonFlatBgdFitTypeTags[5] = {"Linear", "Quadratic", "Gaussian", "Polynomial", "DivideByTherm"};

//enum ThermEventsType
const char* const cThermEventsTypeTags[3] = {"_Me", "_Adam", "_MeAndAdam"};

//enum IncludeResidualsType
const char* const cIncludeResidualsTypeTags[3] = {"_NoRes", "_10Res", "_3Res"};
//enum ChargedResidualsType
const char* const cChargedResidualsTypeTags[3] = {"_UsingXiDataForAll", "_UsingXiDataAndCoulombOnly", "_UsingCoulombOnlyForAll"};
//enum ResPrimMaxDecayType
const char* const cResPrimMaxDecayTypeTags[6] = {"_PrimMaxDecay0fm", "_PrimMaxDecay4fm", "_PrimMaxDecay5fm", "_PrimMaxDecay6fm", "_PrimMaxDecay10fm", "_PrimMaxDecay100fm"};

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
const char* const cParameterNames[10] = {"Lambda", "Radius", "Ref0", "Imf0", "d0", "Ref02", "Imf02", "d02", "Norm", "Bgd"};

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


//-----
const double hbarc = 0.197327;
const double gBohrRadiusXiK = 75.233503766;
const double gBohrRadiusOmegaK = 70.942912779;
const double gBohrRadiusSigStPK = 74.328580067;
const double gBohrRadiusSigStMK = 74.267628885;

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
vector<ParticlePDGType> GetResidualDaughtersAndMothers(AnalysisType aResidualType)
{
  AnalysisType tAnType;
  vector<ParticlePDGType> tReturnVec(4);
  ParticlePDGType tDaughterType1, tMotherType1;
  ParticlePDGType tDaughterType2, tMotherType2;
  switch(aResidualType) {
  //LamKchP-------------------------------
  case kResSig0KchP:
  case kResXi0KchP:
  case kResXiCKchP:
  case kResOmegaKchP:
  case kResSigStPKchP:
  case kResSigStMKchP:
  case kResSigSt0KchP:
  case kResLamKSt0:
  case kResSig0KSt0:
  case kResXi0KSt0:
  case kResXiCKSt0:

    tDaughterType1 = kPDGLam;
    tDaughterType2 = kPDGKchP;
    break;

  //ALamKchM-------------------------------
  case kResASig0KchM:
  case kResAXi0KchM:
  case kResAXiCKchM:
  case kResAOmegaKchM:
  case kResASigStMKchM:
  case kResASigStPKchM:
  case kResASigSt0KchM:
  case kResALamAKSt0:
  case kResASig0AKSt0:
  case kResAXi0AKSt0:
  case kResAXiCAKSt0:

    tDaughterType1 = kPDGALam;
    tDaughterType2 = kPDGKchM;
    break;
  //-------------

  //LamKchM-------------------------------
  case kResSig0KchM:
  case kResXi0KchM:
  case kResXiCKchM:
  case kResOmegaKchM:
  case kResSigStPKchM:
  case kResSigStMKchM:
  case kResSigSt0KchM:
  case kResLamAKSt0:
  case kResSig0AKSt0:
  case kResXi0AKSt0:
  case kResXiCAKSt0:

    tDaughterType1 = kPDGLam;
    tDaughterType2 = kPDGKchM;
    break;
  //-------------

  //ALamKchP-------------------------------
  case kResASig0KchP:
  case kResAXi0KchP:
  case kResAXiCKchP:
  case kResAOmegaKchP:
  case kResASigStMKchP:
  case kResASigStPKchP:
  case kResASigSt0KchP:
  case kResALamKSt0:
  case kResASig0KSt0:
  case kResAXi0KSt0:
  case kResAXiCKSt0:

    tDaughterType1 = kPDGALam;
    tDaughterType2 = kPDGKchP;
    break;
  //-------------

  //LamK0-------------------------------
  case kResSig0K0:
  case kResXi0K0:
  case kResXiCK0:
  case kResOmegaK0:
  case kResSigStPK0:
  case kResSigStMK0:
  case kResSigSt0K0:
  case kResLamKSt0ToLamK0:
  case kResSig0KSt0ToLamK0:
  case kResXi0KSt0ToLamK0:
  case kResXiCKSt0ToLamK0:

    tDaughterType1 = kPDGLam;
    tDaughterType2 = kPDGK0;
    break;
  //-------------

  //ALamK0-------------------------------
  case kResASig0K0:
  case kResAXi0K0:
  case kResAXiCK0:
  case kResAOmegaK0:
  case kResASigStMK0:
  case kResASigStPK0:
  case kResASigSt0K0:
  case kResALamKSt0ToALamK0:
  case kResASig0KSt0ToALamK0:
  case kResAXi0KSt0ToALamK0:
  case kResAXiCKSt0ToALamK0:

    tDaughterType1 = kPDGALam;
    tDaughterType2 = kPDGK0;
    break;
  //-------------

  default:
    cout << "ERROR: Types::GetResidualDaughtersAndMothers:  aResidualType = " << aResidualType << " is not appropriate" << endl << endl;
    assert(0);
  }


  switch(aResidualType) {
  //LamKchP-------------------------------------------------------------------------------
  case kResSig0KchP:
    tMotherType1 = kPDGSigma;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResXi0KchP:
    tMotherType1 = kPDGXi0;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResXiCKchP:
    tMotherType1 = kPDGXiC;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResOmegaKchP:
    tMotherType1 = kPDGOmega;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResSigStPKchP:
    tMotherType1 = kPDGSigStP;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResSigStMKchP:
    tMotherType1 = kPDGSigStM;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResSigSt0KchP:
    tMotherType1 = kPDGSigSt0;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResLamKSt0:
    tMotherType1 = kPDGLam;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResSig0KSt0:
    tMotherType1 = kPDGSigma;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResXi0KSt0:
    tMotherType1 = kPDGXi0;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResXiCKSt0:
    tMotherType1 = kPDGXiC;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------




  //ALamKchM-------------------------------------------------------------------------------
  case kResASig0KchM:
    tMotherType1 = kPDGASigma;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResAXi0KchM:
    tMotherType1 = kPDGAXi0;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResAXiCKchM:
    tMotherType1 = kPDGAXiC;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResAOmegaKchM:
    tMotherType1 = kPDGAOmega;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResASigStMKchM:
    tMotherType1 = kPDGASigStM;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResASigStPKchM:
    tMotherType1 = kPDGASigStP;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResASigSt0KchM:
    tMotherType1 = kPDGASigSt0;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResALamAKSt0:
    tMotherType1 = kPDGALam;
    tMotherType2 = kPDGAKSt0;
    break;
  //-------------
  case kResASig0AKSt0:
    tMotherType1 = kPDGASigma;
    tMotherType2 = kPDGAKSt0;
    break;
  //-------------
  case kResAXi0AKSt0:
    tMotherType1 = kPDGAXi0;
    tMotherType2 = kPDGAKSt0;
    break;
  //-------------
  case kResAXiCAKSt0:
    tMotherType1 = kPDGAXiC;
    tMotherType2 = kPDGAKSt0;
    break;
  //-------------

  //LamKchM-------------------------------------------------------------------------------
  case kResSig0KchM:
    tMotherType1 = kPDGSigma;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResXi0KchM:
    tMotherType1 = kPDGXi0;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResXiCKchM:
    tMotherType1 = kPDGXiC;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResOmegaKchM:
    tMotherType1 = kPDGOmega;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResSigStPKchM:
    tMotherType1 = kPDGSigStP;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResSigStMKchM:
    tMotherType1 = kPDGSigStM;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResSigSt0KchM:
    tMotherType1 = kPDGSigSt0;
    tMotherType2 = kPDGKchM;
    break;
  //-------------
  case kResLamAKSt0:
    tMotherType1 = kPDGLam;
    tMotherType2 = kPDGAKSt0;
    break;
  //-------------
  case kResSig0AKSt0:
    tMotherType1 = kPDGSigma;
    tMotherType2 = kPDGAKSt0;
    break;
  //-------------
  case kResXi0AKSt0:
    tMotherType1 = kPDGXi0;
    tMotherType2 = kPDGAKSt0;
    break;
  //-------------
  case kResXiCAKSt0:
    tMotherType1 = kPDGXiC;
    tMotherType2 = kPDGAKSt0;
    break;
  //-------------


  //ALamKchP-------------------------------------------------------------------------------
  case kResASig0KchP:
    tMotherType1 = kPDGASigma;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResAXi0KchP:
    tMotherType1 = kPDGAXi0;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResAXiCKchP:
    tMotherType1 = kPDGAXiC;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResAOmegaKchP:
    tMotherType1 = kPDGAOmega;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResASigStMKchP:
    tMotherType1 = kPDGASigStM;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResASigStPKchP:
    tMotherType1 = kPDGASigStP;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResASigSt0KchP:
    tMotherType1 = kPDGASigSt0;
    tMotherType2 = kPDGKchP;
    break;
  //-------------
  case kResALamKSt0:
    tMotherType1 = kPDGALam;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResASig0KSt0:
    tMotherType1 = kPDGASigma;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResAXi0KSt0:
    tMotherType1 = kPDGAXi0;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResAXiCKSt0:
    tMotherType1 = kPDGAXiC;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------


  //LamK0-------------------------------------------------------------------------------
  case kResSig0K0:
    tMotherType1 = kPDGSigma;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResXi0K0:
    tMotherType1 = kPDGXi0;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResXiCK0:
    tMotherType1 = kPDGXiC;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResOmegaK0:
    tMotherType1 = kPDGOmega;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResSigStPK0:
    tMotherType1 = kPDGSigStP;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResSigStMK0:
    tMotherType1 = kPDGSigStM;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResSigSt0K0:
    tMotherType1 = kPDGSigSt0;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResLamKSt0ToLamK0:
    tMotherType1 = kPDGLam;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResSig0KSt0ToLamK0:
    tMotherType1 = kPDGSigma;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResXi0KSt0ToLamK0:
    tMotherType1 = kPDGXi0;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResXiCKSt0ToLamK0:
    tMotherType1 = kPDGXiC;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------


  //ALamK0-------------------------------------------------------------------------------
  case kResASig0K0:
    tMotherType1 = kPDGASigma;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResAXi0K0:
    tMotherType1 = kPDGAXi0;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResAXiCK0:
    tMotherType1 = kPDGAXiC;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResAOmegaK0:
    tMotherType1 = kPDGAOmega;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResASigStMK0:
    tMotherType1 = kPDGASigStM;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResASigStPK0:
    tMotherType1 = kPDGASigStP;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResASigSt0K0:
    tMotherType1 = kPDGASigSt0;
    tMotherType2 = kPDGK0;
    break;
  //-------------
  case kResALamKSt0ToALamK0:
    tMotherType1 = kPDGALam;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResASig0KSt0ToALamK0:
    tMotherType1 = kPDGASigma;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResAXi0KSt0ToALamK0:
    tMotherType1 = kPDGAXi0;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------
  case kResAXiCKSt0ToALamK0:
    tMotherType1 = kPDGAXiC;
    tMotherType2 = kPDGKSt0;
    break;
  //-------------

  default:
    cout << "ERROR: Types::GetResidualDaughtersAndMothers:  aResidualType = " << aResidualType << " is not appropriate" << endl << endl;
    assert(0);
  }

  tReturnVec[0] = tMotherType1;
  tReturnVec[1] = tDaughterType1;
  tReturnVec[2] = tMotherType2;
  tReturnVec[3] = tDaughterType2;

  return tReturnVec;
}


