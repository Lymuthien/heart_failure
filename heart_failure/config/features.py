from heart_failure.config.config import (
    SEX,
    RESTING_ECG,
    CHEST_PAIN_TYPE,
    ST_SLOPE,
    AGE,
    CHOLESTEROL,
    MAX_HR,
    FASTING_BS,
    EXERCISE_ANGINA,
    OLDPEAK,
)

# Conjunctive_rules feature
CONJUNCTIVE_RULES = "conjunctive_rules"
CONJ_RULES = {
    "Oldpeak ge 2": lambda X: X[OLDPEAK] >= 2,
    "MaxHR le 160": lambda X: X[MAX_HR] <= 160,
    "MaxHR le 150": lambda X: X[MAX_HR] <= 150,
    "Age ge 50": lambda X: X[AGE] >= 50,
    "Age ge 60": lambda X: X[AGE] >= 60,
    "ChestPain eq ASY": lambda X: X[CHEST_PAIN_TYPE] == "ASY",
    "ExerciseAngina eq Y": lambda X: X[EXERCISE_ANGINA] == "Y",
    "FastingBS eq 1": lambda X: X[FASTING_BS] == 1,
    "ST_Slope eq Down": lambda X: X[ST_SLOPE] == "Down",
    "ST_Slope eq Flat": lambda X: X[ST_SLOPE] == "Flat",
}
CONJ_MIN_MASK_COUNT = 50
CONJ_MIN_TARGET_RATE = 0.85

# Preprocessing step
TE_FEATURES = [RESTING_ECG, CHEST_PAIN_TYPE, ST_SLOPE]
BINARY_CAT_FEATURES = [SEX, FASTING_BS, EXERCISE_ANGINA]
F_BINS = {AGE: [0, 43, 50, 60, 65, 80], CHOLESTEROL: [0, 200, 220, 240, 270, 1000]}

# Categorical features
CAT_FEATURES = [
    SEX,
    CHEST_PAIN_TYPE,
    RESTING_ECG,
    EXERCISE_ANGINA,
    FASTING_BS,
    ST_SLOPE,
]

# Cross-features target encoding
TE_FEATURE_CONFIG = [
    {
        "name": "oldpeak_cpt_te",
        "features": [OLDPEAK, CHEST_PAIN_TYPE],
        "bins": {OLDPEAK: 4},
        "merge_rules": {
            "0__ASY": "0",
            "0__NAP": "0",
            "0__TA": "0",
            "1__TA": "123_TA",
            "2__TA": "123_TA",
            "3__TA": "123_TA",
            "2__ATA": "23_ATA",
            "3__ATA": "23_ATA",
        },
    },
    {
        "name": "st_te",
        "features": [OLDPEAK, CHOLESTEROL, ST_SLOPE],
        "bins": {OLDPEAK: 4, CHOLESTEROL: 3},
        "merge_rules": None,
    },
    {
        "name": "te_by_max_hr",
        "features": [MAX_HR, RESTING_ECG],
        "bins": {MAX_HR: 4},
        "merge_rules": None,
    },
    {
        "name": "sex_te",
        "features": [AGE, CHOLESTEROL, SEX],
        "bins": {AGE: 5, CHOLESTEROL: 3},
        "merge_rules": None
    },
    {
        "name": "sex_cpt_te",
        "features": [CHEST_PAIN_TYPE, SEX],
        "bins": None,
        "merge_rules": None,
    },
    {
        "name": "sex_cat_te",
        "features": [ST_SLOPE, SEX],
        "bins": None,
        "merge_rules": {
            "Down__F": "Down",
            "Down__M": "Down",
        },
    },
    {
        "name": "recg_cpt_te",
        "features": [CHEST_PAIN_TYPE, RESTING_ECG],
        "bins": None,
        "merge_rules": None,
    },
    {
        "name": "recg_stsl_te",
        "features": [ST_SLOPE, RESTING_ECG],
        "bins": None,
        "merge_rules": None,
    },
    {
        "name": "recg_sex_te",
        "features": [SEX, RESTING_ECG],
        "bins": None,
        "merge_rules": {
            "NAP__ST": "NAP-TA_ST",
            "TA__ST": "NAP-TA_ST",
        },
    },
    {
        "name": "oldpeak_recg_te",
        "features": [OLDPEAK, RESTING_ECG],
        "bins": {OLDPEAK: 4},
        "merge_rules": {"0__LVH": "0", "0__Normal": "0", "0__ST": "0"},
    },
]
