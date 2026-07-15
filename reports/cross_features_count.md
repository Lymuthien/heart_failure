# Cross-features count & differences

I look at target encoded combinations of features.

## Oldpeak & ChestPainType

| Oldpeak_bin | ChestPainType | feature_mean | count |
|-------------|---------------|--------------|-------|
| 0.0         | ASY           | 1.000000     | **5** |
|             | NAP           | 0.344827     | **3** |
|             | TA            | 0.552326     | **1** |
| 1.0         | ASY           | 0.588744     | 114   |
|             | ATA           | 0.073148     | 98    |
|             | NAP           | 0.253064     | 85    |
|             | TA            | 0.448892     | 14    |
| 2.0         | ASY           | 0.850543     | 82    |
|             | ATA           | 0.356720     | 17    |
|             | NAP           | 0.425994     | 26    |
|             | TA            | 0.278459     | 8     |
| 3.0         | ASY           | 0.927780     | 137   |
|             | ATA           | 0.404056     | 9     |
|             | NAP           | 0.659088     | 35    |
|             | TA            | 0.489864     | 11    |

The following should be combined (due to small sample size):
- Oldpeak_bin=0;
- CPT=TA, Oldpeak_bin=1 & Oldpeak_bin=2 & Oldpeak_bin=3;
- CPT=ATA, Oldpeak_bin=2 & Oldpeak_bin=3;

The percentage varies across bins and category values, so everything is fine here.

## Oldpeak & Cholesterol & ST_Slope

| Oldpeak_bin | Cholesterol_bin | ST_Slope | feature_mean | count |
|-------------|-----------------|----------|--------------|-------|
| 0.0         | 0.0             | Up       | 0.552326     | **1** |
|             | 1.0             | Down     | 1.000000     | **2** |
|             |                 | Flat     | 1.000000     | **3** |
|             |                 | Up       | 0.678380     | **3** |
| 1.0         | 0.0             | Flat     | 0.805583     | 24    |
|             |                 | Up       | 0.051394     | 101   |
|             | 1.0             | Flat     | 0.926618     | 44    |
|             |                 | Up       | 0.167785     | 47    |
|             | 2.0             | Flat     | 0.750606     | 24    |
|             |                 | Up       | 0.143171     | 71    |
| 2.0         | 0.0             | Down     | 0.552326     | **1** |
|             |                 | Flat     | 0.689874     | 20    |
|             |                 | Up       | 0.325988     | **9** |
|             | 1.0             | Down     | 0.603989     | **5** |
|             |                 | Flat     | 0.785718     | 38    |
|             |                 | Up       | 0.634950     | 16    |
|             | 2.0             | Down     | 0.500000     | **2** |
|             |                 | Flat     | 0.794264     | 34    |
|             |                 | Up       | 0.161205     | **8** |
| 3.0         | 0.0             | Down     | 1.000000     | 10    |
|             |                 | Flat     | 0.773192     | 41    |
|             |                 | Up       | 0.464373     | **7** |
|             | 1.0             | Down     | 0.765628     | **9** |
|             |                 | Flat     | 0.886157     | 37    |
|             |                 | Up       | 0.452588     | 12    |
|             | 2.0             | Down     | 0.834708     | 11    |
|             |                 | Flat     | 0.929268     | 60    |
|             |                 | Up       | 0.143700     | **5** |

The following should be combined (due to small sample size):
- Oldpeak_bin=0;
- Oldpeak_bin=2, ST_Slope=Down (ttest pvalue 0.6365: Ch=1 & Ch=2);
- Cholesterol_bin=2, ST_Slope=Up, Oldpeak_bin=2 & Oldpeak_bin=3 (ttest pvalue 0.1749);
- Oldpeak_bin=3, ST_Slope=Down, Cholesterol_bin=1 & Cholesterol_bin=2;
- Oldpeak_bin=3, ST_Slope=Up, Cholesterol_bin=0 & Cholesterol_bin=1 (ttest pvalue 0.6118);

Feature means vary between different pairs.

## Age & Cholesterol & Sex

| Age_bin | Cholesterol_bin | Sex | feature_mean | count |
|---------|-----------------|-----|--------------|-------|
| 0.0     | 0.0             | F   | 0.071988     | 18    |
|         |                 | M   | 0.282916     | 33    |
|         | 1.0             | F   | 0.287861     | **4** |
|         |                 | M   | 0.500514     | 30    |
|         | 2.0             | F   | 0.213094     | **5** |
|         |                 | M   | 0.441761     | 32    |
| 1.0     | 0.0             | F   | 0.268086     | 13    |
|         |                 | M   | 0.393242     | 27    |
|         | 1.0             | F   | 0.219630     | 11    |
|         |                 | M   | 0.634011     | 30    |
|         | 2.0             | F   | 0.318094     | 10    |
|         |                 | M   | 0.735630     | 25    |
| 2.0     | 0.0             | F   | 0.000000     | **8** |
|         |                 | M   | 0.457842     | 43    |
|         | 1.0             | F   | 0.000000     | **4** |
|         |                 | M   | 0.790839     | 43    |
|         | 2.0             | F   | 0.317786     | 17    |
|         |                 | M   | 0.595870     | 31    |
| 3.0     | 0.0             | F   | 0.287861     | **4** |
|         |                 | M   | 0.599836     | 33    |
|         | 1.0             | F   | 0.678160     | **3** |
|         |                 | M   | 0.811861     | 43    |
|         | 2.0             | F   | 0.415344     | 13    |
|         |                 | M   | 0.749181     | 33    |
| 4.0     | 0.0             | F   | 0.219538     | **6** |
|         |                 | M   | 0.719322     | 29    |
|         | 1.0             | F   | 0.552326     | **2** |
|         |                 | M   | 0.784737     | 46    |
|         | 2.0             | F   | 0.452348     | 18    |
|         |                 | M   | 0.863783     | 31    |

The following should be combined:
- Age_bin=0, Sex=F, Cholesterol_bin=1 & Cholesterol_bin=2 (0.8901);
- Age_bin=2, Sex=F, Cholesterol_bin=0 & Cholesterol_bin=1 (0.9323);
- Cholesterol_bin=0, Sex=F, Age_bin=3 & Age_bin=4 (0.8504);
- Cholesterol_bin=1, Sex=F, Age_bin=3 & Age_bin=4 (0.4835);

Feature means are generally different between groups.

## ChestPainType & Sex

| ChestPainType | Sex | feature_mean | count |
|---------------|-----|--------------|-------|
| ASY           | F   | 0.564625     | 48    |
|               | M   | 0.833758     | 290   |
| ATA           | F   | 0.050082     | 43    |
|               | M   | 0.183085     | 81    |
| NAP           | F   | 0.145764     | 38    |
|               | M   | 0.458770     | 111   |
| TA            | F   | 0.097204     | 7     |
|               | M   | 0.501646     | 27    |

Combine:
- Sex=F, ChestPainType=TA & ChestPainType=ATA (ttest pvalue p=0.3466);

Other percentages differ.

## ST_Slope & Sex

| ST_Slope | Sex | feature_mean | count |
|----------|-----|--------------|-------|
| Down     | F   | 1.000000     | **3** |
|          | M   | 0.781202     | 37    |
| Flat     | F   | 0.538257     | 52    |
|          | M   | 0.896987     | 273   |
| Up       | F   | 0.053137     | 81    |
|          | M   | 0.238252     | 199   |

Should be combined:
- ST_Slope=Down (although pvalue=6.996e-07 << 0.05, there are too few examples, there is no other way)

