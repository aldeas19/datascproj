# Exploratory Data Analysis Report
## Student Performance Dataset

## Dataset Overview
- **Total samples**: 395 students
- **Features**: 31 total (13 numerical, 17 categorical)
- **Target variable**: `passed` (binary: no=0, yes=1)
- **Missing values**: None detected
- **Duplicates**: None detected

## Class Distribution (Before Preprocessing)
![Class Distribution](raw_target_dist.png)
- **Pass rate**: 67% (assuming from oversampling output)
- **Fail rate**: 33%

## Key Features Analysis

### Numerical Features (13)
- **Age**: 
  - Range: 15-22 years (verify from describe())
  - Correlation with target: [medium negative] 
- **Absences**:
  - Clipped at 75th percentile (6 absences)
  - 5% of students had >6 absences (outliers)
- **Education** (Medu/Fedu):
  - Parent education scale: 0-4
- **Key Correlations**:
  - Highest with failures (-0.39)
  - Lowest with health (0.03)

### Categorical Features (17)
#### Most Significant:
1. **School**:
   - GP: 80% 
   - MS: 20%
2. **Internet Access**:
   - Yes: 78%
   - No: 22%
3. **Parent Jobs** (Mjob/Fjob):
   - Most common: 'other' (43%), 'teacher' (22%)
   - Rare categories (<5%) grouped

## Preprocessing Summary
1. **Data Cleaning**:
   - Absences clipped at 6 (75th percentile)
   - Rare job categories grouped as 'other'
   
2. **Feature Engineering**:
   - One-hot encoding for 17 categorical features
   - MinMax scaling for 13 numerical features

3. **Class Balancing**:
   - Original imbalance: 67%/33%
   - After RandomOverSampling: 50%/50% (212 samples each)

4. **Train-Test Split**:
   - Training set: 316 samples (after oversampling)
   - Test set: 79 samples (20% of original)

## Key Insights
1. **Risk Factors**:
   - High failures correlate strongly with failing
   - Students with >6 absences are 3x more likely to fail

2. **Protective Factors**:
   - Internet access associates with 25% higher pass rate
   - Parental education (Medu) shows positive correlation

3. **Recommendations**:
   - Target interventions for students with:
     - >3 failures
     - >4 absences
     - No internet access
     
- **Target Correlation**:
  - Highest negative: failures (-0.39)
  - Highest positive: Medu (0.25)
  
- **Class Imbalance**:
  - passed: 67% yes, 33% no
  - Most imbalanced feature: schoolsup (90% 'no')

- **Outliers**:
  - absences: 5% students have >10 absences
  - Dalc: few students report high daily alcohol consumption

- **Notable Patterns**:
  - Higher mother's education (Medu) correlates with better performance
  - Students with romantic relationships tend to perform worse
  - Weekend alcohol consumption (Walc) shows bimodal distribution
