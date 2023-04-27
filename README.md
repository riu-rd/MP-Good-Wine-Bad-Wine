# MP - Good Wine Bad Wine
### CSMODEL Final Project
```
Each row in the dataset represents an observation of wine. Each column in the dataset represents the physicochemical test 
(e.g. pH Level, Alcohol) output for each observation, then it's respective quality which is rated by wine experts.

- There are 1599 observations for red wine, and 4898 observations for white wine
- The datasets will be combined and will have a total of 6497 rows and 13 columns Both datasets use the same 13 variables. 
  The first 12 variables are the physicochemical properties (inputs) of each wine and and the 13th is their respective quality

1. color - the most easily recognizable characteristics of wines, heavy wines generally have a deeper color
2. fixed acidity - most acids involved with wine or fixed or nonvolatile (do not evaporate readily)
3. volatile acidity - the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
4. citric acid - found in small quantities, citric acid can add ‘freshness’ and flavor to wines
5. residual sugar - the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 
   1 gram/liter and wines with greater than 45 grams/liter are considered sweet
6. chlorides - the amount of salt in the wine
7. free sulfur dioxide - the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) 
   and bisulfite ion; it prevents microbial growth and the oxidation of wine
8. total sulfur dioxide - amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, 
   but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
9. density - the density of water is close to that of water depending on the percent alcohol and sugar content
10. pH - describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
11. sulphates - a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant
12. alcohol - the percent alcohol content of the wine
13. quality - based on the rating of the wine experts, a score between 0 and 10
```
```
> The Jupyter notebook uses pandas, matlab, numpy, and other statistical libraries to analyze the datasets.
> kmeans.py is a python file containing the functions to perform the clustering via k-Means algorithm.
```
