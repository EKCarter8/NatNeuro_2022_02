from itertools import combinations
import pandas as pd
import numpy as np

from scipy import stats
import scikit_posthocs as sp

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


##############################################################################
### Read in Clean Data
fin = 'BlueColumns.csv'
print(fin)
BlueCols = pd.read_csv(fin, sep=',', engine='python', header=0, index_col=0)
print('Size of file read in: N Gene =', BlueCols.shape[0], 'N samples =', BlueCols.shape[1])
# Filter for 50% missing
cleanDat = BlueCols.dropna(thresh=BlueCols.shape[1]/2)
print('Size of data post 50% missing: N Gene =', cleanDat.shape[0], 'N samples =', cleanDat.shape[1])

## If needed, transpose the cleanDat so that the rows are sample and genes are columns
cleanDat = cleanDat.T #rows as samples, genes as columns
print('')

### Read in Traits File
fin = 'GroupInfo.csv'
print(fin)
traits = pd.read_csv(fin, sep=',', engine='python', header=0, index_col=0)

test = cleanDat.index.tolist() == traits.index.tolist()
if test == True:
    print("SampleIDs Match")
else:
    print("SampleIDs DO NOT MATCH")

### Traits file should have a column entitled "Group".
### If it does not, this needs to be made. You can do this within Python or within Excel
if 'Group' not in traits.columns.tolist():
    print('Must Define Group for ANOVA Analysis')
print('')
##############################################################################

##############################################################################
### Select Features to Calcualte ANOVA on.
idxFeatures = cleanDat.columns.tolist()
##############################################################################

##############################################################################
### Use Group Assignments from above to reorganize data
### This will make a dataframe for each defined group
GroupDict = dict(zip(traits.index.tolist(), traits['Group'].tolist()))
by_group = cleanDat.groupby(GroupDict)

values = {}
for k, v in by_group:
    print('Group', k, ': N Samples =', v.shape[0], 'N Genes =', v.shape[1])
    values.update({k: v[idxFeatures]})
##############################################################################


##############################################################################
## Calculate fold change between all combinations of Group variables
groupCombos = combinations(by_group.groups.keys(), 2)
comboList = [sorted(combo) for combo in groupCombos]

results = {}
for combo in comboList:
    print('Running Fold Change Calculation on:', 'Reference =', combo[0], '& Test =', combo[1])
    # Arbitrary: Use first item in combo as reference
    # This can be flipped later by multiply the result by -1 for visualization
    reference = combo[0]
    test = combo[1]

    reference_avg = by_group.get_group(reference).loc[:, :].mean()
    test_avg = by_group.get_group(test).loc[:, :].mean()

    key = "{}-{}.FoldChange".format(combo[0], combo[1])
    # Your cleanDat has already been log2 transformed
    # Therefore, you do not divide the means you subtract: log2(x/y) = log2(x) - log2(y)
    results[key] = test_avg - reference_avg

fold_change_cols = list(results.keys())
##############################################################################

##############################################################################
### Calculate ANOVA
## Use this later for p-value adjustments, this will add the group assignments to the cleanDat
df = cleanDat.merge(traits['Group'], left_index=True, right_index=True)

fvalues = []
pvalues = []
adj_pvalues = {}
for idx in idxFeatures:
    print(idx)

    tmp = []
    for key, value in values.items():
        tmp.append(value.loc[:, idx].dropna())

    ### One Way ANOVA with stats
    fvalue, pvalue = stats.f_oneway(*tmp)
    fvalues.append(fvalue)
    pvalues.append(pvalue)

    ### Remove NaN's from calculation to make it run property
    keepIDX = cleanDat.loc[:, idx].dropna().index.tolist()

    ### Run posthoc ttest  with p_adjust='holm'
    # By Default pool_sd = False
    apv = sp.posthoc_ttest(df.loc[keepIDX], val_col=idx, group_col='Group', p_adjust='holm')

    adj_pvalues.update({idx: apv})
##############################################################################

##############################################################################
### Append results to Dataframe
results['f.value'] = fvalues  ## Confirmed same as f.value from ANOVA in R
results['p.value'] = pvalues  # Confirmed sam as p.value from ANOVA in R
results_df = pd.DataFrame(results)

for combo in comboList:
    per_combo = {}

    for idx in results_df.index.tolist():
        pv = adj_pvalues[idx].loc[tuple(combo)]
        per_combo.update({idx: pv})

    colName = 'Holm Adjusted pvalule {0} to {1}'.format(combo[0], combo[1])
    results_df[colName] = results_df.index.map(per_combo)
##############################################################################

##############################################################################
### Check for number of significant genes in earch combination
check = []
for col in results_df.columns.tolist():
    print(results_df.loc[results_df[col] <= 0.05].shape[0])
    check.append(results_df.loc[results_df[col] <= 0.05].shape[0])

results_df.loc['pValue Check <0.05'] = check
##############################################################################

##############################################################################
### Save Results
results_df.to_csv('ANOVA_Results_HolmAdjusted.csv')
##############################################################################

results_df = results_df.drop(['pValue Check <0.05'])

x = 'AD-Control.FoldChange'
y = 'Holm Adjusted pvalule AD to Control'
print(x, y)

results_df['cmap'] = 'white'
results_df.loc[((results_df[y] <= 0.05) & (results_df[x] < 0)), 'cmap'] = 'red' #(1, 0, 0, 0.75)
results_df.loc[((results_df[y] <= 0.05) & (results_df[x] > 0)), 'cmap'] = 'blue' #'(0, 0, 1, 0.75)
count = results_df.groupby('cmap').count()['AD-Control.FoldChange']

xx = results_df[x] * -1
yy = -1 * np.log10(results_df[y])

fig, ax = plt.subplots(figsize=(8, 10))
ax.scatter(x=xx,
           y=yy,
           s=yy * 2,
           c=results_df['cmap'].tolist(),
           edgecolors='gray',
           linewidths=0.1
           )

ax.text(0.5, (-1 * np.log10(0.05) + 2),  'n = {}'.format(count['red']),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="r", lw=2, alpha=0.5))
ax.text(-0.5, (-1 * np.log10(0.05) + 2), 'n = {}'.format(count['blue']),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2, alpha=0.5))

title = ('Control vs. AD \n N = {}'.format(count['blue'] + count['red']))

plt.axhline(y=-1 * np.log10(0.05), color='r', linestyle='dashed')
plt.axvline(x=0.05, color='grey', linewidth=1.0, linestyle='dashed')
plt.axvline(x=-0.05, color='grey', linewidth=1.0, linestyle='dashed')
plt.axhline(y=0, color='grey', linewidth=0.5, linestyle='-')
plt.axvline(x=0, color='grey', linewidth=0.5, linestyle='-')

plt.xlim([-.6, 1.75])

plt.xlabel("log2(Fold Change)")
plt.ylabel("-log10(pValue) Holm Corrected")
plt.title(title)
plt.savefig('{}.pdf'.format(x.split('.')[0]), bbox_inches='tight', pad_inches=0.1, transparent=True)
