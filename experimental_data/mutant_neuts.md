# Neutralization assays of Moderna and convalescent (HAARVI) sera versus WT and mutant spike-pseudotyped lentiviruses

Andrea, Lauren, and I set up these neuts.

The data that are analyzed in this notebook were pre-processed by Kate's `excel_to_fracinfect` script. 

### Import modules


```python
import collections
import itertools
import math
import os
import re
import string
import warnings
import xml.etree.ElementTree as ElementTree

from IPython.display import display, HTML
from IPython.display import display, SVG
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as plt

import natsort
import numpy as np
import pandas as pd
from plotnine import *

import neutcurve
from neutcurve.colorschemes import CBMARKERS, CBPALETTE
import seaborn

import yaml
```


```python
warnings.simplefilter('ignore')
```


```python
plt.style.use('seaborn-white')
theme_set(theme_seaborn(style='white', context='talk', font='FreeSans', font_scale=1))
```

### Create results directory


```python
resultsdir='results/mutant_neuts_results/'
os.makedirs(resultsdir, exist_ok=True)
```

Read config file


```python
with open('mutant_neuts_config.yaml') as f:
    config = yaml.safe_load(f)

with open('../config.yaml') as f:
    global_config = yaml.safe_load(f)
```

### Read in data
* Concatenate the `frac_infectivity` files.
* Remove samples specified in config file. 
* Also remove specified sample / dilution combinations where something has clearly gone wrong experimentally (like forgetting to add virus or cells). 
* Replace `serum` with `display_name`


```python
frac_infect = pd.DataFrame() # create empty data frame

if config['neut_samples_ignore']:
    neut_samples_ignore = config['neut_samples_ignore']

for f in config['neut_input_files'].keys():
    df = (pd.read_csv(f, index_col=0).assign(date=config['neut_input_files'][f]))
    frac_infect = frac_infect.append(df).reset_index(drop=True)

print(f"Length before dropping anything = {len(frac_infect.index)}")
    
frac_infect = (frac_infect
        .reset_index(drop=True)
       )

print(f"Length after dropping neut_samples_ignore = {len(frac_infect.index)}")

# below deals with samples / dates/ dilutions to ignore, which are currently none

for virus in config['neut_ignore_viruses']:
    dat = config['neut_ignore_viruses'][virus]
    l = len((frac_infect[(frac_infect['virus'] == virus) & (frac_infect['date'] == dat)]))
    print(f"Dropping {l} rows")
    frac_infect = frac_infect.drop(frac_infect[((frac_infect['virus'] == virus) & (frac_infect['date'] == dat))].index)
    print(f"Length after dropping {virus}: {dat} = {len(frac_infect.index)}")


# change names .assign(wildtype=lambda x: x['site'].map(site_to_wt),    
print(config['validated_samples'])
    
frac_infect = frac_infect.assign(serum=lambda x: x['serum'].map(config['validated_samples']))
frac_infect.head(2)
```

    Length before dropping anything = 1984
    Length after dropping neut_samples_ignore = 1984
    Dropping 128 rows
    Length after dropping E484K: 210308 = 1856
    {'M06-day-119': 'M06 (day 119)', 'M11-day-119': 'M11 (day 119)', 'M05-day-119': 'M05 (day 119)', 'M03-day-119': 'M03 (day 119)', 'M12-day-119': 'M12 (day 119)', 'M14-day-119': 'M14 (day 119)', '24C_d104': 'subject C (day 104)', '22C_d104': 'subject E (day 104)', '23C_d102': 'subject I (day 102)', '1C_d113': 'subject B (day 113)', '23_d120': 'subject A (day 120)', '25_d94': 'subject G (day 94)'}





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>serum</th>
      <th>virus</th>
      <th>replicate</th>
      <th>concentration</th>
      <th>fraction infectivity</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M03 (day 119)</td>
      <td>WT</td>
      <td>1</td>
      <td>0.04000</td>
      <td>-0.000007</td>
      <td>210308</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M03 (day 119)</td>
      <td>WT</td>
      <td>1</td>
      <td>0.01333</td>
      <td>-0.000007</td>
      <td>210308</td>
    </tr>
  </tbody>
</table>
</div>



### Use `neutcurve` to fit Hill curves to data.
Get IC50 and calculate NT50. 
Determine if IC50 is bound.


```python
for d in frac_infect['date'].unique():
    print(d)
```

    210308
    210312



```python
fitparams = pd.DataFrame(columns=['serum', 'virus', 'ic50', 'NT50', 'ic50_bound', 'date'])

for d in frac_infect['date'].unique():
    fits = neutcurve.CurveFits(frac_infect.query('date==@d'))

    fp = (
        fits.fitParams()
        .assign(NT50=lambda x: 1/x['ic50'],
                date=d
               )
        .replace({'WT':'wildtype'})
        # get columns of interest
        [['serum', 'virus', 'ic50', 'NT50', 'ic50_bound', 'date']] 
        )

    # couldn't get lambda / conditional statement to work with assign, so try it here:
    fp['ic50_is_bound'] = fp['ic50_bound'].apply(lambda x: True if x!='interpolated' else False)
    fitparams=fitparams.append(fp, ignore_index=True)

fitparams.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>serum</th>
      <th>virus</th>
      <th>ic50</th>
      <th>NT50</th>
      <th>ic50_bound</th>
      <th>date</th>
      <th>ic50_is_bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M03 (day 119)</td>
      <td>wildtype</td>
      <td>0.000561</td>
      <td>1783.440839</td>
      <td>interpolated</td>
      <td>210308</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M03 (day 119)</td>
      <td>E484P</td>
      <td>0.003804</td>
      <td>262.872902</td>
      <td>interpolated</td>
      <td>210308</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M03 (day 119)</td>
      <td>G446V</td>
      <td>0.000847</td>
      <td>1180.832242</td>
      <td>interpolated</td>
      <td>210308</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M03 (day 119)</td>
      <td>K417N</td>
      <td>0.000555</td>
      <td>1801.081391</td>
      <td>interpolated</td>
      <td>210308</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M03 (day 119)</td>
      <td>K417N-G446V-E484K</td>
      <td>0.002573</td>
      <td>388.718850</td>
      <td>interpolated</td>
      <td>210308</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Calculate fold change for each mutant relative to wild type.


```python
fc = (
    fitparams
#     .query('virus != "wildtype"')
#     .rename(columns={'virus': 'mutant'})
    .merge(fitparams.query('virus == "wildtype"')
                    [['serum', 'ic50', 'date']]
                    .rename(columns={'ic50': 'wildtype_ic50'}),
           on=['serum', 'date'],
           how='left',
           validate='many_to_one',
           )
    .assign(fold_change=lambda x: x['ic50'] / x['wildtype_ic50'],
            log2_fold_change=lambda x: np.log(x['fold_change']) / np.log(2),
            sample_type=lambda x: x['serum'].map(config['sample_types'])
           )
    )

fc.to_csv(f'{resultsdir}/fitparams.csv', index=False)
fc.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>serum</th>
      <th>virus</th>
      <th>ic50</th>
      <th>NT50</th>
      <th>ic50_bound</th>
      <th>date</th>
      <th>ic50_is_bound</th>
      <th>wildtype_ic50</th>
      <th>fold_change</th>
      <th>log2_fold_change</th>
      <th>sample_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M03 (day 119)</td>
      <td>wildtype</td>
      <td>0.000561</td>
      <td>1783.440839</td>
      <td>interpolated</td>
      <td>210308</td>
      <td>False</td>
      <td>0.000561</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>vaccine</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M03 (day 119)</td>
      <td>E484P</td>
      <td>0.003804</td>
      <td>262.872902</td>
      <td>interpolated</td>
      <td>210308</td>
      <td>False</td>
      <td>0.000561</td>
      <td>6.784423</td>
      <td>2.762226</td>
      <td>vaccine</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M03 (day 119)</td>
      <td>G446V</td>
      <td>0.000847</td>
      <td>1180.832242</td>
      <td>interpolated</td>
      <td>210308</td>
      <td>False</td>
      <td>0.000561</td>
      <td>1.510325</td>
      <td>0.594859</td>
      <td>vaccine</td>
    </tr>
  </tbody>
</table>
</div>



### Plot the raw neutralization curves


```python
for d in frac_infect['date'].unique():
    fits = neutcurve.CurveFits(frac_infect.query('date==@d'))
    xlab= 'serum dilution'
    name= 'sera'

    fig, axes = fits.plotSera(xlabel=xlab,)

    plotfile = f'./{resultsdir}/{d}_mutant_neuts.pdf'
    print(f"Saving to {plotfile}")
    fig.savefig(plotfile, bbox_inches='tight')
```

    Saving to ./results/mutant_neuts_results//210308_mutant_neuts.pdf
    Saving to ./results/mutant_neuts_results//210312_mutant_neuts.pdf



    
![png](mutant_neuts_files/mutant_neuts_16_1.png)
    



    
![png](mutant_neuts_files/mutant_neuts_16_2.png)
    


## Get depletion NT50s and fold-change


```python
haarvi_samples = config["validated_samples"].values()
haarvi_depletions = (pd.read_csv(config['haarvi_depletions'])
                     [['serum','fold_change', 'NT50_post', 'post-depletion_ic50', 'post_ic50_bound']]
                     .drop_duplicates()
                     .rename(columns={'NT50_post':'NT50', 
                                      'post-depletion_ic50':'ic50', 
                                      'post_ic50_bound':'ic50_is_bound'})
                     .assign(log2_fold_change=lambda x: np.log(x['fold_change']) / np.log(2),
                             virus='RBD antibodies depleted'
                            )
                     .query('serum in @haarvi_samples')
                    )

moderna_depletions = (pd.read_csv(config['moderna_depletions'])
                      [['serum','fold_change', 'NT50_post', 'post-depletion_ic50', 'post_ic50_bound']]
                      .drop_duplicates()
                      .rename(columns={'NT50_post':'NT50', 
                                      'post-depletion_ic50':'ic50', 
                                      'post_ic50_bound':'ic50_is_bound'})
                      .assign(log2_fold_change=lambda x: np.log(x['fold_change']) / np.log(2),
                              serum=lambda x: x['serum'].map(config['validated_samples']),
                              virus='RBD antibodies depleted'
                             )
                      .dropna(subset=['serum'])
                     )

depletion_df = (pd.concat([moderna_depletions, haarvi_depletions], axis=0, ignore_index=True)
               )
display(HTML(depletion_df.to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>serum</th>
      <th>fold_change</th>
      <th>NT50</th>
      <th>ic50</th>
      <th>ic50_is_bound</th>
      <th>log2_fold_change</th>
      <th>virus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>M03 (day 119)</td>
      <td>44.120772</td>
      <td>25.000000</td>
      <td>0.040000</td>
      <td>True</td>
      <td>5.463386</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>M05 (day 119)</td>
      <td>70.659257</td>
      <td>25.000000</td>
      <td>0.040000</td>
      <td>True</td>
      <td>6.142807</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>M06 (day 119)</td>
      <td>13.299399</td>
      <td>72.471105</td>
      <td>0.013799</td>
      <td>False</td>
      <td>3.733289</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>M11 (day 119)</td>
      <td>51.592102</td>
      <td>25.000000</td>
      <td>0.040000</td>
      <td>True</td>
      <td>5.689078</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>M12 (day 119)</td>
      <td>67.771270</td>
      <td>25.000000</td>
      <td>0.040000</td>
      <td>True</td>
      <td>6.082602</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>M14 (day 119)</td>
      <td>52.573884</td>
      <td>53.834278</td>
      <td>0.018576</td>
      <td>False</td>
      <td>5.716274</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>subject A (day 120)</td>
      <td>43.058197</td>
      <td>20.000000</td>
      <td>0.050000</td>
      <td>True</td>
      <td>5.428216</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>14.243019</td>
      <td>20.000000</td>
      <td>0.050000</td>
      <td>True</td>
      <td>3.832183</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>subject C (day 104)</td>
      <td>69.152358</td>
      <td>20.000000</td>
      <td>0.050000</td>
      <td>True</td>
      <td>6.111707</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>subject E (day 104)</td>
      <td>35.318485</td>
      <td>20.000000</td>
      <td>0.050000</td>
      <td>True</td>
      <td>5.142352</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>subject G (day 94)</td>
      <td>14.653491</td>
      <td>20.000000</td>
      <td>0.050000</td>
      <td>True</td>
      <td>3.873172</td>
      <td>RBD antibodies depleted</td>
    </tr>
    <tr>
      <td>subject I (day 102)</td>
      <td>11.695651</td>
      <td>50.224569</td>
      <td>0.019911</td>
      <td>False</td>
      <td>3.547900</td>
      <td>RBD antibodies depleted</td>
    </tr>
  </tbody>
</table>


## Read in escape fractions


```python
escape_fracs_file = os.path.join('..', global_config['escape_fracs'])
escape_fracs = pd.read_csv(escape_fracs_file).query('library == "average"')

escape_metrics = [global_config['site_metric'], global_config['mut_metric']]

display(HTML(escape_fracs.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>selection</th>
      <th>library</th>
      <th>condition</th>
      <th>site</th>
      <th>label_site</th>
      <th>wildtype</th>
      <th>mutation</th>
      <th>protein_chain</th>
      <th>protein_site</th>
      <th>mut_escape_frac_epistasis_model</th>
      <th>mut_escape_frac_single_mut</th>
      <th>site_total_escape_frac_epistasis_model</th>
      <th>site_total_escape_frac_single_mut</th>
      <th>site_avg_escape_frac_epistasis_model</th>
      <th>site_avg_escape_frac_single_mut</th>
      <th>nlibs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>12C_d152_80</td>
      <td>average</td>
      <td>12C_d152_80</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>A</td>
      <td>E</td>
      <td>331</td>
      <td>0.002020</td>
      <td>0.001326</td>
      <td>0.04926</td>
      <td>0.0478</td>
      <td>0.003079</td>
      <td>0.002988</td>
      <td>2</td>
    </tr>
    <tr>
      <td>12C_d152_80</td>
      <td>average</td>
      <td>12C_d152_80</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>D</td>
      <td>E</td>
      <td>331</td>
      <td>0.005616</td>
      <td>0.000537</td>
      <td>0.04926</td>
      <td>0.0478</td>
      <td>0.003079</td>
      <td>0.002988</td>
      <td>1</td>
    </tr>
    <tr>
      <td>12C_d152_80</td>
      <td>average</td>
      <td>12C_d152_80</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>E</td>
      <td>E</td>
      <td>331</td>
      <td>0.002535</td>
      <td>0.000482</td>
      <td>0.04926</td>
      <td>0.0478</td>
      <td>0.003079</td>
      <td>0.002988</td>
      <td>2</td>
    </tr>
    <tr>
      <td>12C_d152_80</td>
      <td>average</td>
      <td>12C_d152_80</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>F</td>
      <td>E</td>
      <td>331</td>
      <td>0.003032</td>
      <td>0.005816</td>
      <td>0.04926</td>
      <td>0.0478</td>
      <td>0.003079</td>
      <td>0.002988</td>
      <td>2</td>
    </tr>
    <tr>
      <td>12C_d152_80</td>
      <td>average</td>
      <td>12C_d152_80</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>G</td>
      <td>E</td>
      <td>331</td>
      <td>0.003113</td>
      <td>0.001273</td>
      <td>0.04926</td>
      <td>0.0478</td>
      <td>0.003079</td>
      <td>0.002988</td>
      <td>2</td>
    </tr>
  </tbody>
</table>



```python
print(config['map_conditions'])
```

    {'M06-day-119_80': 'M06 (day 119)', 'M11-day-119_200': 'M11 (day 119)', 'M05-day-119_500': 'M05 (day 119)', 'M03-day-119_200': 'M03 (day 119)', 'M12-day-119_200': 'M12 (day 119)', 'M14-day-119_500': 'M14 (day 119)', '24C_d104_200': 'subject C (day 104)', '22C_d104_200': 'subject E (day 104)', '23C_d102_80': 'subject I (day 102)', '1C_d113_200': 'subject B (day 113)', '23_d120_500': 'subject A (day 120)', '25_d94_200': 'subject G (day 94)'}



```python
samples = config['map_conditions'].values()
escape_fracs_df = (escape_fracs
                   .replace(config['map_conditions'])
                   .query('selection in @samples')
                   [['selection', 'label_site', 'wildtype', 'mutation', global_config['site_metric'], global_config['mut_metric']]]
                   .rename(columns={'selection':'serum', 
                                    'label_site': 'site', 
                                    global_config['mut_metric']:'mutation escape',
                                    global_config['site_metric']:'site total escape',
                                   }
                          )
                   .assign(virus=lambda x: x['wildtype'] + x['site'].astype(str) + x['mutation'],
                           site_label=lambda x: x['wildtype'] + x['site'].astype(str),
                           site=lambda x: x['site'].astype(str)
                          )
                   .drop(columns=['wildtype', 'mutation', 'site_label', 'site'])
                  )

display(HTML(escape_fracs_df.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>serum</th>
      <th>site total escape</th>
      <th>mutation escape</th>
      <th>virus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>subject B (day 113)</td>
      <td>0.2028</td>
      <td>0.01312</td>
      <td>N331A</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>0.2028</td>
      <td>0.02020</td>
      <td>N331D</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>0.2028</td>
      <td>0.01226</td>
      <td>N331E</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>0.2028</td>
      <td>0.01330</td>
      <td>N331F</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>0.2028</td>
      <td>0.01373</td>
      <td>N331G</td>
    </tr>
  </tbody>
</table>


## Make plot showing NT50 for each genotype (wildtype or mutant) for each serum.
This is actually an important thing we should probably add to the paper. 
Even though the NT50s decrease by ~10-fold or more for some sera against some mutants, the absolute NT50 remaining might still be quite potent. 
It would be nice to have this clearly shown.


```python
muts_depletions = (pd.concat([depletion_df, fc], axis=0, ignore_index=True)
                   .assign(sample_type=lambda x: x['serum'].map(config['sample_types']))
                   .merge(escape_fracs_df,
                          how='left', 
                          on=['serum', 'virus']
                         )
                   .assign(virus=lambda x: pd.Categorical(x['virus'], categories=config['viruses'], ordered=True))
               )

display(HTML(muts_depletions.tail().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>serum</th>
      <th>fold_change</th>
      <th>NT50</th>
      <th>ic50</th>
      <th>ic50_is_bound</th>
      <th>log2_fold_change</th>
      <th>virus</th>
      <th>ic50_bound</th>
      <th>date</th>
      <th>wildtype_ic50</th>
      <th>sample_type</th>
      <th>site total escape</th>
      <th>mutation escape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>subject A (day 120)</td>
      <td>1.771949</td>
      <td>743.912663</td>
      <td>0.001344</td>
      <td>False</td>
      <td>0.825337</td>
      <td>L452R</td>
      <td>interpolated</td>
      <td>210312</td>
      <td>0.000759</td>
      <td>convalescent</td>
      <td>1.156</td>
      <td>0.1171</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>1.928334</td>
      <td>392.047104</td>
      <td>0.002551</td>
      <td>False</td>
      <td>0.947355</td>
      <td>F456A</td>
      <td>interpolated</td>
      <td>210312</td>
      <td>0.001323</td>
      <td>convalescent</td>
      <td>3.510</td>
      <td>0.4938</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>10.647124</td>
      <td>71.004888</td>
      <td>0.014084</td>
      <td>False</td>
      <td>3.412392</td>
      <td>E484K</td>
      <td>interpolated</td>
      <td>210312</td>
      <td>0.001323</td>
      <td>convalescent</td>
      <td>3.019</td>
      <td>0.1182</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>1.000000</td>
      <td>755.997863</td>
      <td>0.001323</td>
      <td>False</td>
      <td>0.000000</td>
      <td>wildtype</td>
      <td>interpolated</td>
      <td>210312</td>
      <td>0.001323</td>
      <td>convalescent</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>subject B (day 113)</td>
      <td>4.559898</td>
      <td>165.792731</td>
      <td>0.006032</td>
      <td>False</td>
      <td>2.189001</td>
      <td>L452R</td>
      <td>interpolated</td>
      <td>210312</td>
      <td>0.001323</td>
      <td>convalescent</td>
      <td>1.505</td>
      <td>0.1556</td>
    </tr>
  </tbody>
</table>



```python
muts_depletions['virus'].drop_duplicates()
```




    0     RBD antibodies depleted
    12                   wildtype
    13                      E484P
    14                      G446V
    15                      K417N
    16          K417N-G446V-E484K
    17                      P384R
    60                      L452R
    61                      F456A
    67                      E484K
    Name: virus, dtype: category
    Categories (10, object): ['wildtype' < 'P384R' < 'K417N' < 'F456A' ... 'E484K' < 'E484P' < 'K417N-G446V-E484K' < 'RBD antibodies depleted']




```python
p = (ggplot((muts_depletions)) +
     aes('virus', 'NT50', shape='ic50_is_bound', fill='site total escape') +
     geom_point(size=2.5, alpha=0.6) +
     scale_x_discrete(name='') +
     scale_y_log10(name='neutralization titer (NT50)', expand=(0.1,0.1)) +
     facet_wrap('~serum', ncol=6) +
     theme_classic() +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(0.2 * muts_depletions['virus'].nunique()*muts_depletions['serum'].nunique()/2, 2.5),
           strip_margin_y=0.35,
           strip_background_x=element_blank(),
           ) +
         geom_hline(yintercept=25, linetype='dashed', size=1,
                    alpha=0.6, color=CBPALETTE[0]) +
     scale_fill_gradient(low='white', high='#BD0026') +
     scale_shape_manual(values=['o','^'], name='below limit\nof detection')
#      scale_fill_manual(values=['gray', 'white'],name='below limit\nof detection')
#          +scale_color_manual(values=['gray', '#D55E00'],name='E484 mutant')
     )

_ = p.draw()

plotfile = f'{resultsdir}/all_neuts_NT50.pdf'
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to results/mutant_neuts_results//all_neuts_NT50.pdf



    
![png](mutant_neuts_files/mutant_neuts_26_1.png)
    


### Plot fold-change IC50 relative to wild type for each mutant.
You could also imagine drawing a dashed line with the fold-change with RBD depletion, which sets an upper limit on what we would expect to see (the max drop in NT50 we could see due to anything RBD-related). 

To do this you would need to:
* Read in foldchange IC50 due to RBD depletion (specify path in config file)
* Merge with mutant `foldchange` dataframe
* Add `geom_hline` with depletion foldchange


```python
serum_order = list(config['sample_types'].keys())
print(serum_order)
```

    ['M06 (day 119)', 'M11 (day 119)', 'M05 (day 119)', 'M03 (day 119)', 'M12 (day 119)', 'M14 (day 119)', 'subject C (day 104)', 'subject E (day 104)', 'subject I (day 102)', 'subject B (day 113)', 'subject A (day 120)', 'subject G (day 94)']



```python
muts_depletions = (muts_depletions
                   .assign(serum=lambda x: pd.Categorical(x['serum'],ordered=True,categories=serum_order))
                  )
    
p = (ggplot(muts_depletions
            .query("virus != 'RBD antibodies depleted' & virus != 'wildtype'")
            ) +
     aes('virus', 'fold_change', fill='site total escape', shape='ic50_is_bound',
        ) +
     geom_point(size=2.5, alpha=1) +
     scale_y_log10(name='fold change IC50') +
     facet_wrap('~serum', ncol=6) +
     theme_classic() +
     theme(axis_text_x=element_text(angle=90),
           axis_title_x=element_blank(),
           strip_margin_y=0.35,
           strip_background_x=element_blank(),
           figure_size=(0.2 * (muts_depletions['virus'].nunique()-1)*muts_depletions['serum'].nunique()/2, 4),
           ) +
     geom_hline(yintercept=1, linetype='dashed', size=1,
                alpha=0.6, color=CBPALETTE[0]) +
     geom_hline(data=muts_depletions.query('virus=="RBD antibodies depleted"'),
                mapping=aes(yintercept='fold_change'),
                color=CBPALETTE[1],
                alpha=0.7,
                size=1,
                linetype='dotted',
               ) +
     scale_fill_gradient(low='white', high='#BD0026') +
     scale_shape_manual(values=['o','^'], name='upper limit')
#      scale_color_manual(values=CBPALETTE[1:],
#                         name='upper limit') +
#      scale_fill_manual(values=['gray', 'white'],name='upper limit')
     )

_ = p.draw()

plotfile = f'{resultsdir}/fold_change_IC50.pdf'
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to results/mutant_neuts_results//fold_change_IC50.pdf



    
![png](mutant_neuts_files/mutant_neuts_29_1.png)
    



```python
p = (ggplot(muts_depletions
            .query("virus != 'RBD antibodies depleted' & virus != 'wildtype'")
            ) +
     aes('virus', 'fold_change', shape='ic50_is_bound', fill='site total escape'
        ) +
     geom_point(size=2.5, alpha=1) +
     scale_y_log10(name='fold change IC50') +
     facet_wrap('~serum', ncol=6) +
     theme_classic() +
     theme(axis_title_x=element_blank(),
           strip_margin_y=0.35,
           strip_background_x=element_blank(),
           figure_size=(0.2 * (muts_depletions['virus'].nunique()-1)*muts_depletions['serum'].nunique()/2, 4),
           ) +
     geom_hline(yintercept=1, linetype='dashed', size=1,
                alpha=0.6, color=CBPALETTE[0]) +
     geom_hline(data=muts_depletions.query('virus=="RBD antibodies depleted"'),
                mapping=aes(yintercept='fold_change'),
                color=CBPALETTE[1],
                alpha=0.7,
                size=1,
                linetype='dotted',
               ) +
     coord_flip()+
     scale_fill_gradient(low='white', high='#BD0026') +
     scale_shape_manual(values=['o','^'], name='upper limit')
#      scale_color_manual(values=CBPALETTE[1:],
#                         name='upper limit') +
#      scale_fill_manual(values=['gray', 'white'],name='upper limit')
     )

_ = p.draw()

plotfile = f'{resultsdir}/fold_change_IC50_rotated.pdf'
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to results/mutant_neuts_results//fold_change_IC50_rotated.pdf



    
![png](mutant_neuts_files/mutant_neuts_30_1.png)
    



```python
p = (ggplot(muts_depletions
            .query("virus != 'wildtype'")
            ) +
     aes('virus', 'fold_change', shape='ic50_is_bound', fill='site total escape'
        ) +
     geom_point(size=2.5, alpha=0.6) +
     scale_y_log10(name='fold change IC50') +
     facet_wrap('~sample_type', ncol=6) +
     theme_classic() +
     theme(axis_title_x=element_blank(),
           strip_margin_y=0.35,
           strip_background_x=element_blank(),
           figure_size=(4, 0.2 * (muts_depletions['virus'].nunique()-1)*muts_depletions['serum'].nunique()/12),
           ) +
     geom_hline(yintercept=1, linetype='dashed', size=1,
                alpha=0.6, color=CBPALETTE[0]) +
#      geom_hline(data=muts_depletions.query('virus=="RBD antibodies depleted"'),
#                 mapping=aes(yintercept='fold_change'),
#                 color=CBPALETTE[1],
#                 alpha=0.7,
#                 size=1,
#                 linetype='dotted',
#                ) +
     coord_flip() +
     scale_fill_gradient(low='white', high='#BD0026') +
     scale_shape_manual(values=['o','^'], name='upper limit')
#      scale_color_manual(values=CBPALETTE[1:],
#                         name='upper limit') +
#      scale_fill_manual(values=['gray', 'white'],name='upper limit')
     )

_ = p.draw()

plotfile = f'{resultsdir}/fold_change_IC50_rotated_grouped.pdf'
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to results/mutant_neuts_results//fold_change_IC50_rotated_grouped.pdf



    
![png](mutant_neuts_files/mutant_neuts_31_1.png)
    



```python
!jupyter nbconvert mutant_neuts.ipynb --to markdown
```