# JEPA results

All JEPA trainings were done using 200k steps unless stated otherwise

## MLP Results 

### Infos & baselines

| **Domain**      | **Task**                 | **Type**    | **Metric**      | **Criterion**            | **Weight** | **dasheng** | **wav2vec2** | **whisper** | **data2vec** |
| :-------------: | :----------------------: | :---------: | :-------------: | :----------------------: | :--------: | :---------: | :----------: | :---------: | :----------: |
| **Environment** |                          |             |                 |                          | **7071**   | 0.480       | 0.265        | 0.270       | 0.148        |
|                 | Clotho                   | contrastive | recallatk_r1    | AudioTextContrastiveLoss | 1045       | 0.029       | 0.014        | 0.038       | 0.008        |
|                 | DESED                    | frame       | segmentf1_macro | BCEWithLogitsLoss        | 1153       | 0.537       | 0.313        | 0.127       | 0.136        |
|                 | ESC-50                   | clip        | accuracy        | CrossEntropyLoss         | 400        | 0.857       | 0.510        | 0.528       | 0.229        |
|                 | FSD18-Kaggle             | clip        | accuracy        | CrossEntropyLoss         | 1600       | 0.534       | 0.241        | 0.241       | 0.153        |
|                 | FSD50k                   | clip        | mAP             | BCEWithLogitsLoss        | 2000       | 0.409       | 0.166        | 0.262       | 0.085        |
|                 | UrbanSound 8k            | clip        | accuracy        | CrossEntropyLoss         | 873        | 0.833       | 0.659        | 0.687       | 0.426        |
| **Music**       |                          |             |                 |                          | **2965**   | 0.678       | 0.451        | 0.537       | 0.334        |
|                 | Free Music Archive Small | clip        | accuracy        | CrossEntropyLoss         | 800        | 0.643       | 0.469        | 0.581       | 0.334        |
|                 | GTZAN Genre              | clip        | accuracy        | CrossEntropyLoss         | 100        | 0.851       | 0.630        | 0.622       | 0.448        |
|                 | MAESTRO                  | frame       | segmentf1_micro | BCEWithLogitsLoss        | 65         | 0.524       | 0.180        | 0.011       | 0.116        |
|                 | NSynth-Instruments       | clip        | accuracy        | CrossEntropyLoss         | 2000       | 0.688       | 0.443        | 0.532       | 0.336        |
| **Speech**      |                          |             |                 |                          | **28716**  | 0.823       | 0.615        | 0.725       | 0.642        |
|                 | ASV2015                  | clip        | accuracy        | CrossEntropyLoss         | 2000       | 0.964       | 0.924        | 0.966       | 0.937        |
|                 | CREMA-D                  | clip        | accuracy        | CrossEntropyLoss         | 1116       | 0.767       | 0.541        | 0.572       | 0.523        |
|                 | Fluent Speech Commands   | clip        | accuracy        | CrossEntropyLoss         | 2000       | 0.946       | 0.468        | 0.776       | 0.978        |
|                 | LibriCount               | clip        | accuracy        | CrossEntropyLoss         | 1144       | 0.681       | 0.583        | 0.549       | 0.492        |
|                 | LibriSpeech-MF           | clip        | accuracy        | CrossEntropyLoss         | 2620       | 0.986       | 0.948        | 0.973       | 0.752        |
|                 | RAVDESS                  | clip        | accuracy        | CrossEntropyLoss         | 360        | 0.749       | 0.442        | 0.459       | 0.467        |
|                 | Speech Commands V1       | clip        | accuracy        | CrossEntropyLoss         | 2000       | 0.969       | 0.714        | 0.933       | 0.927        |
|                 | Vocal Imitation          | clip        | accuracy        | CrossEntropyLoss         | 1867       | 0.253       | 0.147        | 0.180       | 0.128        |
|                 | VocalSound               | clip        | accuracy        | CrossEntropyLoss         | 2000       | 0.910       | 0.768        | 0.860       | 0.803        |
|                 | VoxCeleb1                | clip        | accuracy        | CrossEntropyLoss         | 2000       | 0.780       | 0.340        | 0.388       | 0.105        |
|                 | VoxLingua33              | clip        | accuracy        | CrossEntropyLoss         | 1609       | 0.814       | 0.553        | 0.873       | 0.620        |
|                 | LibriSpeech-100h         | asr         | WER_inv         | WER_inv                  | 10000      | 0.608       | 0.405        | 0.721       | 0.893        |
| **Overall**     |                          |             |                 |                          | **38752**  | 0.699       | 0.490        | 0.632       | 0.598        |


| **Domain**      | **Task**                 | **Type**    | **Criterion**            | **Weight** | ****16kHz / 200k / wu5k / 160mels**** |
| :-------------: | :----------------------: | :---------: | :----------------------: | :--------: | :-----------------------------------: |
| **Environment** |                          |             |                          | 7071       | 0.221                                 |
|                 | Clotho                   | contrastive | AudioTextContrastiveLoss | 1045       | 0.010                                 |
|                 | DESED                    | frame       | BCEWithLogitsLoss        | 1153       | 0.184                                 |
|                 | ESC-50                   | clip        | CrossEntropyLoss         | 400        | 0.366                                 |
|                 | FSD18-Kaggle             | clip        | CrossEntropyLoss         | 1600       | 0.251                                 |
|                 | FSD50k                   | clip        | BCEWithLogitsLoss        | 2000       | 0.129                                 |
|                 | UrbanSound 8k            | clip        | CrossEntropyLoss         | 873        | 0.609                                 |
| **Music**       |                          |             |                          | 2965       | 0.458                                 |
|                 | Free Music Archive Small | clip        | CrossEntropyLoss         | 800        | 0.514                                 |
|                 | GTZAN Genre              | clip        | CrossEntropyLoss         | 100        | 0.591                                 |
|                 | MAESTRO                  | frame       | BCEWithLogitsLoss        | 65         | 0.168                                 |
|                 | NSynth-Instruments       | clip        | CrossEntropyLoss         | 2000       | 0.438                                 |
| **Speech**      |                          |             |                          | 28716      | 0.257                                 |
|                 | ASV2015                  | clip        | CrossEntropyLoss         | 2000       | 0.951                                 |
|                 | CREMA-D                  | clip        | CrossEntropyLoss         | 1116       | 0.409                                 |
|                 | Fluent Speech Commands   | clip        | CrossEntropyLoss         | 2000       | 0.055                                 |
|                 | LibriCount               | clip        | CrossEntropyLoss         | 1144       | 0.471                                 |
|                 | LibriSpeech-100h         | asr         | CrossEntropyLoss         | 10000      |                                       |
|                 | LibriSpeech-MF           | clip        | CrossEntropyLoss         | 2620       | 0.866                                 |
|                 | RAVDESS                  | clip        | CrossEntropyLoss         | 360        | 0.337                                 |
|                 | Speech Commands V1       | clip        | CrossEntropyLoss         | 2000       | 0.241                                 |
|                 | Vocal Imitation          | clip        | CrossEntropyLoss         | 1867       | 0.053                                 |
|                 | VocalSound               | clip        | CrossEntropyLoss         | 2000       | 0.591                                 |
|                 | VoxCeleb1                | clip        | CrossEntropyLoss         | 2000       | 0.040                                 |
|                 | VoxLingua33              | clip        | CrossEntropyLoss         | 1609       | 0.091                                 |
| Overall         |                          |             |                          | 38752      | 0.266                                 |

### 32kHz vs 16kHz

| **Domain**      | **Task**                 | **32kHz** | **16kHz** |
| --------------- | ------------------------ | :-------: | :-------: |
| **Environment** |                          | 0.252     | 0.308     |
|                 | Clotho                   | 0.012     | 0.016     |
|                 | DESED                    | 0.304     | 0.394     |
|                 | ESC-50                   | 0.341     | 0.520     |
|                 | FSD18-Kaggle             | 0.198     | 0.276     |
|                 | FSD50k                   | 0.151     | 0.204     |
|                 | UrbanSound 8k            | 0.585     | 0.742     |
| **Music**       |                          | 0.440     | 0.457     |
|                 | Free Music Archive Small | 0.551     | 0.496     |
|                 | GTZAN Genre              | 0.635     | 0.606     |
|                 | MAESTRO                  | 0.178     | 0.163     |
|                 | NSynth-Instruments       | 0.394     | 0.444     |
| **Speech**      |                          | 0.378     | 0.415     |
|                 | ASV2015                  | 0.923     | 0.974     |
|                 | CREMA-D                  | 0.430     | 0.457     |
|                 | Fluent Speech Commands   | 0.025     | 0.034     |
|                 | LibriCount               | 0.464     | 0.571     |
|                 | LibriSpeech-MF           | 0.887     | 0.904     |
|                 | RAVDESS                  | 0.306     | 0.353     |
|                 | Speech Commands V1       | 0.176     | 0.185     |
|                 | Vocal Imitation          | 0.056     | 0.109     |
|                 | VocalSound               | 0.526     | 0.607     |
|                 | VoxCeleb1                | 0.042     | 0.069     |
|                 | VoxLingua33              | 0.091     | 0.104     |
| **Overall**     |                          | 0.348     | 0.393     |


### Mels bands

| **Domain**      | **Task**                 | **64** | **96** | **128** | **160** | **192** | **224** | **256** |
| --------------- | ------------------------ | :----: | :----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| **Environment** |                          | 0.220  | 0.286  | 0.308   | 0.219   | 0.242   | 0.237   | 0.213   |
|                 | Clotho                   | 0.015  | 0.020  | 0.016   | 0.009   | 0.012   | 0.008   | 0.008   |
|                 | DESED                    | 0.284  | 0.339  | 0.394   | 0.179   | 0.292   | 0.230   | 0.216   |
|                 | ESC-50                   | 0.301  | 0.471  | 0.520   | 0.365   | 0.435   | 0.365   | 0.337   |
|                 | FSD18-Kaggle             | 0.181  | 0.289  | 0.276   | 0.251   | 0.226   | 0.251   | 0.217   |
|                 | FSD50k                   | 0.141  | 0.177  | 0.204   | 0.129   | 0.149   | 0.148   | 0.136   |
|                 | UrbanSound 8k            | 0.594  | 0.694  | 0.742   | 0.603   | 0.604   | 0.642   | 0.565   |
| **Music**       |                          | 0.425  | 0.419  | 0.457   | 0.458   | 0.430   | 0.446   | 0.437   |
|                 | Free Music Archive Small | 0.454  | 0.457  | 0.496   | 0.501   | 0.506   | 0.504   | 0.491   |
|                 | GTZAN Genre              | 0.533  | 0.559  | 0.606   | 0.593   | 0.604   | 0.614   | 0.662   |
|                 | MAESTRO                  | 0.162  | 0.207  | 0.163   | 0.167   | 0.162   | 0.201   | 0.171   |
|                 | NSynth-Instruments       | 0.416  | 0.404  | 0.444   | 0.444   | 0.400   | 0.423   | 0.413   |
| **Speech**      |                          | 0.383  | 0.408  | 0.415   | 0.396   | 0.390   | 0.402   | 0.392   |
|                 | ASV2015                  | 0.943  | 0.970  | 0.974   | 0.943   | 0.931   | 0.944   | 0.947   |
|                 | CREMA-D                  | 0.418  | 0.421  | 0.457   | 0.409   | 0.426   | 0.444   | 0.447   |
|                 | Fluent Speech Commands   | 0.029  | 0.039  | 0.034   | 0.057   | 0.049   | 0.031   | 0.026   |
|                 | LibriCount               | 0.434  | 0.445  | 0.571   | 0.471   | 0.490   | 0.493   | 0.479   |
|                 | LibriSpeech-MF           | 0.926  | 0.934  | 0.904   | 0.882   | 0.906   | 0.893   | 0.923   |
|                 | RAVDESS                  | 0.305  | 0.297  | 0.353   | 0.342   | 0.335   | 0.352   | 0.322   |
|                 | Speech Commands V1       | 0.191  | 0.184  | 0.185   | 0.244   | 0.152   | 0.198   | 0.183   |
|                 | Vocal Imitation          | 0.055  | 0.092  | 0.109   | 0.052   | 0.078   | 0.072   | 0.045   |
|                 | VocalSound               | 0.526  | 0.604  | 0.607   | 0.589   | 0.564   | 0.599   | 0.547   |
|                 | VoxCeleb1                | 0.030  | 0.097  | 0.069   | 0.037   | 0.048   | 0.086   | 0.057   |
|                 | VoxLingua33              | 0.078  | 0.090  | 0.104   | 0.087   | 0.085   | 0.090   | 0.096   |
| **Overall**     |                          | 0.347  | 0.379  | 0.393   | 0.359   | 0.358   | 0.366   | 0.353   |


### Patch size

| **Domain**      | **Task**                 | **16x16** | **8x16** | **8x32** | **4x64** | **4x128** | **2x128** | **8x8** |
| --------------- | ------------------------ | :-------: | :------: | :------: | :------: | :-------: | :-------: | :-----: |
| **Environment** |                          | 0.308     | 0.229    | 0.256    | 0.301    | 0.294     | 0.242     | 0.243   |
|                 | Clotho                   | 0.016     | 0.010    | 0.014    | 0.016    | 0.017     | 0.014     | 0.011   |
|                 | DESED                    | 0.394     | 0.278    | 0.313    | 0.426    | 0.332     | 0.231     | 0.246   |
|                 | ESC-50                   | 0.520     | 0.302    | 0.389    | 0.464    | 0.447     | 0.369     | 0.406   |
|                 | FSD18-Kaggle             | 0.276     | 0.234    | 0.231    | 0.278    | 0.299     | 0.233     | 0.251   |
|                 | FSD50k                   | 0.204     | 0.128    | 0.162    | 0.191    | 0.211     | 0.176     | 0.146   |
|                 | UrbanSound 8k            | 0.742     | 0.614    | 0.668    | 0.694    | 0.684     | 0.645     | 0.652   |
| **Music**       |                          | 0.457     | 0.446    | 0.442    | 0.473    | 0.452     | 0.463     | 0.466   |
|                 | Free Music Archive Small | 0.496     | 0.514    | 0.521    | 0.514    | 0.547     | 0.576     | 0.511   |
|                 | GTZAN Genre              | 0.606     | 0.677    | 0.617    | 0.628    | 0.687     | 0.712     | 0.608   |
|                 | MAESTRO                  | 0.163     | 0.186    | 0.194    | 0.260    | 0.326     | 0.312     | 0.194   |
|                 | NSynth-Instruments       | 0.444     | 0.415    | 0.410    | 0.455    | 0.406     | 0.411     | 0.449   |
| **Speech**      |                          | 0.415     | 0.386    | 0.393    | 0.420    | 0.411     | 0.410     | 0.399   |
|                 | ASV2015                  | 0.974     | 0.937    | 0.938    | 0.931    | 0.929     | 0.953     | 0.925   |
|                 | CREMA-D                  | 0.457     | 0.436    | 0.446    | 0.452    | 0.438     | 0.447     | 0.462   |
|                 | Fluent Speech Commands   | 0.034     | 0.035    | 0.040    | 0.041    | 0.023     | 0.020     | 0.033   |
|                 | LibriCount               | 0.571     | 0.461    | 0.415    | 0.531    | 0.574     | 0.541     | 0.489   |
|                 | LibriSpeech-MF           | 0.904     | 0.866    | 0.939    | 0.915    | 0.953     | 0.948     | 0.924   |
|                 | RAVDESS                  | 0.353     | 0.320    | 0.360    | 0.347    | 0.372     | 0.370     | 0.321   |
|                 | Speech Commands V1       | 0.185     | 0.202    | 0.187    | 0.262    | 0.217     | 0.196     | 0.190   |
|                 | Vocal Imitation          | 0.109     | 0.054    | 0.064    | 0.096    | 0.053     | 0.069     | 0.070   |
|                 | VocalSound               | 0.607     | 0.558    | 0.559    | 0.636    | 0.544     | 0.515     | 0.586   |
|                 | VoxCeleb1                | 0.069     | 0.056    | 0.043    | 0.068    | 0.113     | 0.140     | 0.063   |
|                 | VoxLingua33              | 0.104     | 0.093    | 0.088    | 0.110    | 0.101     | 0.101     | 0.087   |
| **Overall**     |                          | 0.393     | 0.354    | 0.364    | 0.396    | 0.386     | 0.374     | 0.368   |


### Warmup steps

| **Domain**      | **Task**                 | **1k** | **2k** | **5k** | **10k** |
| --------------- | ------------------------ | :----: | :----: | :----: | :-----: |
| **Environment** |                          | 0.308  | 0.296  | 0.291  | 0.258   |
|                 | Clotho                   | 0.016  | 0.013  | 0.016  | 0.011   |
|                 | DESED                    | 0.394  | 0.361  | 0.347  | 0.281   |
|                 | ESC-50                   | 0.520  | 0.495  | 0.455  | 0.428   |
|                 | FSD18-Kaggle             | 0.276  | 0.294  | 0.319  | 0.258   |
|                 | FSD50k                   | 0.204  | 0.187  | 0.177  | 0.161   |
|                 | UrbanSound 8k            | 0.742  | 0.713  | 0.685  | 0.672   |
| **Music**       |                          | 0.457  | 0.484  | 0.497  | 0.448   |
|                 | Free Music Archive Small | 0.496  | 0.488  | 0.498  | 0.494   |
|                 | GTZAN Genre              | 0.606  | 0.627  | 0.651  | 0.628   |
|                 | MAESTRO                  | 0.163  | 0.211  | 0.200  | 0.174   |
|                 | NSynth-Instruments       | 0.444  | 0.484  | 0.498  | 0.429   |
| **Speech**      |                          | 0.415  | 0.417  | 0.428  | 0.406   |
|                 | ASV2015                  | 0.974  | 0.968  | 0.964  | 0.947   |
|                 | CREMA-D                  | 0.457  | 0.455  | 0.469  | 0.447   |
|                 | Fluent Speech Commands   | 0.034  | 0.038  | 0.076  | 0.054   |
|                 | LibriCount               | 0.571  | 0.518  | 0.515  | 0.481   |
|                 | LibriSpeech-MF           | 0.904  | 0.901  | 0.919  | 0.893   |
|                 | RAVDESS                  | 0.353  | 0.356  | 0.394  | 0.359   |
|                 | Speech Commands V1       | 0.185  | 0.253  | 0.263  | 0.238   |
|                 | Vocal Imitation          | 0.109  | 0.075  | 0.104  | 0.091   |
|                 | VocalSound               | 0.607  | 0.622  | 0.614  | 0.593   |
|                 | VoxCeleb1                | 0.069  | 0.085  | 0.070  | 0.040   |
|                 | VoxLingua33              | 0.104  | 0.086  | 0.108  | 0.100   |
| **Overall**     |                          | 0.393  | 0.394  | 0.401  | 0.374   |

## KNN Results

### Infos & baselines

| **Domain**      | **Task**                 | **Type** | **Criterion**    | **Weight** | dasheng   | wav2vec2 | whisper   | data2vec  |
| :-------------: | :----------------------: | :------: | :--------------: | :--------: |:---------:|:--------:|:---------:|:---------:|
| **Environment** |                          |          |                  | 1273       | 0.648	    | 0.258    | 0.207     | 0.120     |
|                 | ESC-50                   | clip     | CrossEntropyLoss | 400        | 0.618     | 0.081    | 0.191     | 0.040     |
|                 | UrbanSound 8k            | clip     | CrossEntropyLoss | 873        | 0.662     | 0.339    | 0.215     | 0.156     |
| **Music**       |                          |          |                  | 2900       | 0.534     | 0.253    | 0.265     | 0.156     |
|                 | Free Music Archive Small | clip     | CrossEntropyLoss | 800        | 0.592     | 0.251    | 0.406     | 0.106     |
|                 | GTZAN Genre              | clip     | CrossEntropyLoss | 100        | 0.758     | 0.303    | 0.350     | 0.108     |
|                 | NSynth-Instruments       | clip     | CrossEntropyLoss | 2000       | 0.499     | 0.251    | 0.205     | 0.179     |
| **Speech**      |                          |          |                  | 18716      | 0.489     | 0.264    | 0.310     | 0.441     |
|                 | ASV2015                  | clip     | CrossEntropyLoss | 2000       | 0.869     | 0.858    | 0.843     | 0.942     |
|                 | CREMA-D                  | clip     | CrossEntropyLoss | 1116       | 0.380     | 0.221    | 0.372     | 0.351     |
|                 | Fluent Speech Commands   | clip     | CrossEntropyLoss | 2000       | 0.260     | 0.017    | 0.032     | 0.630     |
|                 | LibriCount               | clip     | CrossEntropyLoss | 1144       | 0.311     | 0.235    | 0.246     | 0.176     |
|                 | LibriSpeech-MF           | clip     | CrossEntropyLoss | 2620       | 0.791     | 0.606    | 0.617     | 0.724     |
|                 | RAVDESS                  | clip     | CrossEntropyLoss | 360        | 0.408     | 0.169    | 0.296     | 0.313     |
|                 | Speech Commands V1       | clip     | CrossEntropyLoss | 2000       | 0.903     | 0.208    | 0.096     | 0.852     |
|                 | Vocal Imitation          | clip     | CrossEntropyLoss | 1867       | 0.107     | 0.010    | 0.016     | 0.018     |
|                 | VocalSound               | clip     | CrossEntropyLoss | 2000       | 0.382     | 0.269    | 0.405     | 0.308     |
|                 | VoxCeleb1                | clip     | CrossEntropyLoss | 2000       | 0.262     | 0.003    | 0.010     | 0.033     |
|                 | VoxLingua33              | clip     | CrossEntropyLoss | 1609       | 0.376     | 0.034    | 0.360     | 0.058     |
| **Overall**     |                          |          |                  | 22889      | 0.504     | 0.262    | 0.299     | 0.388     |


### Frequency

| **Domain**      | **Task**                 | **32kHz** |**16kHz** |
| --------------- | ------------------------ | :-------: |:-------: |
| **Environment** |                          | 0.230     | 0.457    |
|                 | ESC-50                   | 0.140     | 0.315    |
|                 | UrbanSound 8k            | 0.303     | 0.522    |
| **Music**       |                          | 0.257     | 0.361    |
|                 | Free Music Archive Small | 0.449     | 0.370    |
|                 | GTZAN Genre              | 0.452     | 0.391    |
|                 | NSynth-Instruments       | 0.170     | 0.356    |
| **Speech**      |                          | 0.255     | 0.279    |
|                 | ASV2015                  | 0.927     | 0.810    |
|                 | CREMA-D                  | 0.268     | 0.273    |
|                 | Fluent Speech Commands   | 0.009     | 0.008    |
|                 | LibriCount               | 0.307     | 0.289    |
|                 | LibriSpeech-MF           | 0.550     | 0.747    |
|                 | RAVDESS                  | 0.215     | 0.250    |
|                 | Speech Commands V1       | 0.044     | 0.051    |
|                 | Vocal Imitation          | 0.017     | 0.036    |
|                 | VocalSound               | 0.259     | 0.326    |
|                 | VoxCeleb1                | 0.002     | 0.006    |
|                 | VoxLingua33              | 0.057     | 0.042    |
| **Overall**     |                          | 0.255     | 0.299    |


### Mel bands

| **Domain**      | **Task**                 | **64mels** | **96mels** | **128mels** | **160mels** | **192mels** | **224mels** | **256mels** |
| --------------- | ------------------------ | :--------: | :--------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| **Environment** |                          | 0.182      | 0.329      | 0.457       | 0.225       | 0.330       | 0.343       | 0.335       |
|                 | ESC-50                   | 0.105      | 0.173      | 0.315       | 0.104       | 0.236       | 0.207       | 0.194       |
|                 | UrbanSound 8k            | 0.218      | 0.401      | 0.522       | 0.280       | 0.373       | 0.405       | 0.400       |
| **Music**       |                          | 0.206      | 0.254      | 0.361       | 0.348       | 0.349       | 0.399       | 0.387       |
|                 | Free Music Archive Small | 0.307      | 0.294      | 0.370       | 0.325       | 0.384       | 0.381       | 0.431       |
|                 | GTZAN Genre              | 0.352      | 0.324      | 0.391       | 0.408       | 0.492       | 0.467       | 0.562       |
|                 | NSynth-Instruments       | 0.158      | 0.234      | 0.356       | 0.354       | 0.328       | 0.402       | 0.360       |
| **Speech**      |                          | 0.230      | 0.268      | 0.279       | 0.251       | 0.302       | 0.281       | 0.296       |
|                 | ASV2015                  | 0.715      | 0.847      | 0.810       | 0.775       | 0.887       | 0.939       | 0.949       |
|                 | CREMA-D                  | 0.238      | 0.249      | 0.273       | 0.315       | 0.314       | 0.349       | 0.370       |
|                 | Fluent Speech Commands   | 0.010      | 0.009      | 0.008       | 0.008       | 0.016       | 0.007       | 0.012       |
|                 | LibriCount               | 0.218      | 0.238      | 0.289       | 0.331       | 0.377       | 0.330       | 0.341       |
|                 | LibriSpeech-MF           | 0.633      | 0.712      | 0.747       | 0.603       | 0.786       | 0.608       | 0.724       |
|                 | RAVDESS                  | 0.178      | 0.213      | 0.250       | 0.216       | 0.264       | 0.228       | 0.224       |
|                 | Speech Commands V1       | 0.045      | 0.050      | 0.051       | 0.057       | 0.055       | 0.058       | 0.059       |
|                 | Vocal Imitation          | 0.017      | 0.026      | 0.036       | 0.016       | 0.026       | 0.022       | 0.015       |
|                 | VocalSound               | 0.209      | 0.291      | 0.326       | 0.274       | 0.335       | 0.332       | 0.294       |
|                 | VoxCeleb1                | 0.003      | 0.003      | 0.006       | 0.002       | 0.004       | 0.003       | 0.003       |
|                 | VoxLingua33              | 0.046      | 0.043      | 0.042       | 0.035       | 0.044       | 0.055       | 0.058       |
| **Overall**     |                          | 0.225      | 0.269      | 0.299       | 0.262       | 0.310       | 0.299       | 0.310       |


### Patch size

| **Domain**      | **Task**                 | **16x16** | **8x16** | **8x32** | **4x64** | **4x128** | **2x128** | **8x8** |
| --------------- | ------------------------ | :-------: | :------: | :------: | :------: | :-------: | :-------: | :-----: |
| **Environment** |                          | 0.457     | 0.315    | 0.315    | 0.346    | 0.344     | 0.283     | 0.330   |
|                 | ESC-50                   | 0.315     | 0.125    | 0.159    | 0.200    | 0.214     | 0.164     | 0.197   |
|                 | UrbanSound 8k            | 0.522     | 0.402    | 0.386    | 0.413    | 0.404     | 0.338     | 0.391   |
| **Music**       |                          | 0.361     | 0.424    | 0.327    | 0.420    | 0.338     | 0.303     | 0.357   |
|                 | Free Music Archive Small | 0.370     | 0.414    | 0.389    | 0.394    | 0.451     | 0.519     | 0.343   |
|                 | GTZAN Genre              | 0.391     | 0.563    | 0.427    | 0.447    | 0.551     | 0.666     | 0.440   |
|                 | NSynth-Instruments       | 0.356     | 0.421    | 0.298    | 0.429    | 0.282     | 0.198     | 0.358   |
| **Speech**      |                          | 0.279     | 0.279    | 0.287    | 0.283    | 0.215     | 0.275     | 0.291   |
|                 | ASV2015                  | 0.810     | 0.926    | 0.891    | 0.946    | 0.510     | 0.808     | 0.887   |
|                 | CREMA-D                  | 0.273     | 0.360    | 0.272    | 0.247    | 0.283     | 0.310     | 0.349   |
|                 | Fluent Speech Commands   | 0.008     | 0.011    | 0.013    | 0.007    | 0.009     | 0.009     | 0.011   |
|                 | LibriCount               | 0.289     | 0.352    | 0.253    | 0.195    | 0.276     | 0.313     | 0.298   |
|                 | LibriSpeech-MF           | 0.747     | 0.644    | 0.840    | 0.727    | 0.593     | 0.762     | 0.780   |
|                 | RAVDESS                  | 0.250     | 0.237    | 0.228    | 0.228    | 0.248     | 0.297     | 0.207   |
|                 | Speech Commands V1       | 0.051     | 0.050    | 0.048    | 0.063    | 0.041     | 0.036     | 0.057   |
|                 | Vocal Imitation          | 0.036     | 0.019    | 0.023    | 0.027    | 0.017     | 0.021     | 0.024   |
|                 | VocalSound               | 0.326     | 0.288    | 0.236    | 0.316    | 0.259     | 0.249     | 0.294   |
|                 | VoxCeleb1                | 0.006     | 0.003    | 0.006    | 0.003    | 0.002     | 0.006     | 0.005   |
|                 | VoxLingua33              | 0.042     | 0.038    | 0.035    | 0.051    | 0.047     | 0.048     | 0.032   |
| **Overall**     |                          | 0.299     | 0.300    | 0.293    | 0.304    | 0.238     | 0.279     | 0.302   |


### Warmup steps

| **Domain**      | **Task**                 | **1k** | **2k** | **5k**| **10k** |
| --------------- | ------------------------ | :----: | :----: | :---: | :-----: |
| **Environment** |                          | 0.457  | 0.436  | 0.452 | 0.370   |
|                 | ESC-50                   | 0.315  | 0.258  | 0.322 | 0.225   |
|                 | UrbanSound 8k            | 0.522  | 0.518  | 0.512 | 0.437   |
| **Music**       |                          | 0.361  | 0.369  | 0.483 | 0.361   |
|                 | Free Music Archive Small | 0.370  | 0.379  | 0.462 | 0.397   |
|                 | GTZAN Genre              | 0.391  | 0.502  | 0.583 | 0.507   |
|                 | NSynth-Instruments       | 0.356  | 0.359  | 0.487 | 0.339   |
| **Speech**      |                          | 0.279  | 0.288  | 0.326 | 0.273   |
|                 | ASV2015                  | 0.810  | 0.722  | 0.903 | 0.789   |
|                 | CREMA-D                  | 0.273  | 0.312  | 0.355 | 0.316   |
|                 | Fluent Speech Commands   | 0.008  | 0.013  | 0.021 | 0.016   |
|                 | LibriCount               | 0.289  | 0.389  | 0.445 | 0.319   |
|                 | LibriSpeech-MF           | 0.747  | 0.807  | 0.812 | 0.644   |
|                 | RAVDESS                  | 0.250  | 0.242  | 0.276 | 0.251   |
|                 | Speech Commands V1       | 0.051  | 0.080  | 0.083 | 0.070   |
|                 | Vocal Imitation          | 0.036  | 0.029  | 0.046 | 0.039   |
|                 | VocalSound               | 0.326  | 0.310  | 0.387 | 0.362   |
|                 | VoxCeleb1                | 0.006  | 0.006  | 0.008 | 0.003   |
|                 | VoxLingua33              | 0.042  | 0.048  | 0.046 | 0.043   |
| **Overall**     |                          | 0.299  | 0.306  | 0.353 | 0.290   |