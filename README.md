# Prediction of Transfusion in ICU in patients with gastrointestinal bleeding

This repository contains data and codes used in our paper ["Artificial intelligence-based prediction of transfusion in the intensive care unit in patients with gastrointestinal bleeding"](https://informatics.bmj.com/content/28/1/e100245) published in BMJ Health and Care Informatics.

## Materials

Patients' data were extracted from [Medical Information Mart for Intensive Care-III (MIMIC-III) V.1.4](https://mimic.mit.edu/docs/iii/) and in the [eICU Collaborative Research Database  (eICU-CRD) V.2.0](https://eicu-crd.mit.edu). The access to these datasets is controlled and researchers should request access on the PhysioNet website.
Extracted data for each time frame are stored in the folder "Data"

## Codes

The codes are written in Python V.3.7 and the following libraries are employed:
* Pandas V.0.25.3
* NumPy V.1.17.5 
* SciPy V.1.4.1
* Scikit-learn V.0.22.1
* Hyperopt V.0.2.3

We developed 3 Jupyter Lab notebook, one for each validation dataset. In addition, custum_util.py and pipelines.py contains supporting functions for the execution of the main code.


## Reference

To cite this work, please use the following references:
```
@article{Levi2021,
author = {Levi, Riccardo and Carli, Francesco and Ar{\'{e}}valo, Aldo Robles and Altinel, Yuksel and Stein, Daniel J and Naldini, Matteo Maria and Grassi, Federica and Zanoni, Andrea and Finkelstein, Stan and Vieira, Susana M and Sousa, Jo{\~{a}}o and Barbieri, Riccardo and Celi, Leo Anthony},
doi = {10.1136/bmjhci-2020-100245},
journal = {BMJ Health &amp; Care Informatics},
month = {jan},
number = {1},
pages = {e100245},
title = {{Artificial intelligence-based prediction of transfusion in the intensive care unit in patients with gastrointestinal bleeding}},
url = {http://informatics.bmj.com/content/28/1/e100245.abstract},
volume = {28},
year = {2021}
}

@article{Johnson2016,
author = {Johnson, Alistair E W and Pollard, Tom J and Shen, Lu and Lehman, Li-wei H and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, Peter and {Anthony Celi}, Leo and Mark, Roger G},
doi = {10.1038/sdata.2016.35},
issn = {2052-4463},
journal = {Scientific Data},
number = {1},
pages = {160035},
title = {{MIMIC-III, a freely accessible critical care database}},
url = {https://doi.org/10.1038/sdata.2016.35},
volume = {3},
year = {2016}
}

@article{Pollard2018,
author = {Pollard, Tom J and Johnson, Alistair E W and Raffa, Jesse D and Celi, Leo A and Mark, Roger G and Badawi, Omar},
doi = {10.1038/sdata.2018.178},
issn = {2052-4463},
journal = {Scientific Data},
number = {1},
pages = {180178},
title = {{The eICU Collaborative Research Database, a freely available multi-center database for critical care research}},
url = {https://doi.org/10.1038/sdata.2018.178},
volume = {5},
year = {2018}
}
```




