# Download and convert the dataset as project demanding 

## Law School Admissions Council (LSAC)
Pre-process Law School Admissions Council (LSAC) Dataset and create train and test files:
Download the Law School dataset from: (http://www.seaphe.org/databases.php), convert SAS file to CSV, and save it in the ./datasets/ folder.


- Download it 
```
wget http://www.seaphe.org/databases/LSAC/LSAC_SAS.zip
unzip LSAC_SAS.zip

# Convert LSAC_SAS/lsac.sas7bdat to lsac.csv by https://dumbmatter.com/sas7bdat/.

```
- Preprocess 
```
cd law_school && python CreateLawSchoolDatasetFiles.py
```
Run CreateLawSchoolDatasetFiles.py to process the dataset, and create files required for training.
