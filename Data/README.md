* This folder contains the files used for DeepPhospho model performance test and fine-tuning
* In folder IonModel_TestData/RT_TestData
  * These two folders contain the files used for model performance test
  * The results were shown as the model performance in the paper main figure and SI
* In folder IonModel_FineTuneData/RT_FineTuneData
  * These two folders contain the files used for pre-trained param fine-tuning
  * The fine-tuned params were used to predict spectra and RT for different usages in the paper
  * Notice the files were the whole dataset here, and in practice, we first used 90% and the remained 10% were also used after they worked as the test data to avoid overfitting
* For pre-training data, please [download here](https://drive.google.com/drive/folders/1ETJEG-8lobVJWaYOBMnqUL1G5dUHRI2B).
* In the paper, we used three datasets to test the DeepPhospho
  * RPE1 DDA
  * RPE1 DIA
  * U2OS DIA
* The fine-tuning datasets are
  * RPE1 DIA
  * U2OS DIA
  * Dilution
* And the data stored here were zipped if the original file size is over 10 M to decrease the downloading time

