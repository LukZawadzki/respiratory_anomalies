# Wykrywanie anomalii w analizie dźwięków układu oddechowego

## Cel projektu
Celem projektu jest utworzenie modelu uczenia maszynowego, który byłby odpowiedzialny za analizowanie sekwencji danych dźwiękowych, pochodzących z nagrań układu oddechowego w celu wykrycia anormalnych układów częstotliwości, które wskazywałyby na nienaturalny proces oddychania.

## Członkowie grupy i wstępny podział obowiązków
- **Łukasz Zawadzki**  
  Przygotowanie i preprocessing danych treningowych

- **Damian Moskała**  
  Implementacja algorytmów uczenia modelu

- **Witold Nowogórski**  
  Rozwinięcie niskopoziomowego składu bloków modelu

- **Jakub Kubicki**  
  Specyfikacja portów dla meta-danych warunkujących (meta-dane pobrane ze zbioru danych, niebędące bezpośrednim nagraniem dźwiękowym)

- **Wiktor Prosowicz**  
  Opracowanie wysokopoziomowej struktury modelu

## Źródła danych
Prace wraz ze zbiorami danych, na których się opieramy:
- [Coswara-Data](https://github.com/iiscleap/Coswara-Data)
- [Artykuł Coswara-Data](https://arxiv.org/abs/2005.10548)
- [RespireNet](https://github.com/microsoft/RespireNet)
- [Artykuł RespireNet](https://arxiv.org/abs/2011.00196)

## Harmonogram prac

### 16.04
- Przygotowanie danych i ustalenie portów wejściowych modelu
- Opracowanie wysokopoziomowej struktury modelu

### 21.05
- Wstępna implementacja modelu
- Testy na dostępnym zbiorze danych

### 4.06
- Rozwinięcie modelu
- Poprawa jakości kodu i błędów modelu

### 18.06
- Użytkowa wersja systemu wraz z aplikacją i walidacją danych wejściowych


# Model description
## Authors:
- Witold Nowogórski
- Łukasz Zawadzki
- Jakub Kubicki
- Damian Moskała
- Wiktor Prosowicz

## Work overview

![work_overview](./doc/res/work_overview.drawio.png)

### Data augmentation

Due to the small size of the dataset chosen for this project and due to the fact that a DNN architecture has been proposed, there is a need for additional data samples. To address this problem a set of audio data augmentation procedures shall be applied.

#### Noise addition

It's purpose is to both simulate the real-world conditions and provide slightly different data samples.

#### Tempo modification

It reduces the influence of the disproportion between the encountered tempo range and the number of samples on the model's performance.

#### Spectral clipping

Consists mostly on low-pass filtering. It's role is to cut out background noises. It allows to control the amount of external noise added to a sample. 

#### Time domain clipping/padding

Adopted in order to unify the samples regarding their duration.

- Smart padding - it is a modification of the standard zero-padding technique. It involves repetition of breath cycles across the sample's time domain. 
- Concatenation-based augmentation - filling the empty space with fragments sampled from all classes.

### Feature Extraction

The input port of the model consists of Mel Frequency Cepstral Coefficients. This form of audio-data presentation allows to capture speech-related features of sound. MFCCs are obtained via cosine-transform of log-spectrogram. The resulting samples have the form of a matrix with axes representing time, mel filterbanks respectively.

### Model training

We propose a deep neural architecture consisting mainly of convolutional layers with skip connections. 

![model_architecture](./doc/res/model_architecture.drawio.png)


### Model evaluation

The model's output is evaluated via set of scores. Those check whether the model accurately predicts the expected multi-class result. Due to the skewed distribution of dataset samples among various classes and recording appliances it is crucial to evaluate the model with respect to the classes to which a specific sample belongs.

