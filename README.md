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
