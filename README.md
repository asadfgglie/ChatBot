# ChatBot

基於繁體中文的對話生成模型

---

## 訓練數據來源

使用[lccc-base](https://huggingface.co/datasets/lccc)並且以[OpenCC](https://github.com/BYVoid/OpenCC)翻譯成繁體中文做訓練

---

## 預訓練模型來源

使用[CKIP](https://ckip.iis.sinica.edu.tw/)開源的[ckiplab/gpt2-tiny-chinese](https://huggingface.co/ckiplab/gpt2-tiny-chinese)作為預訓練模型

大小為22,387,028

## 模型數據
目前模型還在訓練開發中

因此還沒有太多數據

### 2023_03_19_dialog_gpt2-tiny
* 訓練loss: 3.478
* 驗證perplexity: 31.20429039001465

詳細訓練過程數據可以參閱`/model/log`，以tensorbroad開啟即可查閱

## 使用方法

先在`train.py` line 68 處設定好訓練時所需的各個參數

訓練好即可透過`talk.py` line 49 處設定使用的模型，即可與模型做對話測試