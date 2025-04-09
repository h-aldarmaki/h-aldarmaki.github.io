---
title: "Arabic Speech Processing"
date: 2024-04-06
last_modified_at: 2024-04-06
categories: projects
tags: [arabic]
layout: single
author_profile: false
read_time: false
share: true
comments: true
toc: true
toc_sticky: true
image: /assets/images/artst.jpg
---

<div>
  <img src="/assets/images/artst.jpg" alt="ArTST" style="float: left; margin-right: 1rem;max-width:140px"/>    
</div>
<div style="font-size: 16px; text-align: justify;">
  <p> Our group is developing various datasets and models for supporting Arabic speech processing, such as Automatic Speech Recognition (ASR), Text-to-Speech synthesis (TTS), and diacritic restoration. We try to widen coverage of spoken varieties by including regional dialects and code-switching. Arabic Speech and Text Transformer (ArTST) is a project built with the idea that optimizing performance for Arabic speech requires building models for Arabic from the get-go, rather than fine-tuning English-centric on multilingual models. While multi-lingual models are impressive, they are inferior for specific target languages compared to monolingual models trained with the same amounts of data. <br>
  Our models are currently all based on the <a href="https://huggingface.co/docs/transformers/en/model_doc/speecht5" >SpeechT5</a> architecture, which has the advantage of multi-modal pre-training: text/speech as both input and output. This framework means that we can pre-train once, then fine-tune the model on ASR, TTS, or other speech/tasks. Most other pre-training frameworks support only one output modality (e.g. Whisper, at the time of this writing, only supports text output).<br>
  Our first version of ArTST, which received the <strong>best paper award</strong> at ArabicNLP 2023, supported only Modern Standard Arabic (MSA). Subsequent versions included multiple dialects (v2) and languages (v3) to support dialectal speech and code-switching. We also recently added a version pre-trained with diacritics.
  </p>
</div>

## ASR and TTS Fine-Tuning: MSA
<div style="font-size: 16px; text-align: justify;">
    <strong>ASR:</strong> In our <a href="https://aclanthology.org/2023.arabicnlp-1.5.pdf">first ArTST paper</a>, we describe ASR fine-tuning experiments on MSA, where we fine-tune using the MGB2 benchmark dataset. The result is comparable to the current SOTA, with 12.8% WER. Compared to supervised multilingual models like Whisper and MMS, both medium and large versions, ArTST v1 results in half the error rate (at least) with a fraction of the parameters. The model showed some potential for recognizing dialects as well, but it wasn't optimized for that. We also trained the model for ADI17 dialect identification benchmark, and got SOTA 94% accuracy. <br><br>
    <strong>TTS:</strong> We also fine-tuned the model for <a style="color:Tomato">Text-to-Speech</a> synthesis on the <a href="/clartts/">ClArTTS</a> dataset. One of our interesting findings is that reasonable TTS performance can be achieved without the inclusion of diacritics. Prior efforts on Arabic TTS rely on diacritic restoration as TTS systems are generally trained with relatively small amounts of data and rely on short contexts for sound styntehsis. The lack of diacritics means that the model has to infer the pronunciation of short vowels from context, which is far-fetched unless large amounts of data (and large models) are used. Due to ArTST's pre-training on ~1000 hours of Arabic speech, we were able to achieve decent TTS performance without the use of any diacritics, using roughly 12 hours of TTS training data. Furthermore, we experimented with using ASR data from MGB2 to do "pre-fine-tuning" for TTS, where we first fine-tune on MGB2 data, then ClArTTS. Note that ASR data are generally not suitable for TTS training since the data is generally noisy and inconsistent in style, speaking rate, emotion, etc. What we find is that this process results in further improvements in TTS quality. We refer to this model a ArTST*; you can <a style="color:Tomato">listen to some samples <a href="https://speechsample.wixsite.com/artsttts">here</a>. </a>
</div>

## Dialectal ASR and code-switching
<div style="font-size: 16px; text-align: justify;">
    In our subsequent <a href="https://arxiv.org/pdf/2411.05872">paper</a> we describe ASR experiments using our v2 checkpoint. This checkpoint is trained with roughly 3000 hours of speech, spanning 17 dialectal varieties (based on country codes). We conducted ASR fine-tuning experiments on 11 dialcts, in addition to MSA. We also kept 3 dialects for zero-shot testing. In the same paper, we also describe the multi-lingual checkpoint, v3, which includes English, French, and Spanish. The multilingual pre-training and fine-tuning enables the model to handle cases of code-switching. The paper details various experiments on pre-training effects, dialect ID forcing, dialect ID inference, and joint dialectal training. Based on these findings, we recommend the following:
</div>

- For MSA, v1 and v2 perform equally well. 
- For dialects, the joint v2 model with dialect inference achieves the best performance on average. 
- Neither v1 nor v2 can handle code-switching with other languages. 
- v3 performs well on Arabic-English and Arabic-French code-switching, at the cost of somewhat lower performance on monolinual Arabic. 

## Diacritics in Speech Processing
<div style="font-size: 16px; text-align: justify;">
Speech presents an interesting avenue for diacritization research. We did <a href="https://www.isca-archive.org/interspeech_2023/aldarmaki23_interspeech.pdf">a study on ASR diacritization performance</a>, published at INTERSPEECH 2023, comparing it with post-processing using text-based diacritic restoration models (the standard diacritic restoration type of model). Diacritization directing using ASR results in far better performance. <br>
<strong>Multimodal Diacritic Restoration:</strong> Subsequently, we explored the potential of speech as an additional signal for diacritic restoration. As a use case, consider all the resources available for Arabic ASR, such as MGB2 (1000 hours) and QASR (2000 hours); these datasets contain speech and text transcripts, but most of the text contains no diacritics. Could we use both the speech and text for accurate diacritic restoration? Indeed, our <a href="https://aclanthology.org/2024.naacl-long.233.pdf">paper describing such model</a> was published at NAACL 2024. The proposed model incorporated a Whisper model fine-tuned on our <a href="/clartts/">ClArTTS</a> dataset to produce diacritized transcripts, in addition to the raw text input. Using cross-attention, the network integrates predictions from ASR, as well as the correct undiacritized reference text, to restore the missing diacritics. Our findings reveal that such approach is effective and reduced diacritic error rates by half on the ClArTTS test set. We also tested on out-of-domain MSA data and observed some reduction in DER, but the effect was smaller. Overall, DER on MSA data was rather high, even when using popular open and closed text-based diacritic restoration model. <br>
<strong>Data Augmentation:</strong>Our <a href="https://aclanthology.org/2024.arabicnlp-1.15.pdf">subsequent paper</a> explored a data augmentation technique to improve the generaalization of the system. The proposed augmentation method consists of random diacritics applied to text, then synthesizing speech based on these randomly diacritized text using a commercial TTS system. The intuition behind this approach is to reduce the dependency of the model's predictions on the textual context to push the model towards better model of the acoustic properties that correspond to a given diacritic. This technique consistently improved diacritic recognition performance on all models and datasets. <br>
<strong>Pre-training with Diacritics:</strong> Since our ArTST models are pre-trained without diacritics (s most speech resourced don't include diacritics), this may have an effect on fine-tuning performance when diacritics are included, as that requires some unlearning of the patterns aquired in the pre-training phase. Indeed, we find that this effect is most evident for TTS, where pre-training hampers the model's performance when diacritics are included. That is one reason why our original TTS models based on ArTST were undiacritized. Since pre-training does not require alignment between the speech and text data, We pre-trained another version of ArTST (v1.5) using MGB2 audio and diacritized text from Tashkeela. We then fine-tuned TTS with diacritics and compared with v1. Our hypothesis was supported, and the model fine-tuned from v1.5 was significantly better than the one fine-tuned on v1, everything else being equal. 
</div>

## Partial Diacritization
<div style="font-size: 16px; text-align: justify;">

</div>

## Dialectal Diacritic Restoration
<div style="font-size: 16px; text-align: justify;">

</div>
