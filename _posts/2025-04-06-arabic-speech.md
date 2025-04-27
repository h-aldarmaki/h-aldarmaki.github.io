---
title: "ArTST: Unified Arabic Speech Recognition and Synthesis Models" # Slightly more descriptive title
date: 2024-04-06
last_modified_at: 2024-04-27
categories: projects
tags: [arabic, asr, tts, speecht5, artst, huggingface, speech recognition, text to speech, natural language processing, ai, mbzuai] # Expanded tags
layout: single
author_profile: false
read_time: true # Enable read time estimation
share: true
comments: true
toc: true
toc_sticky: true
image: /assets/images/artst.jpg # For social sharing
excerpt: "Dive into ArTST, our state-of-the-art Arabic Text & Speech Transformer. Learn about models covering MSA, Dialects, and Code-Switching, with practical usage examples."
author: Amir Djanibekov
---
<div>
  <img src="/assets/images/artst.jpg" alt="ArTST Logo" style="float: left; margin-right: 1rem; max-width:140px"/>
</div>

<div style="font-size: 16px; text-align: justify;">
The Arabic language presents unique challenges for speech technology due to its vast dialectal diversity and the common phenomenon of code-switching. While large multilingual models offer broad language coverage, they often fall short of optimal performance for specific, resource-rich languages like Arabic compared to dedicated models.<br>
</div>
<!-- <div style="font-size: 16px; text-align: justify;">
  To address this, our group at the Speech Lab developed the <a href="https://github.com/mbzuai-nlp/ArTST" target="_blank" rel="noopener noreferrer"><strong>Arabic Speech and Text Transformer (ArTST)</strong></a>. ArTST is built on the principle that high performance requires models trained <em>specifically</em> for Arabic's nuances from the outset. This post provides an overview of the ArTST project, its different versions tailored for various tasks (MSA, dialects, code-switching, diacritics), key research findings, and practical examples of how to use our models.
</div> -->

<div style="font-size: 16px; text-align: justify;">
<p> Our group is developing various datasets and models for supporting Arabic speech processing, such as Automatic Speech Recognition (ASR), Text-to-Speech synthesis (TTS), and diacritic restoration. We try to widen coverage of spoken varieties by including regional dialects and code-switching. <a href="https://github.com/mbzuai-nlp/ArTST"><strong>Arabic Speech and Text Transformer (ArTST)</strong></a> is a project built with the idea that optimizing performance for Arabic speech requires building models for Arabic from the get-go, rather than fine-tuning English-centric on multilingual models. While multi-lingual models are impressive, they are inferior for specific target languages compared to monolingual models trained with the same amounts of data. <br>

Our models are currently all based on the <a href="https://huggingface.co/docs/transformers/en/model_doc/speecht5" target="_blank" rel="noopener noreferrer"><strong>SpeechT5</strong></a> architecture, notable for its unified multi-modal pre-training approach. It can handle both speech and text as input <em>and</em> output, allowing a single pre-trained foundation model to be effectively fine-tuned for diverse tasks like Automatic Speech Recognition (<strong>ASR</strong>) and Text-to-Speech (<strong>TTS</strong>). This offers flexibility compared to many frameworks limited to a single output modality (e.g. Whisper, at the time of this writing, only supports text output). Our first version of ArTST (v1), which received the <strong>best paper award</strong> at ArabicNLP 2023, supported only Modern Standard Arabic (MSA). Subsequent versions include multiple dialects (v2) and languages (v3) to support dialectal speech and code-switching. We also recently added a version pre-trained with diacritics.
</p>
</div>

## ArTST Versions: Choosing the Right Pre-Trained Model

<div style="font-size: 16px; text-align: justify;">
We've released several ArTST versions, each pre-trained and optimized for different scenarios. Understanding their strengths helps in selecting the best model for your specific needs:
</div>

| Version  | Pre-training Data Focus          | Key Highlight                                    | Recommended Use Case                      |
| :------- | :------------------------------- | :----------------------------------------------- | :---------------------------------------- |
| <span class="tooltip" data-tooltip="Pre-trained ~1k hrs MSA (MGB2)">**v1**</span>   | Modern Standard Arabic (MSA)     | Foundational MSA model (12.8% WER on MGB2)       | High-quality MSA ASR/TTS (undiacritized)   |
| <span class="tooltip" data-tooltip="Pre-trained ~3k hrs MSA + 11+ Dialects (MGB2, QASR, MGB3, MGB5, CommonVoice, etc.)">**v2**</span>   | MSA + Arabic Dialects (11+)      | Best average dialectal ASR performance           | General Dialectal ASR                     |
| <span class="tooltip" data-tooltip="Pre-trained like v2 + VoxBlink data + English/French data (CommonVoice)">**v3**</span>   | MSA + Dialects + EN/FR           | Handles Arabic-EN/FR <strong>code-switching</strong>          | Code-Switching ASR                        |
| <span class="tooltip" data-tooltip="Pre-trained ~1k hrs MSA (MGB2 audio + Tashkeela text)">**v1.5**</span> | MSA + <strong>Diacritized</strong> Text       | Optimized for tasks requiring diacritics         | High-quality Diacritized MSA TTS         |

<div style="font-size: 16px; text-align: justify;">
In essence: use <strong>v1</strong> or <strong>v2</strong> for MSA tasks, <strong>v2</strong> for general dialectal ASR, <strong>v3</strong> if you expect code-switching with English or French, and <strong>v1.5</strong> for high-fidelity diacritized TTS.
</div>

## Getting Started with ArTST

[![Star History Chart](https://api.star-history.com/svg?repos=mbzuai-nlp/ArTST&type=Date)](https://www.star-history.com/#mbzuai-nlp/ArTST&Date)
<div style="font-size: 16px; text-align: justify;">
Our pre-trained models are available on the Hugging Face Hub, making them easy to integrate into your projects.
</div>

<div style="margin-top: 20px; margin-bottom: 25px; text-align: center;">
  <a href="https://github.com/mbzuai-nlp/ArTST" class="btn btn--github" target="_blank" rel="noopener noreferrer"><i class="fab fa-github"></i> View ArTST on GitHub</a>
  <a href="https://huggingface.co/collections/MBZUAI/artst-arabic-text-speech-transformer-672cb44bb4215fd38814aeef" class="btn btn--info" target="_blank" rel="noopener noreferrer"><i class="fas fa-box-open"></i> Find ArTST Models on Hugging Face Hub</a>
</div>

### Selecting task specific ArTST

<div style="font-size: 16px; text-align: justify;">
After pre-training, we further train (or "fine-tune") these ArTST models to make them specialized for specific tasks like turning speech into text Automatic Speech Recognition (ASR) or text into speech (TTS). Fine-tuning uses labeled data (like audio paired with its transcription for ASR, or text paired with its spoken version for TTS) to adjust the pre-trained model. For ASR, we train the model to output the correct text transcription for a given audio input. For TTS, we train it to generate natural-sounding speech audio from given text input. Here are some of our main fine-tuned ASR models available on the Hugging Face Hub:
</div>

<div style="margin-top: 20px;"></div>


| Version (Fine-tuned Model ID)                                               | Based On (Pre-trained)           | Fine-tuning Focus                         | Task     |
| :-------------------------------------------------------------------------- | :------------------------------- | :---------------------------------------- |:---------|
| [MBZUAI/artst_asr](https://huggingface.co/MBZUAI/artst_asr)                       | v1 (MSA Pre-trained)           | MGB2 dataset (MSA)                      | ASR   |
| [MBZUAI/artst_asr_v2](https://huggingface.co/MBZUAI/artst_asr_v2)                 | v2 (Dialectal Pre-trained)     | MGB2 dataset (MSA)                      | ASR   |
| [MBZUAI/artst_asr_v3](https://huggingface.co/MBZUAI/artst_asr_v3)                 | v3 (Multilingual Pre-trained)  | MGB2 dataset (MSA)                      | ASR   |
| [MBZUAI/artst_asr_v2_qasr](https://huggingface.co/MBZUAI/artst_asr_v2_qasr)       | v2 (Dialectal Pre-trained)     | QASR dataset (Dialectal/MSA)            | ASR   |
| [MBZUAI/artst_asr_v3_qasr](https://huggingface.co/MBZUAI/artst_asr_v3_qasr)       | v3 (Multilingual Pre-trained)  | QASR dataset (Dialectal/MSA)            | ASR   |
| [MBZUAI/speecht5_tts_clartts_ar](https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar) | v1 (MSA Pre-trained)     | MGB2 + ClArTTS dataset (MSA)            | TTS   |


<div style="font-size: 16px; text-align: justify;">
The easiest way to use these fine-tuned models is with the Hugging Face <code class="language-plaintext highlighter-rouge">transformers</code> library, like in the examples below. If you need to use the Fairseq toolkit methods shown in our papers, you can find code examples in our GitHub demo notebooks:
<ul>
    <li><a href="https://github.com/mbzuai-nlp/ArTST/blob/main/demo-artst-asr.ipynb" target="_blank" rel="noopener noreferrer">ArTST ASR Demo Notebook</a></li>
    <li><a href="https://github.com/mbzuai-nlp/ArTST/blob/main/demo-artst-tts.ipynb" target="_blank" rel="noopener noreferrer">ArTST TTS Demo Notebook</a></li>
</ul>
</div>

#### Example: Automatic Speech Recognition (ASR)

<div style="font-size: 16px; text-align: justify;">
This snippet shows how to transcribe an Arabic audio file using a dialectal ArTST model (v1). Remember to install necessary libraries: <code class="language-plaintext highlighter-rouge">pip install transformers torch datasets soundfile librosa</code>.
</div>

{% highlight python %}
from transformers import pipeline
import soundfile as sf
import librosa # Needed for resampling
import torch
import os

# --- Configuration ---
# 1. Select the appropriate ArTST ASR model ID from Hugging Face Hub
#    Example: Using v1 for MSA ASR. Find more: https://huggingface.co/collections/MBZUAI/artst-arabic-text-speech-transformer-672cb44bb4215fd38814aeef
model_id = "MBZUAI/artst_asr" 

# 2. Specify the path to your audio file
audio_path = "path/to/your/arabic_audio.wav" # IMPORTANT: Replace with your audio file path

# 3. Set target sample rate (ArTST models require 16kHz)
TARGET_SR = 16000
# --- End Configuration ---

# --- Audio Loading and Preprocessing ---
speech = None
if not os.path.exists(audio_path):
    print(f"Error: Audio file not found at {audio_path}")
else:
    try:
        speech, sample_rate = sf.read(audio_path)
        print(f"Loaded audio: {audio_path}, Sample Rate: {sample_rate}Hz, Duration: {len(speech)/sample_rate:.2f}s")
        
        # Ensure mono audio
        if speech.ndim > 1:
            print("Audio appears to be stereo, converting to mono...")
            speech = speech.mean(axis=1)
        
        # Resample if necessary
        if sample_rate != TARGET_SR:
            print(f"Resampling audio from {sample_rate}Hz to {TARGET_SR}Hz...")
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR # Update sample rate after resampling
            print("Resampling complete.")
            
    except Exception as e:
        print(f"Error loading or processing audio file: {e}")
        print("Ensure 'libsndfile' (Linux: sudo apt-get install libsndfile1) and 'ffmpeg' are installed for broader format support.")
# --- End Audio Processing ---

# --- ASR Inference ---
if speech is not None:
    print(f"\nInitializing ASR pipeline with model: {model_id}")
    # Use GPU if available (device=0), otherwise CPU (device=-1)
    device_id = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda:0' if device_id == 0 else 'cpu'}")
    
    asr_pipeline = pipeline(
        "automatic-speech-recognition", 
        model=model_id, 
        device=device_id
    )

    print("Transcribing audio (this may take a moment for longer files)...")
    # Chunking is recommended for long audio files to manage memory
    transcription_result = asr_pipeline(
        speech.copy(), # Pass a copy if you need the original array later
        stride_length_s=(5, 0) # Overlap chunks slightly (5s here) for smoother transcription
    )
    
    print("\n--- Transcription Result ---")
    print(transcription_result["text"])
    print("---------------------------")
else:
    print("\nCannot proceed with transcription due to audio loading/processing errors.")
# --- End ASR Inference ---
{% endhighlight %}

#### Example: Text-to-Speech (TTS)

<div style="font-size: 16px; text-align: justify;">
This snippet demonstrates generating speech from Arabic text using the ArTST* v1 model. Ensure libraries are installed: <code class="language-plaintext highlighter-rouge">pip install transformers datasets torch soundfile</code>.
</div>

{% highlight python %}
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import time

# --- Configuration ---
# 1. Select the ArTST TTS model ID (e.g., v1 for MSA)
#    Find more: https://huggingface.co/models?search=mbzuai-nlp/artst
model_id = "MBZUAI/speecht5_tts_clartts_ar" 
vocoder_id = "microsoft/speecht5_hifigan" # Standard HiFi-GAN vocoder for SpeechT5

# 2. Input text (use diacritized text for v1.5)
text_input = "لأنه لا يرى أنه على السفه ثم من بعد ذلك حديث منتشر" 

# 3. Output audio file path
output_filename = "artst_tts_output.wav"
# --- End Configuration ---

# --- Model Loading ---
start_load_time = time.time()
print(f"Loading TTS components: {model_id} & {vocoder_id}")
try:
    processor = SpeechT5Processor.from_pretrained(model_id)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)

    # Move models to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    vocoder.to(device)
    print(f"Models loaded to {device} in {time.time() - start_load_time:.2f}s")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()
# --- End Model Loading ---

# --- Speaker Embedding Loading ---
print("Loading speaker embeddings (required by SpeechT5)...")
try:
    # Using a standard English dataset for embeddings; quality may vary for Arabic.
    # Fine-tuning with Arabic speaker embeddings would yield better results.
    embeddings_dataset = load_dataset("herwoww/arabic_xvector_embeddings", split="validation")
    # Example speaker embedding (index 7306: female US English). Experiment with different indices.
    speaker_embeddings = torch.tensor(embeddings_dataset[105]["speaker_embeddings"]).unsqueeze(0).to(device) 
    print("Speaker embeddings loaded.")
except Exception as e:
    print(f"Warning: Could not load speaker embeddings dataset: {e}. Using random embeddings as fallback.")
    speaker_embeddings = torch.randn((1, 512)).to(device) # Fallback
# --- End Speaker Embedding Loading ---

# --- Speech Generation ---
print("Processing text and generating speech...")
start_gen_time = time.time()
inputs = processor(text=text_input, return_tensors="pt").to(device)

with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
generation_time = time.time() - start_gen_time
print(f"Speech generated in {generation_time:.2f}s")
# --- End Speech Generation ---

# --- Saving Audio ---
print(f"Saving generated audio to {output_filename}...")
try:
    sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
    print(f"Successfully saved audio file. Duration: {len(speech)/16000:.2f}s")
except Exception as e:
    print(f"Error saving audio file: {e}")
# --- End Saving Audio ---
{% endhighlight %}

### Advanced Usage with Fairseq

<div style="font-size: 16px; text-align: justify;">
<p>For specific research experiments, particularly those involving Language Model (<strong>LM</strong>) fusion during ASR decoding (as described in our papers), the <strong>Fairseq</strong> toolkit was employed. This generally requires a deeper setup, including cloning the ArTST repository and running specific command-line scripts provided therein.</p>

<p>If your work requires these advanced capabilities, please consult the detailed instructions and scripts within the <a href="https://github.com/mbzuai-nlp/ArTST" target="_blank" rel="noopener noreferrer">ArTST GitHub repository</a>.</p>
</div>

{% highlight bash %}
# Note: This is a conceptual example. Refer to the ArTST repo for actual commands.

# Example: Running Fairseq generation for ASR with an external LM
# fairseq-generate /path/to/audio/manifest.tsv \
#   --config-yaml config_asr.yaml \
#   --gen-subset test_set_name \
#   --task speech_to_text \
#   --path /path/to/artst_model_checkpoint.pt \
#   --max-tokens 1000000 \
#   --beam 10 \
#   --scoring wer \
#   --lm-path /path/to/external_language_model.pt \
#   --lm-weight 0.5 \
#   --word-score -1 \
#   --results-path /path/to/output/results
{% endhighlight %}

---

## Research Highlights & Key Findings

Our research with ArTST has yielded several key insights into Arabic speech processing:

### MSA Performance (ArTST v1)

<div style="font-size: 16px; text-align: justify;">
     <strong>ASR:</strong> In our <a href="https://aclanthology.org/2023.arabicnlp-1.5.pdf">first ArTST paper</a>, we describe ASR fine-tuning experiments on MSA, where we fine-tune using the MGB2 benchmark dataset. The result is comparable to the current SOTA, with 12.8% WER. Compared to supervised multilingual models like Whisper and MMS, both medium and large versions, ArTST v1 results in half the error rate (at least) with a fraction of the parameters. The model showed some potential for recognizing dialects as well, but it wasn't optimized for that. We also trained the model for ADI17 dialect identification benchmark, and got SOTA 94% accuracy. <br><br>
     <strong>TTS:</strong> We also fine-tuned the model for <a style="color:Tomato">Text-to-Speech</a> synthesis on the <a href="/clartts/">ClArTTS</a> dataset. One of our interesting findings is that reasonable TTS performance can be achieved without the inclusion of diacritics. Prior efforts on Arabic TTS rely on diacritic restoration as TTS systems are generally trained with relatively small amounts of data and rely on short contexts for sound styntehsis. The lack of diacritics means that the model has to infer the pronunciation of short vowels from context, which is far-fetched unless large amounts of data (and large models) are used. Due to ArTST's pre-training on ~1000 hours of Arabic speech, we were able to achieve decent TTS performance without the use of any diacritics, using roughly 12 hours of TTS training data. Furthermore, we experimented with using ASR data from MGB2 to do "pre-fine-tuning" for TTS, where we first fine-tune on MGB2 data, then ClArTTS. Note that ASR data are generally not suitable for TTS training since the data is generally noisy and inconsistent in style, speaking rate, emotion, etc. What we find is that this process results in further improvements in TTS quality. We refer to this model a ArTST*; you can <a style="color:Tomato">listen to some samples <a href="https://speechsample.wixsite.com/artsttts">here</a>. </a>
 </div>

### Dialectal ASR and Code-Switching (ArTST v2 & v3)

<div style="font-size: 16px; text-align: justify;">
     In our subsequent <a href="https://arxiv.org/pdf/2411.05872">paper</a> we describe ASR experiments using our v2 checkpoint. This checkpoint is trained with roughly 3000 hours of speech, spanning 17 dialectal varieties (based on country codes). We conducted ASR fine-tuning experiments on 11 dialcts, in addition to MSA. We also kept 3 dialects for zero-shot testing. In the same paper, we also describe the multi-lingual checkpoint, v3, which includes English, French, and Spanish. The multilingual pre-training and fine-tuning enables the model to handle cases of code-switching. The paper details various experiments on pre-training effects, dialect ID forcing, dialect ID inference, and joint dialectal training. Based on these findings, we recommend the following:
 </div>
 
 - For MSA, v1 and v2 perform equally well. 
 - For dialects, the joint v2 model with dialect inference achieves the best performance on average. 
 - Neither v1 nor v2 can handle code-switching with other languages. 
 - v3 performs well on Arabic-English and Arabic-French code-switching, at the cost of somewhat lower performance on monolinual Arabic. 

### Diacritics in Speech Processing (ArTST v1.5 & Multimodal Methods)

<div style="font-size: 16px; text-align: justify;">
 Speech presents an interesting avenue for diacritization research. We did <a href="https://www.isca-archive.org/interspeech_2023/aldarmaki23_interspeech.pdf">a study on ASR diacritization performance</a>, published at INTERSPEECH 2023, comparing it with post-processing using text-based diacritic restoration models (the standard diacritic restoration type of model). Diacritization directing using ASR results in far better performance. <br>
 <strong>Multimodal Diacritic Restoration:</strong> Subsequently, we explored the potential of speech as an additional signal for diacritic restoration. As a use case, consider all the resources available for Arabic ASR, such as MGB2 (1000 hours) and QASR (2000 hours); these datasets contain speech and text transcripts, but most of the text contains no diacritics. Could we use both the speech and text for accurate diacritic restoration? Indeed, our <a href="https://aclanthology.org/2024.naacl-long.233.pdf">paper describing such model</a> was published at NAACL 2024. The proposed model incorporated a Whisper model fine-tuned on our <a href="/clartts/">ClArTTS</a> dataset to produce diacritized transcripts, in addition to the raw text input. Using cross-attention, the network integrates predictions from ASR, as well as the correct undiacritized reference text, to restore the missing diacritics. Our findings reveal that such approach is effective and reduced diacritic error rates by half on the ClArTTS test set. We also tested on out-of-domain MSA data and observed some reduction in DER, but the effect was smaller. Overall, DER on MSA data was rather high, even when using popular open and closed text-based diacritic restoration model. <br>
 <strong>Data Augmentation:</strong>Our <a href="https://aclanthology.org/2024.arabicnlp-1.15.pdf">subsequent paper</a> explored a data augmentation technique to improve the generaalization of the system. The proposed augmentation method consists of random diacritics applied to text, then synthesizing speech based on these randomly diacritized text using a commercial TTS system. The intuition behind this approach is to reduce the dependency of the model's predictions on the textual context to push the model towards better model of the acoustic properties that correspond to a given diacritic. This technique consistently improved diacritic recognition performance on all models and datasets. <br>
 <strong>Pre-training with Diacritics:</strong> Since our ArTST models are pre-trained without diacritics (s most speech resourced don't include diacritics), this may have an effect on fine-tuning performance when diacritics are included, as that requires some unlearning of the patterns aquired in the pre-training phase. Indeed, we find that this effect is most evident for TTS, where pre-training hampers the model's performance when diacritics are included. That is one reason why our original TTS models based on ArTST were undiacritized. Since pre-training does not require alignment between the speech and text data, We pre-trained another version of ArTST (v1.5) using MGB2 audio and diacritized text from Tashkeela. We then fine-tuned TTS with diacritics and compared with v1. Our hypothesis was supported, and the model fine-tuned from v1.5 was significantly better than the one fine-tuned on v1, everything else being equal. 
 </div>

---

## Conclusion

The ArTST project provides open-source models specifically designed for the complexities of Arabic speech. By focusing on Arabic data from the start and tailoring versions for MSA, dialects, code-switching, and diacritics, ArTST offers state-of-the-art performance across various tasks. We encourage researchers and developers to explore the models on the [Hugging Face Hub](https://huggingface.co/models?search=mbzuai-nlp/artst) and contribute to the ongoing development on [GitHub](https://github.com/mbzuai-nlp/ArTST).
