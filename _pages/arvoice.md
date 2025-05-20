---
title: ""
layout: splash
permalink: /arvoice/
author_profile: false
---


<div class="dataset-wrapper">
  <div class="dataset-sidebar">
    <img src="/assets/images/arvoice.png" alt="ArVoice Dataset" />
    
  </div>
  <div class="dataset-main">
  <p><strong>ArVoice: A Multi-Speaker Dataset for Arabic Speech Synthesis</strong></p>
    <p>
      <a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
      <img alt="Creative Commons License" style="height: 20px; border-width:0" 
          src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png" />
      </a>
  </p>
    <ul class="dataset-features">
      <li><i data-vi="hourglass" data-vi-size="20"></i> 84 hours</li>
      <li><i data-vi="user"  data-vi-size="20"></i> 7 human speakers (train/test), 4 synthetic speakers (train/test)</li>
      <li><i data-vi="chat"  data-vi-size="20"></i> X utterances</li>
      <li><i data-vi="cog"  data-vi-size="20"></i> TTS, Diacritic Restoration</li>
    </ul>
    <p>
          <i class="fas fa-file-pdf"></i> <a class="pub-link" href="/">ArVoice</a> &nbsp;
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF" style="height: 1em; vertical-align: middle; margin-right: 4px;">
        <a class="pub-link" href="https://huggingface.co/datasets/herwoww/ArVoice">Dataset</a> &nbsp;
        <i class="fas fa-quote-right"></i> <span class="bibtex-toggle pub-link" onclick="this.nextElementSibling.style.display = (this.nextElementSibling.style.display === 'block') ? 'none' : 'block';">ArVoice</span>
        <span class="bibtex-box">
@inproceedings{coming soon.
}
  </span>
    
  <div style="font-size: 16px; text-align: justify;">
    <p>ArVoice is a multi-speaker Modern Standard Arabic (MSA) speech corpus with fully diacritized transcriptions, intended  for multi-speaker speech synthesis,  and can be useful for other tasks such as speech-based diacritic restoration, voice conversion, and deepfake detection. ArVoice comprises: (1) a new professionally recorded set from $6$ voice talents with diverse demographics, (2) a modified subset of the Arabic Speech Corpus; and (3) high-quality synthetic speech from $2$ commercial systems. The complete corpus consists of a total of $83.52$ hours of speech across $11$ voices; around $10$ hours consist of human voices from $7$ speakers.The modified subset and full synthetic subset are available on HuggingFace. To access the new professionally recorded subset, <a href="/" > sign this agreement </a> . If you use the dataset or transcriptions provided in Huggingface, <u>place cite the paper</u>. 
</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/vivid-icons@1.0.10" type="text/javascript"></script>

<style>
.dataset-wrapper {
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
  margin-top: 2rem;
}

.dataset-sidebar {
  flex: 1;
  min-width: 100px;
  max-width: 140px;
}

.dataset-sidebar img {
  width: 100%;
  margin-left: 1em;
  margin-top: 1em;
}

.dataset-main {
  flex: 1;
  min-width: 250px;
  max-width: 750px
}

.dataset-features {
  list-style: none;
  padding: 0;
  margin: 1.5rem 0;
}

.dataset-features li {
  display: flex;
  align-items: center;
  margin-bottom: 0.4rem;
}

.dataset-features i,
.dataset-features svg.vi {
  width: 20px;
  height: 20px;
  margin-right: 0.6rem;
  fill: #8a0303;
  flex-shrink: 0;
}
</style>
