---
title: ""
layout: splash
permalink: /clartts/
author_profile: false
---


<div class="dataset-wrapper">
  <div class="dataset-sidebar">
    <img src="/assets/images/clartts.png" alt="ClArTTS Dataset" />
    
  </div>
  <div class="dataset-main">
  <p><strong>Classical Arabic Text-to-Speech Corpus</strong></p>
    <p>
      <a rel="license" href="https://creativecommons.org/public-domain/">
      <img alt="Creative Commons License" style="height: 20px; border-width:0" 
          src="https://mirrors.creativecommons.org/presskit/buttons/80x15/png/publicdomain.png" />
      </a>
  </p>
    <ul class="dataset-features">
      <li><i data-vi="hourglass" data-vi-size="20"></i> 12 hours</li>
      <li><i data-vi="user"  data-vi-size="20"></i> 1 male speaker</li>
      <li><i data-vi="chat"  data-vi-size="20"></i> 9,705 utterances</li>
      <li><i data-vi="cog"  data-vi-size="20"></i> TTS, ASR</li>
    </ul>
    <p>
          <i class="fas fa-file-pdf"></i> <a class="pub-link" href="https://www.isca-archive.org/interspeech_2023/kulkarni23_interspeech.pdf">PDF</a> &nbsp;
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF" style="height: 1em; vertical-align: middle; margin-right: 4px;">
        <a class="pub-link" href="https://huggingface.co/datasets/MBZUAI/ClArTTS">Dataset</a> &nbsp;
        <i class="fas fa-quote-right"></i> <span class="bibtex-toggle pub-link" onclick="this.nextElementSibling.style.display = (this.nextElementSibling.style.display === 'block') ? 'none' : 'block';">BibTeX</span>
        <span class="bibtex-box">
@inproceedings{kulkarni2023clartts,
  title={ClArTTS: An Open-Source Classical Arabic Text-to-Speech Corpus},
  author={Kulkarni, Ajinkya and Kulkarni, Atharva and Shatnawi, Sara Abedalmon'em Mohammad and Aldarmaki, Hanan},
  booktitle={Proc. Interspeech 2023},
  pages={5511--5515},
  year={2023},
  doi={10.21437/Interspeech.2023-2224}
}
  </span>
  <div style="font-size: 16px; text-align: justify;">
    <p>The Classical Arabic Text-to-Speech corpus is constructed using audio from the <a href="https://librivox.org/pages/public-domain/"> LibriVox</a> project (public domain). Specifically, we used a single audiobook, <em><a href="https://librivox.org/kitab-adab-al-dunya-wal-din-the-ethics-of-religion-and-of-this-world-by-abu-al-hasan-ali-ibn-muhammad-ibn-habib-al-mawardi/">Kitab Adab al-Dunya w'al-Din</a> (972 - 1058 AD)</em>, recorded by a male speaker. The audio is sampled at 40100 Hz. We processed and segmented the original audio into shorter segments from 2 to 10 seconds, and discarded some samples that diverge in speaking style. In total, we kept around 12 hours of audio, and split it into train:test subsets (9,500:205 utterances). Before segmentation, we recruited native Arabic speakers to manually transcribe and validate the audio, including full diacritics. The dataset has been used for research on Arabic text-to-speech, ASR, and diacritic restoration. Check out the paper for more details on dataset construction and text-to-speech baselines. </p>
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
