---
title: ""
layout: splash
permalink: /mixat/
author_profile: false
---


<div class="dataset-wrapper">
  <div class="dataset-sidebar">
    <img src="/assets/images/mixat.png" alt="Mixat Dataset" />
    
  </div>
  <div class="dataset-main">
  <p><strong>Mixat: A Data Set of Bilingual Emirati-English Speech</strong></p>
    <p>
      <a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
      <img alt="Creative Commons License" style="height: 20px; border-width:0" 
          src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png" />
      </a>
  </p>
    <ul class="dataset-features">
      <li><i data-vi="hourglass" data-vi-size="20"></i> 15 hours</li>
      <li><i data-vi="user"  data-vi-size="20"></i> 5+ male speakers (train), 1 female speaker (test)</li>
      <li><i data-vi="chat"  data-vi-size="20"></i> 5,316 utterances</li>
      <li><i data-vi="cog"  data-vi-size="20"></i> ASR, Code-switching</li>
    </ul>
    <p>
          <i class="fas fa-file-pdf"></i> <a class="pub-link" href="https://aclanthology.org/2024.sigul-1.26.pdf">Mixat</a> &nbsp;
           <i class="fas fa-file-pdf"></i> <a class="pub-link" href="https://aclanthology.org/2024.findings-emnlp.356.pdf">PolyWER</a> &nbsp;
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF" style="height: 1em; vertical-align: middle; margin-right: 4px;">
        <a class="pub-link" href="https://huggingface.co/datasets/sqrk/mixat-tri">Dataset</a> &nbsp;
        <i class="fas fa-quote-right"></i> <span class="bibtex-toggle pub-link" onclick="this.nextElementSibling.style.display = (this.nextElementSibling.style.display === 'block') ? 'none' : 'block';">Mixat</span>
        <span class="bibtex-box">
@inproceedings{al-ali-aldarmaki-2024-mixat,
    title = "Mixat: A Data Set of Bilingual Emirati-{E}nglish Speech",
    author = "Al Ali, Maryam Khalifa and Aldarmaki, Hanan",
    booktitle = "Proceedings of the 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages @ LREC-COLING 2024",
    year = "2024",
}
  </span>
    <i class="fas fa-quote-right"></i> <span class="bibtex-toggle pub-link" onclick="this.nextElementSibling.style.display = (this.nextElementSibling.style.display === 'block') ? 'none' : 'block';">PolyWER</span>
        <span class="bibtex-box">
@inproceedings{kadaoui-etal-2024-polywer,
    title = "{P}oly{WER}: A Holistic Evaluation Framework for Code-Switched Speech Recognition",
    author = "Kadaoui, Karima  and Ali, Maryam Al  and Toyin, Hawau Olamide  and Mohammed, Ibrahim  and Aldarmaki, Hanan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.356/",
}
  </span>
  <div style="font-size: 16px; text-align: justify;">
    <p>Mixat is an ASR dataset of Emirati Arabic speech, code-mixed with English. The dataset consists of 15 hours of speech derived from two public podcasts featuring native Emirati speakers, extracted with the hosts' permission (to be used for research purposes). The <a href="https://www.youtube.com/channel/UCZbKz4QeFWbfMVE0fSJeuUw">first podcast</a> (used as a train set) consists of conversations between the host and various guests, from which we extracted 3,728 utterances. The <a href="https://open.spotify.com/show/3yEonEQO8Jfu4plB6B78HE?si=04c16d09c4dd49e2">second podcast</a> (test set) is a single-speaker podcast, from which we extracted 1,588 segments. Check out the <a href="https://aclanthology.org/2024.sigul-1.26.pdf">Mixat paper </a> for more details on dataset construction.</p> <br>   
    <p>In addition to the original reference transcriptions, where code-switched English segments are transcribed in the Latin script, we added two additional transcription types: one with transliterations of English segments into Arabic script, and one with their translation into Emirati Arabic. The transliterated/translated segments are marked with brackets to easily separate them from the original Arabic text. These additional annotations were added to support the implementation of the <a href="https://aclanthology.org/2024.findings-emnlp.356.pdf">PolyWER</a> metric. ❗ Note that the transcriptions in Huggingface have been modified from the original ones provided in Mixat, where transcription errors have been found and corrected. If you use the dataset or transcriptions provided in Huggingface, <u>place cite both papers</u>. 
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
