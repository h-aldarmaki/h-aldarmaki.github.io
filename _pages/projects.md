---
title: ""
permalink: /projects/
layout: single
author_profile: false
---

<div class="project-list">

  <!-- Project 1 -->
  <a href="/projects/arabic-speech/" class="project-row-link">
  <div class="project-row">    <div class="project-image">
      <img src="/assets/images/artst.jpg" alt="Project 1">
    </div>
    <div class="project-text">
      <h2>Arabic Speech Processing</h2>
      <p>We develop tools and datasets for Arabic speech, including automatic speech recognition, speech synthesis, and diacritic restoration. We work on different variants of Arabic: Classical Arabic, Modern Standard Arabic (MSA), dialects, and code-switching.</p>
    </div>
  </div> </a>

  <!-- Project 2 -->
  <a href="https://www.potion.ae/chatty-check" class="project-row-link">
  <div class="project-row reverse">
    <div class="project-image">
      <img src="/assets/images/dld.png" alt="Project 2">
    </div>
    <div class="project-text">
      <h2>Screening for Developmental Language Disorder</h2>
      <p>In collaboration with speech-language pathologists, we investigate the feasibility of automatic early screening of developmental language disorder (DLD) using video games and speech processing tools.</p>
    </div>
  </div>
  </a>

  <!-- Project 3 -->
  <a href="https://mbzuai-nlp.github.io/stutterbank/" class="project-row-link">
  <div class="project-row">
    <div class="project-image">
      <img src="/assets/images/stutter.jpg" alt="Project 3">
    </div>
    <div class="project-text">
      <h2>Stuttering Assessment</h2>
      <p>In collaboration with speech-language pathologists, we develop clinical annotation schemes and models for automatic stuttering severity assessment, and collect/annotate audiovisual data in multiple languages. </p>
    </div>
  </div>
  </a>

  <!-- Project 4 -->
  <a href="https://ramsalab.ae" class="project-row-link">
  <div class="project-row reverse">
    <div class="project-image">
      <img src="/assets/images/emirati.jpg" alt="Project 4">
    </div>
    <div class="project-text">
      <h2>Ramsa Lab: Spoken Emirati Archive</h2>
      <p>We ethically collect diverse Emirati speech data from various regions and age groups by engaging directly with the public. We develop tools, dictionaries, and AI models for effective processing of Emirati dialects. </p>
    </div>
  </div>
  </a>
  

</div>

<style>

.project-row-link {
  text-decoration: none;
  color: inherit;
  display: block;
  transition: background 0.2s, box-shadow 0.2s;
  border-radius: 8px;
}

.project-row-link:hover {
  background: #f9f9f9;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  /*border-bottom: 2px solid #8a0303;*/
}

/* Remove all native underline styling */
.project-row-link h2, .project-row-link p {
  color: black !important;
  text-decoration: none !important;
  border-bottom: 0px solid transparent; /* simulate underline */
  display: inline-block;
  transition: border-color 0.2s ease;
}

.project-row-link:hover .project-image img {
    transform: scale(1.05);
}

.project-list {
  display: flex;
  flex-direction: column;
  gap: 3rem;
  margin-top: 2rem;
}

.project-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 2rem;
}

.project-row.reverse {
  flex-direction: row-reverse;
}

.project-image {
  flex: 1 1 200px;
  max-width: 150px;
}

.project-image img {
  width: 100%;
  height: auto;
  border-radius: 8px;
  display: block;
}

.project-text {
  flex: 2 1 400px;
  text-align: justify;
}

.project-text h2 {
  margin-top: 0;
  font-size: 0.8rem;
}

.project-text p {
  font-size: 0.7rem;
  line-height: 1.5;
}
</style>

