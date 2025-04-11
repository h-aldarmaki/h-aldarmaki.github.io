---
title: ""
permalink: /posts/
layout: posts
author_profile: false
---

## Latest Posts

<div class="post-card-grid">
  {% assign posts = site.posts | slice: 0, 2 %}
  {% for post in posts %}
    <div class="post-card">
      <a href="{{ post.url | relative_url }}">
        <img src="{{ post.image | default: '/assets/images/default.jpg' }}" alt="{{ post.title }}">
        <h3>{{ post.title }}</h3>
        <p>{{ post.excerpt | strip_html | truncate: 100 }}</p>
      </a>
    </div>
  {% endfor %}
</div>


<!-- <p><a href="/quotes/">See all quotes →</a></p> -->

<style>
    .post-card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 300px));
      gap: 2rem;
      margin-top: 2rem;
      justify-content: center;
    }
    
    .post-card {
      border: 1px solid #eee;
      border-radius: 12px;
      overflow: hidden;
      transition: box-shadow 0.2s;
      background: white;
    }
    
    .post-card img {
        display: block;
        margin: 0 auto;
        max-width: 80%;
        height: 100px;
        object-fit: cover;
        box-shadow: none !important;

    }
    .post-card img:hover {
        box-shadow: none !important;
    }
    .post-card h3 {
      margin: 0.5rem 1rem 0 1rem;
      font-size: 0.8rem;
      color: #8a0303;
    }
    
    .post-card p {
      margin: 0.5rem 1rem 1rem 1rem;
      font-size: 0.7rem;
      color: #444;
    }
    .post-card a {
        text-decoration: none;
        color: inherit;
    }
    .post-card a:hover,
    .post-card a:hover h3,
    .post-card a:hover p {
        text-decoration: none;
        color: inherit;
    }

</style>
    
