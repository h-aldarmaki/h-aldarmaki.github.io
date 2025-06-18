---
title: "Clinical Annotations for Automatic Stuttering Severity Assessment"
date: 2025-06-09
last_modified_at: 2025-06-15
categories: projects
tags: [stuttering, speech pathology, clinical annotations, fluencybank, speech disorders, machine learning, automatic assessment, speech therapy, disfluency, clinical expertise]
layout: single
author_profile: false
read_time: true
share: true
comments: true
toc: true
toc_sticky: true
image: /assets/images/stutter.jpg
excerpt: "Explore our enhanced FluencyBank dataset with expert clinical annotations for automatic stuttering severity assessment, featuring multi-modal labeling and consensus-based evaluation standards."
author: Rufael Marew
---

<div>
  <img src="/assets/images/stutter.jpg" alt="Clinical Stuttering Assessment" style="float: left; margin-right: 1rem; max-width:140px"/>
</div>

<div style="font-size: 16px; text-align: justify;">
Stuttering affects millions of people worldwide, yet accurate assessment and treatment require specialized clinical expertise that isn't universally available. While machine learning offers promise for automated assessment tools, the complexity of stuttering disorders demands training data that truly reflects real-world clinical standards and practices.<br>
</div>

<div style="font-size: 16px; text-align: justify;">
<p>Our research addresses this critical gap by enhancing the FluencyBank dataset with comprehensive clinical annotations created by expert speech-language pathologists. Rather than relying on simplified or non-expert labeling, we prioritized clinical validity by hiring certified clinicians to provide multi-modal annotations that capture the full complexity of stuttering assessment. This work represents a significant step toward building machine learning models that can genuinely support clinical practice and improve accessibility to quality stuttering assessment.<br>

The enhanced dataset includes detailed annotations for stuttering moments, secondary behaviors, and tension scores using established clinical standards. By incorporating both audio and visual features, our annotations provide a comprehensive foundation for developing robust automatic assessment tools. Most importantly, we provide expert consensus annotations for reliable model evaluation, ensuring that automated systems can be properly validated against clinical expertise.
</p>
</div>

## Enhanced Clinical Annotations and Dataset

<div style="font-size: 16px; text-align: justify;">
Stuttering assessment involves multiple dimensions that require specialized training to identify and classify accurately. Our annotation scheme captures these clinical complexities through four key components: stuttering moments with precise temporal boundaries and type classification, secondary behaviors detected using audiovisual features, tension scores based on clinical rating scales, and consensus labels providing highly reliable ground truth through expert agreement.
</div>

| Assessment Component | Clinical Focus | Annotation Details |
| :------------------- | :------------- | :----------------- |
| **Stuttering Moments** | Core disfluencies (repetitions, blocks, prolongations) | Precise temporal boundaries and type classification |
| **Secondary Behaviors** | Associated physical movements | Multimodal detection using audiovisual features |
| **Tension Scores** | Severity rating of physical struggle | Clinical rating scales applied by expert annotators |
| **Consensus Labels** | Highly reliable ground truth | Multiple expert agreement for evaluation standards |

<div style="font-size: 16px; text-align: justify;">
The FluencyBank dataset, while valuable, previously lacked the detailed clinical annotations necessary for training robust automatic assessment models. To ensure clinical validity, we implemented a rigorous annotation process involving certified speech-language pathologists with specialized stuttering expertise. Our protocol includes multi-modal analysis of audio and video content, standardized clinical assessment tools, inter-rater reliability measures with consensus-building for complex cases, and regular quality assurance through calibration sessions and expert supervision.
</div>

<div style="margin-top: 20px; margin-bottom: 25px; text-align: center;">
  <a href="https://github.com/mbzuai-nlp/CASA.git" class="btn btn--github" target="_blank" rel="noopener noreferrer"><i class="fab fa-github"></i> View Dataset on GitHub</a>
  <a href="https://arxiv.org/abs/2506.00644" class="btn btn--info" target="_blank" rel="noopener noreferrer"><i class="fas fa-book"></i> View Paper on ArXiv</a>
</div>

<div style="font-size: 16px; text-align: justify;">
Our annotation scheme captures the essential elements that clinicians assess during stuttering evaluation. Each disfluent moment is precisely labeled with temporal boundaries and classified according to clinical categories, enabling models to learn not just detection, but also the specific types of disfluencies that inform severity assessment. Beyond core stuttering moments, our annotations capture secondary behaviors such as physical tension, avoidance strategies, and associated movements that are crucial for comprehensive assessment but often overlooked in automated systems. Tension scores and overall severity ratings are provided by expert clinicians using established clinical scales, enabling training of models that can predict not just the presence of stuttering, but its clinical significance and impact.
</div>

## Inter-Annotator Agreement and Reliability

<div style="font-size: 16px; text-align: justify;">
A critical aspect of our annotation process was establishing and measuring inter-annotator agreement to ensure the reliability and consistency of our clinical labels. Given the subjective nature of stuttering assessment, quantifying agreement between expert clinicians provides essential validation of our annotation quality and demonstrates the feasibility of creating reliable training data for machine learning models.
</div>

<div style="font-size: 16px; text-align: justify;">
Our dataset includes a total of 1,654 and 4,037 non-overlapping stuttering spans in the reading and interview sections of FluencyBank dataset, respectively. Each span was annotated with a tension score, a primary type, and a secondary type, though in some cases only primary or secondary types were observed. We measured Inter-Annotator Agreement (IAA) across multiple annotation dimensions using established methodologies.
</div>


To ensure our annotations are reliable, we systematically calculated inter‑annotator agreement across temporal spans, disfluency types, and tension scores. Here’s how we approached it:

### 1. Grouping / Aligning Annotations  
- We applied *agglomerative clustering* on annotated time segments using **Intersection-over-Union (IoU)** as the distance measure.  
- Annotations that overlapped sufficiently were clustered together, creating comparable units for agreement computation.

### 2. Applying Agreement Metrics  
For each component, we used **Krippendorff’s α (alpha)** — a robust reliability coefficient suitable for different data types and multiple annotators. it is computed as follows

Let **Dₒ** be the observed disagreement and **Dₑ** the expected disagreement:
<div style="text-align: left; margin-left: 30%;">
  α = 1 – (Dₒ / Dₑ)
</div>

> For details, see Krippendorff’s explanation:  
> [Krippendorff, K. (2013). "Computing Krippendorff’s Alpha-Reliability".](https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf)

The disagreements for each category are computed as follows:

| Component              | Data Type        | Distance/Similarity Metric           | Krippendorff’s α |
|------------------------|------------------|--------------------------------------|------------------|
| Temporal spans         | Interval         | 1 – IoU (interval distance)          | 0.68             |
| Primary/Secondary types| Nominal          | Binary distance (match/mismatch)     | Varies by class  |
| Tension scores         | Ordinal          | Normalized rank‑Euclidean distance   | 0.18             |

<!-- 


- For *intervals*, agreement is based on temporal overlap (IoU).
- For *nominal labels*, exact matches count as agreement.
- For *ordinal ratings* (tension), distances are weighted by their rank differences. -->


You can compute α using our implementation on [Github](https://github.com/rufaelfekadu/IAA.git)

Run the following commands

{% highlight bash %}
git clone https://github.com/rufaelfekadu/IAA.git
cd IAA
pip install numpy pandas matplotlib scikit-learn
{% endhighlight %}

Download the dataset from [Here](https://github.com/mbzuai-nlp/CASA)

{% highlight bash %}
wget https://raw.githubusercontent.com/mbzuai-nlp/CASA/refs/heads/main/data/Voices-AWS/total_dataset.csv
{% endhighlight %}

Use the following script to compute the IAA

{% highlight python %}
import pandas as pd
import numpy as np
import json
from agreement import InterAnnotatorAgreement
from granularity import SeqRange

input_path = "total_dataset.csv"
item_col = "item"
annotator_col = "annotator"
LABELS = ['SR','ISR','MUR','P','B', 'V', 'FG', 'HM', 'ME']


def binary_distance(x, y):
    return 1 if x != y else 0

# euclidian distance
def euclidian_distance(a1, a2, max_value=3):
    return abs(a1 - a2) / max_value

# Intersection Over Union
def iou(vrA, vrB):
    xA = max(vrA.start_vector[0], vrB.start_vector[0])
    xB = min(vrA.end_vector[0], vrB.end_vector[0])

    interrange = max(0, xB - xA + 1) 
    unionrange = (vrA.end_vector[0] - vrA.start_vector[0] + 1) + (vrB.end_vector[0] - vrB.start_vector[0] + 1) - interrange
    return (interrange / unionrange)

def inverse_iou(vrA, vrB):
    return 1 - iou(vrA, vrB)

# Read the CSV file
grannodf = pd.read_csv(input_path)
results = {}
grannodf = grannodf[~grannodf['annotator'].isin(['Gold','bau','mas','sad'])]
print(grannodf['annotator'].unique())

grannodf['timevr'] = grannodf[['start','end']].apply(lambda row: SeqRange(row.to_list()), axis=1)

# compute IAA for each class
for label in LABELS[:-1]:
    iaa = InterAnnotatorAgreement(grannodf, 
                                  item_colname=item_col, 
                                  uid_colname=annotator_col, 
                                  label_colname=label,
                                  distance_fn=binary_distance)
    iaa.setup()
    results[label] = {
        'alpha': iaa.get_krippendorff_alpha(),
        'ks': iaa.get_ks(),
        'sigma': iaa.get_sigma(use_kde=False),
    }
    # average  number of annotaions per annotator for a given class
    results[label]['num_annotations'] = grannodf[grannodf[label] == 1].groupby(annotator_col).size().mean()

# compute IAA for tension
iaa = InterAnnotatorAgreement(grannodf, 
                              item_colname=item_col, 
                              uid_colname=annotator_col, 
                              label_colname='T',
                              distance_fn=euclidian_distance)
iaa.setup()
results['T'] = {
    'alpha': iaa.get_krippendorff_alpha(),
    'ks': iaa.get_ks(),
    'sigma': iaa.get_sigma(use_kde=False),
    'num_annotations': grannodf[grannodf['T'] >= 1].groupby(annotator_col).size().mean()
}

# compute IAA for time-intervals
iaa = InterAnnotatorAgreement(grannodf, 
                              item_colname=item_col, 
                              uid_colname=annotator_col, 
                              label_colname='timevr',
                              distance_fn=inverse_iou)
iaa.setup()
results['time-intervals'] = {
    'alpha': iaa.get_krippendorff_alpha(),
    'ks': iaa.get_ks(),
    'sigma': iaa.get_sigma(use_kde=False),
    'num_annotations': grannodf.groupby(annotator_col).size().mean()
}


with open(f'iaa_results.json', 'w') as f:
    json.dump(results, f, indent=4)


{% endhighlight%}
{: .copy-code}


### 3. Interpretation  
<div style="text-align: center;">
  <img src="/assets/images/posts/agreement.png" alt="Clinical Stuttering Assessment" style="width: 70%;"/>
</div>

- A **span α of 0.68** indicates substantial temporal agreement.  
- **Primary/secondary type α values** vary by disfluency class, but generally show moderate consistency.  
- The low **tension score α (0.18)** highlights the difficulty and subjectivity in rating physical struggle.




## Applications and Expert Consensus Standards

<div style="font-size: 16px; text-align: justify;">
The enhanced dataset supports various research directions and practical applications in automated stuttering assessment. With precise temporal annotations, researchers can train models to automatically detect stuttering moments in continuous speech, while the multi-modal nature supports systems that leverage both audio and visual cues, mirroring clinical assessment practices. The inclusion of clinical severity ratings enables development of automated systems that can predict stuttering severity scores, significantly improving accessibility to preliminary assessment tools, particularly in underserved areas.
</div>

<div style="font-size: 16px; text-align: justify;">
A critical contribution of our work is the provision of expert consensus annotations for model evaluation. This consensus dataset addresses a fundamental challenge in stuttering research: establishing reliable ground truth for complex clinical assessments. Our consensus annotations are created through a structured process where multiple expert clinicians provide independent initial annotations, followed by systematic discussion and resolution of annotation differences, resulting in final consensus labels agreed upon by the expert panel. The consensus dataset is thus can be used for standardized evaluation.
</div>

## Research Findings and Clinical Impact

<div style="font-size: 16px; text-align: justify;">
Our research reveals several important insights about automatic stuttering assessment. The analysis demonstrates that stuttering annotation requires extensive clinical expertise, as attempts to use non-expert annotators or simplified labeling schemes consistently produce inadequate training data that fails to capture the nuances essential for clinical validity. This finding underscores the importance of involving qualified speech-language pathologists in dataset development.
</div>

<div style="font-size: 16px; text-align: justify;">
Incorporating both audio and visual features significantly improves automatic detection of stuttering behaviors, particularly secondary behaviors and tension indicators that may not be apparent from audio alone. This mirrors clinical practice where visual observation is essential for comprehensive assessment. The expert consensus annotations provide a robust foundation for model evaluation that addresses the inherent subjectivity in stuttering assessment, with models trained and evaluated on consensus data showing better alignment with clinical judgment and improved generalization across different clinical contexts.
</div>

<div style="font-size: 16px; text-align: justify;">
Our experiments highlight the critical importance of clinical validation in automated stuttering assessment. Models that perform well on traditional metrics may still fail to meet clinical standards, emphasizing the need for expert-annotated evaluation datasets. This work opens several promising avenues for future research and clinical application, including expanded dataset development to include more diverse populations and stuttering severities, development of real-time assessment tools that could support telepractice and remote therapy sessions, and automated treatment monitoring capabilities that provide objective progress tracking and outcome measurement tools.
</div>

---

## Conclusion

The enhancement of FluencyBank with expert clinical annotations represents a significant advancement in stuttering research and assessment. By prioritizing clinical validity and expert knowledge, this work provides a foundation for developing automated assessment tools that can genuinely support clinical practice. The multi-modal annotation scheme, expert consensus process, and rigorous evaluation framework establish new standards for stuttering assessment research.

We encourage researchers and clinicians to explore the enhanced dataset and contribute to the ongoing development of clinically valid automatic stuttering assessment systems. Through continued collaboration between machine learning researchers and speech-language pathologists, we can work toward improving accessibility and quality of stuttering assessment and treatment services worldwide.

For access to the enhanced FluencyBank dataset and annotation guidelines, please visit our project repository and follow the clinical research protocols for responsible use of this sensitive clinical data.