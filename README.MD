# Archetypes!

This is a library developed to run what might be called a "souped-up dictionary method" for psychological text analysis. Or any kind of text analysis, really.

The core idea behind *Archetypes* is that you pre-define a set of prototypical sentences that reflect the construct that you are looking to measure in a body of text. Using modern contextual embeddings, then, this library will aggregate your prototypes into an *archetypal* representation of your construct. Then, you can quantify texts in your corpus for their semantic similarity to your construct(s) of interest.

*Note*: For the curious: no, this approach not inspired by anything [Jungian](https://en.wikipedia.org/wiki/Jungian_archetypes) in nature. In the past, I've [said a few things](https://www.tandfonline.com/doi/full/10.1080/1047840X.2019.1633122?casa_token=cnHLr5uwiXUAAAAA:ACUCFK4tC9HXBBMlfhIFjfPxWLuCEK7owT3z_IBv2rZUa35fq4Z_rVkETqGO0wa1FqYjbmgxcRWxnw) about Jungian archetypes that have inspired scholars to write more than a few frustrated e-mails to me. Apologies to the Jungians.

# Installation

This package is easily installable via pip via the following command:

`pip install archetyper`


# Requirements
If you want to run the library without `pip` installing as shown above, you will need to first install the following packages:
- `numpy`
- `tqdm`
- `torch`
- `sentence_transformers`
- `nltk`

You can try to install these all in one go by running the following command from your terminal/cmd:

`pip install numpy tqdm torch sentence_transformers nltk`



# Examples

I have provided an example notebook in this repo that walks through the basic process of using this library, along with demonstrations of a few important "helper" functions to help you evaluate the statistical/psychometric qualities of your archetypes.


# Citation

This method is originally described in [this paper](https://aclanthology.org/2024.clpsych-1.28/):
```
@inproceedings{varadarajan-etal-2024-archetypes,
    title = "Archetypes and Entropy: Theory-Driven Extraction of Evidence for Suicide Risk",
    author = "Varadarajan, Vasudha  and
      Lahnala, Allison  and
      Ganesan, Adithya V. and
      Dey, Gourab  and
      Mangalik, Siddharth  and
      Bucur, Ana-Maria  and
      Soni, Nikita  and
      Rao, Rajath  and
      Lanning, Kevin  and
      Vallejo, Isabella  and
      Flek, Lucie  and
      Schwartz, H. Andrew  and
      Welch, Charles  and
      Boyd, Ryan L.",
    editor = "Yates, Andrew  and
      Desmet, Bart  and
      Prud{'}hommeaux, Emily  and
      Zirikly, Ayah  and
      Bedrick, Steven  and
      MacAvaney, Sean  and
      Bar, Kfir  and
      Ireland, Molly  and
      Ophir, Yaakov",
    booktitle = "Proceedings of the 9th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2024)",
    month = mar,
    year = "2024",
    address = "St. Julians, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.clpsych-1.28",
    pages = "278--291",
    abstract = "Research on psychological risk factors for suicide has developed for decades. However, combining explainable theory with modern data-driven language model approaches is non-trivial. In this study, we propose and evaluate methods for identifying language patterns aligned with theories of suicide risk by combining theory-driven suicidal archetypes with language model-based and relative entropy-based approaches. Archetypes are based on prototypical statements that evince risk of suicidality while relative entropy considers the ratio of how unusual both a risk-familiar and unfamiliar model find the statements. While both approaches independently performed similarly, we find that combining the two significantly improved the performance in the shared task evaluations, yielding our combined system submission with a BERTScore Recall of 0.906. Consistent with the literature, we find that titles are highly informative as suicide risk evidence, despite the brevity. We conclude that a combination of theory- and data-driven methods are needed in the mental health space and can outperform more modern prompt-based methods.",
}
```
