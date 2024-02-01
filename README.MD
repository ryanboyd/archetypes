# Archetypes!

This is a library developed to run what might be called a "souped-up dictionary method" for psychological text analysis. Or any kind of text analysis, really.

The core idea behind *Archetypes* is that you pre-define a set of prototypical sentences that reflect the construct that you are looking to measure in a body of text. Using modern contextual embeddings, then, this library will aggregate your prototypes into an *archetypal* representation of your construct. Then, you can quantify texts in your corpus for their semantic similarity to your construct(s) of interest.

*Note*: For the curious: no, this approach not inspired by anything [Jungian](https://en.wikipedia.org/wiki/Jungian_archetypes) in nature. In the past, I've [said a few things](https://www.tandfonline.com/doi/full/10.1080/1047840X.2019.1633122) about Jungian archetypes that have inspired scholars to write more than a few frustrated e-mails to me. Apologies to the Jungians.

# Examples

I have provided an example notebook in this repo that walks through the basic process of using this library, along with demonstrations of a few important "helper" functions to help you evaluate the statistical/psychometric qualities of your archetypes.


# Citation

This method is originally described in the following forthcoming paper:
```
@inproceedings{varadarajan_archetypes,
	address = {St. Julians, Malta},
	title = {Archetypes and {Entropy}: {Theory}-{Driven} {Extraction} of {Evidence} for {Suicide} {Risk}},
	booktitle = {Proceedings of the {Tenth} {Workshop} on {Computational} {Linguistics} and {Clinical} {Psychology}},
	publisher = {Association for Computational Linguistics},
	author = {Varadarajan, Vasudha and Lahnala, Allison and Ganesan, Adithya V. and Dey, Gourab and Mangalik, Siddharth and Bucur, Ana-Maria and Soni, Nikita and Rao, Rajath and Lanning, Kevin and Vallejo, Isabella and Flekova, Lucie and Schwartz, H. Andrew and Boyd, Ryan L.},
}
```

The citation above will be updated once the paper is actually published 😊