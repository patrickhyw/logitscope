# Simplescope
![Plot showing improved results](prec1.png)

A simplified and more performant token identity Patchscope from the [Patchscopes paper](https://arxiv.org/abs/2401.06102).

## Background
Section 4.1 of the [Patchscopes paper](https://arxiv.org/abs/2401.06102) describes the token identity Patchscope technique. Briefly, the idea is to take a prompt like `"cat -> cat\n1135 -> 1135\nhello -> hello\n?"`, patch the residual activation from any layer in the last token of some prompt, and see if we can correctly predict the next token. For instance, it would patch the residual of `=` in `"1+1="` into the `?` and would see if it predicts `2`.

## Method
Instead of `"cat -> cat\n1135 -> 1135\nhello -> hello\n?"`, Simplescope just uses `"?"`.

Intuitively, `"cat -> cat\n1135 -> 1135\nhello -> hello\n?"` feels strange:

1. The in-context examples show `->` tokens after the first tokens, so there's a bias to predicting `->` after the `?`.
    - Indeed, the most likely next token after `?` is `->` if early layers are patched, which can explain why `patchscope` has low performance (this differs from the paper's explanation).
2. The in-context examples show a token mapping to itself (e.g. `hello->hello`), not to the next token (e.g. `hello->there`). However, if we patch the residual of `hello` into `?`, we want it to predict `there` and not `hello`.
3. The prompt ends with `?`, not `->`. Performance degrades significantly if you add a `->`, which I think is due to point 2.

`"?"` tries to fix this by just removing the parts that don't make sense. It could perhaps be improved by adding a prompt like `"hello->there;?->"`. However, keeping it simple makes it feel like a more expensive but more accurate companion to the original logit lens, where we do the rest of the forward pass instead of skipping to the decoder.

## Results
`simplescope` performs better than `patchscope` in both precision@1 and surprisal (the two metrics in the Patchscope paper) with the same dataset ([The Pile](https://huggingface.co/datasets/EleutherAI/pile)) and one of the same models ([GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b)). I used the same preprocessing steps with slight changes: 2K examples from the start instead of after an offset of 10K, and reduced the word/character limits to avoid running out of GPU memory.

![Plot showing improved precision@1](prec1.png)
![Plot showing improved surprisal](surprisal.png)

## How to Run
Run `pip install -r requirements.txt` and then all cells in `simplescope.ipynb`. I ran this on an A100 80GB with 100GB of disk space, using Python 3.11.