# Simplescope
![Plot showing improved results](prec1.png)

## Abstract
A simplified and more performant next token Patchscope from the [Patchscopes paper](https://arxiv.org/abs/2401.06102).

## Background
Section 4.1 of the [Patchscopes paper](https://arxiv.org/abs/2401.06102) describes the next token Patchscope approach. Briefly, the idea is to take a prompt like `"cat->cat;135->135;hello->hello;?"`, patch the residual activation from any layer in the last token of some prompt, and see if we can correctly predict the next token.

## Method
Instead of `"cat->cat;135->135;hello->hello;?"`, just use `"?"`.
 - The idea is simple and I'm not claiming novelty, but I haven't seen it published so I wanted to share it.

## Discussion
`"cat->cat;135->135;hello->hello;?"` feels conceptually flawed:

1. The in-context examples show `->` tokens after the first tokens, so there's a bias to predicting `->` after the `?`.
    - Indeed, the most likely next token after `?` is `->` if early layers are patched, which can explain why `patchscope` has low performance (this differs from the paper's explanation).
2. The in-context examples show a token mapping to itself (e.g. `hello->hello`), not to the next token (e.g. `hello->there`). However, if we patch the residual of `hello` into `?`, we want it to predict `there` and not `hello`.
3. The prompt ends with `?`, not `->`. Performance degrades significantly if you add a `->`, which I think is due to point 2.

`"?"` tries to fix this by just removing the parts that don't make sense. As future work, it could perhaps be improved with a different prompt (e.g. `"hello->there;?->"`).

## Results
`simplescope` performs better than `patchscope` in both precision@1 and surprisal (the two metrics in the Patchscope paper) with the same dataset ([The Pile](https://huggingface.co/datasets/EleutherAI/pile)) and one of the same models ([GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b)). I used similar preprocessing steps: 2K instead of 20K examples for cost reasons and reduced the word/character limits to avoid running out of GPU memory.

![Plot showing improved precision@1](prec1.png)
![Plot showing improved surprisal](surprisal.png)

## Running
Run `pip install -r requirements.txt` and then all cells in `simplescope.ipynb`. I ran this on an 80GB A100 with 100GB of disk space, using Python 3.11.