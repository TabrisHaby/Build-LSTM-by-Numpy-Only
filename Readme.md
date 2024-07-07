## About this Project

This is a repo for LSTM algorithm without using PyTorch and Tensorflow in Python.
I got this as an interview question from Sh____ for 3rd round tech interview, which I didn't pass.
So I want to spend some time to figure it out, just in case i meet this in the future. :)

## Environment

- Python: Version 3.11
- VSCode : Version 1.91.0

## About LSTM

- LSTM is a commonly use neural network algorithm, mainly for time series or LLM predition. This algorithm was designed to avoid gradient explosion and gradient vanish in RNN. It use "cell" to decide which info should be kept or forgetten, combining by using approprate activative function, which helps to keep gradient in a reasonable range, to avoid gradient explosion and gradient vanish.

- Wiki: [https://en.wikipedia.org/wiki/Long_short-term_memory#:~:text=Long%20short%2Dterm%20memory%20(LSTM,and%20other%20sequence%20learning%20methods.](Long Short Term Memory)

-![alt text](./lstm.png)
