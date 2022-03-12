# Handwriting Calculator using a CNN

This small programm is recognizing handwriten digits and performs caluclations based on input using a optimized Convolutional Neural Network.

---

## Setup

Dependencies are listed in *environment.yml*. The CNN is build using _[tensforflow](https://www.tensorflow.org)_, the GUI is based on a Template by _antaloaalonsa_ ([github](https://github.com/antaloaalonso/CNN-With-GUI)) and build with tkinter.

* run `conda env create -f environment.yml`
* activate env via `conda activate <env name>`

---

## Usage

Run `python calculatorCNN.py` - a secondary GUI window should open up.

1. Write a digit in the gray square
2. Save the digit via the _Save Digit_ Button, **for multi-character digits (i.e. >9) save each digit individually**
3. Choose a operation (add, subtract, multiple, divide)
4. Repeat Step 1. and 2.
5. After the last digit is save, click '=' to view results

If a number is not correct or to reset the whole programm, find buttons at the bottom.