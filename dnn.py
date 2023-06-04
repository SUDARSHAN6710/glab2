{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "tfPmfTms3995"
   },
   "source": [
    "<h2 style=\"color:blue\" align=\"center\">Handwritten digits classification using neural network</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (4265223854.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[1], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    pip install tensorflow\u001b[0m\n\u001b[1;37m        ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "\n",
    "pip install matplotlib\n",
    "pip intsall numpy\n",
    "pip install pandas\n",
    "pip install seaborn "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tensorflow in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (2.12.0)\n",
      "Requirement already satisfied: tensorflow-intel==2.12.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow) (2.12.0)\n",
      "Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.14.1)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (0.31.0)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.4.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (2.3.0)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (3.3.0)\n",
      "Requirement already satisfied: numpy<1.24,>=1.22 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.23.5)\n",
      "Requirement already satisfied: tensorflow-estimator<2.13,>=2.12.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (2.12.0)\n",
      "Requirement already satisfied: keras<2.13,>=2.12.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (2.12.0)\n",
      "Requirement already satisfied: jax>=0.3.15 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (0.4.10)\n",
      "Requirement already satisfied: h5py>=2.9.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (3.7.0)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.54.2)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (0.4.0)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.16.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (16.0.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: tensorboard<2.13,>=2.12 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (2.12.3)\n",
      "Requirement already satisfied: flatbuffers>=2.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (23.5.26)\n",
      "Requirement already satisfied: packaging in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (22.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (4.4.0)\n",
      "Requirement already satisfied: setuptools in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (65.6.3)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.12.0->tensorflow) (4.23.2)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.12.0->tensorflow) (0.38.4)\n",
      "Requirement already satisfied: scipy>=1.7 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from jax>=0.3.15->tensorflow-intel==2.12.0->tensorflow) (1.10.0)\n",
      "Requirement already satisfied: ml-dtypes>=0.1.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from jax>=0.3.15->tensorflow-intel==2.12.0->tensorflow) (0.1.0)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (0.7.0)\n",
      "Requirement already satisfied: google-auth<3,>=1.6.3 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.19.0)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: google-auth-oauthlib<1.1,>=0.5 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (1.0.0)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.28.1)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.2.2)\n",
      "Requirement already satisfied: rsa<5,>=3.1.4 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (4.9)\n",
      "Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (0.2.8)\n",
      "Requirement already satisfied: urllib3<2.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (1.26.14)\n",
      "Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (5.3.1)\n",
      "Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (1.3.1)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2022.12.7)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (3.4)\n",
      "Requirement already satisfied: charset-normalizer<3,>=2 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.0.4)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.1.1)\n",
      "Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (0.4.8)\n",
      "Requirement already satisfied: oauthlib>=3.0.0 in c:\\users\\jagrutijalkote\\anaconda3\\lib\\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (3.2.2)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "executionInfo": {
     "elapsed": 3632,
     "status": "ok",
     "timestamp": 1683714667905,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "rlx2UtevAW3u"
   },
   "outputs": [],
   "source": [
    "import tensorflow"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OQUUqcRw3999"
   },
   "source": [
    "In this notebook we will classify handwritten digits using a simple neural network which has only input and output layers. We will than add a hidden layer and see how the performance of the model improves"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "executionInfo": {
     "elapsed": 419,
     "status": "ok",
     "timestamp": 1683714697843,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "tFwRkBNo399-"
   },
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 1833,
     "status": "ok",
     "timestamp": 1683714702997,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "WN-vK_9t399_",
    "outputId": "0556602c-c10e-4adc-c39f-eabcf7fcfe88"
   },
   "outputs": [],
   "source": [
    "(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 6,
     "status": "ok",
     "timestamp": 1683714706721,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "_Y8yoZmg39-A",
    "outputId": "2adeca56-8659-48c6-8857-1a7282c0c2f6"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "60000"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 8,
     "status": "ok",
     "timestamp": 1683714708495,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "fxzW950w39-B",
    "outputId": "2fd3dbd3-546c-421c-85cc-45be7992bd32"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10000"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(X_train[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 7,
     "status": "ok",
     "timestamp": 1683714709773,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "VRxBE--m39-C",
    "outputId": "cfb958d9-616d-4007-bb07-3b47cadf6054"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(28, 28)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x25c60fd5120>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AGCGCAEAzBAhAIAZIgQAMEOEAABmiBAAwAwRAgCYIUIAADNECABghggBAMwQIQCAGSIEADBDhAAAZogQAMAMEQIAmCFCAAAzRAgAYIYIAQDMECEAgBkiBAAwQ4QAAGaIEADADBECAJghQgAAM0QIAGBmqPUCruWckyT16JLkjBcDAPDWo0uS/v9/z/vS7yLU1dUlSXpDe41XAgD4PLq6uhQOh/vcJuB+n1TdQpcvX9aZM2cUCoUUCARSXovH4youLlZra6vy8vKMVmiP43AFx+EKjsMVHIcr+sNxcM6pq6tLRUVFuu22vt/16XdXQrfddpvGjBnT5zZ5eXmD+iS7iuNwBcfhCo7DFRyHK6yPw82ugK7igwkAADNECABgJqciFAwGtWbNGgWDQeulmOI4XMFxuILjcAXH4YpcOw797oMJAIDBI6euhAAAAwsRAgCYIUIAADNECABgJqci9MILL6i0tFS33367Jk2apNdff916SbdUTU2NAoFAyiMSiVgvK+sOHjyoBQsWqKioSIFAQLt27Up53TmnmpoaFRUVacSIESovL9fx48dtFptFNzsOS5cuve78mDZtms1is6S2tlZTpkxRKBRSQUGBFi5cqBMnTqRsMxjOh9/nOOTK+ZAzEdqxY4dWrFih1atX6+jRo7r33ntVWVmp06dPWy/tlho/frza2tqSj2PHjlkvKeu6u7s1ceJE1dXV3fD1devWaf369aqrq1NTU5MikYjmzZuXvA/hQHGz4yBJ8+fPTzk/9u4dWPdgbGxsVFVVlQ4fPqz6+nr19PSooqJC3d3dyW0Gw/nw+xwHKUfOB5cjvvrVr7rHH3885bm7777bffe73zVa0a23Zs0aN3HiROtlmJLkXn755eTXly9fdpFIxD377LPJ5z755BMXDofdj3/8Y4MV3hrXHgfnnFuyZIm7//77TdZjpaOjw0lyjY2NzrnBez5cexycy53zISeuhC5evKgjR46ooqIi5fmKigodOnTIaFU2mpubVVRUpNLSUj300EM6efKk9ZJMtbS0qL29PeXcCAaDmj179qA7NySpoaFBBQUFGjdunB599FF1dHRYLymrYrGYJCk/P1/S4D0frj0OV+XC+ZATETp79qx6e3tVWFiY8nxhYaHa29uNVnXrTZ06VVu2bNG+ffv04osvqr29XTNmzFBnZ6f10sxc/ec/2M8NSaqsrNTWrVu1f/9+Pffcc2pqatLcuXOVSCSsl5YVzjlVV1dr5syZKisrkzQ4z4cbHQcpd86HfncX7b5c+6MdnHPXPTeQVVZWJn89YcIETZ8+XXfddZc2b96s6upqw5XZG+znhiQtXrw4+euysjJNnjxZJSUl2rNnjxYtWmS4suxYtmyZ3n33Xb3xxhvXvTaYzofPOg65cj7kxJXQ6NGjNWTIkOv+T6ajo+O6/+MZTEaNGqUJEyaoubnZeilmrn46kHPjetFoVCUlJQPy/Fi+fLl2796tAwcOpPzol8F2PnzWcbiR/no+5ESEhg8frkmTJqm+vj7l+fr6es2YMcNoVfYSiYTee+89RaNR66WYKS0tVSQSSTk3Ll68qMbGxkF9bkhSZ2enWltbB9T54ZzTsmXLtHPnTu3fv1+lpaUprw+W8+Fmx+FG+u35YPihCC/bt293w4YNcz/96U/dr3/9a7dixQo3atQod+rUKeul3TIrV650DQ0N7uTJk+7w4cPuvvvuc6FQaMAfg66uLnf06FF39OhRJ8mtX7/eHT161P32t791zjn37LPPunA47Hbu3OmOHTvmHn74YReNRl08HjdeeWb1dRy6urrcypUr3aFDh1xLS4s7cOCAmz59uvvCF74woI7DE0884cLhsGtoaHBtbW3Jx8cff5zcZjCcDzc7Drl0PuRMhJxz7vnnn3clJSVu+PDh7itf+UrKxxEHg8WLF7toNOqGDRvmioqK3KJFi9zx48etl5V1Bw4ccJKueyxZssQ5d+VjuWvWrHGRSMQFg0E3a9Ysd+zYMdtFZ0Ffx+Hjjz92FRUV7o477nDDhg1zd955p1uyZIk7ffq09bIz6kZ/fklu06ZNyW0Gw/lws+OQS+cDP8oBAGAmJ94TAgAMTEQIAGCGCAEAzBAhAIAZIgQAMEOEAABmiBAAwAwRAgCYIUIAADNECABghggBAMwQIQCAmf8Lw4IYymq+HboAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(X_train[0])   # 1st image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 828,
     "status": "ok",
     "timestamp": 1683714716877,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "Y-clcjQj39-E",
    "outputId": "bd48fa7f-cff9-434f-c612-31cc0404526e"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    6742\n",
       "7    6265\n",
       "3    6131\n",
       "2    5958\n",
       "9    5949\n",
       "0    5923\n",
       "6    5918\n",
       "8    5851\n",
       "4    5842\n",
       "5    5421\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "pd.Series(y_train).value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<module 'matplotlib.pyplot' from 'C:\\\\Users\\\\jagrutijalkote\\\\anaconda3\\\\lib\\\\site-packages\\\\matplotlib\\\\pyplot.py'>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 455
    },
    "executionInfo": {
     "elapsed": 16,
     "status": "ok",
     "timestamp": 1683714711498,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "lfQECGzj39-D",
    "outputId": "b1052b8c-9fb1-4e1a-ad86-72ed905262cb"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x25c64d8e6b0>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 480x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.matshow(X_train[0])  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 7,
     "status": "ok",
     "timestamp": 1683712692212,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "vDh0ZXhX39-E",
    "outputId": "dd580d88-8b9d-4076-c73b-4728b91f3044"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,\n",
       "         18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,\n",
       "        253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,\n",
       "        253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,\n",
       "        253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,\n",
       "        205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,\n",
       "         90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,\n",
       "        190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,\n",
       "        253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,\n",
       "        241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "         81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,\n",
       "        148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,\n",
       "        253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,\n",
       "        253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,\n",
       "        195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,\n",
       "         11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0]], dtype=uint8)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(28, 28)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "executionInfo": {
     "elapsed": 5,
     "status": "ok",
     "timestamp": 1683714717485,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "DyiF9-iv39-E"
   },
   "outputs": [],
   "source": [
    "X_train = X_train / 255     #NORMALIZATION\n",
    "X_test = X_test / 255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 8,
     "status": "ok",
     "timestamp": 1683712970596,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "Hv-YLEsjGUgC",
    "outputId": "3202f790-ad79-48e8-c4b8-84c72a81d51d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 784)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.reshape(60000,28*28).shape\n",
    "#60000,784"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "executionInfo": {
     "elapsed": 535,
     "status": "ok",
     "timestamp": 1683714723615,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "F6alyEI439-F"
   },
   "outputs": [],
   "source": [
    "X_train_flattened = X_train.reshape(len(X_train), 28*28)\n",
    "X_test_flattened = X_test.reshape(len(X_test), 28*28)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "GcGn-7Y939-F",
    "outputId": "86cfa0f5-04fc-4309-ebc8-50606bd9dd0d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 784)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_flattened.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 784)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test_flattened.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "id": "j8dYPYxt39-F",
    "outputId": "8ce10da1-350d-41b5-d8ac-ca9883abcab1"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.01176471, 0.07058824, 0.07058824,\n",
       "       0.07058824, 0.49411765, 0.53333333, 0.68627451, 0.10196078,\n",
       "       0.65098039, 1.        , 0.96862745, 0.49803922, 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.11764706, 0.14117647, 0.36862745, 0.60392157,\n",
       "       0.66666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.99215686, 0.88235294, 0.6745098 , 0.99215686, 0.94901961,\n",
       "       0.76470588, 0.25098039, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.19215686, 0.93333333,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.98431373, 0.36470588,\n",
       "       0.32156863, 0.32156863, 0.21960784, 0.15294118, 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.07058824, 0.85882353, 0.99215686, 0.99215686,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.77647059, 0.71372549,\n",
       "       0.96862745, 0.94509804, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.31372549, 0.61176471, 0.41960784, 0.99215686, 0.99215686,\n",
       "       0.80392157, 0.04313725, 0.        , 0.16862745, 0.60392157,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.05490196,\n",
       "       0.00392157, 0.60392157, 0.99215686, 0.35294118, 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.54509804,\n",
       "       0.99215686, 0.74509804, 0.00784314, 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.04313725, 0.74509804, 0.99215686,\n",
       "       0.2745098 , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.1372549 , 0.94509804, 0.88235294, 0.62745098,\n",
       "       0.42352941, 0.00392157, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.31764706, 0.94117647, 0.99215686, 0.99215686, 0.46666667,\n",
       "       0.09803922, 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.17647059,\n",
       "       0.72941176, 0.99215686, 0.99215686, 0.58823529, 0.10588235,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.0627451 , 0.36470588,\n",
       "       0.98823529, 0.99215686, 0.73333333, 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.97647059, 0.99215686,\n",
       "       0.97647059, 0.25098039, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.18039216, 0.50980392,\n",
       "       0.71764706, 0.99215686, 0.99215686, 0.81176471, 0.00784314,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.15294118,\n",
       "       0.58039216, 0.89803922, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.98039216, 0.71372549, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.09411765, 0.44705882, 0.86666667, 0.99215686, 0.99215686,\n",
       "       0.99215686, 0.99215686, 0.78823529, 0.30588235, 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.09019608, 0.25882353, 0.83529412, 0.99215686,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.77647059, 0.31764706,\n",
       "       0.00784314, 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.07058824, 0.67058824, 0.85882353,\n",
       "       0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.76470588,\n",
       "       0.31372549, 0.03529412, 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.21568627, 0.6745098 ,\n",
       "       0.88627451, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.95686275, 0.52156863, 0.04313725, 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.53333333, 0.99215686, 0.99215686, 0.99215686,\n",
       "       0.83137255, 0.52941176, 0.51764706, 0.0627451 , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        ])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_flattened[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(784,)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_flattened[0].shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7l5eF4kq39-G"
   },
   "source": [
    "<h3 style='color:purple'>Very simple neural network with no hidden layers</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "iLNVARW039-G"
   },
   "source": [
    "<img src=\"https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/1_digits_recognition/digits_nn.jpg?raw=1\" />"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 9,
     "status": "ok",
     "timestamp": 1683715759048,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "L6vx6-X6R34_",
    "outputId": "59f7bc17-5cb3-4069-b1c1-7ba8a8722c96"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 146531,
     "status": "ok",
     "timestamp": 1683716192576,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "0guMSJkW39-G",
    "outputId": "94e7ddc6-aab1-494f-eb13-d06a88ae9f0a"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "1875/1875 [==============================] - 7s 3ms/step - loss: 0.4661 - accuracy: 0.8782\n",
      "Epoch 2/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.3034 - accuracy: 0.9151\n",
      "Epoch 3/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2832 - accuracy: 0.9208\n",
      "Epoch 4/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2732 - accuracy: 0.9238\n",
      "Epoch 5/20\n",
      "1875/1875 [==============================] - 9s 5ms/step - loss: 0.2664 - accuracy: 0.9261\n",
      "Epoch 6/20\n",
      "1875/1875 [==============================] - 8s 4ms/step - loss: 0.2615 - accuracy: 0.9275\n",
      "Epoch 7/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2581 - accuracy: 0.9282\n",
      "Epoch 8/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.2553 - accuracy: 0.9291\n",
      "Epoch 9/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2530 - accuracy: 0.9299\n",
      "Epoch 10/20\n",
      "1875/1875 [==============================] - 5s 2ms/step - loss: 0.2505 - accuracy: 0.9313\n",
      "Epoch 11/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2492 - accuracy: 0.9309\n",
      "Epoch 12/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2476 - accuracy: 0.9324\n",
      "Epoch 13/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2463 - accuracy: 0.9322\n",
      "Epoch 14/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2451 - accuracy: 0.9327\n",
      "Epoch 15/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.2439 - accuracy: 0.9331\n",
      "Epoch 16/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.2431 - accuracy: 0.9329\n",
      "Epoch 17/20\n",
      "1875/1875 [==============================] - 5s 2ms/step - loss: 0.2418 - accuracy: 0.9331\n",
      "Epoch 18/20\n",
      "1875/1875 [==============================] - 4s 2ms/step - loss: 0.2414 - accuracy: 0.9338\n",
      "Epoch 19/20\n",
      "1875/1875 [==============================] - 4s 2ms/step - loss: 0.2406 - accuracy: 0.9332\n",
      "Epoch 20/20\n",
      "1875/1875 [==============================] - 5s 2ms/step - loss: 0.2399 - accuracy: 0.9338\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x25c63ef6740>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# using Keras seqencial(pipeline) method list\n",
    "\n",
    "model = keras.Sequential([\n",
    "    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "model.fit(X_train_flattened, y_train, epochs=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " dense (Dense)               (None, 10)                7850      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 7,850\n",
      "Trainable params: 7,850\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 222ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[5.6476815e-04, 5.4315741e-10, 1.5911874e-03, 9.7002959e-01,\n",
       "        8.1304001e-04, 1.5190595e-01, 4.3483889e-10, 9.9985820e-01,\n",
       "        6.8580128e-02, 6.9592953e-01]], dtype=float32)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred2=model.predict(X_test_flattened[0].reshape(1,784))\n",
    "y_pred2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x25c6407d990>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AGCGCAEAzBAhAIAZIgQAMEOEAABmiBAAwAwRAgCYIUIAADNECABghggBAMwQIQCAGSIEADBDhAAAZogQAMAMEQIAmCFCAAAzRAgAYIYIAQDMECEAgBkiBAAwQ4QAAGaIEADADBECAJghQgAAM0QIAGBmqPUCruWckyT16JLkjBcDAPDWo0uS/v9/z/vS7yLU1dUlSXpDe41XAgD4PLq6uhQOh/vcJuB+n1TdQpcvX9aZM2cUCoUUCARSXovH4youLlZra6vy8vKMVmiP43AFx+EKjsMVHIcr+sNxcM6pq6tLRUVFuu22vt/16XdXQrfddpvGjBnT5zZ5eXmD+iS7iuNwBcfhCo7DFRyHK6yPw82ugK7igwkAADNECABgJqciFAwGtWbNGgWDQeulmOI4XMFxuILjcAXH4YpcOw797oMJAIDBI6euhAAAAwsRAgCYIUIAADNECABgJqci9MILL6i0tFS33367Jk2apNdff916SbdUTU2NAoFAyiMSiVgvK+sOHjyoBQsWqKioSIFAQLt27Up53TmnmpoaFRUVacSIESovL9fx48dtFptFNzsOS5cuve78mDZtms1is6S2tlZTpkxRKBRSQUGBFi5cqBMnTqRsMxjOh9/nOOTK+ZAzEdqxY4dWrFih1atX6+jRo7r33ntVWVmp06dPWy/tlho/frza2tqSj2PHjlkvKeu6u7s1ceJE1dXV3fD1devWaf369aqrq1NTU5MikYjmzZuXvA/hQHGz4yBJ8+fPTzk/9u4dWPdgbGxsVFVVlQ4fPqz6+nr19PSooqJC3d3dyW0Gw/nw+xwHKUfOB5cjvvrVr7rHH3885bm7777bffe73zVa0a23Zs0aN3HiROtlmJLkXn755eTXly9fdpFIxD377LPJ5z755BMXDofdj3/8Y4MV3hrXHgfnnFuyZIm7//77TdZjpaOjw0lyjY2NzrnBez5cexycy53zISeuhC5evKgjR46ooqIi5fmKigodOnTIaFU2mpubVVRUpNLSUj300EM6efKk9ZJMtbS0qL29PeXcCAaDmj179qA7NySpoaFBBQUFGjdunB599FF1dHRYLymrYrGYJCk/P1/S4D0frj0OV+XC+ZATETp79qx6e3tVWFiY8nxhYaHa29uNVnXrTZ06VVu2bNG+ffv04osvqr29XTNmzFBnZ6f10sxc/ec/2M8NSaqsrNTWrVu1f/9+Pffcc2pqatLcuXOVSCSsl5YVzjlVV1dr5syZKisrkzQ4z4cbHQcpd86HfncX7b5c+6MdnHPXPTeQVVZWJn89YcIETZ8+XXfddZc2b96s6upqw5XZG+znhiQtXrw4+euysjJNnjxZJSUl2rNnjxYtWmS4suxYtmyZ3n33Xb3xxhvXvTaYzofPOg65cj7kxJXQ6NGjNWTIkOv+T6ajo+O6/+MZTEaNGqUJEyaoubnZeilmrn46kHPjetFoVCUlJQPy/Fi+fLl2796tAwcOpPzol8F2PnzWcbiR/no+5ESEhg8frkmTJqm+vj7l+fr6es2YMcNoVfYSiYTee+89RaNR66WYKS0tVSQSSTk3Ll68qMbGxkF9bkhSZ2enWltbB9T54ZzTsmXLtHPnTu3fv1+lpaUprw+W8+Fmx+FG+u35YPihCC/bt293w4YNcz/96U/dr3/9a7dixQo3atQod+rUKeul3TIrV650DQ0N7uTJk+7w4cPuvvvuc6FQaMAfg66uLnf06FF39OhRJ8mtX7/eHT161P32t791zjn37LPPunA47Hbu3OmOHTvmHn74YReNRl08HjdeeWb1dRy6urrcypUr3aFDh1xLS4s7cOCAmz59uvvCF74woI7DE0884cLhsGtoaHBtbW3Jx8cff5zcZjCcDzc7Drl0PuRMhJxz7vnnn3clJSVu+PDh7itf+UrKxxEHg8WLF7toNOqGDRvmioqK3KJFi9zx48etl5V1Bw4ccJKueyxZssQ5d+VjuWvWrHGRSMQFg0E3a9Ysd+zYMdtFZ0Ffx+Hjjz92FRUV7o477nDDhg1zd955p1uyZIk7ffq09bIz6kZ/fklu06ZNyW0Gw/lws+OQS+cDP8oBAGAmJ94TAgAMTEQIAGCGCAEAzBAhAIAZIgQAMEOEAABmiBAAwAwRAgCYIUIAADNECABghggBAMwQIQCAmf8Lw4IYymq+HboAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(X_train[0]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 868,
     "status": "ok",
     "timestamp": 1683716232292,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "DNp6JyA_Tcw2",
    "outputId": "0bfe20f3-b119-42b3-be38-c59a74302263"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 2ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(10000, 10)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model.predict(X_test_flattened)\n",
    "y_pred.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 8,
     "status": "ok",
     "timestamp": 1683716296105,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "G9kcnGHBT0Ob",
    "outputId": "229b2c0f-0532-4d6b-e516-04c015e1cd2d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.argmax(y_pred[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "id": "94Je2N_p39-G",
    "outputId": "df670813-918c-46fe-94c9-1a0e1a168588",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 2ms/step - loss: 0.2696 - accuracy: 0.9261\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.2696371078491211, 0.9261000156402588]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(X_test_flattened, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 489,
     "status": "ok",
     "timestamp": 1683716558470,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "gjSF-a8QUo5B",
    "outputId": "a0448f03-4f1a-4521-e8c0-a6548a76af5c"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[7, 2, 1, 0, 4]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# list coprehnsession use\n",
    "y_pred_final=[np.argmax(i) for i in y_pred]\n",
    "y_pred_final[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 8,
     "status": "ok",
     "timestamp": 1683716574930,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "K3wMhrP9U_J9",
    "outputId": "30736b96-88c5-49ae-921c-9cf6c2f9a376"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([7, 2, 1, 0, 4], dtype=uint8)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 7,
     "status": "ok",
     "timestamp": 1683716639898,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "vOaGIms7VDuD",
    "outputId": "8aeac493-e82a-461a-c031-51432f38e5ac"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.94      0.98      0.96       980\n",
      "           1       0.96      0.98      0.97      1135\n",
      "           2       0.93      0.89      0.91      1032\n",
      "           3       0.90      0.91      0.91      1010\n",
      "           4       0.94      0.93      0.93       982\n",
      "           5       0.91      0.86      0.89       892\n",
      "           6       0.95      0.95      0.95       958\n",
      "           7       0.94      0.92      0.93      1028\n",
      "           8       0.87      0.89      0.88       974\n",
      "           9       0.91      0.93      0.92      1009\n",
      "\n",
      "    accuracy                           0.93     10000\n",
      "   macro avg       0.93      0.92      0.92     10000\n",
      "weighted avg       0.93      0.93      0.93     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test,y_pred_final))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "id": "x0nnCgaH39-I",
    "outputId": "bb443bc9-a171-4990-d584-47f4e17d0896"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(95.72222222222221, 0.5, 'Truth')"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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EZGwisxgcCIiIiKioiAfLGkIQb3nrSMiIiIiIkmwokFEREREVACn1xEGKxpERERERCQ4VjSIiIiIiAoomRNMC48VDSIiIiIiEhwrGkREREREBeTLOOuUEFjRICIiIiIiwbGiQURERERUAGedEgYrGkREREREJDhWNIiIiIiICuCsU8JgRYOIiIiIiATHjgYREREREQmOl04RERERERWQz9ltBcGKBhERERERCY4VDSIiIiKiAvLBkoYQWNEgIiIiIiLBsaJBRERERFQAb9gnDFY0iIiIiIhIcKxoEBEREREVwFmnhMGOBhV7GrKS+W7wOidL6giSKJmPNmC+KUTqCJJ4seErqSNIwnDENqkjSCIzN1vqCEQkIHY0iIiIiIgKyJc6gJrgGA0iIiIiIhIcKxpERERERAVw1ilhsKJBRERERESCY0WDiIiIiKgAzjolDFY0iIiIiIhIcKxoEBEREREVwFmnhMGKBhERERERCY4VDSIiIiKiAljREAYrGkREREREJDhWNIiIiIiICpBz1ilBsKJBRERERESCY0eDiIiIiIgEx0uniIiIiIgK4GBwYbCiQUREREREgmNFg4iIiIioAFY0hMGKBhERERERCY4dDSIiIiKiAuQiLp/Ky8sLMpkMnp6e/+SWyzF37lzY2tpCT08PrVu3xp07d5R+LysrCxMmTIC5uTkMDAzg7u6OmJiYz0jyfuxoEBEREREVI9evX8eGDRtQt25dpfYff/wRK1euxE8//YTr16/D2toa7dq1Q1pammIbT09PHDhwALt27YK/vz/S09Ph5uaGvLw8wXOyo0FEREREVEC+TLzlY6Wnp2PAgAHYuHEjypQpo2iXy+VYtWoVZs6ciR49eqB27drw9fXF69evsXPnTgBAamoqNm/ejBUrVsDV1RUNGjTAjh07EBISgjNnzgj136fAjsYnaNHCEQf+2IrHkTeQnRUDd/cOSutnz5qEkODzSEm+j4T4UBw//jsaN24gUVrVGT1qEIICTyM5KQzJSWHwv3AYHTu0kTqWKEqXNsDy5XPx4P4VpL54CL/zB+HgUE/qWCrl3MIRBw/4IOpxIHKzYws979XVg/tXkJMdW2hZs3qR1NEE1aKFI/7YvwWRETeQlRkN967Kj+/GjSuRlRmttFzwOyRR2k/3KisHP566jU5rj8Nx6UEM8jmP0KfJivVyuRzr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\n",
      "text/plain": [
       "<Figure size 1000x700 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sn\n",
    "plt.figure(figsize = (10,7))\n",
    "cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred_final)\n",
    "\n",
    "sn.heatmap(cm, annot=True, fmt='d')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Truth')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uRjXHgzu39-I"
   },
   "source": [
    "<h3 style='color:purple'>Using hidden layer</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 83053,
     "status": "ok",
     "timestamp": 1683717043304,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "LTuFtxq_39-J",
    "outputId": "e738a39f-3eda-41a1-cda3-91e1ff45f02b",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "1875/1875 [==============================] - 7s 3ms/step - loss: 0.2670 - accuracy: 0.9247\n",
      "Epoch 2/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.1180 - accuracy: 0.9649\n",
      "Epoch 3/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0823 - accuracy: 0.9757\n",
      "Epoch 4/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0635 - accuracy: 0.9806\n",
      "Epoch 5/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0506 - accuracy: 0.9847\n",
      "Epoch 6/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0401 - accuracy: 0.9872\n",
      "Epoch 7/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0328 - accuracy: 0.9896\n",
      "Epoch 8/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0267 - accuracy: 0.9918\n",
      "Epoch 9/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0221 - accuracy: 0.9934\n",
      "Epoch 10/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0201 - accuracy: 0.9936\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x25c02c5ad70>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = keras.Sequential([\n",
    "    keras.layers.Dense(100, input_shape=(784,), activation='relu'),\n",
    "    keras.layers.Dense(10, activation='sigmoid')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "model.fit(X_train_flattened, y_train, epochs=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 1168,
     "status": "ok",
     "timestamp": 1683717118197,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "9F3tnOIrWs1D",
    "outputId": "9e57e2e2-105d-4f77-b831-99830d69cb3c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 2ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "10000"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model.predict(X_test_flattened)\n",
    "y_pred_final=[np.argmax(i) for i in y_pred]\n",
    "len(y_pred_final)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 9,
     "status": "ok",
     "timestamp": 1683717145104,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "E3jlaSXkXHbB",
    "outputId": "7cb1f94a-b33b-422d-b686-de398f067020"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.99      0.99       980\n",
      "           1       0.99      0.99      0.99      1135\n",
      "           2       0.96      0.99      0.98      1032\n",
      "           3       0.98      0.97      0.98      1010\n",
      "           4       0.99      0.96      0.98       982\n",
      "           5       0.98      0.98      0.98       892\n",
      "           6       0.97      0.99      0.98       958\n",
      "           7       0.97      0.98      0.98      1028\n",
      "           8       0.98      0.96      0.97       974\n",
      "           9       0.98      0.97      0.97      1009\n",
      "\n",
      "    accuracy                           0.98     10000\n",
      "   macro avg       0.98      0.98      0.98     10000\n",
      "weighted avg       0.98      0.98      0.98     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,y_pred_final))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " dense_1 (Dense)             (None, 100)               78500     \n",
      "                                                                 \n",
      " dense_2 (Dense)             (None, 10)                1010      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 79,510\n",
      "Trainable params: 79,510\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 7308,
     "status": "ok",
     "timestamp": 1683717249242,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "6pSixlYzXQA8",
    "outputId": "1929bbdc-1bb4-4b13-d7c0-f34d15dd3cf2"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1875/1875 [==============================] - 5s 2ms/step\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00      5923\n",
      "           1       1.00      1.00      1.00      6742\n",
      "           2       0.99      1.00      0.99      5958\n",
      "           3       1.00      0.99      1.00      6131\n",
      "           4       1.00      0.99      1.00      5842\n",
      "           5       1.00      1.00      1.00      5421\n",
      "           6       0.99      1.00      1.00      5918\n",
      "           7       0.99      1.00      1.00      6265\n",
      "           8       1.00      0.99      0.99      5851\n",
      "           9       1.00      1.00      1.00      5949\n",
      "\n",
      "    accuracy                           1.00     60000\n",
      "   macro avg       1.00      1.00      1.00     60000\n",
      "weighted avg       1.00      1.00      1.00     60000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred_train=model.predict(X_train_flattened)\n",
    "y_pred_train_final=[np.argmax(i) for i in y_pred_train]\n",
    "print(classification_report(y_train,y_pred_train_final))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "executionInfo": {
     "elapsed": 6,
     "status": "ok",
     "timestamp": 1683717566383,
     "user": {
      "displayName": "Shivam Pandey",
      "userId": "09364170959834321998"
     },
     "user_tz": -330
    },
    "id": "VWQxm2kHYtLO"
   },
   "outputs": [],
   "source": [
    "\n",
    "#model deploy & load\n",
    "model.save(\"final_model.h5\")   # online site pe kam krta code  herte h5 is code.  in binary format\n",
    "#model1=tf.Keras.load_model(\"#path dedo model ka jo colab me save ki hai\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "id": "3Guf7bWC39-J",
    "outputId": "333837f9-4506-4afb-aeae-f6596a5a138b",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 3ms/step - loss: 0.0817 - accuracy: 0.9782\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.08169794082641602, 0.9782000184059143]"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(X_test_flattened,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "id": "i4-Ohs8b39-J",
    "outputId": "23f86cc4-79d3-4690-d65e-716a7bb4e2bc"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 2ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(95.72222222222221, 0.5, 'Truth')"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1000x700 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "y_predicted = model.predict(X_test_flattened)\n",
    "y_predicted_labels = [np.argmax(i) for i in y_predicted]\n",
    "cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)\n",
    "\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(cm, annot=True, fmt='d')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Truth')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "U_HzYZlp39-J"
   },
   "source": [
    "<h3 style='color:purple'>Using Flatten layer so that we don't have to call .reshape on input dataset</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "id": "pRpwYNzc39-J",
    "outputId": "a89900f5-1a78-4c16-d303-6baaa4adf3b0",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "1875/1875 [==============================] - 7s 3ms/step - loss: 0.2807 - accuracy: 0.9197\n",
      "Epoch 2/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.1288 - accuracy: 0.9617\n",
      "Epoch 3/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0900 - accuracy: 0.9731\n",
      "Epoch 4/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0670 - accuracy: 0.9800\n",
      "Epoch 5/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0540 - accuracy: 0.9833\n",
      "Epoch 6/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0425 - accuracy: 0.9873\n",
      "Epoch 7/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0352 - accuracy: 0.9890\n",
      "Epoch 8/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0302 - accuracy: 0.9914\n",
      "Epoch 9/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0255 - accuracy: 0.9921\n",
      "Epoch 10/10\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0203 - accuracy: 0.9939\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x25c054e5840>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = keras.Sequential([\n",
    "    keras.layers.Flatten(input_shape=(28, 28)),\n",
    "    keras.layers.Dense(100, activation='relu'),\n",
    "    keras.layers.Dense(10, activation='sigmoid') # sigmoid or softmax use karo\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "model.fit(X_train, y_train, epochs=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_2\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " flatten (Flatten)           (None, 784)               0         \n",
      "                                                                 \n",
      " dense_3 (Dense)             (None, 100)               78500     \n",
      "                                                                 \n",
      " dense_4 (Dense)             (None, 10)                1010      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 79,510\n",
      "Trainable params: 79,510\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "id": "-0romdSJ39-K",
    "outputId": "e22408d7-3754-428b-b417-58af72fd5d5c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 2ms/step - loss: 0.0848 - accuracy: 0.9766\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.08481539040803909, 0.9765999913215637]"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(X_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "gpuClass": "standard",
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
