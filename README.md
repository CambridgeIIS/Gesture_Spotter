## Gesture Spotter: A Rapid Prototyping Tool for Key Gesture Spotting in Virtual and Augmented Reality Applications
The code in this repository is an implementation of the Graphical User Interface (GUI) and Application Programming Interface (API) described in the TVCG 2022 paper:

	Junxiao Shen, John Dudley, Geroge Mo, Per Ola Kristensson,
	Gesture Spotter: A Rapid Prototyping Tool for Key Gesture Spotting in Virtual and Augmented Reality Applications, 
	TVCG 2022
    
    
We present Gesture Spotter: A Rapid Prototyping Tool for Key Gesture Spotting in Virtual and Augmented Reality Applications for developers. 

In this paper we examine the task of key gesture spotting: accurate and timely online recognition of hand gestures. We specifically seek to address two key challenges faced by developers when integrating key gesture spotting functionality into their applications. These are: i) achieving high accuracy and zero or negative activation lag with single-time activation; and ii) avoiding the requirement for deep domain expertise in machine learning. We address the first challenge by proposing a key gesture spotting architecture consisting of a novel gesture classifier model and a novel single-time activation algorithm. This key gesture spotting architecture was evaluated on four separate hand skeleton gesture datasets, and achieved high recognition accuracy with early detection. We address the second challenge by encapsulating different data processing and augmentation strategies, as well as the proposed key gesture spotting architecture, into a graphical user interface and an application programming interface. Two user studies demonstrate that developers are able to efficiently construct custom recognizers using both the graphical user interface and the application programming interface.
 

### Trained generative model for data augmentation: 
unzip generator_model.zip


#### GUI Usage
- **Setting dependencies** 
Use pip to set up the environments.
```
  $ pip install -r requrements.txt
```

- **Launch the GUI**
```
  $ python gui_main.py
```

#### API Examples
- **Synthesize the data for data augmentation**
```
  Synthesize_Data.ipynb
```

- **Model training and evaluation**
```
  train_eval.ipynb
```

- **Online evaluation of the model**
```
  eval_online.ipynb
```

- **Gesture visualization**
```
  GestureDisplay.ipynb
```


