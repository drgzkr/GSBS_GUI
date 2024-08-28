# GSBS_GUI
A make-shift tkinter GUI for viewing results of the Greedy State Boundary Search algorithm from 

[pip install statesegmentation](https://github.com/lgeerligs/statesegmentation)

Has horrible implementation, but is convenient.
I don't know what I'm doing and have no idea how to build a proper GUI. No elegance here.

## Requirements
statesegmentation package needs to be installed

## Usage
Simply 
```bash
python run run_gsbs_gui.py

- Can perform GSBS if the path of a 2d numpy array of shape (ntimepoints,nchannels) is entered with the given parameters.
- Can also display results of a saved GSBS object.
- Lets you explore segmentation results from the whole search, plotting boundary locations and average state patterns.
- If you have as many images as there are timepoints in your data with 1-1 correspondance, and the images are named 'n.jpg', can show images before and after each boudnary.

![image](https://github.com/user-attachments/assets/74192d8d-0f8e-45fd-94a0-848ba51d13e4)

# ToDo:
- [ ] Fix figure scale problem for MacOs
- [ ] Fix save gsbs object button
- [ ] Expand input parameters
- [ ] Fix needing to adjsut kmax when loading saved results
- [ ] Show boundary strengths
- [ ] add more to do's
- [ ] Find someone who knows what they are doing for a proper version
