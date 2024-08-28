# GSBS_GUI
A make-shift tkinter GUI for viewing results of the Greedy State Boundary Search algorithm from 

[pip install statesegmentation](https://github.com/lgeerligs/statesegmentation)

Has horrible implementation, but is convenient.
I don't know what I'm doing and have no idea how to build a proper GUI. No elegance here.

Can perform GSBS if a 2d numpy array of shape (ntimepoints,nchannels) given the parameters.
Can also display results of a GSBS object if saved after running.

If you have as many images as there are timepoints in your data with 1-1 correspondance, and the images are named 'n.jpg', can show images before and after each boudnary.

ToDo:
- [ ] Add save gsbs object button
- [ ] Expand input parameters
- [ ] Fix needing to adjsut kmax when loading saved results
- [ ] add more to do's
- [ ] Find someone who knows what they are doing for a proper version

![image](https://github.com/user-attachments/assets/42bd4715-ba85-4747-82a6-9a9ba0a34b7d)

