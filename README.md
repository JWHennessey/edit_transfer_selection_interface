# edit_transfer_selection_interface

## Preresquites
- cmake 3.2 or higher
- c++14 compatible compiler (tested using clang 3.4)

## Download, Compile and Run
```
git clone git@github.com:JWHennessey/edit_transfer_selection_interface.git
cd edit_transfer_selection_interface
mkdir build
cd build
cmake ..
make -j
./edit_transfer_interface
```
Warning: The repo is quite large as it comes with several libraries as dependences. However, this hopefully means these will be no dependency issues. 

## Selection Tool Usage

- On the right hand panel, press the folder icon to open a scene
- Select the scene .txt file you you like to view (demo comes with car and wine scenes)
- On the right side panel, select the "Selected Channel" tab. This will show all of the render channels
- On the image use the mouse to click and drag to select a region (alternatively you can use a none rectangle selection mask, described in more detail below)
- After a few seconds of processing the channels the render channels icons will be ordered from most to least unique. The channels that the method thinks the user would like to edit will be outlined in green. Alternatively you can hold the 'a' key to solo these selected channels in the main window. 


Selection Masks
- If the region you would like to select isn't well approximated by a rectangle you can load in a selection mask.
- On the right side panel, press the far rectangle button
- Locate your selection mask (car scene has some examples) and open it
- Selection process will be same as above

## Scene Folder Structure

To create a new scene you need to follow the following folder structure and naming conversion.

Where *foo* is used this can be replaced with relevant channel names. 

- folder/
- - _.*scene name*.txt
- - _.exr (Beauty Pass)
- - _.\**channel1\**.exr (Render Channel 1)
- - _.*channel2*.exr (Render Channel 2)
- - png/
- - - _.channel1.png (Render Channel 1 png icon)
- - - _.channel2.png (Render Channel 1 png icon)
- - material_ids
- - - _.Material_ID*i*.png (All material IDs output to individual channels pure red)
- - selection_polygon (Helper folder to store any selection masks)
- - candidate_patches (The first time you load a scene these will be created and saved here)
- - edits (Other parts of the interface allow you to make edits, these will be stored here)

To help create a new scene we provide a blank_template folder


