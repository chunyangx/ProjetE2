Texture synthesis using optimization
------------------------------------

Students:

  - ChunYang Xiao, chunyang.xiao@student.ecp.fr
  - Loïc Fagot-Bouquet, loic.fagot-bouquet@student.ecp.fr

Install dependencies
--------------------

OpenCV :

  Download OpenCV at http://opencv.org/downloads.html (version 2.4.2).
  Extract the files in a target directory and let OPENCV denote this directory.

  cd OPENCV
  mkdir Build
  cd Build\
  cmake ..
  make
  sudo make install

Build
-----

  mkdir build/
  cd build/
  cmake ..
  make

Run the texture generation
--------------------------

  cd build
  mkdir test
  ./Texture_Synthesis input_image test/

  The results would be added into test directory

  There are no options for Texture_Synthesis to run different optimization process, to do this, go to Texure_synthesis.cpp, about line 63, you could manually comment the default basic solver and call others. Rebuild the project as specied in Build section and run the same command.
