# Data
In order to reproduce the DMVN performance score on the paried CASIA dataset, one has to manually download the **CASIA TIDEv2.0** dataset (http://forensics.idealtest.org/casiav2/), and decompress it under the current directory (i.e. `data/`).

### Step-by-step instructions

-1. Go to website http://forensics.idealtest.org/casiav2/join/

-2. Click the **Register** button close the webpage bottom.

-3. Register and create your account, and download `CASIA2.rar` (2.2GB)

-4. Unrar `CASIA2.rar` to the data directory, i.e. `data/` using the following command

    mkdir CASIA2; unrar e CASIA2.rar ./CASIA2 
-5. Make sure that the number of items under `CASIA2` matches the number below

    ls CASIA2 | wc -l
    12615
-6. Create a full path image list for all images in CASIA2, namely

    find $PWD/CASIA2 -name "*.*" > casia2.images
-7. Use predefined ID pairs to create the paired CASIA2 dataset
 
    python script/create_paired_casia_dataset.py
-8. The resulting paired CASIA2 dataset can be seen in file `local_paired_CASIA_files.csv`

